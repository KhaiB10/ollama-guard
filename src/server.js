// server.js — the proxy. /v1/chat/completions is translated to /api/chat (so
// num_ctx is honoured); everything else is passed through untouched.
import http from 'node:http';
import { randomBytes } from 'node:crypto';
import { toOllamaChat, toChatCompletion, toChunks, usageOf, TranslateError } from './translate.js';
import { looksTruncated, estimateTokens, rescue, holdIndex } from './guard.js';

const HOP = new Set(['connection', 'keep-alive', 'transfer-encoding', 'content-length', 'host', 'upgrade']);

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on('data', c => chunks.push(c));
    req.on('end', () => resolve(Buffer.concat(chunks)));
    req.on('error', reject);
  });
}

function sendJson(res, status, obj, headers = {}) {
  const body = JSON.stringify(obj);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(body), ...headers });
  res.end(body);
}

const openaiError = (message, type = 'invalid_request_error') => ({ error: { message, type } });

async function passthrough(req, res, body, cfg) {
  const headers = {};
  for (const [k, v] of Object.entries(req.headers)) if (!HOP.has(k)) headers[k] = v;
  let up;
  try {
    up = await fetch(new URL(req.url, cfg.upstream), {
      method: req.method, headers, body: ['GET', 'HEAD'].includes(req.method) ? undefined : body, duplex: 'half',
    });
  } catch (e) {
    return sendJson(res, 502, openaiError(`ollama-guard: upstream ${cfg.upstream} unreachable: ${e.message}`, 'upstream_error'));
  }
  const out = {};
  up.headers.forEach((v, k) => { if (!HOP.has(k)) out[k] = v; });
  res.writeHead(up.status, out);
  if (!up.body) return res.end();
  for await (const chunk of up.body) res.write(chunk);
  res.end();
}

async function* ndjson(stream) {
  const dec = new TextDecoder();
  let buf = '';
  for await (const chunk of stream) {
    buf += dec.decode(chunk, { stream: true });
    let nl;
    while ((nl = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, nl).trim(); buf = buf.slice(nl + 1);
      if (line) yield JSON.parse(line);
    }
  }
  if (buf.trim()) yield JSON.parse(buf.trim());
}

async function chat(req, res, raw, cfg, log) {
  const t0 = Date.now();
  let body;
  try { body = JSON.parse(raw.toString('utf8') || '{}'); } catch { return sendJson(res, 400, openaiError('invalid JSON body')); }

  let tr;
  try { tr = toOllamaChat(body, { numCtx: cfg.numCtx, modelDefaults: cfg.modelDefaults }); }
  catch (e) {
    if (e instanceof TranslateError) return sendJson(res, 400, openaiError(e.message));
    throw e;
  }
  const { request, numCtx } = tr;
  const tools = request.tools || [];
  const est = estimateTokens(request);
  const guardHeaders = { 'x-ollama-guard-num-ctx': String(numCtx ?? 'server-default') };
  const entry = { model: request.model, num_ctx: numCtx, est_prompt_tokens: est, stream: request.stream, tools: tools.length };

  if (numCtx && est > numCtx) {
    const msg = `ollama-guard: this prompt is ~${est} tokens but num_ctx is ${numCtx}; Ollama would silently drop the middle of it (tool definitions included). Raise --num-ctx.`;
    if (cfg.onOverflow === 'error') { log({ ...entry, blocked: 'overflow' }); return sendJson(res, 400, openaiError(msg, 'context_length_exceeded'), guardHeaders); }
    guardHeaders['x-ollama-guard-warning'] = 'prompt-likely-exceeds-num-ctx';
  }

  let up;
  try {
    up = await fetch(new URL('/api/chat', cfg.upstream), {
      method: 'POST', headers: { 'content-type': 'application/json', ...(req.headers.authorization ? { authorization: req.headers.authorization } : {}) },
      body: JSON.stringify(request),
    });
  } catch (e) {
    return sendJson(res, 502, openaiError(`ollama-guard: upstream ${cfg.upstream} unreachable: ${e.message}`, 'upstream_error'));
  }
  if (!up.ok) {
    const text = await up.text();
    let msg = text; try { msg = JSON.parse(text).error ?? text; } catch { /* keep text */ }
    log({ ...entry, upstream_status: up.status, error: String(msg).slice(0, 200) });
    return sendJson(res, up.status, openaiError(String(msg), 'upstream_error'), guardHeaders);
  }

  const id = `chatcmpl-${randomBytes(6).toString('hex')}`;

  // ── Non-streaming ──────────────────────────────────────────────────────────
  if (!request.stream) {
    const r = await up.json();
    const completion = toChatCompletion(id, r);
    const truncated = looksTruncated(r.prompt_eval_count, numCtx);
    const msg = completion.choices[0].message;
    let rescued = 0;
    if (cfg.rescue && !msg.tool_calls) {
      const rs = rescue(msg.content, tools);
      if (rs.toolCalls.length) {
        msg.tool_calls = rs.toolCalls; msg.content = rs.content;
        completion.choices[0].finish_reason = 'tool_calls';
        rescued = rs.toolCalls.length;
      }
    }
    if (truncated) guardHeaders['x-ollama-guard-truncated'] = `prompt cut to ${r.prompt_eval_count} tokens`;
    if (rescued) guardHeaders['x-ollama-guard-rescued'] = String(rescued);
    log({ ...entry, prompt_tokens: r.prompt_eval_count, truncated, rescued, ms: Date.now() - t0 });
    if (truncated && cfg.onOverflow === 'error') {
      return sendJson(res, 400, openaiError(`ollama-guard: Ollama truncated this prompt to ${r.prompt_eval_count} tokens (num_ctx ${numCtx}). Raise --num-ctx.`, 'context_length_exceeded'), guardHeaders);
    }
    return sendJson(res, 200, completion, guardHeaders);
  }

  // ── Streaming ──────────────────────────────────────────────────────────────
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...guardHeaders });
  const send = (obj) => res.write(`data: ${JSON.stringify(obj)}\n\n`);
  const holding = cfg.rescue && tools.length > 0;
  let text = '', streamedUpTo = 0, toolCallSent = false, last = null, model = request.model;

  for await (const r of ndjson(up.body)) {
    if (r.error) { res.write(`data: ${JSON.stringify(openaiError(String(r.error), 'upstream_error'))}\n\n`); break; }
    model = r.model || model;
    const content = r.message?.content || '';
    if (r.done) { last = r; }
    if (r.message?.tool_calls?.length) toolCallSent = true;

    if (!holding) {
      if (!r.done) for (const c of toChunks(id, r, { toolCallSent })) send(c);
      continue;
    }
    // Hold text back from the first character that could start tool-call
    // markup; stream everything before it.
    text += content;
    const h = holdIndex(text);
    const safeEnd = h < 0 ? text.length : h;
    if (safeEnd > streamedUpTo || r.message?.thinking || r.message?.tool_calls?.length) {
      const piece = text.slice(streamedUpTo, Math.max(safeEnd, streamedUpTo));
      streamedUpTo = Math.max(safeEnd, streamedUpTo);
      const clone = { ...r, done_reason: undefined, message: { ...r.message, content: piece } };
      if (!r.done && (piece || clone.message.thinking || clone.message.tool_calls?.length)) for (const c of toChunks(id, clone, { toolCallSent })) send(c);
    }
  }

  let rescued = 0;
  if (holding) {
    const rest = text.slice(streamedUpTo);
    const rs = !toolCallSent ? rescue(text, tools) : { toolCalls: [] };
    if (rs.toolCalls.length) {
      rescued = rs.toolCalls.length;
      const prose = rs.content.slice(streamedUpTo);
      if (prose) send({ id, object: 'chat.completion.chunk', created: Math.floor(Date.now() / 1000), model, system_fingerprint: 'fp_ollama', choices: [{ index: 0, delta: { role: 'assistant', content: prose }, finish_reason: null }] });
      send({ id, object: 'chat.completion.chunk', created: Math.floor(Date.now() / 1000), model, system_fingerprint: 'fp_ollama', choices: [{ index: 0, delta: { role: 'assistant', tool_calls: rs.toolCalls }, finish_reason: null }] });
      toolCallSent = true;
    } else if (rest) {
      send({ id, object: 'chat.completion.chunk', created: Math.floor(Date.now() / 1000), model, system_fingerprint: 'fp_ollama', choices: [{ index: 0, delta: { role: 'assistant', content: rest }, finish_reason: null }] });
    }
  }
  if (last) {
    const finalChunk = toChunks(id, { ...last, message: { role: 'assistant', content: '' } }, { toolCallSent }).pop();
    send(finalChunk);
    const truncated = looksTruncated(last.prompt_eval_count, numCtx);
    if (truncated) res.write(`: ollama-guard: prompt was truncated to ${last.prompt_eval_count} tokens (num_ctx ${numCtx})\n\n`);
    if (body.stream_options?.include_usage) send({ id, object: 'chat.completion.chunk', created: Math.floor(Date.now() / 1000), model, system_fingerprint: 'fp_ollama', choices: [], usage: usageOf(last) });
    log({ ...entry, prompt_tokens: last.prompt_eval_count, truncated, rescued, ms: Date.now() - t0 });
  }
  res.write('data: [DONE]\n\n');
  res.end();
}

export function createGuard(opts = {}) {
  const cfg = {
    upstream: opts.upstream || 'http://127.0.0.1:11434',
    numCtx: opts.numCtx ?? 32768,
    rescue: opts.rescue !== false,
    onOverflow: opts.onOverflow || 'warn',
    modelDefaults: !!opts.modelDefaults,
  };
  const log = opts.log || ((e) => console.log(`[ollama-guard] ${JSON.stringify(e)}`));
  return http.createServer(async (req, res) => {
    try {
      const raw = await readBody(req);
      const path = new URL(req.url, 'http://x').pathname;
      if (req.method === 'POST' && path === '/v1/chat/completions') return await chat(req, res, raw, cfg, log);
      return await passthrough(req, res, raw, cfg);
    } catch (e) {
      if (!res.headersSent) sendJson(res, 500, openaiError(`ollama-guard: ${e.message}`, 'server_error'));
      else res.end();
    }
  });
}
