import { test, describe, before, after } from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { toOllamaChat, toChatCompletion, toChunks, TranslateError } from '../src/translate.js';
import { truncationLimits, looksTruncated, rescue, holdIndex } from '../src/guard.js';
import { createGuard } from '../src/server.js';

const TOOLS = [{ type: 'function', function: { name: 'bash', parameters: { type: 'object', properties: { command: { type: 'string' } } } } }];

describe('translate: OpenAI -> Ollama native', () => {
  test('num_ctx is applied, and Ollama /v1 defaults are mirrored', () => {
    const { request } = toOllamaChat({ model: 'm', messages: [{ role: 'user', content: 'hi' }] }, { numCtx: 32768 });
    assert.deepEqual(request.options, { temperature: 1.0, top_p: 1.0, num_ctx: 32768 });
    assert.equal(request.stream, false);
  });
  test('modelDefaults leaves unsent temperature/top_p to the model', () => {
    const { request } = toOllamaChat({ model: 'm', messages: [], temperature: 0.2 }, { numCtx: 8192, modelDefaults: true });
    assert.deepEqual(request.options, { temperature: 0.2, num_ctx: 8192 });
  });
  test("the client's own num_ctx wins — the one /v1 silently drops", () => {
    assert.equal(toOllamaChat({ model: 'm', messages: [], options: { num_ctx: 65536 } }, { numCtx: 32768 }).request.options.num_ctx, 65536);
    assert.equal(toOllamaChat({ model: 'm', messages: [], num_ctx: 16384 }, { numCtx: 32768 }).request.options.num_ctx, 16384);
  });
  test('tool calls: string arguments become objects; tool results get their tool name', () => {
    const { request } = toOllamaChat({ model: 'm', messages: [
      { role: 'assistant', content: null, tool_calls: [{ id: 'c1', type: 'function', function: { name: 'bash', arguments: '{"command":"ls"}' } }] },
      { role: 'tool', tool_call_id: 'c1', content: 'a.txt' },
    ] });
    assert.deepEqual(request.messages[0].tool_calls[0].function, { index: 0, name: 'bash', arguments: { command: 'ls' } });
    assert.equal(request.messages[1].tool_name, 'bash');
    assert.equal(request.messages[1].tool_call_id, 'c1');
  });
  test('content parts, images, stop, max_tokens, response_format, reasoning', () => {
    const { request } = toOllamaChat({ model: 'm', stop: 'END', max_tokens: 50, response_format: { type: 'json_object' }, reasoning_effort: 'low',
      messages: [{ role: 'user', content: [{ type: 'text', text: 'look' }, { type: 'image_url', image_url: { url: 'data:image/png;base64,AAAA' } }] }] });
    assert.deepEqual(request.messages, [{ role: 'user', content: 'look' }, { role: 'user', content: '', images: ['AAAA'] }]);
    assert.deepEqual(request.options.stop, ['END']);
    assert.equal(request.options.num_predict, 50);
    assert.equal(request.format, 'json');
    assert.equal(request.think, 'low');
  });
  test('errors Ollama would also reject', () => {
    assert.throws(() => toOllamaChat({ model: 'm', messages: [{ role: 'user', content: [{ type: 'image_url', image_url: 'https://x/y.png' }] }] }), TranslateError);
    assert.throws(() => toOllamaChat({ model: 'm', messages: [], reasoning_effort: 'extreme' }), TranslateError);
  });
});

describe('translate: Ollama -> OpenAI', () => {
  test('completion with tool calls', () => {
    const c = toChatCompletion('chatcmpl-abc12345', { model: 'm', created_at: '2026-09-29T00:00:00Z', done_reason: 'stop', prompt_eval_count: 10, eval_count: 3,
      message: { role: 'assistant', content: '', thinking: 'hmm', tool_calls: [{ function: { index: 0, name: 'bash', arguments: { command: 'ls' } } }] } });
    assert.equal(c.choices[0].finish_reason, 'tool_calls');
    assert.equal(c.choices[0].message.tool_calls[0].function.arguments, '{"command":"ls"}');
    assert.equal(c.choices[0].message.reasoning, 'hmm');
    assert.deepEqual(c.usage, { prompt_tokens: 10, completion_tokens: 3, total_tokens: 13 });
  });
  test('a chunk carrying thinking and content is split in two, like Ollama does', () => {
    const cs = toChunks('id', { model: 'm', message: { content: 'x', thinking: 't' } });
    assert.equal(cs.length, 2);
    assert.equal(cs[0].choices[0].delta.reasoning, 't');
    assert.equal(cs[1].choices[0].delta.content, 'x');
    assert.equal(cs[1].choices[0].delta.reasoning, undefined);
  });
});

describe('guard helpers', () => {
  test("Ollama's truncation fingerprint (measured: 2050 at num_ctx 4096)", () => {
    assert.deepEqual(truncationLimits(4096), [2050, 2051]);
    assert.equal(looksTruncated(2050, 4096), true);
    assert.equal(looksTruncated(2049, 4096), false);
    assert.equal(looksTruncated(2050, null), false);
  });
  test('rescue turns a Qwen text call into a real one and keeps the prose', () => {
    const r = rescue("I'll run it.\n<function=bash>\n<parameter=command>\nsha256sum a.tgz\n</parameter>\n</function>", TOOLS);
    assert.equal(r.toolCalls[0].function.name, 'bash');
    assert.equal(r.toolCalls[0].function.arguments, '{"command":"sha256sum a.tgz"}');
    assert.equal(r.content, "I'll run it.");
  });
  test('rescue never invents a call to a tool that was not offered', () => {
    assert.deepEqual(rescue('<function=rm><parameter=path>/</parameter></function>', TOOLS).toolCalls, []);
    assert.deepEqual(rescue('just prose', TOOLS).toolCalls, []);
  });
  test('holdIndex finds the first character that could begin markup', () => {
    assert.equal(holdIndex('abc <function='), 4);
    assert.equal(holdIndex('plain text'), -1);
  });
});

// ── End to end, through real HTTP, against a fake Ollama ─────────────────────
describe('proxy end to end', () => {
  let fake, guard, base, lastUpstream;
  let reply = {};
  before(async () => {
    fake = http.createServer(async (req, res) => {
      let b = ''; for await (const c of req) b += c;
      if (req.url === '/api/tags') { res.writeHead(200, { 'content-type': 'application/json' }); return res.end('{"models":[{"name":"m"}]}'); }
      lastUpstream = JSON.parse(b || '{}');
      if (!lastUpstream.stream) { res.writeHead(200, { 'content-type': 'application/json' }); return res.end(JSON.stringify(reply.final)); }
      res.writeHead(200, { 'content-type': 'application/x-ndjson' });
      for (const line of reply.stream) res.write(JSON.stringify(line) + '\n');
      res.end();
    });
    await new Promise(r => fake.listen(0, '127.0.0.1', r));
    guard = createGuard({ upstream: `http://127.0.0.1:${fake.address().port}`, numCtx: 4096, log: () => {} });
    await new Promise(r => guard.listen(0, '127.0.0.1', r));
    base = `http://127.0.0.1:${guard.address().port}`;
  });
  after(() => { fake.close(); guard.close(); });

  const post = (body) => fetch(`${base}/v1/chat/completions`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });

  test('non-streaming: num_ctx reaches Ollama; truncation is flagged; a text call is rescued', async () => {
    reply.final = { model: 'm', done: true, done_reason: 'stop', prompt_eval_count: 2050, eval_count: 9,
      message: { role: 'assistant', content: 'Sure.\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>' } };
    const r = await post({ model: 'm', messages: [{ role: 'user', content: 'list files' }], tools: TOOLS });
    const j = await r.json();
    assert.equal(lastUpstream.options.num_ctx, 4096);
    assert.equal(r.headers.get('x-ollama-guard-truncated'), 'prompt cut to 2050 tokens');
    assert.equal(r.headers.get('x-ollama-guard-rescued'), '1');
    assert.equal(j.choices[0].finish_reason, 'tool_calls');
    assert.equal(j.choices[0].message.content, 'Sure.');
    assert.equal(JSON.parse(j.choices[0].message.tool_calls[0].function.arguments).command, 'ls');
  });

  test('streaming without tools: plain SSE, finish reason, usage, [DONE]', async () => {
    reply.stream = [{ model: 'm', message: { role: 'assistant', content: 'Hel' } }, { model: 'm', message: { role: 'assistant', content: 'lo' } },
      { model: 'm', done: true, done_reason: 'stop', prompt_eval_count: 5, eval_count: 2, message: { role: 'assistant', content: '' } }];
    const r = await post({ model: 'm', stream: true, stream_options: { include_usage: true }, messages: [{ role: 'user', content: 'hi' }] });
    const events = (await r.text()).split('\n\n').filter(e => e.startsWith('data: ')).map(e => e.slice(6));
    assert.equal(events.at(-1), '[DONE]');
    const chunks = events.slice(0, -1).map(JSON.parse);
    assert.equal(chunks.map(c => c.choices[0]?.delta?.content || '').join(''), 'Hello');
    assert.equal(chunks.find(c => c.choices[0]?.finish_reason)?.choices[0].finish_reason, 'stop');
    assert.equal(chunks.at(-1).usage.total_tokens, 7);
  });

  test('streaming with tools: prose streams, the text call arrives as a real tool call', async () => {
    reply.stream = [{ model: 'm', message: { content: 'Running it now. ' } }, { model: 'm', message: { content: '<function=bash>\n<parameter=command>\n' } },
      { model: 'm', message: { content: 'ls\n</parameter>\n</function>' } }, { model: 'm', done: true, done_reason: 'stop', prompt_eval_count: 9, eval_count: 9, message: { content: '' } }];
    const r = await post({ model: 'm', stream: true, messages: [{ role: 'user', content: 'x' }], tools: TOOLS });
    const chunks = (await r.text()).split('\n\n').filter(e => e.startsWith('data: {')).map(e => JSON.parse(e.slice(6)));
    const text = chunks.map(c => c.choices[0]?.delta?.content || '').join('');
    assert.equal(text.includes('<function='), false, 'markup must not leak to the client');
    assert.equal(text.trim(), 'Running it now.');
    const call = chunks.find(c => c.choices[0]?.delta?.tool_calls)?.choices[0].delta.tool_calls[0];
    assert.equal(call.function.name, 'bash');
    assert.equal(chunks.find(c => c.choices[0]?.finish_reason).choices[0].finish_reason, 'tool_calls');
  });

  test('streaming with tools but no call: held text is released intact', async () => {
    reply.stream = [{ model: 'm', message: { content: 'Use a [bracket] ' } }, { model: 'm', message: { content: 'like this.' } },
      { model: 'm', done: true, done_reason: 'stop', prompt_eval_count: 3, eval_count: 3, message: { content: '' } }];
    const r = await post({ model: 'm', stream: true, messages: [{ role: 'user', content: 'x' }], tools: TOOLS });
    const chunks = (await r.text()).split('\n\n').filter(e => e.startsWith('data: {')).map(e => JSON.parse(e.slice(6)));
    assert.equal(chunks.map(c => c.choices[0]?.delta?.content || '').join(''), 'Use a [bracket] like this.');
  });

  test('--on-overflow error refuses a prompt that clearly will not fit', async () => {
    const strict = createGuard({ upstream: `http://127.0.0.1:${fake.address().port}`, numCtx: 100, onOverflow: 'error', log: () => {} });
    await new Promise(r => strict.listen(0, '127.0.0.1', r));
    const r = await fetch(`http://127.0.0.1:${strict.address().port}/v1/chat/completions`, { method: 'POST', body: JSON.stringify({ model: 'm', messages: [{ role: 'user', content: 'x'.repeat(4000) }] }) });
    strict.close();
    assert.equal(r.status, 400);
    assert.equal((await r.json()).error.type, 'context_length_exceeded');
  });

  test('every other endpoint passes straight through', async () => {
    const r = await fetch(`${base}/api/tags`);
    assert.deepEqual(await r.json(), { models: [{ name: 'm' }] });
  });
});
