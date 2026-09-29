// translate.js — OpenAI /v1/chat/completions  <->  Ollama native /api/chat.
//
// Why this exists: Ollama's OpenAI-compatible endpoint ignores `num_ctx`, so a
// model runs at the server's VRAM-based default (4096 tokens under 23 GiB) and an
// agent prompt bigger than that is silently truncated. The native endpoint DOES
// honour `options.num_ctx`. So we translate, and the only behavioural difference
// from Ollama's own /v1 is the context size.
//
// The mapping mirrors Ollama's own converter (openai/openai.go, v0.32.1:
// FromChatRequest / ToChatCompletion / ToChunks), including its defaults
// (temperature 1.0 and top_p 1.0 when the client sends none).

const REASONING_EFFORTS = ['high', 'medium', 'low', 'max', 'none'];

export class TranslateError extends Error {}

function nameFromToolCallId(messages, id) {
  for (let i = messages.length - 1; i >= 0; i--) {
    for (const tc of messages[i].tool_calls || []) if (tc.id === id) return tc.function?.name || '';
  }
  return '';
}

function fromOpenAIToolCalls(toolCalls) {
  if (!Array.isArray(toolCalls)) return undefined;
  return toolCalls.map((tc, index) => {
    let args = tc.function?.arguments ?? {};
    if (typeof args === 'string') {
      try { args = args.trim() ? JSON.parse(args) : {}; } catch { throw new TranslateError(`tool call arguments are not valid JSON: ${args.slice(0, 80)}`); }
    }
    return { ...(tc.id ? { id: tc.id } : {}), function: { index, name: tc.function?.name, arguments: args } };
  });
}

function imageFromUrl(url) {
  const m = /^data:[^;,]+;base64,(.*)$/s.exec(url);
  if (!m) throw new TranslateError('only base64 data: image URLs are supported (same as Ollama)');
  return m[1];
}

/**
 * OpenAI chat request -> Ollama /api/chat request.
 * `numCtx` is applied unless the client asked for one itself (options.num_ctx
 * or top-level num_ctx — which Ollama's /v1 silently drops).
 */
export function toOllamaChat(body, { numCtx, modelDefaults = false } = {}) {
  if (!body || typeof body !== 'object') throw new TranslateError('request body must be a JSON object');
  const src = Array.isArray(body.messages) ? body.messages : [];
  const messages = [];
  for (const msg of src) {
    const role = msg.role;
    let toolName;
    if (String(role).toLowerCase() === 'tool') toolName = msg.name || (msg.tool_call_id ? nameFromToolCallId(src, msg.tool_call_id) : '');
    const extra = {
      ...(msg.reasoning ? { thinking: msg.reasoning } : {}),
      ...(toolName ? { tool_name: toolName } : {}),
      ...(msg.tool_call_id ? { tool_call_id: msg.tool_call_id } : {}),
    };
    const toolCalls = fromOpenAIToolCalls(msg.tool_calls);
    if (typeof msg.content === 'string') {
      messages.push({ role, content: msg.content, ...extra, ...(toolCalls ? { tool_calls: toolCalls } : {}) });
    } else if (Array.isArray(msg.content)) {
      const start = messages.length;
      for (const part of msg.content) {
        if (part?.type === 'text') messages.push({ role, content: String(part.text ?? '') });
        else if (part?.type === 'image_url') {
          const url = typeof part.image_url === 'string' ? part.image_url : part.image_url?.url;
          messages.push({ role, content: '', images: [imageFromUrl(String(url || ''))] });
        } else if (part?.type === 'input_audio') {
          messages.push({ role, content: '', images: [String(part.input_audio?.data || '')] });
        } else throw new TranslateError(`unsupported content part: ${part?.type}`);
      }
      if (messages.length > start && toolCalls) Object.assign(messages[messages.length - 1], { tool_calls: toolCalls, ...extra });
    } else if (toolCalls) {
      messages.push({ role, content: '', ...extra, tool_calls: toolCalls });
    } else if (msg.content == null) {
      messages.push({ role, content: '', ...extra });
    } else throw new TranslateError(`invalid message content type: ${typeof msg.content}`);
  }

  const options = {};
  if (typeof body.stop === 'string') options.stop = [body.stop];
  else if (Array.isArray(body.stop)) options.stop = body.stop.filter(s => typeof s === 'string');
  const maxTokens = body.max_tokens ?? body.max_completion_tokens;
  if (maxTokens != null) options.num_predict = maxTokens;
  // Ollama's /v1 forces temperature 1.0 and top_p 1.0 when the client sends
  // none, overriding the sampling a model's Modelfile recommends. Mirrored by
  // default for parity; modelDefaults leaves them to the model instead.
  if (body.temperature != null || !modelDefaults) options.temperature = body.temperature ?? 1.0;
  if (body.seed != null) options.seed = body.seed;
  if (body.frequency_penalty != null) options.frequency_penalty = body.frequency_penalty;
  if (body.presence_penalty != null) options.presence_penalty = body.presence_penalty;
  if (body.top_p != null || !modelDefaults) options.top_p = body.top_p ?? 1.0;

  const clientCtx = body.options?.num_ctx ?? body.num_ctx;
  const ctx = Number.isInteger(clientCtx) && clientCtx > 0 ? clientCtx : numCtx;
  if (ctx) options.num_ctx = ctx;

  let format;
  const rf = body.response_format;
  if (rf) {
    const t = String(rf.type || '').trim().toLowerCase();
    if (t === 'json_object') format = 'json';
    else if (t === 'json_schema' && rf.json_schema?.schema) format = rf.json_schema.schema;
  }

  let think;
  const effort = body.reasoning?.effort ?? body.reasoning_effort;
  if (effort) {
    if (!REASONING_EFFORTS.includes(effort)) throw new TranslateError(`invalid reasoning value: '${effort}'`);
    think = effort === 'none' ? false : effort;
  }

  return {
    request: {
      model: body.model,
      messages,
      ...(format !== undefined ? { format } : {}),
      options,
      stream: !!body.stream,
      ...(Array.isArray(body.tools) && body.tools.length ? { tools: body.tools } : {}),
      ...(think !== undefined ? { think } : {}),
      ...(body.keep_alive !== undefined ? { keep_alive: body.keep_alive } : {}),
    },
    numCtx: ctx || null,
    clientSetCtx: Number.isInteger(clientCtx) && clientCtx > 0,
  };
}

function toOpenAIToolCalls(toolCalls, idPrefix) {
  return (toolCalls || []).map((tc, i) => ({
    id: tc.id || `${idPrefix}_${i}`,
    index: tc.function?.index ?? i,
    type: 'function',
    function: { name: tc.function?.name, arguments: JSON.stringify(tc.function?.arguments ?? {}) },
  }));
}

export function usageOf(r) {
  const p = r.prompt_eval_count || 0, c = r.eval_count || 0;
  return { prompt_tokens: p, completion_tokens: c, total_tokens: p + c };
}

/** Ollama /api/chat final (non-streaming) response -> OpenAI chat.completion. */
export function toChatCompletion(id, r) {
  const toolCalls = toOpenAIToolCalls(r.message?.tool_calls, `call_${id.slice(-8)}`);
  let finish = r.done_reason || null;
  if (toolCalls.length) finish = 'tool_calls';
  return {
    id,
    object: 'chat.completion',
    created: Math.floor(Date.parse(r.created_at || '') / 1000) || Math.floor(Date.now() / 1000),
    model: r.model,
    system_fingerprint: 'fp_ollama',
    choices: [{
      index: 0,
      message: {
        role: r.message?.role || 'assistant',
        content: r.message?.content ?? '',
        ...(toolCalls.length ? { tool_calls: toolCalls } : {}),
        ...(r.message?.thinking ? { reasoning: r.message.thinking } : {}),
      },
      finish_reason: finish,
    }],
    usage: usageOf(r),
  };
}

/** One Ollama stream line -> OpenAI chunk(s), same split rule as Ollama's ToChunks. */
export function toChunks(id, r, { toolCallSent = false } = {}) {
  const base = () => ({
    id, object: 'chat.completion.chunk', created: Math.floor(Date.now() / 1000), model: r.model,
    system_fingerprint: 'fp_ollama',
    choices: [{ index: 0, delta: { role: 'assistant' }, finish_reason: null }],
  });
  const toolCalls = toOpenAIToolCalls(r.message?.tool_calls, `call_${id.slice(-8)}`);
  const finish = r.done_reason ? ((toolCallSent || toolCalls.length) ? 'tool_calls' : r.done_reason) : null;
  const thinking = r.message?.thinking || '';
  const content = r.message?.content || '';

  const chunk = base();
  const d = chunk.choices[0].delta;
  if (content) d.content = content;
  if (toolCalls.length) d.tool_calls = toolCalls;
  if (thinking) d.reasoning = thinking;
  chunk.choices[0].finish_reason = finish;
  if (!(thinking && (content || toolCalls.length))) return [chunk];

  const reasoning = base();
  reasoning.created = chunk.created;
  reasoning.choices[0].delta.reasoning = thinking;
  delete d.reasoning;
  return [reasoning, chunk];
}
