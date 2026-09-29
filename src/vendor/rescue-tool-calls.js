// Vendored from rescue-tool-calls v1.2.0 (MIT, same author): https://github.com/KhaiB10/rescue-tool-calls
// rescue-tool-calls — recover tool calls that an LLM wrote as plain TEXT
// instead of using the structured tool-calling API.
//
// Local models (Ollama, llama.cpp) and some cheap cloud models frequently do this:
// they print  {"name":"get_weather","parameters":{"city":"Paris"}},
// <tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>  or
// <function=get_weather><parameter=city>Paris</parameter></function>  as the
// message content. Your agent then never runs the tool and the user just sees a blob.
// This library salvages those calls so your agent keeps working across any model.
//
// No dependencies. Works in Node and the browser. MIT licensed.

/**
 * Scan text and return every top-level, balanced JSON object/array substring.
 * String-aware, so braces inside quoted strings don't break the balance count.
 * @param {string} text
 * @returns {string[]} JSON substrings, in order of appearance
 */
export function extractJsonFragments(text) {
  const out = [];
  if (typeof text !== 'string') return out;
  for (let i = 0; i < text.length; i++) {
    const open = text[i];
    if (open !== '{' && open !== '[') continue;
    let depth = 0, inStr = false, esc = false;
    for (let j = i; j < text.length; j++) {
      const ch = text[j];
      if (esc) { esc = false; continue; }
      if (ch === '\\') { esc = true; continue; }
      if (ch === '"') inStr = !inStr;
      if (inStr) continue;
      if (ch === '{' || ch === '[') depth++;
      else if (ch === '}' || ch === ']') {
        depth--;
        if (depth === 0) { out.push(text.slice(i, j + 1)); i = j; break; }
      }
    }
  }
  return out;
}

function coerceInput(raw) {
  let input = raw ?? {};
  if (typeof input === 'string') {
    try { input = JSON.parse(input); } catch { input = {}; }
  }
  return (input && typeof input === 'object') ? input : {};
}

/**
 * Read tool NAMES out of whatever shape you keep your tool definitions in.
 *
 * This exists because of a real production failure. A caller passed its tool
 * list through `tools.map(t => t.name)` — but the list was in OpenAI shape,
 * where the name lives at `t.function.name`. Every entry became `undefined`, the
 * allow-list matched nothing, and a PERFECTLY VALID rescued call was rejected.
 * The user was told the model's reply "came out malformed". Nothing logged a
 * problem, because from the library's point of view the caller had simply
 * allowed no tools.
 *
 * So the names may be given as any of:
 *   ['get_weather']                                        plain strings
 *   [{ name: 'get_weather' }]                              Anthropic / native
 *   [{ type: 'function', function: { name: '…' } }]        OpenAI / Ollama
 *   new Set([...])                                         any of the above
 *
 * @param {any} toolNames
 * @returns {{ requested: number, names: Set<string> } | null} null when no filter was asked for
 */
export function normalizeToolNames(toolNames) {
  if (toolNames === null || toolNames === undefined) return null;
  const src = toolNames instanceof Set ? [...toolNames]
    : Array.isArray(toolNames) ? toolNames
      : [toolNames];
  const names = new Set();
  for (const t of src) {
    if (typeof t === 'string') { if (t) names.add(t); continue; }
    if (t && typeof t === 'object') {
      const n = t.name || t.function?.name || t.tool || t.tool_name;
      if (typeof n === 'string' && n) names.add(n);
    }
  }
  return { requested: src.length, names };
}

let warned = false;
function warnUnreadableToolList(onWarn, requested) {
  const msg = `rescue-tool-calls: received ${requested} tool definition(s) but could not read a `
    + 'name from any of them, so the allow-list would reject every rescued call. '
    + 'Filtering has been DISABLED for this call. This usually means the tool list was '
    + 'mapped with `t => t.name` while holding OpenAI-shaped tools (name is at '
    + '`t.function.name`) — pass the tool objects themselves and let the library read them.';
  if (typeof onWarn === 'function') { onWarn(msg); return; }
  if (!warned) { warned = true; try { console.warn(msg); } catch { /* browser-safe */ } }
}

/**
 * Parse tool calls out of raw model text.
 *
 * @param {string} text  The model's text output (its "content").
 * @param {string[]|object[]|Set<any>} [toolNames]  The tools you offered — names, or the tool
 *   objects themselves in any common shape (see normalizeToolNames). If given, only calls to
 *   these names are returned (prevents false positives). If omitted, any object that looks like
 *   a tool call is accepted. Passing an EMPTY array means "allow nothing" and is respected.
 * @param {object} [options]
 * @param {boolean} [options.all=false]  Return every call found instead of just the first.
 * @param {(msg: string) => void} [options.onWarn]  Receive diagnostics instead of console.warn.
 * @returns {{ id: string, name: string, input: object }[]}
 */
export function parseToolCalls(text, toolNames, options = {}) {
  if (!text || typeof text !== 'string') return [];
  const all = !!options.all;

  const norm = normalizeToolNames(toolNames);
  let names = norm ? norm.names : null;
  // A non-empty list that yields no readable names is a caller bug, not a policy
  // of "allow nothing" — and silently rejecting everything is the worst possible
  // outcome, because the agent looks broken and nothing explains why.
  if (norm && norm.requested > 0 && norm.names.size === 0) {
    warnUnreadableToolList(options.onWarn, norm.requested);
    names = null;
  }

  const found = [];
  const push = (o) => {
    if (!o || typeof o !== 'object') return;
    const name = o.name || o.function?.name || o.tool || o.tool_name;
    if (!name) return;
    if (names && !names.has(name)) return;
    const input = coerceInput(
      o.parameters ?? o.arguments ?? o.input ?? o.args ?? o.function?.arguments
    );
    found.push({ id: `rtc_${found.length}`, name, input });
  };

  // 1) JSON object/array fragments anywhere in the text (also covers ```json fences,
  //    Mistral's [TOOL_CALLS] arrays, <tool_call>{...}</tool_call>, etc.)
  for (const frag of extractJsonFragments(text)) {
    let obj;
    try { obj = JSON.parse(frag); } catch { continue; }
    if (Array.isArray(obj)) obj.forEach(push);
    else push(obj);
    if (found.length && !all) break; // first call wins unless caller wants them all
  }

  // 1b) Mistral:  [TOOL_CALLS]NAME[ARGS]{...}  — the v11+ tokenizer format — and the
  //     drifted forms a weakened prompt produces: NAME{...}, NAME: {...}, NAME,{...},
  //     NAME\n{...}. The arguments must be real JSON; prose after the name
  //     ("bash: sha256sum x") is left alone, because guessing arguments out of prose
  //     is inventing a tool call, not rescuing one.
  if (!found.length && text.includes('[TOOL_CALLS]')) {
    for (const mm of text.matchAll(/\[TOOL_CALLS\]\s*([a-zA-Z_][\w.-]*)\s*(?:\[ARGS\]|[:,])?\s*(?=\{)/g)) {
      const nm = mm[1];
      if (names && !names.has(nm)) continue;
      const frag = extractJsonFragments(text.slice(mm.index + mm[0].length))[0];
      let args;
      try { args = JSON.parse(frag); } catch { continue; }
      if (!args || typeof args !== 'object' || Array.isArray(args)) continue;
      found.push({ id: `rtc_${found.length}`, name: nm, input: args });
      if (!all) break;
    }
  }

  // 2) XML-ish style:  <tool_call>NAME<arg_key>k</arg_key><arg_value>v</arg_value>...
  if (!found.length && text.includes('<tool_call>')) {
    const tc = text.split('<tool_call>')[1] || '';
    const nm = (tc.match(/^\s*([a-zA-Z_][a-zA-Z0-9_]*)/) || [])[1];
    if (nm && (!names || names.has(nm))) {
      const input = {};
      for (const mt of tc.matchAll(
        /<arg_key>([\s\S]*?)<\/arg_key>\s*<arg_value>([\s\S]*?)(?:<\/arg_value>|$)/g
      )) {
        input[mt[1].trim()] = mt[2];
      }
      found.push({ id: 'rtc_0', name: nm, input });
    }
  }

  // 3) Qwen3-Coder / Qwen 3.5+ / Granite 4.2 style:
  //      <function=NAME><parameter=KEY>VALUE</parameter>...</function>
  //    Measured 2026-09-28: 57 of 57 text-format calls from qwen3-coder:30b under a
  //    truncated prompt were in this shape, and none were recovered before this
  //    branch existed. The closing tags are optional: a reply cut off by the
  //    output limit still yields the call it was writing.
  if (!found.length && /<function=/.test(text)) {
    for (const fm of text.matchAll(/<function=\s*([a-zA-Z_][\w.-]*)\s*>([\s\S]*?)(?=<\/function>|<function=|$)/g)) {
      const nm = fm[1];
      if (names && !names.has(nm)) continue;
      const input = {};
      for (const pm of fm[2].matchAll(/<parameter=\s*([^>\s]+)\s*>([\s\S]*?)(?=<\/parameter>|<parameter=|$)/g)) {
        input[pm[1]] = decodeParameterValue(pm[2]);
      }
      found.push({ id: `rtc_${found.length}`, name: nm, input });
      if (!all) break;
    }
  }

  return all ? found : found.slice(0, 1);
}

/**
 * A <parameter=…> value arrives as raw text, framed by the template's newlines.
 * Strip exactly that framing, then decode only what is unambiguous JSON:
 * numbers, booleans, null, objects and arrays. Anything else stays a string —
 * so '007' and 'Boston' are untouched, while '15' becomes 15 for a `limit`.
 */
function decodeParameterValue(raw) {
  const v = raw.replace(/^\r?\n/, '').replace(/\r?\n\s*$/, '');
  const t = v.trim();
  if (/^(-?(0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?|true|false|null)$/.test(t) || /^[[{]/.test(t)) {
    try { return JSON.parse(t); } catch { /* not JSON after all: keep the text */ }
  }
  return v;
}

/**
 * Report what a rescue attempt found AND what it rejected.
 *
 * Use this when an agent is behaving strangely and you need to know whether the
 * model called something you never offered. A call to an unknown tool and a
 * model that produced no call at all look identical from parseToolCalls, but
 * they are completely different bugs: the first is a stale tool list or a
 * hallucinated name, the second is a model that simply answered in prose.
 *
 * @returns {{ calls: {id,name,input}[], rejected: string[], sawToolLikeText: boolean }}
 */
export function inspectToolCalls(text, toolNames, options = {}) {
  const calls = parseToolCalls(text, toolNames, { ...options, all: true });
  const anyCall = parseToolCalls(text, undefined, { ...options, all: true });
  const kept = new Set(calls.map(c => c.name));
  return {
    calls: options.all ? calls : calls.slice(0, 1),
    rejected: [...new Set(anyCall.map(c => c.name).filter(n => !kept.has(n)))],
    sawToolLikeText: anyCall.length > 0,
  };
}

/**
 * The one you probably want: use the model's STRUCTURED tool calls if it produced any,
 * and only fall back to text-salvage when it didn't. So it never fights a well-behaved
 * model, and rescues the ones that misbehave.
 *
 * @param {object} message  A chat message, e.g. OpenAI/Ollama shape
 *   { content?: string, tool_calls?: [{ id?, function: { name, arguments } }] }.
 * @param {string[]|object[]|Set<any>} [toolNames]
 * @param {object} [options]  Passed through to parseToolCalls (e.g. { all: true, onWarn }).
 * @returns {{ id: string, name: string, input: object }[]}
 */
export function resolveToolCalls(message, toolNames, options = {}) {
  if (!message || typeof message !== 'object') return [];

  const structured = message.tool_calls;
  if (Array.isArray(structured) && structured.length) {
    return structured.map((c, i) => ({
      id: c.id || `rtc_${i}`,
      name: c.function?.name || c.name,
      input: coerceInput(c.function?.arguments ?? c.arguments ?? c.input),
    })).filter((c) => c.name);
  }

  return parseToolCalls(message.content || '', toolNames, options);
}

export default { parseToolCalls, resolveToolCalls, extractJsonFragments, normalizeToolNames, inspectToolCalls };
