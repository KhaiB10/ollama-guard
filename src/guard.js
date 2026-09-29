// guard.js — the two protections: notice truncation, and rescue text tool calls.
import { parseToolCalls } from './vendor/rescue-tool-calls.js';

/**
 * Ollama's truncation keeps num_keep tokens (4, +1 when the tokenizer adds BOS)
 * and cuts the prompt to numCtx - max((numCtx - numKeep) / 2, 1) tokens
 * (llm/llama_server.go, contextShiftPromptLimit, v0.32.1). A prompt_eval_count
 * equal to that number is its fingerprint: 2050 or 2051 at num_ctx 4096.
 */
export function truncationLimits(numCtx) {
  if (!numCtx || numCtx <= 1) return [];
  return [4, 5].map(k => numCtx - Math.max(Math.floor((numCtx - Math.min(k, numCtx - 1)) / 2), 1));
}

export function looksTruncated(promptEvalCount, numCtx) {
  return Number.isInteger(promptEvalCount) && truncationLimits(numCtx).includes(promptEvalCount);
}

/**
 * Rough token estimate for a pre-flight check (~4 characters per token for
 * English and code). Deliberately simple: it only has to catch prompts that are
 * clearly too big; the fingerprint above catches the rest after the fact.
 */
export function estimateTokens(ollamaRequest) {
  let chars = 0;
  for (const m of ollamaRequest.messages || []) chars += String(m.content || '').length + 16;
  if (ollamaRequest.tools) chars += JSON.stringify(ollamaRequest.tools).length;
  return Math.ceil(chars / 4);
}

const CALL_START = /<tool_call>|<function=|\[TOOL_CALLS\]|```(?:json)?\s*\{|\{\s*"(?:name|function)"/;

/** Where the tool-call markup begins in a reply (so the prose before it can be kept). */
export function callStart(text) {
  const m = CALL_START.exec(text || '');
  return m ? m.index : -1;
}

/** First character that could begin tool-call markup — streaming holds text back from here. */
export function holdIndex(text) {
  const m = /[<[{`]/.exec(text || '');
  return m ? m.index : -1;
}

/**
 * If the model wrote its tool call(s) as text, turn them into real ones.
 * Only calls to tools the client actually offered are accepted.
 * Returns the rescued OpenAI tool_calls (possibly empty) and the prose to keep.
 */
export function rescue(content, tools) {
  if (!content || !Array.isArray(tools) || !tools.length) return { toolCalls: [], content };
  const calls = parseToolCalls(content, tools, { all: true });
  if (!calls.length) return { toolCalls: [], content };
  const at = callStart(content);
  return {
    toolCalls: calls.map((c, i) => ({
      id: `call_rescued_${Date.now().toString(36)}_${i}`, index: i, type: 'function',
      function: { name: c.name, arguments: JSON.stringify(c.input ?? {}) },
    })),
    content: at > 0 ? content.slice(0, at).trim() : '',
  };
}
