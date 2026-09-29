# ollama-guard

**Stop Ollama from silently breaking your AI agent.**

Point your agent at `ollama-guard` instead of Ollama. Nothing else changes, and three silent failures go away:

1. **`num_ctx` finally works on `/v1`.** Ollama's OpenAI-compatible endpoint ignores it and runs models at a default context chosen by your VRAM — **4,096 tokens on GPUs under 23 GiB**. A coding agent's prompt is bigger than that, so Ollama quietly drops the middle of it, tool definitions included. ollama-guard sends the request to Ollama's native API, where `num_ctx` is honoured.
2. **Tool calls written as text become real tool calls.** Local models often print `<function=bash><parameter=command>…` or `[TOOL_CALLS]name{…}` instead of making a structured call, and your agent never runs it. ollama-guard turns them back into proper `tool_calls` (only for tools you actually offered).
3. **Truncation stops being silent.** If a prompt still doesn't fit, the response is flagged — or refused with `--on-overflow error`.

Plus `ollama-guard doctor`, which checks your setup for these problems in a few seconds.

## Measured

Same request both ways: OpenCode's real system prompt and 18 tools (8,385 tokens), `qwen3-coder:30b`, Ollama 0.32.1 on a 24 GB Tesla P40.

| | tokens that reached the model | correct tool call |
|---|---|---|
| straight to Ollama `/v1` | **2,050** of 8,385 (every time) | **2 / 6** — the rest called the wrong tool (including `shell`, which doesn't exist) or wrote the call as text |
| through ollama-guard | **8,385** (every time) | **12 / 12** — 4 of them rescued from text |

Why this happens, measured across 936 pre-registered trials: [tool-call-truncation-bench](https://github.com/KhaiB10/tool-call-truncation-bench).

## Use it

Requires Node 18.17+.

```bash
npm install -g github:KhaiB10/ollama-guard
ollama-guard serve --num-ctx 32768
```

Then point your agent at `http://localhost:11435/v1` instead of `http://localhost:11434/v1`. For OpenCode, change the provider's `baseURL`:

```json
"options": { "baseURL": "http://localhost:11435/v1" }
```

A `num_ctx` your client sends itself (`options.num_ctx` in `opencode.json`, for example) now takes effect and wins over `--num-ctx`.

### Check your setup

```bash
ollama-guard doctor --model qwen3-coder:30b
```

```
  ✓ Ollama 0.32.1 is reachable
  ✓ qwen3-coder:30b supports tool calling
  · qwen3-coder:30b was trained for up to 262,144 tokens of context
  ⚠ requests to /v1 (what OpenAI-style clients and most agents use) run qwen3-coder:30b with a 4,096-token context — /v1 ignores num_ctx sent by the client
  ✗ a prompt the size of OpenCode (system prompt + tools, measured) (~8,385 tokens) would be silently truncated to ~2,048 tokens
  ✗ Ollama truncated 586 prompt(s) in the last 24 h. Last: "truncating input prompt" limit=2050 prompt=8385 keep=4 new=2050
```

It sends one tiny `/v1` request so Ollama reports the context it really uses (`--no-probe` skips it; that loads the model). It also flags models without tool support — in testing, models given no tools *invented* tool results, like a fake sha256, in up to 8 of 8 tries.

## Options

| flag | default | |
|---|---|---|
| `--num-ctx N` | `32768` | context for every `/v1` chat request (env `OLLAMA_GUARD_NUM_CTX`). Bigger costs VRAM. |
| `--on-overflow warn\|error` | `warn` | `warn` adds a header and a log line; `error` refuses a prompt that clearly won't fit, and a response Ollama truncated |
| `--no-rescue` | rescue on | leave text-format tool calls as text |
| `--model-defaults` | off | when the client sends no `temperature`/`top_p`, use the model's own settings instead of Ollama `/v1`'s forced `1.0` (in testing it made no difference to tool calling) |
| `--port`, `--host` | `11435`, `127.0.0.1` | there is no authentication, so keep it on localhost unless you put one in front |
| `--upstream URL` | `http://127.0.0.1:11434` | also reads `OLLAMA_HOST` |

Response headers tell you what happened: `x-ollama-guard-num-ctx`, `x-ollama-guard-rescued`, `x-ollama-guard-truncated`, `x-ollama-guard-warning`.

## How it works, and its limits

- Only `POST /v1/chat/completions` is translated; every other path (`/api/*`, `/v1/models`, embeddings…) passes straight through. The translation mirrors Ollama's own `/v1` converter (v0.32.1), including its defaults, so the only intended differences are the context size and the rescued calls.
- **Truncation detection** uses Ollama's own fingerprint: its truncation cuts a prompt to exactly `num_ctx − (num_ctx − 4) / 2` tokens (2,050 at 4,096). With prompt caching the reported count can differ, so a truncation can occasionally go unflagged. The pre-flight size check is a rough estimate (~4 characters per token).
- **Streaming with tools:** text streams normally until the first character that could begin tool-call markup (`<`, `[`, `{`, or a backtick). From there it is held until the reply finishes, then sent either as a tool call or as the original text. A reply that uses brackets early on streams its tail all at once.
- Images must be base64 `data:` URLs, the same limit as Ollama's `/v1`.

Tool-call rescue comes from [rescue-tool-calls](https://github.com/KhaiB10/rescue-tool-calls), vendored — 89% of text-format calls recovered on fresh data, zero false positives.

## Test

```bash
npm test
```

18 tests, including the proxy end to end over real HTTP (streaming included) against a fake Ollama.

## License

MIT
