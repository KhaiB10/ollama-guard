#!/usr/bin/env node
import { createGuard } from '../src/server.js';
import { doctor } from '../src/doctor.js';

const HELP = `ollama-guard — stop Ollama from silently breaking your agent.

  ollama-guard serve   [--port 11435] [--host 127.0.0.1] [--upstream http://127.0.0.1:11434]
                       [--num-ctx 32768] [--on-overflow warn|error] [--no-rescue] [--model-defaults] [--quiet]
  ollama-guard doctor  [--model NAME] [--upstream URL] [--no-probe]

serve   An OpenAI-compatible proxy. Point your agent at http://localhost:11435/v1.
        /v1/chat/completions is sent to Ollama's native API so num_ctx is honoured,
        tool calls the model wrote as text are turned back into real tool calls,
        and truncated prompts are flagged (or refused with --on-overflow error).
        --model-defaults: when the client sends no temperature/top_p, use the
        model's own settings instead of Ollama /v1's forced 1.0.
doctor  Checks an Ollama install for the silent failures: the context /v1 really
        uses, models without tool support, and truncations in the server log.

Environment: OLLAMA_GUARD_NUM_CTX, OLLAMA_GUARD_UPSTREAM, OLLAMA_GUARD_PORT`;

function flags(argv) {
  const f = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith('--')) continue;
    const [k, inline] = a.slice(2).split('=');
    if (inline !== undefined) f[k] = inline;
    else if (argv[i + 1] && !argv[i + 1].startsWith('--')) f[k] = argv[++i];
    else f[k] = true;
  }
  return f;
}

const [cmd, ...rest] = process.argv.slice(2);
const f = flags(rest);
const upstream = f.upstream || process.env.OLLAMA_GUARD_UPSTREAM || process.env.OLLAMA_HOST?.replace(/^(?!https?:\/\/)/, 'http://') || 'http://127.0.0.1:11434';

if (cmd === 'serve') {
  const port = Number(f.port || process.env.OLLAMA_GUARD_PORT || 11435);
  const host = f.host || '127.0.0.1';
  const numCtx = Number(f['num-ctx'] || process.env.OLLAMA_GUARD_NUM_CTX || 32768);
  const onOverflow = f['on-overflow'] === 'error' ? 'error' : 'warn';
  const server = createGuard({ upstream, numCtx, onOverflow, rescue: !f['no-rescue'], modelDefaults: !!f['model-defaults'], log: f.quiet ? () => {} : undefined });
  server.listen(port, host, () => {
    console.log(`ollama-guard on http://${host}:${port}/v1  ->  ${upstream}   num_ctx ${numCtx}, rescue ${f['no-rescue'] ? 'off' : 'on'}, overflow ${onOverflow}`);
  });
} else if (cmd === 'doctor') {
  const r = await doctor({ upstream, model: typeof f.model === 'string' ? f.model : undefined, probe: !f['no-probe'] });
  process.exitCode = r.findings.some(x => x.level === 'bad') ? 1 : 0;
} else {
  console.log(HELP);
  process.exitCode = cmd && cmd !== 'help' && cmd !== '--help' ? 2 : 0;
}
