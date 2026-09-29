// doctor.js — find the silent failures in an Ollama setup before an agent does.
import { execFile } from 'node:child_process';

const AGENT_PROMPTS = [
  ['OpenCode (system prompt + tools, measured)', 8385],
  ['a small agent (system prompt + ~5 tools)', 3000],
];

async function j(url, init) {
  const r = await fetch(url, { ...init, signal: AbortSignal.timeout(init?.timeoutMs ?? 30_000) });
  if (!r.ok) throw new Error(`${r.status} ${await r.text().catch(() => '')}`.slice(0, 200));
  return r.json();
}

function journalTruncations() {
  return new Promise((resolve) => {
    execFile('journalctl', ['-u', 'ollama', '--since', '24 hours ago', '--no-pager', '-q'], { timeout: 15_000, maxBuffer: 64 << 20 }, (err, out) => {
      if (err || !out) return resolve(null);
      const lines = out.split('\n').filter(l => l.includes('truncating input prompt'));
      resolve({ count: lines.length, last: lines.at(-1)?.replace(/^.*msg=/, '') || null });
    });
  });
}

function contextFromShow(show) {
  const info = show.model_info || {};
  const key = Object.keys(info).find(k => k.endsWith('.context_length'));
  const trained = key ? info[key] : null;
  const m = /num_ctx\s+(\d+)/.exec(show.parameters || '');
  return { trained, modelfile: m ? Number(m[1]) : null };
}

/**
 * Run the checks. Returns { findings: [{level, text}], facts }.
 * `probe` sends one tiny /v1 request so /api/ps reveals the context /v1 really uses.
 */
export async function doctor({ upstream = 'http://127.0.0.1:11434', model, probe = true, out = console.log } = {}) {
  const findings = [];
  const say = (level, text) => { findings.push({ level, text }); out(`${{ ok: '  ✓', warn: '  ⚠', bad: '  ✗', info: '  ·' }[level]} ${text}`); };

  out(`ollama-guard doctor — ${upstream}\n`);
  let version;
  try { version = (await j(new URL('/api/version', upstream))).version; say('ok', `Ollama ${version} is reachable`); }
  catch (e) { say('bad', `cannot reach Ollama at ${upstream}: ${e.message}`); return { findings, facts: {} }; }

  const tags = (await j(new URL('/api/tags', upstream))).models || [];
  if (!tags.length) { say('warn', 'no models installed'); return { findings, facts: { version } }; }
  model = model || tags[0].name;
  if (!tags.some(t => t.name === model || t.model === model)) { say('bad', `model "${model}" is not installed`); return { findings, facts: { version } }; }

  const show = await j(new URL('/api/show', upstream), { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ model }) });
  const caps = show.capabilities || [];
  if (caps.includes('tools')) say('ok', `${model} supports tool calling`);
  else say('bad', `${model} does NOT support tool calling — an agent's tools will not reach it, and in testing models with no tools invented tool results (e.g. a fake sha256) in up to 8/8 tries`);
  const { trained, modelfile } = contextFromShow(show);
  if (trained) say('info', `${model} was trained for up to ${trained.toLocaleString()} tokens of context`);
  if (modelfile) say('info', `its Modelfile sets num_ctx ${modelfile.toLocaleString()}`);

  let v1Ctx = null;
  if (probe) {
    try {
      await j(new URL('/v1/chat/completions', upstream), {
        method: 'POST', headers: { 'content-type': 'application/json' }, timeoutMs: 300_000,
        body: JSON.stringify({ model, messages: [{ role: 'user', content: 'Reply with OK.' }], max_tokens: 1 }),
      });
      const ps = (await j(new URL('/api/ps', upstream))).models || [];
      v1Ctx = ps.find(p => p.name === model || p.model === model)?.context_length ?? null;
    } catch (e) { say('warn', `probe request failed: ${e.message}`); }
    if (v1Ctx) {
      say(v1Ctx >= 32768 ? 'ok' : 'warn', `requests to /v1 (what OpenAI-style clients and most agents use) run ${model} with a ${v1Ctx.toLocaleString()}-token context — /v1 ignores num_ctx sent by the client`);
      for (const [who, size] of AGENT_PROMPTS) {
        if (size > v1Ctx) say('bad', `a prompt the size of ${who} (~${size.toLocaleString()} tokens) would be silently truncated to ~${Math.ceil(v1Ctx / 2).toLocaleString()} tokens`);
      }
    }
  } else say('info', 'skipped the /v1 probe (--no-probe)');

  const jt = await journalTruncations();
  if (jt === null) say('info', 'could not read the Ollama log (not local, or no journalctl access) — check it for "truncating input prompt"');
  else if (jt.count) say('bad', `Ollama truncated ${jt.count} prompt(s) in the last 24 h. Last: ${jt.last}`);
  else say('ok', 'no "truncating input prompt" in the Ollama log in the last 24 h');

  out('');
  if (findings.some(f => f.level === 'bad' || f.level === 'warn')) {
    out('Fixes, pick one:');
    out('  • run agents through ollama-guard (num_ctx honoured on /v1, text tool calls rescued):');
    out('      ollama-guard serve --num-ctx 32768     then point the agent at http://localhost:11435/v1');
    out('  • or raise the server default:  OLLAMA_CONTEXT_LENGTH=32768 ollama serve');
    out('  • or bake it into the model:     a Modelfile with  FROM <model>  and  PARAMETER num_ctx 32768');
  } else out('No silent-failure risks found.');
  return { findings, facts: { version, model, v1Ctx, trained, modelfile } };
}
