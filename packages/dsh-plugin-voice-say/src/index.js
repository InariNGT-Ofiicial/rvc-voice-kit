/**
 * dsh-voice-say —— 把「文本 → 目标音色语音文件」封成**一个**工具。
 *
 * 设计约束（来自实测，别绕）：
 *   1. 绝不改 dsh-qqbot 的 dist/（插件重装会被覆盖）。
 *   2. 产物必须落在 agent cwd（默认 <AGENT_CWD>）里，
 *      否则 qqbot_send_file 的 sendFile.restrictPaths 会拒绝。
 *   3. 三套 Python 环境不能合并 → 走两个常驻 HTTP 服务（8788 TTS / 8789 RVC）。
 *   4. 零运行时依赖：只用 node: 内置模块（与 dsh-plugin-cover 的惯例一致）。
 *
 * 契约：
 *   export const name = 'voice-say'
 *   export const inject = ['tools']
 *   export async function apply(ctx, config)
 */
import { existsSync, mkdirSync, statSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

export const name = 'voice-say';
export const inject = ['tools'];

const DEFAULTS = {
  ttsUrl: 'http://127.0.0.1:8788/say',
  rvcUrl: 'http://127.0.0.1:8789/convert',
  outDir: '',            // 空 = <cwd>/tts
  maxChars: 80,
  timeoutMs: 120000,
  keepIntermediate: false,
};

function cfgOf(config) {
  return { ...DEFAULTS, ...(config ?? {}) };
}

async function postJson(url, body, timeoutMs) {
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), timeoutMs);
  try {
    const r = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: ac.signal,
    });
    return r;
  } finally {
    clearTimeout(timer);
  }
}

/** 一句话：文本 → wav 文件绝对路径。抛错即失败，不静默降级。 */
async function synthToFile(text, cfg, cwd) {
  const t0 = Date.now();
  const rTts = await postJson(cfg.ttsUrl, { text }, cfg.timeoutMs);
  if (!rTts.ok) throw new Error(`TTS 服务返回 ${rTts.status}: ${(await rTts.text()).slice(0, 300)}`);
  const ttsWav = Buffer.from(await rTts.arrayBuffer());
  const tTts = Date.now();

  const outDir = cfg.outDir || join(cwd, 'tts');
  if (!existsSync(outDir)) mkdirSync(outDir, { recursive: true });
  const stem = `voice_${new Date().toISOString().replace(/[:.]/g, '-')}`;
  const dry = join(outDir, `${stem}_dry.wav`);
  const wet = join(outDir, `${stem}.wav`);
  writeFileSync(dry, ttsWav);

  const rRvc = await postJson(cfg.rvcUrl, {
    in: dry.replace(/\\/g, '/'),
    out: wet.replace(/\\/g, '/'),
  }, cfg.timeoutMs);
  const j = await rRvc.json().catch(() => ({}));
  if (!rRvc.ok || !j.ok) throw new Error(`RVC 服务失败: ${JSON.stringify(j).slice(0, 300)}`);
  const tRvc = Date.now();

  if (!existsSync(wet) || statSync(wet).size < 1000) throw new Error('RVC 产物缺失或过小');
  return {
    path: wet.replace(/\\/g, '/'),
    dry: dry.replace(/\\/g, '/'),
    ttsMs: tTts - t0,
    rvcMs: tRvc - tTts,
    totalMs: tRvc - t0,
    bytes: statSync(wet).size,
  };
}

export async function apply(ctx, config) {
  const tools = ctx.get('tools');
  const logger = ctx.logger ?? console;
  const cfg = cfgOf(config);
  if (!tools?.register) {
    logger.warn('voice-say: tools 服务不可用，未注册任何工具');
    return;
  }

  tools.register({
    name: 'voice_say',
    description: 'Synthesize the given Chinese text into a voice file using the configured character voice '
      + '(local TTS + RVC). Returns an absolute path that is already inside the agent workspace, so it can be '
      + 'sent with qqbot_send_file immediately. Call this when the user asks you to SPEAK / send a voice message.',
    parameters: {
      type: 'object',
      properties: {
        text: { type: 'string', description: '要念的中文文本，建议 < 40 字；超过 maxChars 会被拒绝。' },
      },
      required: ['text'],
      additionalProperties: false,
    },
    output: {
      schema: {
        type: 'object',
        properties: {
          file_path: { type: 'string' },
          seconds: { type: 'number' },
          totalMs: { type: 'integer' },
          ttsMs: { type: 'integer' },
          rvcMs: { type: 'integer' },
        },
        required: ['file_path', 'totalMs'],
        additionalProperties: false,
      },
      render: (_a, v) => [{
        type: 'text',
        text: `语音已生成：${v.file_path}（${v.totalMs} ms，TTS ${v.ttsMs} / RVC ${v.rvcMs}）。`
          + '下一步必须调用 qqbot_send_file 发送这个路径。',
      }],
    },
    async execute(args, exec) {
      const text = String(args?.text ?? '').trim();
      if (!text) throw new Error('voice_say: text 不能为空');
      if (text.length > cfg.maxChars) {
        throw new Error(`voice_say: 文本 ${text.length} 字，超过上限 ${cfg.maxChars}；请拆成更短的句子`);
      }
      const cwd = exec?.cwd ?? cfg.cwd ?? process.cwd();
      const r = await synthToFile(text, cfg, cwd);
      logger.info(`voice-say: ${text.length} 字 → ${r.path} (${r.totalMs}ms)`);
      return {
        file_path: r.path,
        totalMs: r.totalMs,
        ttsMs: r.ttsMs,
        rvcMs: r.rvcMs,
        seconds: Math.round((r.bytes / 96000) * 100) / 100, // 48k/16bit/mono 估算，仅展示
      };
    },
  });

  tools.register({
    name: 'voice_doctor',
    description: 'Check whether the local TTS (8788) and RVC (8789) services are alive.',
    parameters: { type: 'object', properties: {}, additionalProperties: false },
    output: {
      schema: {
        type: 'object',
        properties: { tts: { type: 'string' }, rvc: { type: 'string' }, outDir: { type: 'string' } },
        required: ['tts', 'rvc', 'outDir'],
        additionalProperties: false,
      },
      render: (_a, v) => [{ type: 'text', text: `TTS: ${v.tts}\nRVC: ${v.rvc}\n输出目录: ${v.outDir}` }],
    },
    async execute(_args, exec) {
      const cwd = exec?.cwd ?? process.cwd();
      const probe = async (u) => {
        try {
          const base = u.replace(/\/[^/]*$/, '');
          const r = await fetch(`${base}/health`, { signal: AbortSignal.timeout(3000) });
          return `${r.status} ${(await r.text()).slice(0, 120)}`;
        } catch (e) {
          return `DOWN (${e?.name ?? e})`;
        }
      };
      return {
        tts: await probe(cfg.ttsUrl),
        rvc: await probe(cfg.rvcUrl),
        outDir: cfg.outDir || join(cwd, 'tts'),
      };
    },
  });

  logger.info('voice-say: voice_say / voice_doctor 已注册');
}
