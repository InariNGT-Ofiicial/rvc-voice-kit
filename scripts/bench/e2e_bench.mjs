// 端到端：文本 → TTS(8788) → RVC(8789) → SILK，逐阶段墙钟计时
// 用法：node e2e_bench.mjs "今天天气真不错，我们出去走走吧。" 3
import { execFile } from "node:child_process";
import { writeFileSync, mkdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { promisify } from "node:util";

const run = promisify(execFile);
const FF = process.env.FFMPEG_PATH
  || (process.env.RVCX_FFMPEG_DIR ? `${process.env.RVCX_FFMPEG_DIR}/ffmpeg.exe` : "ffmpeg");
const text = process.argv[2] ?? "今天天气真不错，我们出去走走吧。";
const runs = Number(process.argv[3] ?? 3);
const outDir = process.argv[4] ?? join(process.cwd(), "e2e");
mkdirSync(outDir, { recursive: true });

// silk-wasm 的位置不写死：先试包名，再试同级目录，最后试环境变量
const { encode } = await (async () => {
  const candidates = [
    () => import("silk-wasm"),
    () => import(new URL("./node_modules/silk-wasm/lib/index.mjs", import.meta.url).href),
    () => import(new URL("../../silk-test/node_modules/silk-wasm/lib/index.mjs", import.meta.url).href),
    () => import(process.env.SILK_WASM_PATH),
  ].filter(Boolean);
  for (const c of candidates) {
    try { return await c(); } catch { /* 继续试下一个 */ }
  }
  throw new Error("找不到 silk-wasm：npm i silk-wasm，或设 SILK_WASM_PATH 指向 lib/index.mjs");
})();

const rows = [];
for (let i = 0; i < runs; i++) {
  const t0 = performance.now();
  const r = await fetch("http://127.0.0.1:8788/say", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!r.ok) throw new Error(`tts ${r.status} ${await r.text()}`);
  const wav = Buffer.from(await r.arrayBuffer());
  const tTts = performance.now();
  const ttsPath = join(outDir, `e2e_${i}_tts.wav`);
  writeFileSync(ttsPath, wav);

  const rvcOut = join(outDir, `e2e_${i}_rvc.wav`);
  const rr = await fetch("http://127.0.0.1:8789/convert", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ in: ttsPath.replace(/\\/g, "/"), out: rvcOut.replace(/\\/g, "/") }),
  });
  const rvcJson = await rr.json();
  if (!rvcJson.ok) throw new Error(`rvc ${JSON.stringify(rvcJson)}`);
  const tRvc = performance.now();

  const { stdout } = await run(FF, ["-v", "error", "-i", rvcOut, "-f", "s16le", "-ac", "1", "-ar", "24000", "pipe:1"],
    { encoding: "buffer", maxBuffer: 64 * 1024 * 1024 });
  const { data, duration } = await encode(new Uint8Array(stdout), 24000);
  const tSilk = performance.now();
  const silkPath = join(outDir, `e2e_${i}.silk`);
  writeFileSync(silkPath, Buffer.from(data));

  rows.push({
    run: i,
    warm: i > 0,
    tts_ms: Math.round(tTts - t0),
    rvc_ms: Math.round(tRvc - tTts),
    silk_ms: Math.round(tSilk - tRvc),
    total_ms: Math.round(tSilk - t0),
    audio_sec: Number(duration.toFixed(2)),
    silk_bytes: data.length,
    tts_wav_bytes: wav.length,
    silk_path: silkPath.replace(/\\/g, "/"),
  });
  console.log(`[run ${i}] total=${rows[i].total_ms}ms (tts ${rows[i].tts_ms} + rvc ${rows[i].rvc_ms} + silk ${rows[i].silk_ms}) audio=${rows[i].audio_sec}s`);
}
console.log(JSON.stringify({ text, runs: rows }));
