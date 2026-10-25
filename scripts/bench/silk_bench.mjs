// SILK 转码耗时基线：wav → ffmpeg 解 24k mono s16le PCM → silk-wasm encode
// 用法：node silk_bench.mjs <in.wav> [runs]
import { execFile } from "node:child_process";
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { basename, dirname, join } from "node:path";
import { promisify } from "node:util";

const run = promisify(execFile);
const FF = process.env.FFMPEG_PATH
  || (process.env.RVCX_FFMPEG_DIR ? `${process.env.RVCX_FFMPEG_DIR}/ffmpeg.exe` : "ffmpeg");
const inWav = process.argv[2];
const runs = Number(process.argv[3] ?? 3);
const outDir = join(dirname(inWav), "silk");
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
  const { stdout } = await run(FF, [
    "-v", "error", "-i", inWav, "-f", "s16le", "-ac", "1", "-ar", "24000", "pipe:1",
  ], { encoding: "buffer", maxBuffer: 64 * 1024 * 1024 });
  const tFfmpeg = performance.now();
  const pcm = stdout;
  const { data, duration } = await encode(new Uint8Array(pcm), 24000);
  const tEncode = performance.now();
  const silkPath = join(outDir, `out_${i}.silk`);
  writeFileSync(silkPath, Buffer.from(data));
  rows.push({
    run: i,
    ffmpeg_ms: Math.round(tFfmpeg - t0),
    silk_encode_ms: Math.round(tEncode - tFfmpeg),
    total_ms: Math.round(tEncode - t0),
    pcm_bytes: pcm.length,
    silk_bytes: data.length,
    seconds: Number(duration.toFixed(2)),
    silk_path: silkPath.replace(/\\/g, "/"),
  });
}
console.log(JSON.stringify({ input: inWav, input_bytes: readFileSync(inWav).length, runs: rows }, null, 0));
