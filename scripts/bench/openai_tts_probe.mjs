// 模拟 AstrBot 的 ProviderOpenAITTSAPI.get_audio()：
//   client.audio.speech.create(model=..., voice=..., response_format="wav", input=text)
// 只验证协议与产物，不验证 AstrBot 本体（本机未安装）。
import { writeFileSync } from "node:fs";

const base = "http://127.0.0.1:8790/v1";
const text = "今天天气真不错，我们出去走走吧。";
const out = "D:/WorkData/2026-10-19-03-29-21/rvc-research/audio/openai_tts_out.wav";

const t0 = Date.now();
const r = await fetch(`${base}/audio/speech`, {
  method: "POST",
  headers: { "Content-Type": "application/json", Authorization: "Bearer local" },
  body: JSON.stringify({ model: "fangno120", voice: "fangno120", response_format: "wav", input: text }),
});
const ms = Date.now() - t0;
if (!r.ok) {
  console.error("FAIL", r.status, (await r.text()).slice(0, 400));
  process.exit(1);
}
const buf = Buffer.from(await r.arrayBuffer());
writeFileSync(out, buf);
const riff = buf.toString("ascii", 0, 4) === "RIFF" && buf.toString("ascii", 8, 12) === "WAVE";
console.log(JSON.stringify({
  status: r.status,
  content_type: r.headers.get("content-type"),
  bytes: buf.length,
  riff_ok: riff,
  elapsed_ms: ms,
  out,
}));
