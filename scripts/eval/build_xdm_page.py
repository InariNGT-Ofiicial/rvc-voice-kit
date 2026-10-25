# -*- coding: utf-8 -*-
"""生成《下等马》DS娘换声 · 独立对照页（v2：含本轮高音补正与母带）。

纪律 D1：**不拼接**多系统输出成一条 wav。每条独立成文件、独立播放，
并在页面上逐条标注「① 它是什么 ② 哪个模型 ③ 什么参数」；
并提供「原始输出 / 统一响度」切换（统一响度 = RMS −20 dBFS + 限峰 0.95）。

条目（每条一个独立文件 + 独立播放器）：
  A 泠鸢原唱干声（标尺，不是产品）
  B DS娘 e20 + index_rate 0.75（换声干声 · 上一轮，高音有缺陷）
  C DS娘 e40 + index_rate 0    （换声干声 · 本轮，高音补正）   ★
  D 成品母带 e20（上一轮成品）
  E 成品母带 e40（本轮最终交付）                              ★
  F 芳乃 e120（横向参照 · 成熟音色基准）
  G 原伴奏（MSST instrumental.wav，裸伴奏）

用法：python build_xdm_page.py
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rvcx import paths as P  # noqa: E402

OUT = Path(r"D:/WorkData/2026-10-19-03-29-21/rvc-research")
AUD = OUT / "audio" / "xdm_v2"
XDM = Path(r"E:/AI-RVC/work/dsx_xdm")
KARA = Path(r"D:/AI-Song/_pipeline_xiadengma/01_sep/step2_kara/vocals")
FANGNO_RAW = Path(r"D:/AI-Song/_pipeline_xiadengma/02_conv/out_lead/lead.wav")
FFMPEG = Path(r"D:/AI-Song/01_人声分离_MSST-GUI/env")

NORM_RMS_DB = -20.0        # 统一响度目标（D1）
NORM_PEAK = 0.95           # 统一响度限峰（D1）

SYSTEMS = [
    dict(
        sid="A", tag="标尺", name="泠鸢原唱《下等马》",
        what="被换声的输入本身：原曲分离出来的人声干声。**不是产品，是标尺**——任何换声结果都要和它比。",
        model="无模型（源素材自身）",
        params=["44.1 kHz / 单声道", "MSST 分离 → kara vocals", "187.51 s 全曲"],
        detail="★ 换声只改音色，不改咬字与时值；和 A 逐句对齐听最能判断换声质量。"
               "★ 注意 A 本身含和声（MSST 的 kara 模式没把主唱/和声拆开）。",
        src=KARA / "Vocals.wav",
    ),
    dict(
        sid="B", tag="换声·旧", name="DS娘 e20 · 换声干声（index_rate 0.75）",
        what="泠鸢原唱整轨换声成 DS娘（晓伊A3）音色，**未混伴奏**。上一轮版本。",
        model="dsxiaoyi_a3_v1_e20_s2640.pth ＋ added_IVF1929…_v2.index",
        params=["F0: rmvpe", "index_rate = 0.75 / protect = 0.33", "pitch = 0（不变调）",
                "输出 48 kHz / 单声道"],
        detail="★ 这一版开着检索索引，**高音区会匹配到低八度邻居**："
               "缺陷扫描 D1 八度错 136 帧、缺陷总时长 4.02 s。",
        src=XDM / "lead_dsx_e20.wav",
        metrics="e20",
    ),
    dict(
        sid="C", tag="换声·新", name="DS娘 e40 · 换声干声（index_rate 0）",
        what="同上，但**关掉检索索引**（index_rate = 0）并多训到 e40。**本轮高音补正后的换声干声。**",
        model="dsxiaoyi_a3_v1_e40_s5280.pth（**不使用索引**）",
        params=["F0: rmvpe", "index_rate = 0 / protect = 0.33", "pitch = 0（不变调）",
                "输出 44.1 kHz / 单声道（分段渲染拼接）"],
        detail="★ 关索引后 D1 八度错降到 41 帧（−70%）、缺陷总时长 1.93 s（−52%）；"
               "而且关索引**不损音色反更准**（300–3000 Hz 频带误差 +1.65/+1.13 dB → +0.90/+0.36 dB）。",
        src=XDM / "lead_dsx_e40_ir0.wav",
        metrics="e40",
    ),
    dict(
        sid="D", tag="成品·旧", name="DS娘 e20 · 成品母带（上一轮）",
        what="B 与原伴奏混合，走 mix_master 母带链（实测原曲配比 + 侧链让位 + 短混响 + 限幅 + LUFS 归一）。",
        model="同 B（换声结果 + 原始 instrumental.wav）",
        params=["LUFS −9.0 / 峰值 0.9478 / 削波 0", "人声配比 0.0 dB（实测）", "侧链让位 −1.21 dB"],
        detail="作为「上一轮成品」的基线，与 E 逐段对齐听。",
        src=XDM / "mix" / "下等马_DS娘_e20_mix_master.wav",
    ),
    dict(
        sid="E", tag="成品·新", name="DS娘 e40 · 成品母带（本轮最终交付）",
        what="C 与原伴奏混合，走同一条 mix_master 母带链。**本轮的最终交付物。**",
        model="dsxiaoyi_a3_v1_e40_s5280.pth（index_rate 0）＋ 原始 instrumental.wav",
        params=["LUFS −9.0（实测 −9.03）/ 峰值 0.9485 / 削波 0 样本",
                "人声配比 0.0 dB（实测原曲配比）", "侧链让位 −1.21 dB", "静音占比 0.00%"],
        detail="★ 人声轨没有独立和声轨（MSST kara 未拆开主唱/和声），"
               "和声链按静音处理（harm_muted = true），不放大噪声。",
        src=XDM / "mix" / "下等马_DS娘_e40_mix_master.wav",
    ),
    dict(
        sid="F", tag="横向参照", name="芳乃 e120 · 换声干声",
        what="同一首歌、同一套参数，换成**已验证成熟的芳乃音色**——用来看 DS娘 差在哪。",
        model="fangno_p2_e120_s13440.pth ＋ added_IVF1419…_v2.index",
        params=["F0: rmvpe", "index_rate = 0.75 / protect = 0.33", "pitch = 0"],
        detail="★ 芳乃 e120 已过完整验收（Δr = +0.0409，同信道自测上限 0.9863），可当「成熟音色」基准线。",
        src=FANGNO_RAW,
    ),
    dict(
        sid="G", tag="伴奏", name="原伴奏（MSST instrumental.wav）",
        what="原曲的纯伴奏（无人声）。听这条可以看出：换声轨里哪些内容属于伴奏残留。",
        model="无模型（MSST 分离产物）",
        params=["44.1 kHz / 立体声", "与 A / B / C 同源（同一份分离产物）"],
        detail="★ MSST 的 kara 模式只把「人声 vs 伴奏」拆开，主唱与和声仍在同一条 vocals 里。",
        src=KARA / "instrumental.wav",
    ),
]


def rms(y):
    return float(np.sqrt(np.mean(np.square(y, dtype=np.float64))))


def to_mp3(src: Path, dst: Path):
    env = dict(os.environ)
    env["PATH"] = str(FFMPEG) + os.pathsep + env.get("PATH", "")
    cmd = [str(FFMPEG / "ffmpeg.exe"), "-y", "-v", "error", "-i", str(src),
           "-c:a", "libmp3lame", "-b:a", "320k", str(dst)]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg 失败：{r.stderr[-400:]}")


def make_audio():
    AUD.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for s in SYSTEMS:
        p = s["src"]
        if not p.exists():
            print(f"  [skip] 缺文件 {p}")
            continue
        y, sr = sf.read(p, dtype="float32", always_2d=False)
        keep2d = y.ndim > 1
        if keep2d:
            ym = y.mean(axis=1)
        else:
            ym = y
        stem = f"xdm_{s['sid']}"
        # 原始输出（保留原声道结构）。
        # ★ 用 FLOAT 子类型：源 vocals.wav 的 peak = 1.0065（>1.0），
        #   写 PCM_16 会被硬削顶 ⇒ 「原始」必须是真的原始。
        raw_wav = AUD / f"{stem}_raw.wav"
        sf.write(raw_wav, y, sr, subtype="FLOAT")
        to_mp3(raw_wav, AUD / f"{stem}_raw.mp3")
        # 统一响度：RMS → −20 dBFS，再限峰 0.95
        r = rms(ym)
        g = (10 ** (NORM_RMS_DB / 20)) / max(r, 1e-12)
        yn = y * g
        pk = float(np.max(np.abs(yn)))
        if pk > NORM_PEAK:
            yn = yn * (NORM_PEAK / pk)
        norm_wav = AUD / f"{stem}_norm.wav"
        sf.write(norm_wav, yn, sr, subtype="PCM_16")
        to_mp3(norm_wav, AUD / f"{stem}_norm.mp3")
        manifest[stem] = dict(
            sid=s["sid"], name=s["name"], tag=s["tag"],
            sr=sr, dur=round(len(y) / sr, 2), channels=int(y.shape[1]) if keep2d else 1,
            rms_raw_db=round(20 * np.log10(max(r, 1e-12)), 2),
            gain_db=round(20 * np.log10(max(g, 1e-12)), 2),
            peak_raw=round(float(np.max(np.abs(ym))), 4),
            peak_norm=round(float(np.max(np.abs(yn))), 4),
            src=str(p),
        )
        print(f"  {stem:6s} {sr} Hz {manifest[stem]['dur']:6.1f}s "
              f"raw_rms={manifest[stem]['rms_raw_db']:+7.2f} dB  "
              f"补偿={manifest[stem]['gain_db']:+6.2f} dB  "
              f"peak_raw={manifest[stem]['peak_raw']:.4f}")
        raw_wav.unlink(missing_ok=True)
        norm_wav.unlink(missing_ok=True)
    (OUT / "audio_manifest_xdm_v2.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def load_metrics():
    """把缺陷扫描 / 音准 / 母带报告读进来，供指标表使用。"""
    m = {}
    for key, fn in [("e20", XDM / "defects_e20ir075_same.json"),
                    ("e40", XDM / "defects_e40ir0.json")]:
        if fn.exists():
            d = json.loads(fn.read_text(encoding="utf-8"))
            m[key] = d["counts"] | {"total_defect_dur_sec": d["total_defect_dur_sec"]}
    fn = XDM / "pitch_diag_e40ir0.json"
    if fn.exists():
        m["pitch"] = json.loads(fn.read_text(encoding="utf-8"))
    for key, fn in [("mix_e20", XDM / "mix" / "mix_report.json"),
                    ("mix_e40", XDM / "mix" / "mix_report_e40.json")]:
        if fn.exists():
            m[key] = json.loads(fn.read_text(encoding="utf-8"))
    for key, fn in [("cer_e20", XDM / "song_cer_e20ir075.json"),
                    ("cer_e40", XDM / "song_cer_e40ir0.json")]:
        if fn.exists():
            m[key] = json.loads(fn.read_text(encoding="utf-8"))
    return m


CSS = """
:root{
  --bg:#0e1116; --panel:#161b22; --panel2:#1b222c; --line:#252d38;
  --tx:#e6edf3; --tx2:#9aa7b4; --tx3:#6e7b8a;
  --acc:#4c9aff; --ok:#3fb950; --warn:#d29922; --bad:#f85149;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--tx);
  font-family:"Microsoft YaHei","PingFang SC",system-ui,sans-serif;line-height:1.78}
.wrap{max-width:1100px;margin:0 auto;padding:32px 20px 80px}
header{border-bottom:1px solid var(--line);padding-bottom:20px;margin-bottom:26px}
h1{font-size:26px;margin:0 0 8px}
h3{font-size:17px;margin:26px 0 10px;color:var(--tx)}
.sub{color:var(--tx2);font-size:14px}
.meta{margin-top:14px;font-size:13px;color:var(--tx3);font-family:ui-monospace,Consolas,monospace}
.note{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--warn);
  border-radius:8px;padding:14px 18px;margin:18px 0;font-size:14px;color:var(--tx2)}
.note b{color:var(--tx)}
.note.ok{border-left-color:var(--ok)}
.note.bad{border-left-color:var(--bad)}
.sys{background:var(--panel);border:1px solid var(--line);border-radius:12px;
  padding:20px 22px;margin:0 0 20px}
.sys.new{border-color:#2d5a8f;box-shadow:0 0 0 1px rgba(76,154,255,.15) inset}
.hd{display:flex;align-items:center;gap:12px;margin-bottom:6px;flex-wrap:wrap}
.badge{background:var(--panel2);border:1px solid var(--line);border-radius:6px;
  padding:2px 9px;font-size:12px;color:var(--tx2)}
.badge.new{background:rgba(76,154,255,.15);border-color:#2d5a8f;color:#9ecbff}
.sid{background:var(--acc);color:#06121f;font-weight:700;border-radius:6px;
  width:30px;height:30px;display:inline-flex;align-items:center;justify-content:center;font-size:16px}
h2{font-size:18px;margin:0}
.row{display:grid;grid-template-columns:84px 1fr;gap:8px 14px;font-size:14px;margin-top:14px}
.k{color:var(--tx3);font-size:13px}
.v{color:var(--tx)}
.v code{background:var(--panel2);padding:1px 6px;border-radius:4px;font-size:12.5px;
  font-family:ui-monospace,Consolas,monospace;color:#c9d5e0}
.v ul{margin:4px 0;padding-left:18px}
.plays{margin-top:16px;padding-top:14px;border-top:1px dashed var(--line)}
.pl{display:flex;align-items:center;gap:12px;margin:8px 0;flex-wrap:wrap}
.pl label{font-size:12px;color:var(--tx3);min-width:96px}
audio{flex:1;min-width:260px;height:34px}
table{width:100%;border-collapse:collapse;font-size:13.5px;margin:14px 0}
th,td{padding:9px 12px;border-bottom:1px solid var(--line);text-align:left}
th{color:var(--tx2);font-weight:600;font-size:12.5px;background:var(--panel2)}
td.num{font-family:ui-monospace,Consolas,monospace}
td.good{color:var(--ok)} td.bad{color:var(--bad)}
pre{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px 16px;
  overflow-x:auto;font-size:12.5px;color:#c9d5e0;line-height:1.6}
footer{margin-top:36px;padding-top:18px;border-top:1px solid var(--line);
  color:var(--tx3);font-size:12.5px}
a{color:var(--acc)}
"""

HTML = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>《下等马》DS娘换声 · 独立对照页 v2</title>
<style>{css}</style></head><body><div class="wrap">
<header>
  <h1>《下等马》· DS娘（晓伊A3）换声对照页 <span style="font-size:14px;color:var(--tx3)">v2</span></h1>
  <div class="sub">泠鸢原唱 → DS娘音色 整轨换声　｜　每条独立成文件、独立播放，不拼接</div>
  <div class="meta">{meta}</div>
</header>

<div class="note ok">
<b>本轮三件事：</b>
① <b>伴奏补进去</b> —— 换声干声与 MSST 原伴奏走 mix_master 母带链（实测原曲配比 0.0 dB、
侧链让位 −1.21 dB、LUFS −9.0、限峰 0.9485、削波 0 样本）。
② <b>修音调 → 判定为「不动」</b> —— 音准按<b>能量分桶</b>看：最强 20% 能量桶中位偏差仅
<b>2.41 音分</b>（人耳容差约 ±20），弱音桶的几十音分是 f0 提取伪影，硬修会改坏准确的部分。
③ <b>高音补正 → 已做</b> —— 根因是<b>检索索引在高音区匹配到低八度邻居</b>，
关掉索引（index_rate 0）并多训到 e40 后：D1 八度错 136 → <b>41 帧</b>、
缺陷总时长 4.02 → <b>1.93 s</b>。
</div>

<div class="note">
<b>怎么听：</b>先按 <b>A（标尺）</b> 的段落定位，再在同位置切到 B / C / E 逐条对；
每条都有「原始输出 / 统一响度」两版——<b>判断音色与音准用原始版</b>，
统一响度（−20 dBFS + 限峰 0.95）只用来快速对齐音量，避免「谁更响谁赢」。
</div>

<h3>指标表（换声轨口径：同一录音信道，源 = 同一份 MSST vocals.wav）</h3>
{metrics_table}

<h3>各条身份牌</h3>
{systems}

<h3>本轮的三条根因教训（别再踩）</h3>
<pre>1) 检索索引会在高音区「找错邻居」
   added_IVF1929 检索在高音帧匹配到低八度的相似向量 ⇒ RVC 照着错音高合成。
   对照实验：同一 15 s 窗口，index_rate=0 → 769.7 Hz（正确）；=0.75 → 410.5 Hz（低八度）。
   ⇒ 中文/台词类任务可一律 --index-rate 0；关索引还更准（300–3000 Hz 误差 +1.65/+1.13 dB → +0.90/+0.36 dB）。

2) 分段渲染必须显式对齐采样率
   RVC 模型输出采样率 = checkpoint 的 config[-1] = 48000，而源是 44100。
   直接按样本索引写回 ⇒ 时间轴被压缩 8.84 %，段内拉慢、段尾丢失，整体逐段错位。
   ⇒ 统一传 --resample-sr &lt;源采样率&gt;，让 RVC 自己重采样；并按「上一段实际输出长度」前向推进拼接。

3) RMVPE 必须喂 16 kHz
   RMVPE 的 mel 滤波器组按 16 kHz 设计。直接喂 44.1 kHz ⇒ 频率轴压缩 2.756 倍，
   所有 f0 读数偏低 2.756 倍 ⇒ 一切「音分」判据被放大 2.756 倍（八度错、音准偏差全是伪影）。
   降采样到 16 kHz 后帧间隔 = 160/16000 = 10 ms，恰好与 parselmouth 的 to_pitch_ac(time_step=0.01) 同频。
   ★ RVC 本身没事：infer/vc/modules.py 的 load_audio(path, 16000) 已经先降过采样。</pre>

<h3>读法与限制</h3>
<div class="note">
<b>① 缺陷计数怎么来的：</b>把「换声轨」与「同一份 MSST vocals.wav」逐帧比 f0
（两口径经 parselmouth/Praat 独立复核 + 谐波能量侧证）。
D1 = 八度错（|音分| &gt; 300）；D3 = 高音区跑调（|音分| &gt; 30）；D2 = 原唱有声换声无声。
<b>同一信道内比</b>才有意义——跨信道（TTS vs 真声）结论不可搬。
</div>
<div class="note">
<b>② 音准为什么分桶看：</b>偏差随能量单调下降（弱音桶几十音分 → 最强 20% 桶 2.41 音分）。
弱音区 f0 提取本身就不可靠 ⇒ 那是测量伪影，不是需要修的音准。
所以本轮<b>没有做任何 pitch shift 或音高校正</b>。
</div>
<div class="note">
<b>③ Δr（音色位移）为什么不在本页：</b>同信道自测上限 0.9863（芳乃留一法），
跨系统比较无分辨力；且本轮的改动（关索引）对音色是<b>正向</b>的（频带误差反而更小），
这部分结论见指标表与报告，不在此重复。
</div>
<div class="note bad">
<b>④ 已知残留（判定为不修）：</b>115.69–116.07 s（0.38 s）原唱是一段 <b>700→842 Hz 的假声滑音</b>，
换声后被平滑成稳定的 ~703 Hz（−133 音分），700–870 Hz 频带能量占比 59 % → 33 %。
<b>不是「画不出来」</b>（高频能量仍在），而是颤音被抹平 + 电平低约 8 dB；
改用 Praat 口径重渲只把占比提到 44 % 且整体更弱 ⇒ 按「修音九成不该动手」的纪律保留。
</div>
<div class="note">
<b>⑤ 母带规格：</b>LUFS −9.0（实测 −9.03）/ 峰值 0.9485 / 削波 0 样本 / 静音占比 0.00 %。
人和声没有独立轨（MSST kara 只拆「人声 vs 伴奏」），和声链按静音处理（harm_muted），
不放大伴奏残留。
</div>

<footer>
生成于 {stamp}　｜　脚本 <code>scripts/eval/build_xdm_page.py</code>（rvc-voice-kit）　｜
音频：<code>audio/xdm_v2/</code>（每条独立 mp3 320k，原始/统一响度各一版）
</footer>
</div></body></html>
"""


def build_metrics_table(m):
    def row(name, key, extra=None):
        c = m.get(key, {})
        if not c:
            return f"<tr><td>{name}</td><td colspan='5' class='num'>—</td></tr>"
        d1 = c.get("D1_octave_error_frames")
        d3 = c.get("D3_high_dev_frames")
        d4 = c.get("D4_high_flat_frames")
        dur = c.get("total_defect_dur_sec")
        n_frames = 18751  # 全曲帧数（10 ms 步长）
        def cls(v, good_max, bad_min):
            if v is None:
                return ""
            if v <= good_max:
                return "good"
            if v >= bad_min:
                return "bad"
            return ""
        pct = lambda v: f"{v}（{v / n_frames * 100:.2f}%）"
        return (f"<tr><td>{name}</td>"
                f"<td class='num {cls(d1, 50, 200)}'>{pct(d1)}</td>"
                f"<td class='num {cls(d3, 300, 600)}'>{pct(d3)}</td>"
                f"<td class='num {cls(d4, 100, 300)}'>{pct(d4)}</td>"
                f"<td class='num'>{dur} s</td>"
                f"<td>{extra or '—'}</td></tr>")

    pitch = m.get("pitch", {})
    buckets = pitch.get("buckets", [])
    strong = next((b for b in buckets if "最强" in b.get("bucket", "")), None)
    strong_txt = (f"{strong['median_abs_cents']} 音分（中位）"
                  if strong else "—")
    hi = pitch.get("high_notes", {})
    hi_txt = (f"中位 {hi.get('median_abs_cents')} 音分 / 带符号 {hi.get('signed_mean_cents')}"
              if hi else "—")

    mx20 = m.get("mix_e20", {})
    mx40 = m.get("mix_e40", {})
    cer20 = m.get("cer_e20", {}).get("src_vs_conv", {})
    cer40 = m.get("cer_e40", {}).get("src_vs_conv", {})

    rows = [
        "<tr><th>条目</th><th>D1 八度错帧</th><th>D3 跑调帧</th><th>D4 偏低帧</th>"
        "<th>缺陷总时长</th><th>索引</th></tr>",
        row("B · e20 + index 0.75（旧）", "e20", "开着检索索引"),
        row("C · e40 + index 0（新）", "e40", "<b>关掉检索索引</b>"),
    ]
    tbl = "".join(rows)
    extra = f"""
<table>
<tr><th>母带项</th><th>e20 成品</th><th>e40 成品（本轮）</th></tr>
<tr><td>LUFS</td><td class="num">{mx20.get('lufs_out','—')}</td><td class="num good">{mx40.get('lufs_out','—')}</td></tr>
<tr><td>峰值</td><td class="num">{mx20.get('peak','—')}</td><td class="num">{mx40.get('peak','—')}</td></tr>
<tr><td>削波样本</td><td class="num">{mx20.get('clipped_samples','—')}</td><td class="num good">{mx40.get('clipped_samples','—')}</td></tr>
<tr><td>人声配比（实测原曲）</td><td class="num">{mx20.get('vocal_db_actual','—')} dB</td><td class="num">{mx40.get('vocal_db_actual','—')} dB</td></tr>
<tr><td>侧链让位</td><td class="num">{mx20.get('duck_min_db','—')} dB</td><td class="num">{mx40.get('duck_min_db','—')} dB</td></tr>
<tr><td>静音占比</td><td class="num">{mx20.get('silent_ratio','—')} %</td><td class="num good">{mx40.get('silent_ratio','—')} %</td></tr>
</table>
<table>
<tr><th>ASR 回读 CER（整曲 · faster-whisper small）</th><th>e20 + index 0.75</th><th>e40 + index 0</th><th>怎么读</th></tr>
<tr><td>汉字 CER</td><td class="num">{cer20.get('han_cer','—')}</td>
    <td class="num good">{cer40.get('han_cer','—')}</td>
    <td rowspan="3">★ 唱歌的 ASR 绝对 CER 必然很难看（不是换声的锅）。<b>只看相对差</b>：
        以「源音频自己的 ASR」为参照，新版三项全部下降 ⇒ 本轮改动没伤可懂度。</td></tr>
<tr><td>拼音 CER（同音字不罚）</td><td class="num">{cer20.get('pinyin_cer','—')}</td>
    <td class="num good">{cer40.get('pinyin_cer','—')}</td></tr>
<tr><td>声调 CER（只比调号）</td><td class="num">{cer20.get('tone_cer','—')}</td>
    <td class="num good">{cer40.get('tone_cer','—')}</td></tr>
</table>
<table>
<tr><th>音准（e40 + index 0）</th><th>数值</th><th>怎么读</th></tr>
<tr><td>最强 20% 能量桶</td><td class="num good">{strong_txt}</td>
    <td>人声最实的地方＝最可信的音准读数，远在人耳容差内 ⇒ <b>不动音调</b></td></tr>
<tr><td>高音区（&gt; {hi.get('threshold_hz','—')} Hz）</td><td class="num">{hi_txt}</td>
    <td>系统性轻微偏低，是「模型画不到那么高」而非「画错」</td></tr>
<tr><td>换声最高音 / 原唱最高音</td><td class="num">{hi.get('test_max_hz','—')} / {hi.get('ref_max_hz','—')} Hz</td>
    <td>上界被削 ≈ 115 音分，对应 115.69 s 那段假声滑音</td></tr>
</table>"""
    return tbl + extra


def build_systems(manifest):
    out = []
    for s in SYSTEMS:
        stem = f"xdm_{s['sid']}"
        if stem not in manifest:
            continue
        m = manifest[stem]
        new = " new" if s["sid"] in ("C", "E") else ""
        params = "".join(f"<li>{p}</li>" for p in s["params"])
        out.append(f"""
<div class="sys{new}">
  <div class="hd">
    <span class="sid">{s['sid']}</span>
    <h2>{s['name']}</h2>
    <span class="badge{new}">{s['tag']}</span>
    <span class="badge">{m['dur']:.2f} s · {m['sr']} Hz · {m['channels']} ch</span>
  </div>
  <div class="row">
    <div class="k">它是什么</div><div class="v">{s['what']}</div>
    <div class="k">哪个模型</div><div class="v"><code>{s['model']}</code></div>
    <div class="k">什么参数</div><div class="v"><ul>{params}</ul></div>
    <div class="k">要注意</div><div class="v">{s['detail']}</div>
  </div>
  <div class="plays">
    <div class="pl"><label>原始输出</label>
      <audio controls preload="none" src="audio/xdm_v2/{stem}_raw.mp3"></audio></div>
    <div class="pl"><label>统一响度</label>
      <audio controls preload="none" src="audio/xdm_v2/{stem}_norm.mp3"></audio></div>
    <div class="pl"><label>电平</label>
      <span style="font-size:12.5px;color:var(--tx3);font-family:ui-monospace,Consolas,monospace">
        原始 RMS {m['rms_raw_db']:+.2f} dB · 峰值 {m['peak_raw']:.4f} ·
        统一响度补偿 {m['gain_db']:+.2f} dB</span></div>
  </div>
</div>""")
    return "".join(out)


def main():
    manifest = make_audio()
    metrics = load_metrics()
    meta = (f"RVC 2.3.260718 (81eed5e) · torch 2.7.1+cu128 · 权重 e40 s5280 · "
            f"f0=rmvpe · index_rate=0 · pitch=0 · 母带 mix_master/preset=match · "
            f"{len(manifest)} 条")
    html = (HTML
            .replace("{css}", CSS)
            .replace("{meta}", meta)
            .replace("{metrics_table}", build_metrics_table(metrics))
            .replace("{systems}", build_systems(manifest))
            .replace("{stamp}", "2026-10-26"))
    dst = OUT / "下等马_DS娘换声对照页_v2.html"
    dst.write_text(html, encoding="utf-8")
    print(f"\n对照页 -> {dst}")
    print(f"音频   -> {AUD}")


if __name__ == "__main__":
    main()
