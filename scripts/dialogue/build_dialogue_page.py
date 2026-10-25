# -*- coding: utf-8 -*-
"""生成「台词换声独立对照页」。

纪律 D1：每条音频独立成文件、独立播放，逐条标注「是什么 / 哪个模型 / 什么参数」；
不拼接串烧。
"""
import sys as _sys
from pathlib import Path as _pth

_sys.path.insert(0, str(_pth(__file__).resolve().parents[2]))
from rvcx import paths as _paths  # noqa: E402

import json
from pathlib import Path

import numpy as np
import soundfile as sf

DLG = _paths.WORK / "dialogue"
OUT = _paths.WORK / "report"
AUD = OUT / "dialogue" / "audio"

TEXT = {
    "ja01": "私の質問だけに答えてください。",
    "ja02": "顔見ずに、そのまま聞いてください。",
    "ja03": "ああ、申し遅れました。私は日立真子と申します。",
    "ja04": "朝はまず、お風呂場の掃除をしているのですが、私、お風呂が好きなんです。",
}
REAL_FILE = {"ja01": "m4.wav", "ja02": "m13.wav", "ja03": "m15.wav", "ja04": "m19.wav"}
LIDS = ["ja01", "ja02", "ja03", "ja04"]

ARMS = {
    "T": dict(
        tag="TTS 通道", short="TTS",
        name="GPT-SoVITS 合成（kurumi v3）",
        what="用本机 GPT-SoVITS 的另一个角色音色（kurumi / 胡桃）按同一文本合成日语台词，再送进 RVC。",
        model="kurumi_v3_e12_s624.pth（SoVITS）＋ kurumi_v3-e15.ckpt（GPT）",
        params=["v3 / text_split = cut0（不切句）", "参考音 4.90 s，原样语气",
                "输出 32 kHz → 峰值归一化 0.9 后存盘"],
        src_dir="src_tts",
        detail="★ v2 会截断长句（2.5 s 只剩前半句），必须 v3；且 v3 直接写 PCM_16 会被削顶，必须先归一化。",
    ),
    "H": dict(
        tag="真人通道", short="真人",
        name="真人声优原声（茉子 / 日立真子）",
        what="游戏内真人语音，同一批台词由真人念出（与 TTS 通道文本完全一致）。",
        model="无模型（源素材自身）",
        params=["44.1 kHz / 单声道", "游戏原生语音，未做任何处理"],
        src_dir="src_real",
        detail="现实里最接近「录音棚成品」的源；也是 RVC 训练集所在的同一个录音信道。",
    ),
}

SYSTEMS = [
    dict(sid="src", name="源（未换声）", short="源", model="—",
         params=["标尺，不换声"]),
    dict(sid="e120", name="RVC · 芳乃 e120", short="e120",
         model="fangno_p2_e120_s13440.pth", params=["rmvpe · index_rate 0.75 · pitch 0"]),
    dict(sid="e200", name="RVC · 芳乃 e200", short="e200",
         model="fangno_p2_e200_s22400.pth", params=["rmvpe · index_rate 0.75 · pitch 0"]),
]


def src_path(arm, lid):
    return DLG / ARMS[arm]["src_dir"] / f"{lid}.wav"


def row_path(arm, lid, mdl):
    if mdl == "src":
        return src_path(arm, lid)
    return DLG / "out" / f"{lid}_{arm}_{mdl}.wav"


def rms(y):
    return float(np.sqrt(np.mean(np.square(y, dtype=np.float64))))


def make_audio():
    AUD.mkdir(parents=True, exist_ok=True)
    n = 0
    for arm in ARMS:
        for lid in LIDS:
            for s in SYSTEMS:
                p = row_path(arm, lid, s["sid"])
                if not p.exists():
                    continue
                y, sr = sf.read(p, dtype="float32", always_2d=False)
                if y.ndim > 1:
                    y = y.mean(1)
                key = f"{lid}_{arm}_{s['sid']}"
                sf.write(AUD / f"{key}_raw.wav", y, sr, subtype="PCM_16")
                g = 10 ** (-20 / 20) / max(rms(y), 1e-9)   # 统一 -20 dBFS
                yn = y * g
                pk = np.abs(yn).max()
                if pk > 0.95:
                    yn = yn * (0.95 / pk)
                sf.write(AUD / f"{key}_norm.wav", yn, sr, subtype="PCM_16")
                n += 1
    print(f"音频 {n} 组 × 2 版")


CSS = """
:root{--bg:#0e1116;--panel:#161b22;--panel2:#1b222c;--line:#252d38;--txt:#e6edf3;
--dim:#8b98a5;--dim2:#6b7885;--acc:#4ec9b0;--warn:#e3b341;--bad:#f85149;--good:#3fb950;--src:#7d8590}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--txt);font-size:15px;line-height:1.65;
font-family:-apple-system,"Segoe UI","Microsoft YaHei",system-ui,sans-serif;-webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:44px 26px 90px}
h1{font-size:27px;margin:0 0 8px;font-weight:650;letter-spacing:.3px}
.lead{color:var(--dim);font-size:14px;margin:0 0 26px;max-width:82ch}
.rule{background:linear-gradient(90deg,rgba(78,201,176,.12),transparent);border-left:3px solid var(--acc);
padding:13px 16px;border-radius:0 8px 8px 0;font-size:13.5px;color:#c7d3de;margin:0 0 32px}
h2{font-size:18px;margin:44px 0 14px;padding-bottom:9px;border-bottom:1px solid var(--line);font-weight:600}
h3{font-size:14.5px;margin:24px 0 10px;color:var(--acc);font-weight:600}
table{width:100%;border-collapse:collapse;font-size:13.2px;margin:12px 0 6px}
th,td{padding:9px 11px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--dim);font-weight:550;font-size:12.3px;background:var(--panel2)}
td.num{text-align:right;font-variant-numeric:tabular-nums;font-family:ui-monospace,Consolas,monospace}
tbody tr:hover{background:rgba(255,255,255,.022)}
code{background:#0b0f14;border:1px solid var(--line);padding:1px 5px;border-radius:4px;
font-family:ui-monospace,Consolas,monospace;font-size:12.2px;color:#a8c7d8}
.note{color:var(--dim);font-size:12.7px;margin:8px 0 0}
.note b{color:var(--warn);font-weight:600}
.arms{display:grid;grid-template-columns:repeat(2,1fr);gap:13px;margin-top:14px}
.arm{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:15px;position:relative}
.arm.t{border-color:rgba(227,179,65,.35)}
.arm.h{border-color:rgba(78,201,176,.35)}
.tag{position:absolute;top:12px;right:12px;font-size:10.8px;color:var(--dim);
border:1px solid var(--line);padding:1px 7px;border-radius:20px;background:var(--panel2);white-space:nowrap}
.head{display:flex;align-items:flex-start;gap:9px;padding-right:74px}
.badge{flex:0 0 auto;display:inline-flex;align-items:center;justify-content:center;width:24px;height:24px;
margin-top:1px;border-radius:7px;font-weight:700;font-size:14px;font-family:ui-monospace,Consolas,monospace;
background:#243040;color:#cfe0ee}
.arm.t .badge{background:rgba(227,179,65,.16);color:var(--warn)}
.arm.h .badge{background:rgba(78,201,176,.16);color:var(--acc)}
.armname{font-weight:600;font-size:14.2px;line-height:1.35}
.what{font-size:12.9px;color:#bcc8d3;margin:9px 0}
.arm dl{margin:0;font-size:12.3px}
.arm dt{color:var(--dim2);font-size:11.4px;margin-top:6px}
.arm dd{margin:1px 0 0;color:#cdd8e2}
.arm .m{font-family:ui-monospace,Consolas,monospace;font-size:11.5px;color:#9db4c6;
word-break:normal;overflow-wrap:anywhere;line-height:1.55}
.switch{display:flex;gap:8px;align-items:center;margin:16px 0 20px;font-size:13px;color:var(--dim)}
.switch button{background:var(--panel);border:1px solid var(--line);color:var(--dim);padding:6px 14px;
border-radius:20px;cursor:pointer;font-size:12.7px;font-family:inherit}
.switch button.on{background:rgba(78,201,176,.14);border-color:rgba(78,201,176,.5);color:var(--acc)}
.clipblk{background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:16px 17px 18px;margin-bottom:14px}
.cliphead{display:flex;align-items:baseline;gap:11px;margin-bottom:6px;flex-wrap:wrap}
.clipid{font-family:ui-monospace,Consolas,monospace;font-size:13px;color:var(--acc);
background:rgba(78,201,176,.1);padding:2px 9px;border-radius:6px}
.dlgtext{font-size:13.5px;color:#dbe4ec;margin:0 0 12px}
.side{font-size:11.8px;color:var(--dim2);letter-spacing:.3px;margin:12px 0 7px}
.pls{display:grid;grid-template-columns:repeat(3,1fr);gap:9px}
.pl{background:#11161d;border:1px solid var(--line);border-radius:9px;padding:10px 11px}
.pl.t{border-color:rgba(227,179,65,.28)} .pl.h{border-color:rgba(78,201,176,.28)}
.pl .lbl{display:flex;align-items:center;gap:7px;margin-bottom:7px}
.pl .mini{width:20px;height:20px;border-radius:5px;display:flex;align-items:center;justify-content:center;
font-size:11px;font-weight:700;background:#243040;color:#cfe0ee;flex:0 0 auto;font-family:ui-monospace,Consolas,monospace}
.pl.t .mini{background:rgba(227,179,65,.16);color:var(--warn)}
.pl.h .mini{background:rgba(78,201,176,.16);color:var(--acc)}
.pl .nm{font-size:12.6px;font-weight:550}
audio{width:100%;height:32px;margin-top:6px}
.pl .meta{font-size:11.1px;color:var(--dim2);margin-top:5px;font-family:ui-monospace,Consolas,monospace;line-height:1.5}
.up{color:var(--good)} .down{color:var(--bad)} .flat{color:var(--dim)}
footer{margin-top:54px;padding-top:18px;border-top:1px solid var(--line);color:var(--dim2);font-size:12.2px}
@media(max-width:820px){.arms,.pls{grid-template-columns:1fr}}
"""

JS = """
const R={},N={};
document.querySelectorAll('audio').forEach(a=>{R[a.id]=a.dataset.raw;N[a.id]=a.dataset.norm;});
function setMode(m){
  document.querySelectorAll('.switch button').forEach(b=>b.classList.toggle('on',b.dataset.m===m));
  document.querySelectorAll('audio').forEach(a=>{a.src=(m==='raw'?R:N)[a.id];});
  document.getElementById('mh').textContent=(m==='raw')
    ?'原始输出：各系统各自电平（TTS 源峰值归一 0.9，真人源是游戏原始电平），响度不统一。'
    :'统一响度：每条单独拉到 −20 dBFS、峰值限 0.95。跨通道横向比听感请用这一档。';
}
"""


CONCLUSIONS = """
<h3>① 两条通道都能用，但「像不像」的判据在两条通道上不等价</h3>
<p class="note">
TTS 通道：源 <code>r=0.8655</code> → 换声后 <code>0.9568</code>，<b>Δr +0.091</b>，音色确实朝芳乃移动了一大截。<br>
真人通道：源已经是 <code>0.9629</code>，换声后 <code>0.9632</code>，<b>Δr ≈ 0</b>。
<b>不是真人通道没换成功</b>——它本来就贴着目标：茉子和芳乃是同一个游戏、同一套录音链路的声优，
长时平均谱（LTAS）被<b>录音信道</b>主导，语音内容只占很小一部分。参考集自测上限是 0.9863，
真人源一进来就已经 0.963，剩下的 0.023 才是「音色」，Δr 根本不够分辨。
</p>
<p class="note">
⇒ <b>方法论结论</b>：Δr 只能在<b>同一信道内部</b>比较（比如都是同一个原唱干声时比 RVC 与 DDSP），
<b>不能跨信道比</b>。之前「DDSP Δr 为负所以更差」的结论要加上这个限定，否则会被误用。
</p>

<h3>② 可懂度：TTS 通道不掉字，真人通道反而掉一点</h3>
<p class="note">
TTS 通道 CER <b>0.0375 → 0.0375</b>（换声前后完全一致，一个字没多丢）；
真人通道 <b>0.000 → 0.092</b>。反直觉，但说得通：真人台词里有更多气口、弱读、说话尾巴，
RVC 换的是音色，气口处没有可靠的 f0/内容，容易被"抹平"；TTS 源干净、字字清晰，反而更抗换声。<br>
<b>但别把这条读成「TTS 更好」</b>——CER 只测「听得清」，测不出语气自然度。TTS 的念白有合成腔，
这一点 ASR 完全看不见。
</p>

<h3>③ e120 vs e200：客观指标偏 e120，主观听感偏 e200</h3>
<p class="note">
Δr 两者几乎相同（+0.0912 / +0.0911），但 CER <b>e120 更低</b>（T 通道 0.0375 vs 0.0581，真人 0.0917 vs 0.1167）。
也就是说：<b>多训 80 轮没有让字更清楚，反而略微更糊</b>。
这和你的听感判断（e200 略好但边际效率低）并不矛盾——你听的是自然度/音色质感，CER 听的是"能不能听懂"，
这两个指标在本任务上<b>不重叠</b>。所以本页不给"哪个更好"的裁决，只给分层事实：
<b>要清楚 → e120；要质感 → 依你耳朵；两者差距都很小。</b>
</p>

<h3>④ 一个必须写下来的陷阱：这次翻车过一次</h3>
<p class="note">
第一版 TTS 通道换声后 CER <b>1.0</b>（完全不可懂），看起来像"TTS+RVC 这条路线走不通"。
实际根因是 <b>GPT-SoVITS v3 输出的是 int16 量级数组（峰值 22352），
而 <code>soundfile</code> 按 float 写入 PCM_16 会硬削顶</b>——削波产生的谐波把 HuBERT 特征彻底污染，
换声自然全错。归一化后 CER 立刻回到 0.0375。<br>
<b>教训</b>：跨工具链传音频，写盘前必须查 dtype 与峰值；看到"换声把人声搞崩了"，
先怀疑输入电平，再怀疑模型。
</p>
"""


def build():
    met = json.load(open(DLG / "metrics.json", encoding="utf-8"))
    cer = json.load(open(DLG / "cer.json", encoding="utf-8"))
    C = {(r["arm"], r["lid"], r["model"]): r for r in cer}
    M = {(r["arm"], r["lid"], r["model"]): r for r in met}

    def agg(arm, mdl):
        rows = [M[(arm, l, mdl)] for l in LIDS if (arm, l, mdl) in M]
        cs = [C[(arm, l, mdl)]["cer"] for l in LIDS if (arm, l, mdl) in C]
        return dict(
            r=float(np.mean([x["r_lta"] for x in rows])),
            dr=float(np.mean([x["dr"] for x in rows])),
            dur=float(np.mean([x["dur_ratio"] for x in rows])),
            zero=float(np.mean([x["zero"] for x in rows])),
            cer=float(np.mean(cs)) if cs else float("nan"),
        )

    A = {(arm, s["sid"]): agg(arm, s["sid"]) for arm in ARMS for s in SYSTEMS}

    # 通道卡
    acard = []
    for k, v in ARMS.items():
        cls = "arm t" if k == "T" else "arm h"
        acard.append(f"""
      <div class="{cls}">
        <span class="tag">{v['tag']}</span>
        <div class="head"><span class="badge">{k}</span><span class="armname">{v['name']}</span></div>
        <div class="what">{v['what']}</div>
        <dl><dt>模型 / 权重</dt><dd class="m">{v['model']}</dd>
            <dt>参数</dt><dd>{'<br>'.join(v['params'])}</dd></dl>
      </div>""")

    # 播放块
    blocks = []
    for lid in LIDS:
        sides = []
        for arm in ("T", "H"):
            pls = []
            for s in SYSTEMS:
                key = f"{lid}_{arm}_{s['sid']}"
                if not (AUD / f"{key}_raw.wav").exists():
                    continue
                m = M.get((arm, lid, s["sid"]))
                c = C.get((arm, lid, s["sid"]))
                cls = "pl t" if arm == "T" else "pl h"
                if s["sid"] == "src":
                    d = '<div class="meta">标尺 · 不换声</div>'
                else:
                    dr = m["dr"]
                    cl = "up" if dr > 0.005 else ("down" if dr < -0.005 else "flat")
                    sg = "↑" if dr > 0.005 else ("↓" if dr < -0.005 else "≈")
                    d = (f'<div class="meta {cl}">{sg} Δr {dr:+.4f} · CER {c["cer"]:.2f}</div>')
                extra = ""
                if c:
                    extra = f'<div class="meta" style="color:#5f6b78">ASR: {c["hyp"][:26]}</div>'
                pls.append(f"""
            <div class="{cls}">
              <div class="lbl"><span class="mini">{s['short']}</span><span class="nm">{s['name']}</span></div>
              <audio id="{key}" controls preload="none" src="dialogue/audio/{key}_raw.wav"
                data-raw="dialogue/audio/{key}_raw.wav" data-norm="dialogue/audio/{key}_norm.wav"></audio>
              {d}{extra}
            </div>""")
            sides.append(f"""
          <div class="side">{'TTS 通道' if arm == 'T' else '真人通道'}　{ARMS[arm]['name']}</div>
          <div class="pls">{''.join(pls)}</div>""")

        blocks.append(f"""
    <div class="clipblk">
      <div class="cliphead"><span class="clipid">台词 {lid}</span>
        <span class="clipdesc" style="font-size:12.5px;color:#8b98a5">
          真人来源 {REAL_FILE[lid]}</span></div>
      <p class="dlgtext">「{TEXT[lid]}」</p>
      {''.join(sides)}
    </div>""")

    # 指标表
    trs = []
    for arm in ("T", "H"):
        for s in SYSTEMS:
            a = A[(arm, s["sid"])]
            cls = "—" if s["sid"] == "src" else ("up" if a["dr"] > 0.005 else ("down" if a["dr"] < -0.005 else "flat"))
            trs.append(f"""<tr>
        <td><b>{arm}</b></td><td>{s['name']}</td>
        <td class="num">{a['r']:+.4f}</td>
        <td class="num {cls}">{a['dr']:+.4f}</td>
        <td class="num">{a['cer']:.3f}</td>
        <td class="num">{a['dur']:.3f}×</td>
        <td class="num">{a['zero']:.4f}</td>
      </tr>""")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>台词换声对照页 · TTS+RVC vs 真人+RVC</title>
<style>{CSS}</style></head><body><div class="wrap">

<h1>台词换声对照页 · TTS+RVC vs 真人+RVC</h1>
<p class="lead">同一批台词、同一个 RVC 模型，只换「源」：一条是 TTS 合成，一条是真人声优原声。
每条音频独立成文件、独立播放，逐条标注它是什么、哪个模型、什么参数。</p>

<div class="rule">
  <b>读法</b>：每个台词块里，上面三格是 <b>TTS 通道</b>（源 → e120 → e200），下面三格是 <b>真人通道</b>。
  想横向比听感，先切「统一响度」；要听真实电平，留在「原始输出」。
</div>

<h2>1. 两条通道分别是什么</h2>
<div class="arms">{''.join(acard)}</div>
<p class="note">台词文本由真人录音 ASR 转写后，用 TTS 按<b>完全相同的文本</b>合成 ⇒ 两条通道的差异只剩「谁说」，
不掺「说了什么」的变量。台词 4 句，长度 2.3 s ~ 7.6 s。</p>

<h2>2. 逐条试听</h2>
<div class="switch"><span>版本：</span>
  <button class="on" data-m="raw" onclick="setMode('raw')">原始输出</button>
  <button data-m="norm" onclick="setMode('norm')">统一响度</button>
</div>
<p class="note" id="mh" style="margin:-8px 0 18px">原始输出：各系统各自电平（TTS 源峰值归一 0.9，真人源是游戏原始电平）。</p>
{''.join(blocks)}

<h2>3. 客观指标（4 句均值）</h2>
<table>
  <thead><tr><th>通道</th><th>系统</th>
    <th style="text-align:right">r_lta</th><th style="text-align:right">Δr</th>
    <th style="text-align:right">ASR 回读 CER</th><th style="text-align:right">时长比</th>
    <th style="text-align:right">精确归零</th></tr></thead>
  <tbody>{''.join(trs)}</tbody>
</table>
<p class="note">参考集自测上限 <code>r = 0.9863</code>。
<b>Δr</b> = 换声后与芳乃参考集的 LTAS 相关 − 换声前；<b>CER</b> 越低越听得清，但含 ASR 自身误差 ⇒
只看相对变化（同一通道内比），不要当绝对可懂度。</p>

<h2>4. 这批数字说明什么</h2>
{CONCLUSIONS}

<footer>
模型：<code>models\芳乃_RVC_步数120</code> 与 <code>…\\芳乃_RVC_步数200</code><br>
脚本：<code>dialogue_tts_ja.py</code>（TTS）· <code>dialogue_conv.py</code>（换声）·
<code>dialogue_metrics.py</code>（音色）· <code>dialogue_cer.py</code>（可懂度）· <code>build_dialogue_page.py</code>（本页）
</footer>

</div><script>{JS}</script></body></html>"""
    p = OUT / "台词换声对照页.html"
    p.write_text(html, encoding="utf-8")
    print("->", p)


if __name__ == "__main__":
    make_audio()
    build()
