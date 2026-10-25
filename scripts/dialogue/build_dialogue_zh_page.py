# -*- coding: utf-8 -*-
"""生成「中文台词换声独立对照页」。

纪律 D1：每条音频独立成文件、独立播放、逐条标注「是什么 / 哪个模型 / 什么参数」；
不拼接串烧。
"""
import sys as _sys
from pathlib import Path as _pth

_sys.path.insert(0, str(_pth(__file__).resolve().parents[2]))
from rvcx import paths as _paths  # noqa: E402

import json
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

DLG = _paths.WORK / "dialogue_zh"
OUT = _paths.WORK / "report"
AUD = OUT / "dialogue_zh" / "audio"

TEXT = {
    "zh01": "请只回答我的问题。",
    "zh02": "别看我的脸，就这样听我说。",
    "zh03": "啊，失礼了。我叫日立真子。",
    "zh04": "早上我首先会打扫浴室，因为我挺喜欢泡澡的。",
    "zh05": "真是的，都说了三遍了，你怎么还不明白啊？",
    "zh06": "那个人明明什么都知道，却什么都不肯说，就一直笑着。",
}
NOTE = {
    "zh01": "对应 ja01 · 短陈述",
    "zh02": "对应 ja02 · 祈使 + 否定",
    "zh03": "对应 ja03 · 双句 + 语气词（真/贞 同音，汉字层 CER 会假报警）",
    "zh04": "对应 ja04 · 长宾语从句",
    "zh05": "中文特有 · 语气词 + 数字 + 反问",
    "zh06": "中文特有 · 多分句长句 + 连续齿音 sh/zh/ch",
}
LIDS = ["zh01", "zh02", "zh03", "zh04", "zh05", "zh06"]

MAIN = ["src", "e120", "e200"]
IR = ["e120_ir0", "e120_ir050", "e120"]
IR_LABEL = {"e120_ir0": "芳乃 e120 · 关索引（ir 0）",
            "e120_ir050": "芳乃 e120 · 半索引（ir 0.50）",
            "e120": "芳乃 e120 · 全索引（ir 0.75）"}

SYSTEMS = {
    "src":          dict(short="源", name="源（CosyVoice2 未换声）",
                         params=["cross_lingual · 晓伊 A3 参考音色", "24 kHz 单声道"]),
    "e120":         dict(short="e120", name="RVC · 芳乃 e120",
                         params=["rmvpe · index_rate 0.75 · pitch 0"]),
    "e200":         dict(short="e200", name="RVC · 芳乃 e200",
                         params=["rmvpe · index_rate 0.75 · pitch 0"]),
    "e120_ir0":     dict(short="ir 0", name="RVC · 芳乃 e120 · 关索引",
                         params=["rmvpe · index_rate 0.0 · pitch 0"]),
    "e120_ir050":   dict(short="ir .5", name="RVC · 芳乃 e120 · 半索引",
                         params=["rmvpe · index_rate 0.50 · pitch 0"]),
}

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
margin-top:1px;border-radius:7px;font-weight:700;font-size:13px;font-family:ui-monospace,Consolas,monospace;
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
.pl .mini{min-width:20px;height:20px;padding:0 6px;border-radius:5px;display:inline-flex;align-items:center;
justify-content:center;font-size:10.5px;font-weight:700;background:#243040;color:#cfe0ee;flex:0 0 auto;
font-family:ui-monospace,Consolas,monospace;white-space:nowrap}
.pl.t .mini{background:rgba(227,179,65,.16);color:var(--warn)}
.pl.h .mini{background:rgba(78,201,176,.16);color:var(--acc)}
.pl .nm{font-size:12.6px;font-weight:550}
audio{width:100%;height:32px;margin-top:6px}
.pl .meta{font-size:11.1px;color:var(--dim2);margin-top:5px;font-family:ui-monospace,Consolas,monospace;line-height:1.5}
.up{color:var(--good)} .down{color:var(--bad)} .flat{color:var(--dim)}
.kpi{display:grid;grid-template-columns:repeat(4,1fr);gap:11px;margin:16px 0 6px}
.k{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:13px 14px}
.k .v{font-size:20px;font-weight:650;font-family:ui-monospace,Consolas,monospace;color:var(--acc)}
.k .l{font-size:11.6px;color:var(--dim);margin-top:3px;line-height:1.45}
.k .s{font-size:11.2px;color:var(--dim2);margin-top:5px}
footer{margin-top:54px;padding-top:18px;border-top:1px solid var(--line);color:var(--dim2);font-size:12.2px}
@media(max-width:820px){.arms,.pls,.kpi{grid-template-columns:1fr}}
"""

JS = """
const R={},N={};
document.querySelectorAll('audio').forEach(a=>{R[a.id]=a.dataset.raw;N[a.id]=a.dataset.norm;});
function setMode(m){
  document.querySelectorAll('.switch button').forEach(b=>b.classList.toggle('on',b.dataset.m===m));
  document.querySelectorAll('audio').forEach(a=>{a.src=(m==='raw'?R:N)[a.id];});
  document.getElementById('mh').textContent=(m==='raw')
    ?'原始输出：源是 TTS 输出电平（峰值 0.38~0.64），换声输出再加 +1.5~6.5 dB。'
    :'统一响度：每条单独拉到 −20 dBFS、峰值限 0.95。比听感请用这一档。';
}
"""


def rms(y):
    return float(np.sqrt(np.mean(np.square(y, dtype=np.float64))))


def make_audio():
    AUD.mkdir(parents=True, exist_ok=True)
    n = 0
    for lid in LIDS:
        for sid in SYSTEMS:
            p = (DLG / "src_tts" / f"{lid}.wav") if sid == "src" \
                else (DLG / "out" / f"{lid}_{sid}.wav")
            if not p.exists():
                continue
            y, sr = sf.read(str(p), dtype="float32", always_2d=False)
            if y.ndim > 1:
                y = y.mean(1)
            key = f"{lid}_{sid}"
            sf.write(AUD / f"{key}_raw.wav", y, sr, subtype="PCM_16")
            g = 10 ** (-20 / 20) / max(rms(y), 1e-9)
            yn = y * g
            pk = np.abs(yn).max()
            if pk > 0.95:
                yn = yn * (0.95 / pk)
            sf.write(AUD / f"{key}_norm.wav", yn, sr, subtype="PCM_16")
            n += 1
    print(f"音频 {n} 组 × 2 版")


def card(lid, sid, M, C, cls="pl", label=None):
    key = f"{lid}_{sid}"
    if not (AUD / f"{key}_raw.wav").exists():
        return ""
    s = SYSTEMS[sid]
    nm = label or s["name"]
    mini = s["short"]
    m, c = M.get((lid, sid)), C.get((lid, sid))
    if sid == "src":
        meta = '<div class="meta">标尺 · 未换声</div>'
    else:
        dr, tr = m["dr"], m["tone_r"]
        cl = "up" if dr > 0.005 else ("down" if dr < -0.005 else "flat")
        sg = "↑" if dr > 0.005 else ("↓" if dr < -0.005 else "≈")
        meta = (f'<div class="meta {cl}">{sg} Δr {dr:+.4f} · 声调 r {tr:.2f} · '
                f'CER {c["cer_py"]:.2f}</div>')
    hyp = f'<div class="meta" style="color:#5f6b78">ASR: {c["hyp"][:24]}</div>' if c else ""
    return f"""
            <div class="{cls}">
              <div class="lbl"><span class="mini">{mini}</span><span class="nm">{nm}</span></div>
              <audio id="{key}" controls preload="none" src="dialogue_zh/audio/{key}_raw.wav"
                data-raw="dialogue_zh/audio/{key}_raw.wav" data-norm="dialogue_zh/audio/{key}_norm.wav"></audio>
              {meta}{hyp}
            </div>"""


def build():
    met = json.load(open(DLG / "metrics.json", encoding="utf-8"))
    cer = json.load(open(DLG / "cer.json", encoding="utf-8"))
    tts = json.load(open(DLG / "tts_manifest.json", encoding="utf-8"))
    M = {(r["lid"], r["sid"]): r for r in met}
    C = {(r["lid"], r["sid"]): r for r in cer}
    T = {r["lid"]: r for r in tts}

    def agg(sid, key):
        v = [M[(l, sid)][key] for l in LIDS if (l, sid) in M]
        v = [x for x in v if x is not None]
        return float(np.mean(v)) if v else float("nan")

    def agg_cer(sid, key="cer_py"):
        v = [C[(l, sid)][key] for l in LIDS if (l, sid) in C]
        return float(np.mean(v)) if v else float("nan")

    # 播放块（主）
    blocks = []
    for lid in LIDS:
        cards = "".join(card(lid, s, M, C, "pl t" if s == "src" else "pl") for s in MAIN)
        blocks.append(f"""
    <div class="clipblk">
      <div class="cliphead"><span class="clipid">台词 {lid}</span>
        <span style="font-size:12.5px;color:#8b98a5">{NOTE[lid]} · 源时长 {T[lid]['dur']:.2f} s</span></div>
      <p class="dlgtext">「{TEXT[lid]}」</p>
      <div class="pls">{cards}</div>
    </div>""")

    # index_rate 对照块
    irblocks = []
    for lid in LIDS:
        cards = "".join(card(lid, s, M, C, label=IR_LABEL[s]) for s in IR)
        irblocks.append(f"""
    <div class="clipblk">
      <div class="cliphead"><span class="clipid">{lid}</span>
        <span style="font-size:12.5px;color:#8b98a5">「{TEXT[lid]}」</span></div>
      <div class="pls">{cards}</div>
    </div>""")

    # 指标表
    trs = []
    for sid in ["src"] + ["e120", "e200", "e120_ir0", "e120_ir050"]:
        s = SYSTEMS[sid]
        dr = agg(sid, "dr")
        cls = "—" if sid == "src" else ("up" if dr > 0.005 else ("down" if dr < -0.005 else "flat"))
        trs.append(f"""<tr>
        <td>{s['name']}</td>
        <td class="num">{agg(sid,'r_lta'):+.4f}</td>
        <td class="num {cls}">{dr:+.4f}</td>
        <td class="num">{agg_cer(sid,'cer_py'):.3f}</td>
        <td class="num">{agg_cer(sid,'cer_tone'):.3f}</td>
        <td class="num">{agg(sid,'tone_r'):.3f}</td>
        <td class="num">{agg(sid,'tone_rmse'):.2f}</td>
        <td class="num">{agg(sid,'dur_ratio'):.3f}×</td>
        <td class="num">{agg(sid,'zero'):.4f}</td>
      </tr>""")

    srcdur = np.mean([T[l]["dur"] for l in LIDS])

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>中文台词换声对照页 · 日语训练的芳乃 RVC 换中文</title>
<style>{CSS}</style></head><body><div class="wrap">

<h1>中文台词换声对照页 · 日语训练的芳乃 RVC 换中文</h1>
<p class="lead">芳乃的 RVC 模型（e120 / e200）训练集<b>只有日语</b>语音。本页只换一个变量：
把源从日语换成中文，看换声还成不成立——像不像、听不听得清、<b>声调有没有被抹平</b>。
每条音频独立成文件、独立播放，逐条标注它是什么、哪个模型、什么参数。</p>

<div class="rule">
  <b>读法</b>：每块三格 = 源（未换声）→ e120 → e200。想横向比听感先切「统一响度」。
  <b>重点听两件事</b>：① 换声后还听得出是中文吗；② 四个声调（妈麻马骂）有没有走调。
</div>

<div class="kpi">
  <div class="k"><div class="v">+{agg('e120','dr'):.3f}</div><div class="l">Δr（e120）</div>
    <div class="s">源 r={agg('src','r_lta'):.3f} → 换声后 {agg('e120','r_lta'):.3f}</div></div>
  <div class="k"><div class="v">{agg_cer('e120','cer_py'):.3f} / {agg('e120','tone_r'):.2f}</div>
    <div class="l">拼音 CER / 声调轮廓 r</div>
    <div class="s">源 {agg_cer('src','cer_py'):.3f} / 1.00 → 掉字与走调都没出现</div></div>
  <div class="k"><div class="v">{agg_cer('e120','cer_tone'):.3f}</div><div class="l">声调 CER（ASR 判定）</div>
    <div class="s">24 条换声输出里，没有一个字的调被听错</div></div>
  <div class="k"><div class="v">{agg('e120','dur_ratio'):.3f}×</div><div class="l">时长比</div>
    <div class="s">句长基本不变（{srcdur:.2f} s 平均源）</div></div>
</div>

<h2>1. 这批的源是什么</h2>
<div class="arms">
  <div class="arm t">
    <span class="tag">源引擎</span>
    <div class="head"><span class="badge">T</span><span class="armname">CosyVoice2-0.5B（中文，晓伊 A3 参考音色）</span></div>
    <div class="what">为什么不用日语批那套 GPT-SoVITS：本机 kurumi / yoshino 都是日语数据训的，喂中文它自己先念错，
    测出来的是「TTS 不会中文」而不是「RVC 能不能换中文」。</div>
    <dl><dt>模型 / 权重</dt><dd class="m">CosyVoice2-0.5B（本地部署）</dd>
        <dt>参数</dt><dd>cross_lingual · 24 kHz · speed 1.0 · seed 20261025<br>参考音 reference.wav（13.94 s，非芳乃音色）</dd></dl>
  </div>
  <div class="arm h">
    <span class="tag">目标模型</span>
    <div class="head"><span class="badge">R</span><span class="armname">芳乃 RVC v2（48k，日语训练）</span></div>
    <div class="what">train/** 链路自训，训练集 359 条日语语音（SenrenBanka 芳乃）。
    中文从来没进过训练集，这正是本批要测的<b>跨语言迁移</b>。</div>
    <dl><dt>模型 / 权重</dt><dd class="m">models\芳乃_RVC_步数{{120,200}}\\model.pth<br>
    + added_IVF1419_Flat_nprobe_1_fangno_p2_v2.index</dd>
        <dt>参数</dt><dd>rmvpe · pitch 0（保留源 f0）· protect 0.33</dd></dl>
  </div>
</div>
<p class="note">6 条台词：前 4 条是日语那 4 条的<b>对应中译</b>（语义对齐，便于跨语言比较），
后 2 条是中文特有难点（语气词/数字/反问、多分句长句与连续齿音）。</p>

<h2>2. 逐条试听（源 → e120 → e200）</h2>
<div class="switch"><span>版本：</span>
  <button class="on" data-m="raw" onclick="setMode('raw')">原始输出</button>
  <button data-m="norm" onclick="setMode('norm')">统一响度</button>
</div>
<p class="note" id="mh" style="margin:-8px 0 18px">原始输出：源是 TTS 输出电平（峰值 0.38~0.64），换声输出再加 +1.5~6.5 dB。</p>
{''.join(blocks)}

<h2>3. index_rate 在中文上到底有没有用</h2>
<p class="note">索引库（<code>added_*.index</code>）里全是<b>日语</b>帧向量。中文音素在库里找不到近邻时，
检索会把特征拉向日语分布——这是「中文被日语音系带跑」最可能的机制，所以单独扫一遍：
同一句、同一个 e120 模型，只改 index_rate（0 / 0.50 / 0.75）。</p>
{''.join(irblocks)}

<h2>4. 客观指标（6 句均值）</h2>
<table>
  <thead><tr><th>系统</th>
    <th style="text-align:right">r_lta</th><th style="text-align:right">Δr</th>
    <th style="text-align:right">拼音 CER</th><th style="text-align:right">声调 CER</th>
    <th style="text-align:right">声调轮廓 r</th><th style="text-align:right">半音 RMSE</th>
    <th style="text-align:right">时长比</th><th style="text-align:right">精确归零</th></tr></thead>
  <tbody>{''.join(trs)}</tbody>
</table>
<p class="note">
  <b>拼音 CER</b>：汉字先转带声调拼音再算编辑距离 ⇒ 同音字（真/贞）不算错，<b>同时把声调算进去</b>。<br>
  <b>声调 CER</b>：只比声调序列（妈=1 麻=2 马=3 骂=4），测「调有没有被换掉」。<br>
  <b>声调轮廓 r</b>：源与出的 f0 曲线（转半音）线性对齐后求相关；<b>半音 RMSE</b> 是它的幅度误差。
  两个都用 <code>rmvpe</code>（与换声器同一个 f0 提取器）取 f0。
</p>

<h2>5. 这批数字说明什么</h2>
<h3>① 跨语言迁移成立：日语训练的模型换中文，<b>不掉字</b></h3>
<p class="note">
拼音 CER 源 {agg_cer('src','cer_py'):.3f} → e120 {agg_cer('e120','cer_py'):.3f} / e200 {agg_cer('e200','cer_py'):.3f}。
差的这一点点是 ASR 自己认错字（<code>却</code>→<code>这</code>），不是换声糊掉的：
源那一侧同样有 <code>真子→贞子</code> 这类同音误判。换声前后<b>掉字量级不变</b>，
和日语通道（CER 0.0375 → 0.0375）是同一个结论。
<span style="color:#6b7885">（日语那边是汉字层 CER，与这里的拼音层不可直接比数字——比的是结论，不是数字。）</span>
</p>

<h3>② 声调是<b>被保住</b>的，但要看清是怎么保住的</h3>
<p class="note">
声调 CER = <b>{agg_cer('e120','cer_tone'):.3f}</b>（全 0），声调轮廓 r = <b>{agg('e120','tone_r'):.3f}</b>、
半音 RMSE <b>{agg('e120','tone_rmse'):.2f}</b>——四个调型原样过。<br>
<b>但这不是模型「懂汉语」</b>：<code>pitch 0</code> 的语义是「直接用源 f0 驱动」，f0 轮廓根本没经过模型决策。
换句话说，中文能保住调，靠的是这条工程约定，<b>不是跨语言能力</b>。
一旦你为了变调去开 pitch shift（±N 半音），保的是轮廓、丢的是音区——唱中文歌时这一点要单独验。
</p>

<h3>③ Δr 比日语通道大一倍，但<b>不能读成「中文换得更好」</b></h3>
<p class="note">
中文源 r=0.756 → 换声 0.971，<b>Δr +{agg('e120','dr'):.3f}</b>；日语 TTS 通道只 +0.091。
差在<b>起点</b>：这次源是 CosyVoice 的晓伊音色，与芳乃既不同人也不是同一套录音链路，
LTAS 离得远（0.756），所以有 0.21 的空间可挪；日语那批的 kurumi 起点本来就近（0.866）。<br>
⇒ <b>Δr 是「挪了多远」，不是「像不像」的绝对值</b>。跨源比 Δr 大小没有意义，
这与日语批里「Δr 受信道污染、不能跨通道比」是同一条纪律。
</p>

<h3>④ index_rate 在中文上<b>没有收益</b>，甚至可以关掉</h3>
<p class="note">
同一句、同一个 e120：ir 0 / 0.50 / 0.75 的 Δr 是
{agg('e120_ir0','dr'):+.4f} / {agg('e120_ir050','dr'):+.4f} / {agg('e120','dr'):+.4f}，
声调轮廓 r 是 {agg('e120_ir0','tone_r'):.3f} / {agg('e120_ir050','tone_r'):.3f} / {agg('e120','tone_r'):.3f}，
拼音 CER 是 {agg_cer('e120_ir0','cer_py'):.3f} / {agg_cer('e120_ir050','cer_py'):.3f} / {agg_cer('e120','cer_py'):.3f}。
差异全在 1~2 个字的量级，属于噪声——<b>索引检索在中文上既没帮忙也没显著坏事</b>。
工程含义：中文台词场景可以直接 <code>--index-rate 0</code>，省掉索引文件、也省掉「日语帧被误检索」的理论风险。
（但听感上索引通常让音色更「贴」，这一条<b>请以你自己的耳朵为准</b>，客观指标看不见它。）
</p>

<h3>⑤ e120 vs e200：第三次得到同一个答案</h3>
<p class="note">
Δr {agg('e120','dr'):+.4f} vs {agg('e200','dr'):+.4f}，声调轮廓 {agg('e120','tone_r'):.3f} vs {agg('e200','tone_r'):.3f}，
拼音 CER {agg_cer('e120','cer_py'):.3f} vs {agg_cer('e200','cer_py'):.3f}——<b>客观指标不可分</b>。
与日语批、与唱歌批的结论一致：<b>新角色 100~150 轮封顶</b>，200 轮买不到可测量的东西。
</p>

<h3>⑥ 还缺一环：<b>真人中文源</b></h3>
<p class="note">
日语批是 TTS / 真人双通道对撞；中文这批只有 TTS 通道，因为本机<b>没有任何中文真人录音素材</b>
（所有 GPT-SoVITS 角色、芳乃数据集都是日语）。要补齐，录 3 句就够（16 kHz 以上、安静房间、正常语速）：
<br><code>别看我的脸，就这样听我说。</code> / <code>早上我首先会打扫浴室，因为我挺喜欢泡澡的。</code> / <code>那个人明明什么都知道，却什么都不肯说，就一直笑着。</code>
<br>存成 <code>E:\\AI-RVC\\dialogue_zh\\src_real\\zh02.wav</code> 等（文件名对齐），
<code>dialogue_conv_zh.py</code>、<code>dialogue_metrics_zh.py</code>、<code>dialogue_cer_zh.py</code> 里把 <code>src_tts</code> 换成 <code>src_real</code> 即可复跑同一套判据。
</p>

<h2>6. 方法上踩到的三个坑（写下来免得重犯）</h2>
<p class="note">
<b>坑 1 · CosyVoice 的 instruct2 会把指令念出来。</b>
第一次合成时 zh01 回读成「请只回答我的问题，<b>语速适中，句间自然停顿</b>」，CER 1.25。
根因：instruct 必须带 <code>You are a helpful assistant. </code> 前缀 + <code>&lt;|endofprompt|&gt;</code> 结尾，
少了就被当成正文继续念。补上前缀后长句仍会漏尾巴（zh04 多出「语速适中」）
⇒ 本批改用 <code>cross_lingual</code>（只要音色、无指令文本，没有可泄漏的东西）。
</p>
<p class="note">
<b>坑 2 · 用 pyin 测声调会假报警。</b>
第一版声调指标算出 zh04 的 r=0.44、RMSE 4 半音（等于换调型），但 ASR 的声调 CER 是 0——
两者不可能同时为真。查下去发现换声后 pyin 的半音 std 从 2.94 跳到 6.92（八度跳变），
是<b>估计器</b>坏了。改用换声器自身的 <code>rmvpe</code> 后 r=0.99。
<b>教训</b>：两个判据打架时，先怀疑测量工具，别急着改结论。
</p>
<p class="note">
<b>坑 3 · DTW 在短句上会把曲线拉错。</b>
zh02 只有 1.2 s 浊音，DTW(subseq) 给出 r=0.64；而时长比只有 0.990（差 1%），
根本不需要弯曲。改线性对齐后同一批稳定在 0.976~0.990。
</p>

<footer>
目标模型：<code>models\芳乃_RVC_步数120</code> 与 <code>…\\芳乃_RVC_步数200</code><br>
本地素材：<code>CosyVoice2</code>（TTS）· <code>目标角色训练集</code>（LTAS 参考集，自测上限 r=0.9863）<br>
脚本：<code>dialogue_tts_zh.py</code>（中文源）· <code>dialogue_conv_zh.py</code>（换声 + index_rate 扫描）·
<code>dialogue_metrics_zh.py</code>（音色/声调）· <code>dialogue_cer_zh.py</code>（三层 CER）· <code>build_dialogue_zh_page.py</code>（本页）
</footer>

</div><script>{JS}</script></body></html>"""
    p = OUT / "中文台词对照页.html"
    p.write_text(html, encoding="utf-8")
    print("->", p)


if __name__ == "__main__":
    make_audio()
    build()
