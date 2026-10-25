# -*- coding: utf-8 -*-
"""批量合成训练集：句表 → 每条独立 wav（+ 归一化 + 质检清单）。

两个后端，音色链完全不同，别混着跑：

  edge  zh-CN-XiaoyiNeural（Microsoft Edge 神经 TTS）
        ★ 本课题的**正源**：DS娘 的 reference.wav 就是它合成的（rate -5%），
          同引擎同 voice ⇒ 音色分布天然对齐，无漂移。
        ★ 需要 edge-tts **>= 7.3.0**。7.2.8 会在 WSS 握手时返回 403
          （`Sec-MS-GEC-Version` 太旧被服务端拒绝），报错信息完全不像版本问题。
        ★ 输出是 24 kHz / 48 kbps mp3（服务端唯一给得起的无损格式 riff-*-pcm 被拒）。
          实测同句 48 kbps 与 96 kbps 在 0–11 kHz 差 ≤0.2 dB，只在 11–12 kHz 差 1.7 dB，
          而 HuBERT 吃的是 16 kHz（上限 8 kHz）⇒ 默认码率够用，不必折腾高码率补丁。

  cosy  CosyVoice2-0.5B zero-shot / cross_lingual（拿 reference.wav 克隆音色）
        ★ 只用于**交叉验证集**与「表现力补充」，不用于主训练集：
          每次合成都有音色抖动，混进训练集会污染目标音色分布（详见方案文档）。
        ★ 中文台词一律 cross_lingual：instruct2 即使补了 `You are a helpful assistant. `
          前缀 + `<|endofprompt|>`，长句仍会漏念尾巴（本机踩过）。

写盘前三道闸（历史坑，缺一条就会白训一晚）：
  1. **dtype 换算**：TTS 常返回 int16 量级数组（峰值 2 万+）。直接按 float 写
     PCM_16 会硬削顶，削顶谐波污染 HuBERT ⇒ 换声后 CER 直接 1.0。
  2. **峰值检查**：归一化后 peak 必须 ≤ 0.951，且不许出现 |x| ≥ 0.999 的样本。
  3. **静音修剪 + 最短时长**：边缘静音会让 preprocess 切出一堆半空切片。

用法：
  # 主训练集（edge，需 edge-tts>=7.3.0 的 env）
  <env>/Scripts/python.exe scripts/dataset/synth_corpus.py \
      --texts E:/AI-RVC/corpus/dsxiaoyi/text.txt --out E:/AI-RVC/dataset/dsxiaoyi_a3

  # 交叉验证集（cosy，用 CosyVoice env）
  <cosy-env>/Scripts/python.exe scripts/dataset/synth_corpus.py --backend cosy \
      --texts E:/AI-RVC/corpus/dsxiaoyi_holdout/text.txt --out E:/AI-RVC/dataset/dsxiaoyi_xval
"""
import sys as _sys
from pathlib import Path as _pth

_sys.path.insert(0, str(_pth(__file__).resolve().parents[2]))
from rvcx import paths as _paths  # noqa: E402

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_RMS_DBFS = -20.0
PEAK_CAP = 0.95
TRIM_DB = -45.0          # 相对峰值的静音门限
TRIM_PAD_S = 0.03


# --------------------------------------------------------------------------- #
# 质检 + 归一化（后端无关）
# --------------------------------------------------------------------------- #
def normalize(a: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
    """dtype 换算 → 静音修剪 → -20 dBFS 归一 → 限峰。返回 (音频, 质检记录)。"""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    qc: dict = {}

    # ① dtype / 量级：int16 量级数组会被误当 float 写盘，必须先摊平
    raw_peak = float(np.max(np.abs(a))) if a.size else 0.0
    qc["raw_peak"] = round(raw_peak, 4)
    if raw_peak > 4.0:
        a = a / 32768.0
        qc["rescaled_from"] = "int16"
    elif raw_peak > 1.0:
        a = a / max(raw_peak, 1e-9)
        qc["rescaled_from"] = "float_over_1"
    else:
        qc["rescaled_from"] = "none"

    # ② 去直流
    a = a - float(np.mean(a)) if a.size else a

    # ③ 首尾静音修剪（按 20 ms 帧能量，门限相对本段峰值）
    if a.size > sr // 10:
        hop = max(1, int(0.02 * sr))
        n = a.size // hop
        frames = a[: n * hop].reshape(n, hop)
        db = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-12)
        rel = db - db.max()
        loud = np.where(rel > TRIM_DB)[0]
        if loud.size:
            pad = int(TRIM_PAD_S * sr)
            s = max(0, loud[0] * hop - pad)
            e = min(a.size, (loud[-1] + 1) * hop + pad)
            qc["trim_s"] = round((a.size - (e - s)) / sr, 3)
            a = a[s:e]
        else:
            qc["trim_s"] = 0.0

    # ④ 归一化：先按 RMS 目标，再限峰（宁可略低于 -20，也不削顶）
    rms = float(np.sqrt(np.mean(a ** 2))) if a.size else 0.0
    cur_db = 20 * np.log10(rms + 1e-12)
    gain = 10 ** ((TARGET_RMS_DBFS - cur_db) / 20) if rms > 0 else 1.0
    pk = float(np.max(np.abs(a))) * gain if a.size else 0.0
    if pk > PEAK_CAP:
        gain = PEAK_CAP / (pk / gain)
    a = a * gain

    qc.update(gain_db=round(20 * np.log10(gain + 1e-12), 2),
              rms_dbfs=round(20 * np.log10(np.sqrt(np.mean(a ** 2)) + 1e-12), 2),
              peak=round(float(np.max(np.abs(a))), 4) if a.size else 0.0,
              clipped=int(np.sum(np.abs(a) >= 0.999)),
              dc=round(float(np.mean(a)), 5) if a.size else 0.0,
              dur=round(a.size / sr, 3))
    return a.astype(np.float32), qc


def qc_ok(qc: dict) -> tuple[str, str]:
    """三态：ok / drop（丢弃但不算失败）/ fail（必须停下来查）。

    ★ 太短不是「错误」：RVC 的切片窗口是 3.7 s，1 s 以下的句子本来训不进去，
      丢就丢了。削顶和直流才是硬错误——它们会静默污染 HuBERT 特征。
    """
    if qc["clipped"] > 0:
        return "fail", f"削顶 {qc['clipped']} 个样本"
    if qc["peak"] > PEAK_CAP + 1e-3:
        return "fail", f"峰值 {qc['peak']} 超上限"
    if abs(qc["dc"]) > 0.01:
        return "fail", f"直流偏置 {qc['dc']}"
    if qc["dur"] < 1.0:
        return "drop", f"太短 {qc['dur']}s（<1s，RVC 切片留不住）"
    return "ok", ""


def write_wav(path: Path, a: np.ndarray, sr: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), a, sr, subtype="PCM_16")
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# 后端：edge
# --------------------------------------------------------------------------- #
def synth_edge(texts, out: Path, a) -> list[dict]:
    try:
        import edge_tts
    except ModuleNotFoundError:
        raise SystemExit(
            "[synth] 当前 python 没有 edge-tts。用装了它的 env（如 CosyVoice env），\n"
            "        或 pip install \"edge-tts>=7.3.0\"。"
        )
    ver = tuple(int(x) for x in edge_tts.__version__.split(".")[:2])
    if ver < (7, 3):
        # ★ 7.2.8 会在 WSS 握手时 403（Sec-MS-GEC 算法落后），不是网络问题。
        #   环境里常常同时存在两个版本：CosyVoice env 的 site-packages 里是 7.2.8，
        #   而新版是用 pip --target 装在别处、靠 PYTHONPATH 抢先加载的。
        #   ⇒ 报错时把「怎么用上新版」直接给出来，别只说「升级」。
        inject = os.environ.get("RVCX_EDGETTS_LIB", "").strip()
        hint = []
        if inject:
            hint.append(f"        已设 RVCX_EDGETTS_LIB={inject}")
        hint += [
            "        注入新版（不改任何已装 env）：",
            "          pip install --target <注入目录> \"edge-tts>=7.3.0\"",
            "          set PYTHONPATH=<注入目录>            # Windows: set PYTHONPATH=...",
            "        也可设 RVCX_EDGETTS_LIB=<注入目录> 后由调用方注入 sys.path",
        ]
        raise SystemExit(
            f"[synth] edge-tts {edge_tts.__version__} < 7.3，握手会 403。\n"
            + "\n".join(hint)
        )

    async def one(sem, t, tmp: Path):
        async with sem:
            last = None
            for attempt in range(a.retries):
                try:
                    c = edge_tts.Communicate(t, a.voice, rate=a.rate, proxy=a.proxy or None)
                    await c.save(str(tmp))
                    return True, ""
                except Exception as e:                     # noqa: BLE001
                    last = f"{type(e).__name__}: {e}"
                    await asyncio.sleep(1.5 * (attempt + 1))
            return False, str(last)

    async def run(rows):
        sem = asyncio.Semaphore(a.workers)
        tmpdir = out / "_mp3"
        tmpdir.mkdir(parents=True, exist_ok=True)
        tasks = []
        for r in rows:
            tmp = tmpdir / f"{r['id']}.mp3"
            tasks.append((r, tmp, asyncio.create_task(one(sem, r["text"], tmp))))
        for r, tmp, tk in tasks:
            ok, err = await tk
            r["_ok"], r["_err"], r["_tmp"] = ok, err, tmp
        return rows

    rows = asyncio.run(run(texts))
    res = []
    for r in rows:
        if not r["_ok"]:
            res.append(dict(id=r["id"], text=r["text"], ok=False, error=r["_err"]))
            continue
        raw, sr = sf.read(str(r["_tmp"]), dtype="float64", always_2d=False)
        if raw.ndim > 1:
            raw = raw.mean(axis=1)
        res.append(finish_one(r, raw, sr, out))
    return res


# --------------------------------------------------------------------------- #
# 后端：cosy
# --------------------------------------------------------------------------- #
def synth_cosy(texts, out: Path, a) -> list[dict]:
    cosy_root, src, model = _paths.COSY, None, None
    if not cosy_root:
        raise SystemExit("[synth] rvcx.paths 没解析到 COSY，检查 rvcx.local.toml 的 cosy_dir")
    src = Path(cosy_root) / "CosyVoice_源码"
    model = src / "pretrained_models" / "CosyVoice2-0.5B"
    ref = Path(a.cosy_ref or _paths.COSY_REF or "")
    if not ref.is_file():
        raise SystemExit(f"[synth] 参考音频不存在：{ref}")

    os.chdir(cosy_root)
    for p in (Path(cosy_root), src, src / "third_party" / "Matcha-TTS"):
        sys.path.insert(0, str(p))
    try:
        import wetext_local_patch
        wetext_local_patch.apply(verbose=False)
    except Exception as e:                                  # noqa: BLE001
        print(f"[synth] wetext 补丁未生效（数字可能读错）: {e}", file=sys.stderr)

    import torch
    from cosyvoice.cli.cosyvoice import AutoModel

    torch.manual_seed(a.seed)
    cosy = AutoModel(model_dir=str(model))
    res = []
    for r in texts:
        t0 = time.time()
        parts = [j["tts_speech"].squeeze(0).cpu().numpy()
                 for j in cosy.inference_cross_lingual(r["text"], str(ref),
                                                       stream=False, speed=a.cosy_speed)]
        raw = np.concatenate(parts).astype(np.float64)
        row = finish_one(r, raw, cosy.sample_rate, out)
        row["synth_s"] = round(time.time() - t0, 2)
        res.append(row)
        print(f"  {r['id']} {row.get('dur', 0):5.2f}s {r['text'][:24]}", flush=True)
    return res


# --------------------------------------------------------------------------- #
def finish_one(r: dict, raw: np.ndarray, sr: int, out: Path) -> dict:
    a, qc = normalize(raw, sr)
    state, why = qc_ok(qc)
    p = out / f"{r['id']}.wav"
    sha = write_wav(p, a, sr) if state == "ok" else ""
    return dict(id=r["id"], text=r["text"], ok=state == "ok", state=state, error=why,
                path=str(p) if state == "ok" else "", sr=sr, sha16=sha,
                bucket=r.get("bucket", ""), **qc)


def load_texts(path: Path, limit: int | None) -> list[dict]:
    """支持 text.txt（一行一句）或 corpus.jsonl。"""
    rows = []
    if path.suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                rows.append(dict(id=d.get("id", f"x{len(rows):04d}"),
                                 text=d["text"], bucket=d.get("bucket", "")))
    else:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(dict(id=f"x{len(rows):04d}", text=line.strip(), bucket=""))
    if limit:
        rows = rows[:limit]
    if not rows:
        raise SystemExit(f"[synth] {path} 里没有句子")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", required=True, help="text.txt 或 corpus.jsonl")
    ap.add_argument("--out", required=True, help="输出目录（每条一个 wav）")
    ap.add_argument("--backend", default="edge", choices=["edge", "cosy"])
    ap.add_argument("--voice", default="zh-CN-XiaoyiNeural")
    ap.add_argument("--rate", default="-5%", help="与 reference.wav 一致：-5%%")
    ap.add_argument("--proxy", default=os.environ.get("HTTPS_PROXY", ""),
                    help="edge 后端用；默认取 HTTPS_PROXY，直连也可")
    ap.add_argument("--workers", type=int, default=4, help="edge 并发（4 稳，8 容易被限）")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cosy-ref", default=None)
    ap.add_argument("--cosy-speed", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    texts = load_texts(Path(a.texts), a.limit)
    if a.skip_existing:
        keep = [r for r in texts if not (out / f"{r['id']}.wav").exists()]
        if len(keep) < len(texts):
            print(f"[synth] 跳过已存在 {len(texts) - len(keep)} 条，剩 {len(keep)}")
        texts = keep

    t0 = time.time()
    print(f"[synth] backend={a.backend} 句数 {len(texts)} -> {out}", flush=True)
    rows = synth_edge(texts, out, a) if a.backend == "edge" else synth_cosy(texts, out, a)
    dt = time.time() - t0

    bad = [r for r in rows if r.get("state") == "fail" or (not r["ok"] and r.get("state") != "drop")]
    drops = [r for r in rows if r.get("state") == "drop"]
    durs = [r["dur"] for r in rows if r["ok"]]
    total = sum(durs)
    summary = dict(backend=a.backend, texts=str(a.texts), out=str(out),
                   n=len(rows), ok=sum(1 for r in rows if r["ok"]),
                   dropped=len(drops), failed=len(bad),
                   total_sec=round(total, 2), total_min=round(total / 60, 3),
                   mean_sec=round(total / len(durs), 2) if durs else 0,
                   wall_sec=round(dt, 1),
                   rt_factor=round(total / dt, 2) if dt else 0,
                   qc=dict(target_rms_dbfs=TARGET_RMS_DBFS, peak_cap=PEAK_CAP,
                           clipped_total=sum(r.get("clipped", 0) for r in rows),
                           rescaled=sum(1 for r in rows
                                        if r.get("rescaled_from") not in (None, "none"))),
                   rows=rows)
    (out / "_synth_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[synth] 成功 {summary['ok']}/{summary['n']}（丢弃 {len(drops)} 条过短）  "
          f"总时长 {summary['total_min']:.2f} min  平均 {summary['mean_sec']}s/句")
    print(f"[synth] 墙钟 {dt:.1f}s  实时倍率 {summary['rt_factor']}x  "
          f"削顶样本 {summary['qc']['clipped_total']}  "
          f"量级修正 {summary['qc']['rescaled']} 条")
    for r in drops:
        print(f"  - {r['id']} {r.get('error')}  {r['text'][:16]}")
    for r in bad:
        print(f"  ✗ {r['id']} {r.get('error')}  {r['text'][:20]}", file=sys.stderr)
    if bad:
        print(f"[synth] ★ 有 {len(bad)} 条质检不过，未写盘。清单见 {out}/_synth_manifest.json",
              file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
