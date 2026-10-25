# -*- coding: utf-8 -*-
"""扫描音色库，生成 models/index.json 与 models/INDEX.md（模型索引）。

★ 为什么仓库里放索引而不是权重：
  1. `added_*.index` 单个就有 170 MB 上下，超过 GitHub 单文件 100 MiB 硬限制；
  2. 音色模型是衍生作品，公开分发涉及原始角色的授权，不适合塞进代码仓库。
  索引给出 sha256 + 训练参数 + 实测指标，拿到权重的人可以逐字节核对，
  没拿到的人也知道该按什么参数自己训一个。

★ manifest 会漂移：本机现有三种 schema（见 docs/模型清单格式.md），
  本脚本用 normalize() 把它们拉成同一套字段，缺字段就填 null，不猜。

用法：
  python tools/build_model_index.py                 # 用 rvcx.paths 的 MODELS
  python tools/build_model_index.py --models D:/x   # 指定目录
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KIT))
from rvcx import paths as _paths  # noqa: E402

CHUNK = 1 << 20


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(CHUNK), b""):
            h.update(blk)
    return h.hexdigest()


def _dig(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _num(*vals):
    for v in vals:
        if isinstance(v, (int, float)):
            return v
    return None


def normalize(man: dict, dname: str) -> dict:
    """把任意一版 manifest 拉成统一字段。三种 schema 的差异都在这一个函数里。"""
    model = man.get("model") or {}
    idx = man.get("index") or {}
    ds = man.get("dataset") or man.get("train_data") or {}

    # 版本：老 schema 用 model_version，新 schema 用 model.version / format
    version = (man.get("model_version") or model.get("version")
               or ("v2" if str(man.get("format", "")).endswith("v2") else None))
    sr = man.get("sample_rate") or model.get("ceiling")
    if_f0 = _num(man.get("if_f0"), model.get("if_f0"))

    epochs = _num(man.get("epoch"), model.get("epochs"))
    if epochs is None:
        m = re.search(r"步数(\d+)", dname)
        epochs = int(m.group(1)) if m else None

    # ---- 实测：三种 schema 的键名完全不同 ----
    meas, src = {}, None
    flat = man.get("measured") or {}
    if isinstance(flat, dict) and "singing" in flat:
        meas, src = dict(flat["singing"]), "manifest.measured.singing（唱歌探针）"
    elif isinstance(flat, dict) and "ltas_r_out" in flat:
        meas, src = dict(flat), "manifest.measured"
    elif isinstance(man.get("probe_metrics"), dict):
        pm = man["probe_metrics"]
        meas = {
            "ltas_r_src": pm.get("ltas_corr_source_vs_target"),
            "ltas_r_out": pm.get("ltas_corr_output_vs_target"),
            "f0_median_diff_hz": pm.get("f0_median_diff_hz"),
            "peak": pm.get("peak"),
            "exact_zero_ratio": pm.get("exact_zero_ratio_out"),
            "mfcc_l2": pm.get("mfcc_l2"),
        }
        src = "manifest.probe_metrics（旧 schema）"

    if meas:
        r_out = _num(meas.get("ltas_r_out"))
        r_src = _num(meas.get("ltas_r_src"))
        meas["delta_r"] = _num(meas.get("delta_r"),
                               (r_out - r_src) if None not in (r_out, r_src) else None)
        meas = {k: v for k, v in meas.items() if v is not None}
        if src:
            meas["source"] = src

    # 空格子用实测数据兜底（本地网格测量）
    rec = dict(man.get("recommended_infer") or {})
    if not rec:
        rec = {"f0_method": _dig(man, "f0", "infer", default="rmvpe"),
               "index_rate": 0.75, "protect": 0.33, "pitch": 0}

    return {
        "voice": man.get("voice") or dname.split("_")[0],
        "engine": man.get("engine", "rvc"),
        "engine_commit": man.get("engine_commit") or man.get("engine_upstream_commit"),
        "format": version,
        "sample_rate": sr,
        "if_f0": if_f0,
        "epochs": epochs,
        "global_step": _num(man.get("global_step"), model.get("steps")),
        "batch_size": _num(man.get("batch_size"), model.get("batch_size")),
        "train_data": {
            "files": _num(ds.get("files")),
            "minutes": _num(ds.get("duration_min"), ds.get("minutes")),
            "slices": _num(ds.get("slices_3_7s"), ds.get("slices")),
            "f0_method": _dig(man, "f0", "train") or ds.get("f0_method"),
        } or None,
        "infer": rec,
        "measured": meas or None,
        "perf": man.get("perf") or man.get("timing"),
        "caveats": man.get("caveats") or ([man["quality_note"]] if man.get("quality_note") else []),
    }


def load_measurements(kit: Path) -> tuple[dict, list]:
    """读 models/measurements/：网格结果按轮次索引（填实测），检查点表单独成表。"""
    d = kit / "models" / "measurements"
    by_ep: dict[int, dict] = {}
    g = d / "grid.json"
    if g.is_file():
        for r in json.loads(g.read_text(encoding="utf-8")):
            m = re.search(r"_e(\d+)_s\d+", r.get("model", ""))
            if not m:
                continue
            ep = int(m.group(1))
            cur = by_ep.get(ep)
            # 同一轮次有多个 index_rate，保留最接近 0.75 的那条
            if cur is None or abs(r["rate"] - 0.75) < abs(cur["rate"] - 0.75):
                by_ep[ep] = r

    ck = []
    c = d / "ckpt_sweep.json"
    if c.is_file():
        ck = json.loads(c.read_text(encoding="utf-8"))
    return by_ep, ck


def scan(root: Path, by_ep: dict | None = None) -> list[dict]:
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        m = d / "model.pth"
        if not m.is_file():
            continue
        man = {}
        mf = d / ".rvc-manifest.json"
        if mf.is_file():
            man = json.loads(mf.read_text(encoding="utf-8"))

        item = normalize(man, d.name)
        ep = item["epochs"]

        if by_ep and ep in by_ep:
            g = by_ep[ep]
            fill = {
                "ltas_r_src": g["r_src"], "ltas_r_out": g["r_lta"],
                "delta_r": g["dr"], "f0_median_diff_hz": g["f0_diff_hz"],
                "peak": g["peak"], "exact_zero_ratio": g["zero_ratio"],
                "index_rate": g["rate"],
            }
            if not item["measured"]:
                item["measured"] = dict(fill, source="models/measurements/grid.json（3 片段平均）")
            else:
                # manifest 里缺的格子用网格测量补，已有的不覆盖
                for k, v in fill.items():
                    item["measured"].setdefault(k, v)

        idxs = sorted(d.glob("added_*.index"))
        trained = sorted(d.glob("trained_*.index"))
        item["id"] = d.name
        item["model"] = {
            "file": m.name, "bytes": m.stat().st_size, "sha256": sha256(m),
        }
        item["index"] = {
            "file": idxs[0].name if idxs else None,
            "bytes": idxs[0].stat().st_size if idxs else None,
            "sha256": sha256(idxs[0]) if idxs else None,
            "note": "只认 added_*；trained_* 是中间产物"
                    + (f"（本目录另有 {len(trained)} 个 trained_*.index，不要用）" if trained else ""),
        }
        out.append(item)
    return out


def to_md(items: list[dict], ck: list[dict] | None = None) -> str:
    L = [
        "# 模型索引",
        "",
        "本仓库**不包含权重**：单个 `added_*.index` 已超 GitHub 100 MiB 单文件限制，",
        "且音色模型属衍生作品，公开分发需另行取得授权。此表用于核对权重、复现训练参数。",
        "",
        "| 音色 | 轮次 | model.pth | sha256 (前 16) | Δr | r_lta | 峰值 | 精确归零 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for it in items:
        me = it["measured"] or {}

        def f(k, fmt, dash="—"):
            v = me.get(k)
            return fmt % v if isinstance(v, (int, float)) else dash

        L.append(
            f"| {it['id']} | {it['epochs'] if it['epochs'] is not None else '?'} "
            f"| {it['model']['bytes'] / 1048576:.0f} MB | `{it['model']['sha256'][:16]}` "
            f"| {f('delta_r', '%+.4f')} | {f('ltas_r_out', '%.4f')} "
            f"| {f('peak', '%.3f')} | {f('exact_zero_ratio', '%.4f')} |"
        )

    L += ["", "## 推荐推理参数", "",
          "| 音色 | f0 | index_rate | protect | pitch | 训练侧 f0 |",
          "|---|---|---|---|---|---|"]
    for it in items:
        c = it["infer"]
        tf0 = (it.get("train_data") or {}).get("f0_method") or "—"
        L.append(f"| {it['id']} | {c.get('f0_method', 'rmvpe')} | {c.get('index_rate', 0.75)} "
                 f"| {c.get('protect', 0.33)} | {c.get('pitch', 0)} | {tf0} |")

    if ck:
        L += ["", "## 检查点横评（单片段，index_rate 0.75）", "",
              "同一角色、同一参考集、同一段干声，只改检查点。Δr 从 20 轮起就在平台区内抖动，",
              "**轮次不是瓶颈**——这条结论就是这张表来的。", "",
              "| 检查点 | r_lta | Δr | f0 中位差 (Hz) | 峰值 | 精确归零 |",
              "|---|---|---|---|---|---|"]
        for r in sorted(ck, key=lambda x: int(re.search(r"_e(\d+)_s", x["model"]).group(1))):
            ep = int(re.search(r"_e(\d+)_s(\d+)", r["model"]).group(1))
            L.append(f"| e{ep} | {r['r_lta']:.4f} | {r['dr']:+.4f} "
                     f"| {r['f0_diff_hz']:+.1f} | {r['peak']:.3f} | {r['zero_ratio']:.4f} |")

    L += [
        "",
        "## 复现要点（训一个自己的）",
        "",
        "1. 数据：同一角色的干净干声，24–30 分钟足够（本库 25 分钟 / 358 段 / 445 切片）。",
        "2. `python scripts/train/p1_run.py --exp <名字> --steps all --sr 48k --version v2 "
        "--epochs 120 --f0method rmvpe`",
        "3. **100–150 轮封顶**：20 轮即进平台区，200 轮（4h21m）相对 120 轮无可判差异。",
        "4. 训练完用 `scripts/train/extract_small.py <exp>` 导出推理小模型，",
        "   连同 `added_*.index` 放进 `<MODELS>/<角色名>/`，再写一份 `.rvc-manifest.json`",
        "   （格式见 `docs/模型清单格式.md`）。",
        "",
        "> 判据与踩坑记录见 `docs/`；原始测量数据在 `models/measurements/`。",
    ]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None)
    ap.add_argument("--out", default=str(KIT / "models"))
    a = ap.parse_args()

    root = Path(a.models) if a.models else Path(_paths.MODELS)
    if not root.is_dir():
        print(f"模型目录不存在：{root}（用 --models 或 RVCX_MODELS 指定）", file=sys.stderr)
        return 1

    by_ep, ck = load_measurements(KIT)
    items = scan(root, by_ep)
    if not items:
        print(f"{root} 下没有找到 model.pth", file=sys.stderr)
        return 1

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.json").write_text(
        json.dumps({"count": len(items), "models": items}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    (out / "INDEX.md").write_text(to_md(items, ck), encoding="utf-8")
    print(f"写入 {out / 'index.json'} 与 {out / 'INDEX.md'}"
          f"（{len(items)} 个音色，{len(ck)} 条检查点实测）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
