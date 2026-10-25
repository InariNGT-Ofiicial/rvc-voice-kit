# -*- coding: utf-8 -*-
"""诊断：同一句话，换切句方式 / 换模型版本，看哪个 TTS 配置能稳定可懂。

背景：kurumi v2 + cut5 在 ja03/ja04（长句）上 CER 0.6~0.7，短句 ja01 却是 0.0。
      ⇒ 不是模型不会说，是「长文本 + 切句」这条链在掉字。
"""
import sys as _sys
from pathlib import Path as _pth

_sys.path.insert(0, str(_pth(__file__).resolve().parents[2]))
from rvcx import paths as _paths  # noqa: E402

import os
import sys
from pathlib import Path

ROOT = _paths.SOVITS
OUT = _paths.WORK / "dialogue" / "probe"
REF = ROOT / "logs" / "kurumi_v2" / "5-wav32k" / "ep01_0003.wav"
REF_TEXT = "そんな話題ばかりで中学生だった私には"

TEXT = "ああ、申し遅れました。私は日立真子と申します。"

VARIANTS = [
    ("v2_cut5", "v2", ROOT / "GPT_weights_v2/kurumi_v2-e15.ckpt",
     ROOT / "SoVITS_weights_v2/kurumi_v2_e8_s224.pth", "cut5"),
    ("v2_cut0", "v2", ROOT / "GPT_weights_v2/kurumi_v2-e15.ckpt",
     ROOT / "SoVITS_weights_v2/kurumi_v2_e8_s224.pth", "cut0"),
    ("v2_cut1", "v2", ROOT / "GPT_weights_v2/kurumi_v2-e15.ckpt",
     ROOT / "SoVITS_weights_v2/kurumi_v2_e8_s224.pth", "cut1"),
    ("v3_cut0", "v3", ROOT / "GPT_weights_v2/kurumi_v3-e15.ckpt",
     ROOT / "SoVITS_weights_v2/kurumi_v3_e12_s624.pth", "cut0"),
]

YAML = """custom:
  bert_base_path: GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large
  cnhuhbert_base_path: GPT_SoVITS/pretrained_models/chinese-hubert-base
  device: cuda
  is_half: true
  t2s_weights_path: {t2s}
  version: {ver}
  vits_weights_path: {vits}
"""


def main():
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "GPT_SoVITS"))
    import numpy as np
    import soundfile as sf
    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

    OUT.mkdir(parents=True, exist_ok=True)
    for name, ver, t2s, vits, split in VARIANTS:
        if not Path(vits).exists():
            print(f"skip {name}: 缺 {Path(vits).name}", flush=True)
            continue
        cfg = OUT / f"_cfg_{name}.yaml"
        cfg.write_text(YAML.format(t2s=t2s, ver=ver, vits=vits), encoding="utf-8")
        print(f"\n===== {name} (version={ver}, split={split}) =====", flush=True)
        try:
            tts = TTS(TTS_Config(str(cfg)))
        except Exception as e:
            print(f"  !! 加载失败 {type(e).__name__}: {e}", flush=True)
            continue
        try:
            sr, audio = None, None
            for sr_, a in tts.run(dict(
                text=TEXT, text_lang="ja",
                ref_audio_path=str(REF), prompt_text=REF_TEXT, prompt_lang="ja",
                top_k=15, top_p=1.0, temperature=1.0,
                text_split_method=split, speed_factor=1.0, seed=42,
                parallel_infer=True, repetition_penalty=1.35,
            )):
                sr, audio = sr_, a
            p = OUT / f"{name}.wav"
            sf.write(p, audio.astype(np.float32), sr)
            print(f"  -> {p.name} {len(audio)/sr:.2f}s {sr}Hz", flush=True)
        except Exception as e:
            print(f"  !! 合成失败 {type(e).__name__}: {e}", flush=True)
        del tts


if __name__ == "__main__":
    main()
