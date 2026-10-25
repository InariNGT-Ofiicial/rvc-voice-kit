# -*- coding: utf-8 -*-
"""TTS 通道：用本机 GPT-SoVITS 的「另一个角色」音色合成台词。

为什么不用 edge-tts：微软已封（WSS 403，直连/代理都是 403）。
为什么不用芳乃自己的 TTS：那是「已经是对的音色」，再走 RVC 等于什么都没测。
⇒ 用 kurumi（胡桃，日语，另一角色）当 TTS 源，让 RVC 有真正要干的活。
"""
import sys as _sys
from pathlib import Path as _pth

_sys.path.insert(0, str(_pth(__file__).resolve().parents[2]))
from rvcx import paths as _paths  # noqa: E402

import json
import os
import sys
from pathlib import Path

ROOT = _paths.SOVITS
CFG = ROOT / "GPT_SoVITS" / "configs" / "tts_infer_kurumi_tmp.yaml"
OUT = _paths.WORK / "dialogue" / "src_tts"

LINES = [
    dict(lid="ja01", text="私の質問だけに答えてください。"),
    dict(lid="ja02", text="顔見ずに、そのまま聞いてください。"),
    dict(lid="ja03", text="ああ、申し遅れました。私は日立真子と申します。"),
    dict(lid="ja04", text="朝はまず、お風呂場の掃除をしているのですが、私、お風呂が好きなんです。"),
]

REF = ROOT / "logs" / "kurumi_v2" / "5-wav32k" / "ep01_0003.wav"
REF_TEXT = "そんな話題ばかりで中学生だった私には"

YAML = f"""custom:
  bert_base_path: GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large
  cnhuhbert_base_path: GPT_SoVITS/pretrained_models/chinese-hubert-base
  device: cuda
  is_half: true
  t2s_weights_path: {ROOT / 'GPT_weights_v2' / 'kurumi_v3-e15.ckpt'}
  version: v3
  vits_weights_path: {ROOT / 'SoVITS_weights_v2' / 'kurumi_v3_e12_s624.pth'}
"""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CFG.write_text(YAML, encoding="utf-8")
    os.chdir(ROOT)
    # ★ 两个路径都要加：GPT_SoVITS 下的脚本用裸导入（AR / feature_extractor / module）
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "GPT_SoVITS"))
    # 只有环境里真缺 jieba_fast 时才挂兼容层（GPT-SoVITS 自己的 .venv 里有真货）
    import importlib.util
    if importlib.util.find_spec("jieba_fast") is None:
        sys.path.insert(0, str(Path(__file__).parent / "shim"))
        print("[shim] 用 jieba 顶替 jieba_fast", flush=True)

    import numpy as np
    import soundfile as sf
    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

    tts = TTS(TTS_Config(str(CFG)))
    print("TTS 就绪:", tts.configs.version, flush=True)

    rows = []
    for ln in LINES:
        inputs = dict(
            text=ln["text"], text_lang="ja",
            ref_audio_path=str(REF), prompt_text=REF_TEXT, prompt_lang="ja",
            top_k=15, top_p=1.0, temperature=1.0,
            text_split_method="cut0", speed_factor=1.0, seed=42,
            parallel_infer=True, repetition_penalty=1.35,
        )
        sr, audio = None, None
        for sr_, a in tts.run(inputs):
            sr, audio = sr_, a
        p = OUT / f"{ln['lid']}.wav"
        # ★ v3 输出峰值会超过 1.0；直接写 PCM_16 会被硬削顶（实测 rms 0.95 / 平坦度 0.15 全是失真），
        #   削顶后的谐波污染会让 RVC 的 HuBERT 特征彻底失配 ⇒ 必须先归一化再写。
        a = audio.astype(np.float32)
        pk = float(np.max(np.abs(a))) or 1.0
        if pk > 0.95:
            a = a * (0.9 / pk)
        sf.write(p, a, sr)
        print(f"  峰值 {pk:.3f} -> 归一化 0.9", flush=True)
        rows.append(dict(lid=ln["lid"], text=ln["text"], path=str(p),
                         sr=sr, dur=round(len(audio) / sr, 2)))
        print(f"{ln['lid']} -> {len(audio)/sr:.2f}s {sr}Hz  {ln['text']}", flush=True)

    (OUT.parent / "tts_manifest.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("done")


if __name__ == "__main__":
    main()
