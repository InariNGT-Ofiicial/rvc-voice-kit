# -*- coding: utf-8 -*-
"""astrbot_plugin_voice_patch —— 按需把 LLM 回复换成「现有 RVC 音色」的语音。

⚠ 本文件**未在实机 AstrBot 上跑过**（本机未安装 AstrBot），是按官方文档
   dev/star/guides/listen-message-event.html 与 astrbot/core/star/register/star_handler.py
   的契约写的。上机后请先只开 `enabled` 在私聊里试。

为什么需要它（而不是只用内置 provider_tts_settings）：
  内置 TTS 只管「全局开/关 + 概率」，不能按群、按关键词、按前缀精细控制。
  本插件在 `on_decorating_result`（发送前）就地改消息链，内置 TTS 的判断逻辑不受影响。

依赖：只需要 aiohttp（AstrBot 自带）。真正的合成在本机两个常驻服务里：
  8788 CosyVoice（TTS） → 8789 RVC（换声），见 rvc-research/bench/svc_*.py
"""
import re

import aiohttp

import astrbot.api.message_components as Comp
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register

DEFAULTS = {
    "enabled": False,
    "shim_url": "http://127.0.0.1:8790/v1/audio/speech",
    "model": "fangno120",
    "voice": "fangno120",
    "trigger_prefix": "#语音",
    "keyword": "",            # 非空时：命中关键词才转
    "session_whitelist": [],  # 空 = 全部会话
    "max_chars": 60,
    "timeout_sec": 180,
    "keep_text": True,        # 语音之外是否保留原文
}


@register(
    "astrbot_plugin_voice_patch",
    "rvc-research",
    "把 LLM 回复转成本机 RVC 音色语音",
    "0.1.0",
)
class VoicePatch(Star):
    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context)
        self.cfg = {**DEFAULTS, **(config or {})}
        logger.info(f"[voice_patch] enabled={self.cfg['enabled']} url={self.cfg['shim_url']}")

    # ---- 触发判定：满足任一即可 -------------------------------------------
    def _should_speak(self, event: AstrMessageEvent, text: str) -> bool:
        if not self.cfg["enabled"]:
            return False
        umo = event.unified_msg_origin
        wl = self.cfg["session_whitelist"]
        if wl and umo not in wl:
            return False
        prefix = self.cfg["trigger_prefix"]
        if prefix and event.message_str.strip().startswith(prefix):
            return True
        kw = self.cfg["keyword"]
        if kw and kw in event.message_str:
            return True
        return not prefix and not kw  # 两者都空 = 全量转

    async def _speak(self, text: str) -> bytes | None:
        payload = {
            "model": self.cfg["model"],
            "voice": self.cfg["voice"],
            "response_format": "wav",
            "input": text,
        }
        try:
            timeout = aiohttp.ClientTimeout(total=self.cfg["timeout_sec"])
            async with aiohttp.ClientSession(timeout=timeout) as sess:
                async with sess.post(self.cfg["shim_url"], json=payload) as r:
                    if r.status != 200:
                        logger.error(f"[voice_patch] shim {r.status}: {(await r.text())[:200]}")
                        return None
                    return await r.read()
        except Exception as e:
            logger.error(f"[voice_patch] 合成失败 {type(e).__name__}: {e}")
            return None

    @filter.on_decorating_result()
    async def on_decorating_result(self, event: AstrMessageEvent):
        """发送前就地改 chain：把 Plain 换成 Record。

        ★ 必须原地改 `result.chain`，不能 `event.result = 新对象` —— 框架后续阶段
          持有的是同一个引用（官方文档与社区实践一致）。
        """
        result = event.get_result()
        if result is None or not result.is_llm_result():
            return
        plain = "".join(c.text for c in result.chain if isinstance(c, Comp.Plain)).strip()
        if len(plain) < 2:
            return
        plain = re.sub(r"[\U0001F300-\U0001FAFF]", "", plain)  # 去 emoji，避免念出来
        if not plain:
            return
        if not self._should_speak(event, plain):
            return
        if len(plain) > int(self.cfg["max_chars"]):
            logger.info(f"[voice_patch] {len(plain)} 字超上限，跳过（避免 20 s+ 的等待）")
            return

        wav = await self._speak(plain)
        if not wav:
            return  # 失败就保持纯文本，绝不吞掉回复

        # Record 只接受 wav；这里直接落 AstrBot 数据目录的 temp
        from astrbot.core.utils.astrbot_path import get_astrbot_temp_path
        import os, time

        path = os.path.join(get_astrbot_temp_path(), f"voice_patch_{int(time.time()*1000)}.wav")
        with open(path, "wb") as f:
            f.write(wav)

        result.chain.clear()
        result.chain.append(Comp.Record(file=path, url=path, text=plain))
        if self.cfg["keep_text"]:
            result.chain.append(Comp.Plain(plain))

    async def terminate(self):
        logger.info("[voice_patch] terminated")
