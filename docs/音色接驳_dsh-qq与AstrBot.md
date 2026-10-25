<!-- 本文件由工作区技术报告迁移而来，原始绝对路径已替换为占位符：
     <ROOT> 工作根 / <REPO> 上游 clone / <PY> venv 解释器 / <MODELS> 音色库 /
     <FFMPEG_DIR> ffmpeg 目录 / <COSY> CosyVoice2 / <COSY_SRC> 其源码目录 /
     <COSY_ENV> 其 venv / <PRESETS> 音色预设 / <DATASET_SRC> 原始训练集 /
     <REPORT> 交付目录 / <WORKSPACE> 工作区 / <AGENT_CWD> agent 工作目录 /
     <DSH_HOME> DSH 配置目录 / <CODEX> Codex 工作区。
     本仓库实际以 rvcx/paths.py + rvcx.local.toml 解析这些路径。 -->

# 现有 RVC / DDSP 音色模型接驳 dsh-qq 与 AstrBot

> 研究日期：2026-10-26　环境：Windows 11 + git bash　验证方式：全部在本机真跑
> 结论先行：**「文本 → 目标音色语音」已经跑通（实测 7.4 s/句，常驻），但真发 QQ 这一步在沙箱内无法验证**（缺 group_openid）。

---

## 0. 一页纸结论

| 问题 | 结论 |
|---|---|
| dsh-qq 能发语音吗 | **能**，`qqbot_send_file` 已内置，`.wav` 走 `sendVoice` → `file_type=3` |
| 需要转 SILK 吗 | **不需要**。SDK 把 `.wav/.mp3/.silk` 当作「原生上传格式」直接 base64 上传；它自带的 `audioFileToSilkBase64()` **在 dist 里没有任何调用点**（详见 §5 纠错） |
| 延迟可接受吗 | 冷启动 **≈47 s/句**（不可接受）；常驻 **7.4 s/句**（群聊勉强可接受，需配「正在录…」占位或限长） |
| 最推荐接法（dsh-qq） | **路线 C：把「TTS+RVC」封成一个工具 `voice_say`**，产物落 agent cwd，再机械调用一次 `qqbot_send_file` |
| 最推荐接法（AstrBot） | **零插件**：起一个 OpenAI 兼容门面，用内置 `openai_tts_api` provider + `provider_tts_settings.enable` |
| 三套 Python 环境 | 合并会炸：CosyVoice env **没有** faiss/rmvpe，做不了 RVC → 必须**两个常驻进程**，各用各的 env |

---

## 1. 架构（文字版）

```
QQ 群消息
   │
   ├─(A) dsh-qq 插件（@tencent-connect/dsh-qqbot@0.5.2，官方 Bot API / WebSocket）
   │        │  agent 决定要说话
   │        ├─▶ 工具 voice_say(text)                     ← 本方案新增的独立插件包
   │        │        │ HTTP POST 127.0.0.1:8788/say
   │        │        ▼
   │        │   [常驻进程 1] svc_tts.py   CosyVoice2-0.5B（<AI_VOICE>/.../env）
   │        │        │ 返回 wav（24 kHz）
   │        │        ▼ HTTP POST 127.0.0.1:8789/convert
   │        │   [常驻进程 2] svc_rvc.py   RVC（<ROOT>/venv，模型常驻）
   │        │        │ 返回 wav（48 kHz）
   │        │        ▼ 落盘 <agent cwd>/tts/voice_*.wav
   │        └─▶ 工具 qqbot_send_file(file_path)   ← 插件内置，一行不改
   │                 │ 白名单校验（media / cwd / extraRoots）
   │                 ▼
   │            sendVoice → uploadMedia(file_type=3, base64 原字节) → 语音气泡
   │
   └─(B) AstrBot（本机未安装；推荐 aiocqhttp/NapCat 接个人号）
            │  LLM 回复（is_llm_result）
            ▼
         ResultDecorateStage  → tts_provider.get_audio(text)
            │  api_base = http://127.0.0.1:8790/v1  （内置 openai_tts_api provider）
            ▼
         [常驻进程 3] svc_openai_shim.py  →  8788 → 8789  （纯标准库，无模型）
            │ 返回 wav
            ▼
         Comp.Record(file=path, url=path) → 平台适配器发语音
```

三套 python 环境的分工（**不能合并**，见 §6 坑③）：

| 进程 | 解释器 | 为什么必须独立 |
|---|---|---|
| `svc_tts.py` | `<COSY_ENV>/Scripts/python.exe` | 有 cosyvoice/torch，**没有 faiss、没有 rmvpe** |
| `svc_rvc.py` | `<PY>` | 有 faiss + rmvpe + RVC repo，没有 cosyvoice |
| `svc_openai_shim.py` | 任意（纯标准库） | 只做协议翻译 |
| dsh / AstrBot | 各自的 node / python | 不 import 任何模型库 |

---

## 2. 通道 A：dsh-qq 接入点（已实测）

### 2.1 事实核对表

| 事实 | 复核结果 | 证据位置 |
|---|---|---|
| 插件内置 `qqbot_send_file` | ✅ | `dist/media/send-file-tool.js:14,131` |
| 语音白名单 `.mp3 .wav .ogg .aac .silk .amr` | ✅ | `send-file-tool.js:21` |
| `restrictPaths` 默认 true，允许 media+cwd+extraRoots | ✅ | `send-file-tool.js:39-44`；`dist/config.js:63-69` |
| media 目录 = `~/.dsh-qqbot/media` | ✅ | `dist/media/media-cleaner.js:12` |
| 语音上限 20 MB | ✅ | `qqbot-nodejs/src/protocol/utils/file-utils.ts:25` |
| SDK 自动把 mp3/wav 转 SILK | ❌ **不成立**，见 §5 |
| `sendVoice` 前会转码 | ❌ 没有，直接 base64 原文件 | `qqbot-nodejs/dist/protocol/api/media.js:44-46` |

### 2.2 接入点（只用 patch + 独立包，**不碰 dist**）

1. 独立插件包：`rvc-research/packages/dsh-plugin-voice-say/`（已写好并冒烟通过）
2. profile 侧只加一段 patch（**不要现在合进生产**）：

```yaml
# <DSH_HOME>\profiles\qqbot\cordis.patch.yml 追加（整行替换语义，注意保留原有键）
- insert:
    - id: voice-say
      name: 'dsh-plugin-voice-say'
      config:
        ttsUrl: 'http://127.0.0.1:8788/say'
        rvcUrl: 'http://127.0.0.1:8789/convert'
        outDir: ''          # 空 = <agent cwd>/tts
        maxChars: 80
        timeoutMs: 120000
```

3. `groupPrompt` 里只需加**一行**（把「生成」和「发送」焊死，避免模型只生成不发）：

```
需要发语音时调用 voice_say(text)，拿到 file_path 后必须立刻调用 qqbot_send_file 发送它，不要复述路径。
```

安装命令（本地目录装，不 push、不发 npm）：

```bash
# 只装到临时 profile 验证；不要动生产 qqbot profile
cd "<REPORT>/packages/dsh-plugin-voice-say"
dsh plugin add . --profile voice-say-test   # 若 dsh CLI 不支持 --profile，用 DSH_PROFILE 环境变量
```

> 冒烟（已验证，不需要 dsh 本体）：
> ```bash
> cd packages/dsh-plugin-voice-say && node test/smoke.mjs
> # 输出：registered: voice_say, voice_doctor / totalMs 8290 / 长文本拒绝 OK
> ```

### 2.3 三条触发路线对比

| | A 提示词驱动 | B 独立 hook | C 工具封装（推荐） |
|---|---|---|---|
| 实现位置 | profile `groupPrompt` | 需要 dsh 的「回复后」事件 | 独立插件包 + 一次 `qqbot_send_file` |
| 改 dist 吗 | 否 | 否 | 否 |
| 延迟 | 1 次工具往返 + 生成 | 回复后异步追发（不阻塞文本） | 1 次 `voice_say` + 1 次机械 `send` |
| 失败面 | 模型可能不调、可能编路径 | **dsh-qq 不对外暴露 bot/session**：`dist/index.js` 只 `inject=['agents']`，`bot` 与 `SessionManager` 都是闭包内变量 → 外部拿不到 replyTarget，B 实际不可行（除非自己再连一条 WS 去抢消息，会有双收风险） | 路径固定、失败即抛错，模型只决定「说什么」 |
| 结论 | 作兜底 | **否决** | **采用** |

C 的关键设计：`voice_say` 的产物**必须**写进 `exec.cwd`（= `<AGENT_CWD>`），这样 `qqbot_send_file` 的 `restrictPaths` 才放行 —— 不要写到 `<AI_SONG>/...`。

---

## 3. 通道 B：AstrBot（本机未安装，方案按官方源码核对）

### 3.1 官方 TTS 机制（具体到类名/方法/配置项）

| 项 | 值 | 来源 |
|---|---|---|
| TTS 基类 | `TTSProvider`（`astrbot/core/provider/provider.py`） | 源码 |
| 必须实现 | `async def get_audio(self, text: str) -> str`（返回**音频文件路径**） | 源码 |
| 可选 | `support_stream() -> bool`、`get_audio_stream(text_queue, audio_queue)` | 源码 |
| 注册装饰器 | `@register_provider_adapter("<type_id>", "<显示名>", provider_type=ProviderType.TEXT_TO_SPEECH)` | `astrbot/core/provider/register.py` |
| 内置同款 | `openai_tts_api` → `ProviderOpenAITTSAPI`；`edge_tts` → `ProviderEdgeTTS` | `provider/sources/openai_tts_api_source.py` |
| 全局开关 | `provider_tts_settings.enable` / `.provider_id` / `.dual_output` / `.use_file_service` | 配置文档 |
| 隐藏项 | `provider_tts_settings.trigger_probability`（0–1，代码读取，文档未列） | `pipeline/result_decorate/stage.py` |
| 触发条件 | `provider_tts_settings.enable && result.is_llm_result() && should_process_tts_request && random()<=trigger_probability && tts_provider`，且**只对 `Plain` 段、长度>1** 生效 | 同上 |
| 发语音组件 | `Record(file=path, url=path, text=原文字)` | 同上 |
| 装饰钩子 | `@filter.on_decorating_result()` → `event.get_result().chain` 就地改 | 文档 `dev/star/guides/listen-message-event.html` |
| 其它可用钩子 | `on_llm_request` / `on_llm_response` / `on_agent_done` / `after_message_sent` / `llm_tool` | `star/register/star_handler.py` |

文档链接：
- 配置项：<https://docs.astrbot.app/dev/astrbot-config.html>（`provider_tts_settings` 一节）
- 插件钩子：<https://docs.astrbot.app/dev/star/guides/listen-message-event.html>
- 旧版一站式：<https://docs.astrbot.app/dev/star/plugin.html>（含 `TTSProvider` 定义原文）
- Provider 注册机制：<https://deepwiki.com/AstrBotDevs/AstrBot/5.2-provider-implementations>

### 3.2 推荐路线：**内置 `openai_tts_api` + 本地 OpenAI 兼容门面（零插件）**

AstrBot 的 `ProviderOpenAITTSAPI.get_audio()` 实际就是：

```python
client.audio.speech.create(model=self.model_name, voice=self.voice,
                           response_format="wav", input=text)
```

只要本机有个说这个协议的端点，AstrBot **不用写一行插件**就能用上现有音色。已写好并通过协议验证的门面：
`rvc-research/bench/svc_openai_shim.py`（纯标准库）。

WebUI 里新增服务提供商：

| 字段 | 值 |
|---|---|
| 类型 | `OpenAI TTS API`（type id `openai_tts_api`） |
| `api_base` | `http://127.0.0.1:8790/v1` |
| `api_key` | `local`（门面不校验，但 SDK 要求非空） |
| `model` | `fangno120` |
| `openai-tts-voice` | `fangno120` |
| `timeout` | `180` |

然后：`provider_tts_settings.enable=true`、`provider_id=<该提供商 id>`、`dual_output=true`（同时发文字+语音）。

**为什么不用自定义 Provider 插件**：自定义 provider 只在「要出现在 WebUI 下拉框 / 要流式 / 要在 AstrBot 进程内直接持有模型」时才值得；而模型必须留在各自 env 的独立进程里，注册进 AstrBot 只会把 8 GB 显存挤爆。**为什么不用 `on_decorating_result` 插件**：内置管线已经处理了 `dual_output`、分段、失败回落文本，自己重写一遍只会更脆。若日后要「按群/按关键词」控制，再加一个 `on_decorating_result` 插件覆盖（骨架见 `packages/astrbot_plugin_voice_patch/`，**未实机验证**）。

### 3.3 致命前提：AstrBot 必须用 `aiocqhttp`（NapCat），不能用 QQ 官方接口

文档明确：`Comp.Record` 目前**只接受 wav**，且「在 QQ 个人号(aiocqhttp)、Telegram、飞书、企业微信支持；**QQ 官方接口、钉钉不支持**」。而 dsh-qq 走的正是 QQ 官方 Bot API —— **两条通道不是同一套 QQ 接入方式**，不要把 AstrBot 的结论套到 dsh-qq 上，反之亦然。

### 3.4 部署形态

- AstrBot 自己的 python env（`uv`/pip 一套），**不装 torch**，不要试图 import cosyvoice。
- 跨进程用 **localhost HTTP**：延迟 2–5 ms，故障面清晰（服务挂了立刻 500，能回落到纯文本）。子进程调用（每句 spawn）延迟 18 s 起，否决。消息队列对单机单卡无收益，否决。

---

## 4. 延迟预算表（全部实测，RTX 5060 Laptop 8 GB）

### 4.1 分阶段

| 阶段 | 冷（每句起进程） | 热（常驻） | 备注 |
|---|---|---|---|
| TTS `python` 起步（import torch/cosyvoice） | 12.5 s | 0 | `tts_bench.py` |
| TTS 载模型 | 10.6 s | 0 | 一次 |
| TTS 合成（14 字 → 2.8–3.5 s 音频） | 5.0–6.6 s | 5.0–6.6 s | **RTF 1.17–1.92，比实时还慢**，是最大头 |
| RVC 起步 + 载模型 | 约 7 s（含在 18.5 s 里） | 0.6 s（页缓存热） | `svc_rvc` 启动 |
| RVC 换声 | 11–12 s（`rvcx` CLI 全链路 18.5 s） | **0.45–0.57 s**（单独）/ **1.6–1.8 s**（紧跟 TTS 之后） | 见 4.2 争用 |
| SILK 转码 | 0.43 s（ffmpeg 首启） | 0.08–0.25 s | **可跳过**：wav 直发 |
| **端到端** | **≈ 47 s** | **7.4 s**（实测 e2e，含争用） | 见 4.3 |

### 4.2 一个反直觉的实测发现：同卡双模型会互相拖慢

同一份 wav、同一个常驻 RVC 进程：

- TTS 空闲时调用：`446 ms` / `566 ms`
- 紧接着 CosyVoice 刚跑完时调用：`1605 ms` / `1797 ms`

差 3 倍。CosyVoice 把显存吃到高位后，RVC 的 rmvpe/faiss 抢不到显存。**结论：8 GB 卡上「TTS 与 RVC 都常驻」的真实代价要按 1.7 s 算，不要按 0.5 s 算。**

### 4.3 端到端实测记录

```
[run 0] total=12411ms (tts 5104 + rvc 6600 + silk 707)   ← RVC 首调用（rmvpe/cuda 预热）
[run 1] total= 7430ms (tts 5719 + rvc 1605 + silk 106)
[run 2] total= 7475ms (tts 5426 + rvc 1797 + silk 252)
```
插件骨架全链路（`voice_say` → 落盘到 `qqbot-workspace`）：`8290 ms`（tts 6615 + rvc 1675）。
OpenAI 门面（AstrBot 走的那条）：热 `9451 ms`，冷首调 `12799 ms`。

### 4.4 群聊可接受吗

- 单句 10–20 字：**7–9 s**。QQ 里 7 秒无响应＝用户以为没反应 → 必须先发一条文字/表情占位（`event.send("…")` / 先回文本再追语音）。
- 30 字以上或多句：线性放大（TTS 是 RTF>1 的），**不可接受** → `maxChars` 卡 40–80，长回复只发文字。
- 想进 3 s 内，只有两条路：① 换成「训练好音色的 TTS」（GPT-SoVITS `api_v2.py` 直出目标音色，省掉 RVC，且支持流式；本机 `<SOVITS>` 已装、有权重，**本次未实测**）；② 加显存/换卡把 RTF 压到 0.3 以下。

---

## 5. 纠错：SDK **不会**自动把 wav/mp3 转 SILK

原始前提写的是「`sendVoice` 内部 `audio.ts` 会自动把 mp3/wav 转成 SILK」。实测源码后否定：

1. `dist/protocol/utils/audio.js` 里 `audioFileToSilkBase64()` 确实存在，`QQ_NATIVE_UPLOAD_FORMATS = [".wav",".mp3",".silk"]`，**这三个扩展名直接跳过转码**。
2. 更关键：`grep audioFileToSilkBase64 dist/` 除了 `protocol/index.js` 的 `export {}` 再导出，**没有任何调用点**；`shouldTranscodeVoice` / `pcmToSilk` 同样无人调用。
3. 真正的上传路径是 `QQBot.js:675 uploadMedia → protocol/api/media.js:44-46`：
   ```js
   const buf = await fs.promises.readFile(opts.localPath);
   fileData = buf.toString("base64");
   ```
   —— 原文照传，`file_type=3`。

**实践含义（好消息）**：直接发 `.wav` 就行，不需要 SILK，也不需要 ffmpeg/silk-wasm。

**但要注意两个连带事实**：
- 本机 profile `node_modules` 里 **没有** `silk-wasm` / `mpg123-decoder`，`ffmpeg` 也不在 PATH（真 ffmpeg 在 `<FFMPEG_DIR>/ffmpeg.exe`，SDK 的 `detectFfmpeg()` 不会去那里找）。所以**一旦将来真的需要转码，现有环境会直接失败**；要用就设 `FFMPEG_PATH` 或把 `silk-wasm` 装进自己的插件包。
- `.wav` 直发是否被 QQ 服务端接受，**只能真人真群验证**（SDK 自己认为是原生格式，但没在本机跑通过一次真实上传）。

---

## 6. 三个必须避开的坑（外加两个实测新坑）

### 坑① 改 `dist/` 会被插件重装覆盖
`node_modules/@tencent-connect/dsh-qqbot/dist/**` 是构建产物。任何「在 send-file-tool.js 里加一句 TTS」的想法都是错的：`npm/pnpm` 重装、`dsh plugin update` 一到就没了，而且升级后行为静默回退。
→ **只能**走 profile 的 `cordis.patch.yml` + 独立插件包（`dsh plugin add` 自己那一套 `src/`）。

### 坑② `sendFile.restrictPaths` 会挡住音频路径
白名单 = `~/.dsh-qqbot/media` + `config.cwd`（`<AGENT_CWD>`）+ `sendFile.extraRoots`。
把音频生成到 `<AI_SONG>/...`、`<ROOT>/...`、或 `%TEMP%` 都会被 `isPathAllowed()` 拒（报 `file_path 不在允许的目录内`）。
→ 产物**必须**落 `<agent cwd>/tts/`。已实测：`voice_say` 写到 `<AGENT_CWD>/tts-test-rvc-research/tts/*.wav` 成功（370 KB）。

### 坑③ 三套 python 环境的依赖冲突
- CosyVoice env（py 3.10 + torch 2.8）**没有 faiss、没有 rmvpe** → 装不了 RVC。
- RVC venv **没有 cosyvoice**，而且它依赖 `fairseq 0.12.2`，与 CosyVoice 的 `torch 2.8 / torchaudio` 极易打架。
→ **不要试图合成一个 env**。用两个常驻进程 + localhost HTTP。想省一个进程，代价是同时 pin 两套重量级依赖，升级任一侧都会互相炸。

### 坑④（实测新发现）faiss 打不开含中文的 index 路径
```
RuntimeError: Error in faiss::FileIOReader::FileIOReader(const char *) ... 
Error: 'f' failed: could not open <MODELS>/芳乃_RVC_步数120/added_IVF1419_Flat_nprobe_1_fangno_p2_v2.index
```
路径存在、权限正常，就是 faiss 的 C++ 文件层不认非 ASCII。而现有音色库全在中文目录下。
**后果很隐蔽**：不抛到上层，只打 traceback，然后**静默按 index_rate=0 跑**（音色相似度明显下降，但你会以为"换声就是这样"）。
→ 服务启动时把 `model.pth` / `*.index` 复制/软链到 ASCII 目录（`svc_rvc.py` 里的 `ascii_mirror()` 已实现，实测生效）。**装 174 MB index 只需一次。**

### 坑⑤（实测新发现）Edge TTS 在本机 403
`edge-tts 7.2.8` 存在（CosyVoice env），但握手被拒：
```
WSServerHandshakeError: 403, url='wss://speech.platform.bing.com/consumer/speech/synthesize/readaloud/edge/v1?...Sec-MS-GEC=...'
```
→ 不要把它当「快速 TTS 备选」。要么升级 edge-tts 让 `Sec-MS-GEC` 算法跟上，要么直接放弃。

---

## 7. 本期能做 / 不能做

**能做（已验证）**
- 文本 → 目标音色 wav：`7–9 s/句`（常驻两服务）
- 产物落 `agent cwd`，满足 `restrictPaths`
- `voice_say` / `voice_doctor` 工具骨架，`node --check` + 冒烟通过
- OpenAI 兼容门面：`POST /v1/audio/speech` → `200 audio/wav`，RIFF 头正确（336 KB / 4.2 s）
- RVC 常驻服务：模型 0.6 s 加载、0.45 s 热换声、中文路径镜像绕过 faiss 限制

**不能做 / 待人工**
- **真发一条语音到 QQ**：无 `group_openid`（QQ Bot API 不提供群列表，`replyTarget` 只能由入站消息派生），且需要在线 WS 会话。凭据本身有效（已成功换到 `access_token`，`expires_in=2306`），但「能拿 token」≠「能发消息」。
- 真群验证 `.wav` 是否被 QQ 接受为语音气泡（见 §8）
- AstrBot 全链路（本机未安装，`openai_tts_api` 的配置键与请求格式已按源码核对，但未跑过 AstrBot 本体）
- `dual_output`、`trigger_probability` 的实际观感

---

## 8. 人工验证清单（必须在真实 QQ 群做）

**dsh-qq 通道**
1. 起两个服务（见 §9），`curl 127.0.0.1:8788/health` 与 `:8789/health` 都返回 `ok:true`。
2. 在**测试 profile**（非 `qqbot`）里装入 `dsh-plugin-voice-say`，启动后确认日志有 `voice-say: voice_say / voice_doctor 已注册`。
3. 群里 @ 机器人发「用语音说：今天辛苦了」→ 观察是否生成 `<AGENT_CWD>/tts/*.wav`，再自动调 `qqbot_send_file`。
4. **核对气泡形态**：是「语音条（可点播放、显示秒数）」还是「文件」？若是文件 → QQ 不接受 wav 作 `file_type=3`，需回到 SILK 路线（把 `silk-wasm` 装进插件包，发 `.silk`）。
5. 记录：从发消息到气泡出现的实际秒数（预期 8–12 s，含模型思考）。
6. 边界：发 100 字长句 → 应被 `maxChars` 拒绝并提示拆分，不应静默截断。
7. 并发：连发两条 → 第二个请求应在 TTS 锁上排队（预期第二条更慢，但不串音）。

**AstrBot 通道（装好后）**
8. 安装 AstrBot（uv 或 Windows 一键包），平台适配器选 **aiocqhttp**（NapCat），**不要**选 QQ 官方接口。
9. 加提供商：类型 `OpenAI TTS API`，`api_base=http://127.0.0.1:8790/v1`，`model` / `openai-tts-voice` 都填 `fangno120`，`timeout=180`。
10. `provider_tts_settings.enable=true`、`provider_id` 选它、`dual_output=true`。
11. 私聊发一句短话 → 应收到「文字 + 语音」两条；再关 `dual_output` 只留语音。
12. 关键回归：**不要在群里同时开 dsh-qq 和 AstrBot 两个 bot 抢同一个群**（两边都会回，且 QQ 官方接口与 NapCat 是两套账号体系）。

---

## 9. 起服务 & 回滚

```bash
# 服务 1：TTS（CosyVoice env，~28 s 就绪）
cd "<REPORT>/bench"
"<COSY_ENV>/Scripts/python.exe" svc_tts.py --port 8788

# 服务 2：RVC（RVC venv，~11 s 就绪）
"<PY>" svc_rvc.py --port 8789 \
  --model "<MODELS>/芳乃_RVC_步数120/model.pth" \
  --index "<MODELS>/芳乃_RVC_步数120/added_IVF1419_Flat_nprobe_1_fangno_p2_v2.index"

# 服务 3：OpenAI 门面（只用 AstrBot 时；任意 python）
"<PY>" svc_openai_shim.py --port 8790
```

回滚：删掉 profile patch 里 `voice-say` 那一段 → 重载 profile；杀掉三个进程。插件包与产物目录可直接删，**不影响** dsh-qqbot 本体（从未改过 `dist/`）。

---

## 10. 文件清单

```
rvc-research/
  音色接驳_dsh-qq与AstrBot.md      ← 本文
  bench/
    tts_bench.py                   CosyVoice 冷/热基线（含 JSON 输出）
    rvc_daemon_bench.py            RVC 常驻基线（Config() 需清 sys.argv、weight_root 要指对）
    silk_bench.mjs                 wav→PCM→SILK 计时
    edge_bench.py                  Edge TTS（本机 403，留作证据）
    e2e_bench.mjs                  端到端三阶段计时
    svc_tts.py                     常驻 TTS 服务（8788）
    svc_rvc.py                     常驻 RVC 服务（8789，含 ascii_mirror）
    svc_openai_shim.py             OpenAI 兼容门面（8790，纯标准库）
    openai_tts_probe.mjs           模拟 AstrBot get_audio 的协议验证
  packages/
    dsh-plugin-voice-say/          dsh 插件骨架（已冒烟）
      package.json  cordis.patch.yml  src/index.js  test/smoke.mjs
    astrbot_plugin_voice_patch/    AstrBot 按需覆盖插件（未实机验证）
  audio/                           实验产物（tts / rvc / e2e / silk / edge 日志）
  ascii-models/                    中文路径镜像兜底
```
