# AstrBot 本机部署说明（E 盘）

> 生成时间：2026-10-26　｜　AstrBot v4.29.3（commit 832fc6d）
> 部署位置：`E:\AstrBot\AstrBot`

## 一、访问入口

| 项 | 值 |
|---|---|
| WebUI | **http://localhost:6185** |
| 初始账号 | `astrbot` |
| 初始密码 | `v6xz9rK2ISXx0rPJlOaqklLO`（**首次登录后请立刻改**） |
| 配置目录 | `E:\AstrBot\AstrBot\data\cmd_config.json`（**带 UTF-8 BOM**） |
| 数据/插件 | `E:\AstrBot\AstrBot\data\`（`plugins/`、`config/`、`logs/`、`data_v4.db`） |

## 二、目录与环境

| 项 | 路径 |
|---|---|
| 源码 | `E:\AstrBot\AstrBot`（`git clone --depth 1 https://github.com/AstrBotDevs/AstrBot.git`） |
| venv | `E:\AstrBot\AstrBot\.venv`（Python **3.12.14**，来自 `D:\uv-python\cpython-3.12.14`） |
| uv 缓存 | `E:\uv-cache`（★ D 盘只剩 19 G / 98%，新装一律落 E 盘） |

## 三、启动与停止

```bash
# 启动（前台；Ctrl+C 停）
cd /e/AstrBot/AstrBot && ./.venv/Scripts/python.exe main.py

# 停止（PowerShell，先按端口找 PID）
Get-NetTCPConnection -LocalPort 6185 -State Listen | Select-Object OwningProcess
Stop-Process -Id <PID> -Force
```

★ **改配置必须重启进程才生效**（TTS provider、平台适配器都是启动时读一次）。

## 四、接本机 RVC 音色（零插件路线）

AstrBot 内置 `openai_tts_api` provider，走 OpenAI `audio.speech` 协议，
所以**只要本机有个说这个协议的端点就行**，不需要写插件。

### 链路

```
AstrBot ──POST /v1/audio/speech──▶ :8790 shim（OpenAI 兼容门面）
                                          ├──▶ :8788 CosyVoice2 TTS（24 kHz 干声）
                                          └──▶ :8789 RVC 换声（48 kHz）
```

### 已写入 `cmd_config.json` 的内容

```jsonc
"provider": [
  {
    "id": "local_rvc_tts",
    "type": "openai_tts_api",
    "provider": "openai",
    "provider_type": "text_to_speech",
    "enable": true,
    "api_key": "local",
    "api_base": "http://127.0.0.1:8790/v1",
    "model": "dsxiaoyi_e20",
    "openai-tts-voice": "dsxiaoyi_e20",
    "timeout": "180",
    "proxy": ""
  }
],
"provider_tts_settings": {
  "enable": true,
  "provider_id": "local_rvc_tts",
  "dual_output": false,
  "use_file_service": false,
  "trigger_probability": 1.0
},
"no_proxy": "localhost,127.0.0.1,::1"
```

### 三个常驻服务

| 端口 | 作用 | 启动命令 |
|---|---|---|
| 8788 | CosyVoice2 TTS（DS娘参考音） | 见下 |
| 8789 | RVC 换声（DS娘 e20 权重） | 见下 |
| 8790 | OpenAI 兼容门面（shim） | 见下 |

```bash
cd /d/WorkData/2026-10-19-03-29-21/rvc-voice-kit

# 8788 CosyVoice（★ 必须带 PYTHONPATH，源码在 CosyVoice_源码/ 子目录）
export PYTHONPATH="D:/AI-Voice/01_情感配音_CosyVoice/CosyVoice_源码;D:/AI-Voice/01_情感配音_CosyVoice/CosyVoice_源码/third_party/Matcha-TTS;D:/AI-Voice/01_情感配音_CosyVoice"
/d/AI-Voice/01_情感配音_CosyVoice/env/Scripts/python.exe scripts/service/svc_tts.py \
  --port 8788 --src-root "D:/AI-Voice/01_情感配音_CosyVoice/CosyVoice_源码" \
  --prompt "D:/AI-Voice/音色预设/DS娘_晓伊A3/reference.wav"

# 8789 RVC（DS娘 e20）
export PATH="/d/AI-Song/01_人声分离_MSST-GUI/env:$PATH"
/e/AI-RVC/venv/Scripts/python.exe scripts/service/svc_rvc.py --port 8789 \
  --model "E:/AI-Song/03_音色模型_SenrenBanka/DS娘_晓伊A3_RVC_e20/model.pth" \
  --index "E:/AI-Song/03_音色模型_SenrenBanka/DS娘_晓伊A3_RVC_e20/added_IVF1929_Flat_nprobe_1_dsxiaoyi_a3_v1_v2.index"

# 8790 shim
/e/AI-RVC/venv/Scripts/python.exe scripts/service/svc_openai_shim.py --port 8790 --voice-id dsxiaoyi_e20
```

## 五、已验证

| 验证项 | 结果 |
|---|---|
| WebUI 可达 | `HTTP 200` |
| 登录 | `POST /api/auth/login` 返回 JWT ✅ |
| provider 加载 | 日志 `Loading model openai_tts_api(local_rvc_tts) ...` ✅ |
| shim 合成 | `POST /v1/audio/speech` → **HTTP 200**，5.06 s，48 kHz，peak 0.6973 ✅ |
| **AstrBot 同款 SDK 调用** | `AsyncOpenAI(...).audio.speech.with_streaming_response.create(...)` → 200 / 570284 bytes ✅ |
| 换声结果 | `E:\AstrBot\work\astrbot_tts_test.wav`（5.94 s / 48 kHz / peak 0.7363） |

★ 用 AstrBot 自己的 openai SDK 复测通过，才算证明 **provider 契约真兼容**（只 curl 通不算）。

## 六、已装插件

`astrbot_plugin_voice_patch`（在 `data/plugins/`）——按会话/关键词/前缀把 LLM 回复转成语音，
比内置 TTS 的「全局开关 + 概率」更精细。**默认 `enabled=false`**，配置在 WebUI 插件页。

## 七、尚未验证（需人工）

1. **真实 QQ 群语音气泡**：`aiocqhttp` 适配器要接 NapCat 个人号；
   SDK **不会**自动把 wav/mp3 转 SILK，`sendVoice` 路径上没有转换调用点 ⇒ 需真人真群验。
2. **AstrBot + LLM 主链路**：目前只配了 TTS provider，还没配 LLM provider，
   所以「发消息 → LLM 回 → 转语音」这条完整链路尚未跑通。
3. `dual_output=false`：目前**只发语音、不发原文**；想同时保留文字改这个开关。

## 八、坑位

- ★ **`cmd_config.json` 带 UTF-8 BOM** ⇒ 读写都要 `encoding="utf-8-sig"`，
  否则 `json.load` 直接 `UnicodeDecodeError: Unexpected UTF-8 BOM`。
- ★ **uv 在 Windows 上的 safe-delete bug**：`SHFileOperationW 失败: 0x78`
  ⇒ `uv sync` 在装 sdist 包（jieba / aiocqhttp）时会失败。解法：
  ```bash
  uv pip install setuptools wheel
  uv pip install <包名> --no-build-isolation-package <包名>
  uv sync   # 再跑一次就通过
  ```
- ★ **GPU 是独占资源**：训练 RVC 时**必须停掉 8788 CosyVoice**（它占 2.7 GB）。
  并存时训练从 50 s/轮掉到 **10 min/轮**（backward 被显存挤占，不是死锁）。
- ★ AstrBot 启动时若检测到 `http_proxy`，会自己清掉代理变量并把
  `no_proxy` 设为 localhost/127.0.0.1/::1 ⇒ 本机服务调用不受影响。
