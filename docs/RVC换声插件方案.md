<!-- 本文件由工作区技术报告迁移而来，原始绝对路径已替换为占位符：
     <ROOT> 工作根 / <REPO> 上游 clone / <PY> venv 解释器 / <MODELS> 音色库 /
     <FFMPEG_DIR> ffmpeg 目录 / <REF> 参考集 / <REPORT> 交付目录。
     本仓库实际以 rvcx/paths.py + rvcx.local.toml 解析这些路径。 -->

# RVC 换声插件方案（从训练到换声）

> 参考对象：本机 `<AI_SONG>\dsh-plugin-train`（DDSP 训练，19 工具）+ `dsh-plugin-cover`（DDSP 换声）
> 目标：把「训练 → 换声」整条链换成 RVC，并保持现有插件形态与用户话术不变
> 核查时间：2026-10-25 ｜ 上游：RVC-Project **2.3.260718**（2026-07 发布）

---

## 0. 一句话结论

**不要新建一个平行的 RVC 插件，给现有插件加一层 backend 适配器。**
取材层（找素材 → 出数据集 → 清洗）与交付层（混音 → 母带 → 验收）**与模型无关，可以 100% 复用**；
真正要换的只有两处：**训练六步** 和 **换声那一行命令**。

DDSP 的天花板是**机制性**的，不是步数不够——本机 2 万步 DDSP 与 RVC 已做同源同参考集对拍：
DDSP **−0.0189**（把频谱推离目标）、RVC **+0.0409**（推近目标），三句一致，方向相反（见 §7c）。
RVC 走的是「HuBERT 内容特征 + f0 + top1 检索替换」，
检索项**直接来自目标角色的训练集**，音色不是「学出来」而是「查出来」的——这是它能越过源音色的原因。
⇒ **值得做，且本机硬件已被官方覆盖**（2.3 明确提供 RTX 50 系 cu128 路线）。
⇒ **P1 + P2 均已跑完**：200 轮 / 22400 步 / 4 h 21 min，8 GB 显存无 OOM，链路全程无 GUI。

---

## 1. 本机现状核查

| 项 | 状态 |
|---|---|
| RVC 本体 | **未安装**（`D:\` 全盘无 RVC 目录） |
| DDSP-SVC 6.2 | 已装，`02_模型推理_DDSP-SVC-6.2\DDSP-barbara-6.2`，共享 env 在 `02\...\env` |
| 训练插件 | `dsh-plugin-train` v0.4.0，19 工具，agent 驱动取材 |
| 换声插件 | `dsh-plugin-cover`，`runtime/{convert,mix,mix_master,validate_model,verify_mix}.py` |
| 训练数据 | `04_训练工作区/dataset_raw`、`datasets`、`corpus`，已有角色级数据集（实测单角色 895 句 / 24.8 分钟） |
| GPU | RTX 5060 Laptop **8 GB**（sm_120，Blackwell） |
| 另一套新 torch | `<SOVITS>\` 已有 torch **2.7.1+cu128** 独立 venv —— 证明本机 50 系可用，但**不要复用那个 venv**（依赖冲突） |

---

## 2. RVC 2.3 的真实链路（六段，全部可 CLI）

★ 2.3 把目录从 `infer/modules/train/**` 重构成顶层 `train/**`，并新增正式离线入口 `infer/cli.py`。
网上 2023–2024 的教程路径**全部失效**（本次核查亲自撞到两次 404）。

以下 `<RVC>` = RVC 根目录，所有命令 **cwd 必须是 `<RVC>`**（`train.py` 里 `./logs` 是硬编码相对路径）。

### ① 预处理（切片 + 双采样率）
```bash
<RVC>/env/python.exe train/preprocess.py <数据集目录> 48000 <进程数> logs/<实验名> False 3.7
```
- 位置参数：`inp_root, sr, n_p, exp_dir, noparallel, per[, manifest]`
- ★★ **第 5 位语义是反的**：传 `True` 表示**不并行**。要并行必须传 `False`。
- 切片：阈值 −42 dB，最短 1500 ms，间隔 400 ms，最长保留静音 500 ms，片段 **3.7 s**，重叠 0.3
- 产出 `logs/<exp>/0_gt_wavs/*.wav`（48k）+ `1_16k_wavs/*.wav`（16k，给 HuBERT）

### ② F0 提取
```bash
# ★ 必须以 cpu / cuda 开头，不是直接传 exp 目录
#   cuda <n_part> <i_part> <i_gpu> <exp_dir> <is_half>
<RVC>/env/python.exe train/dataset/extract_f0.py cuda 1 0 0 logs/<实验名> True
# 或用 CPU 并行：cpu <exp_dir> <n_p> <f0method>
<RVC>/env/python.exe train/dataset/extract_f0.py cpu logs/<实验名> 8 rmvpe
```
产出 `2a_f0/*.wav.npy` 与 `2b-f0nsf/*.wav.npy`。歌声一律 **rmvpe**（2.3 训练侧只留 rmvpe / pm）。

### ③ HuBERT 特征（v2 = 768 维）
```bash
<RVC>/env/python.exe train/dataset/extract_hubert_feature.py cuda:0 1 0 logs/<实验名> v2 True
#                                         device n_part i_part exp_dir      version is_half
```
- 支持断点续跑（已存在 `.npy` 自动跳过）
- 产出 `3_feature768/*.npy`

### ④ 生成 `filelist.txt`（★ 官方没给 CLI，必须自己造）
DDSP 没有这一步，WebUI 里是按钮，CLI 路线必须自己补。格式 5 列、`|` 分隔：
```
0_gt_wavs/a01_0.wav|3_feature768/a01_0.npy|2a_f0/a01_0.wav.npy|2b-f0nsf/a01_0.wav.npy|0
```
第 5 列是 speaker id，**单角色填 0**。四个路径必须**同时存在**才写进去。

### ⑤ 训练
```bash
<RVC>/env/python.exe train/train.py \
  -e <实验名> -sr 48k -f0 1 -bs 4 -g 0 -te 200 -se 20 \
  -pg assets/pretrained_v2/f0G48k.pth -pd assets/pretrained_v2/f0D48k.pth \
  -l 1 -c 0 -sw 1 -v v2
```
- ★ `-e` 是 **`logs/` 下的名字，不是绝对路径**；`training_files` 被强制指向 `logs/<exp>/filelist.txt`
- ★ `-sw 1`：存档时**自动把小模型导出到 `assets/weights/`** —— 这是绕开下面「坑 7」的正路
- `-l 1` 只留最新 G/D（省盘），`-c 0` 不把数据缓存进显存（8 G 卡必须 0）
- 单卡默认不走 DDP

### ⑥ 建索引（faiss）
```bash
<RVC>/env/python.exe train/train_index.py <实验名> v2 <RVC>/assets/indices <n_cpu>
#                 位置参数：exp_name version outside_index_root n_cpu [index_mode]
```
- 产出 `added_IVF*_Flat_nprobe_1_<exp>_v2.index`，并**硬链接**一份到 `assets/indices/`
- ★ `trained_*.index` 是中间产物，**推理要的是 `added_*`**
- 支持断点续（已有 added 直接跳过）

### ⑦ 换声
```bash
<RVC>/env/python.exe infer/cli.py \
  --model "<小模型.pth 绝对路径>" --index "<added_*.index 绝对路径>" \
  --input <干声目录> --output <输出目录> \
  --pitch 0 --f0-method rmvpe --index-rate 0.75 --protect 0.33 --rms-mix-rate 1.0 --format wav
```
CLI 参数全表（已核对源码）：`--model`（必填，.pth）、`--input`、`--output`、`--speaker-id`、
`--list-speakers`、`--pitch`（半音，默认 0）、`--f0-method`（**仅 pm / rmvpe**）、`--index`、
`--index-rate`（0–1，默认 0.75）、`--resample-sr`、`--rms-mix-rate`（0–1，默认 1.0）、
`--protect`（**0–0.5**，默认 0.33）、`--format`（wav/flac/mp3/m4a）、`--recursive`、`--overwrite`。

---

## 3. 已核实的九个坑（每一个都会白跑一轮）

| # | 坑 | 事实 | 对策 |
|---|---|---|---|
| 1 | 目录已重构 | 2.3 是 `train/`、`train/dataset/`、`infer/cli.py`；老教程 `infer/modules/train/...` 直接 404 | 版本锁 2.3.260718，路径常量集中一处 |
| 1b | `extract_f0.py` 的第一位不是路径 | 必须以 `cpu` / `cuda` 开头：`cuda <n_part> <i_part> <i_gpu> <exp_dir> <is_half>`，或 `cpu <exp_dir> <n_p> <method>` | 照「extract_f0.py <exp> <n_p> rmvpe」写必崩 |
| 1c | 依赖里已无 fairseq | 2.3 改用 `transformers>=4.49,<4.50` 加载 HuBERT（`assets/hubert_base/*` 三件） | 源码直装可行，不必用整合包 |
| 2 | 并行参数反向 | `preprocess.py` 第 5 位 `"True"` = **不并行** | 固定传 `False` |
| 3 | `filelist.txt` 缺失 | 官方无 CLI 生成器；缺了 `train.py` 直接读不到数据 | 自己造（见 ④），四路径齐备才写入 |
| 4 | `-e` 不是路径 | 被 `os.path.join("./logs", name)` 二次拼接 | 只传名字；**实验名全局唯一**（`<角色>_<日期>`） |
| 5 | index 命名 | `trained_` 是中间产物，`added_` 才能推理；`cli.py` 会自动把含 `trained` 的名字替换成 `added`，替换失败就报 FileNotFound | 只认 `added_*`；显式 `--index` 绝对路径 |
| 6 | 不给 index 直接崩 | `index_rate > 0` 且找不到 index ⇒ `FileNotFoundError`，**不像 DDSP 可以不给** | 传 index，或显式 `--index-rate 0` |
| 7 | 大 pth 不能推理 | `logs/<exp>/G_*.pth` 含优化器状态，不是推理模型 | `-sw 1` 让训练自己导出；**别用 GUI 的 ckpt 提取页** |
| 8 | ★ 安全 | 2.3 的 checkpoint 提取界面会对同目录 `train.log` 末字段执行 `eval()`（上游 issue #2866，2026-10-01 仍存在） | 插件**永不调用**该接口；不把第三方模型包丢进 `assets/weights` |
| 9 | 电平被改过 | `preprocess.py` 对切片做混合归一化：`x/max*(0.9*0.75) + 0.25*x` ⇒ 峰值 ≈ 0.675 | ★ 换声输出电平与 DDSP 不同，`mix_master.py` 的增益与 `silent_ratio` 判据**必须重新标定** |

补充：多说话人模型的默认 speaker id 是**最小已声明 id**（不一定是 0）；`--protect` 上限 0.5。

---

## 3b. ★ 实跑补充的七个坑（纯 CLI 化才会遇到，网上查不到）

以上九个是读源码能看出来的。下面七个是**真机跑 P1 时逐条撞出来的**，WebUI 用户一辈子碰不到——因为 WebUI 全替你做了。

| # | 坑 | 现象 | 对策 |
|---|---|---|---|
| **10** | 必须 `-m` 包方式调用 | `python train/preprocess.py` → `ModuleNotFoundError: No module named 'infer'`；就算把 repo 根塞进 `PYTHONPATH`，又会 `ImportError: cannot import name 'utils' from partially initialized module 'train'`——因为 `sys.path[0]` 是**脚本所在目录**，`import train` 直接命中了 `train/train.py` 自身，形成循环导入 | 一律 `python -m train.preprocess` / `-m train.train` / `-m infer.cli`（`-m` 让 `sys.path[0]` 落在 cwd=repo 根） |
| **11** | `logs/<exp>/` 必须**预先存在** | `preprocess.py` 一开始就 `open("logs/<exp>/preprocess.log","a")` → `FileNotFoundError`；它不会自建 | 训练前 `mkdir -p logs/<exp>` |
| **12** | `logs/<exp>/config.json` 必须自己生成 | `train.py` 的 `get_hparams()` 直接读它 → `FileNotFoundError`；WebUI 训练前从 `configs/v2/48k.json` 复制过去 | 单说话人**原样复制** `configs/v2/48k.json`（`spk_embed_dim` 保持 **109**，只去掉 `speaker_info`；多说话人才改 110） |
| **13** | filelist 必须**绝对路径** | `data_utils._filter()` 直接 `os.path.getsize(audiopath)`，相对路径按 cwd 解析 → `FileNotFoundError: '0_gt_wavs/0_2.wav'` | 写绝对路径（WebUI 里就是 `"%s/logs/..." % now_dir`） |
| **14** | ★★ filelist 必须**注入静音样本** | WebUI 给**每个说话人往 filelist 追加 2 条** `logs/mute/*`；README 让解压 `mute.zip` 正是为此。只造真实样本的行，训练照跑，但**模型会在静音/气口处喷噪**——RVC 最经典的毛病 | 下 `mute.zip`（仅 305 KB）解压到 `logs/`，追加两行 `mute<sr>.wav`（48k 就是 `mute48k.wav`） |
| **15** | `ffmpeg` 必须在 PATH | `infer/audio.py` 里硬编码 `run(cmd=["ffmpeg","-nostdin"])`，不做路径探测 → `WinError 2 系统找不到指定的文件`，切片「成功 0 / 失败 45」 | 把完整版 ffmpeg/ffprobe 所在目录**前置**进子进程 PATH（本机用 `<FFMPEG_DIR>`） |
| **16** | `preprocess.py` 吞异常 | 所有切片都失败，**退出码依然是 0**，脚本以为成功 | 只能数产物（`1_16k_wavs/*.wav` 数量）判断成功与否 |
| **17** | `assets/weights/` 必须存在 | `savee()` 里直接 `torch.save(opt, "assets/weights/%s.pth")`，目录不在就抛 `RuntimeError: Parent directory assets/weights does not exist`；**异常被吞掉，训练照常继续，只是小模型永远导不出来** | 训练前 `mkdir -p assets/weights`；已训练的可用 `train.process_ckpt.savee()` 手动补提（注意 `hps` 里**没有** `if_f0`/`version`，得单独传） |

> 这七条合起来说明一件事：**RVC 的「WebUI 一键训练」不是 UI 糖，它承担了六项隐含职责**（建目录、写 config、拼绝对路径、补静音样本、注入 mute 行、判成败）。
> 谁想把它 CLI 化，这六项必须逐条补回来——这也是本方案建议「封装成工具」而不是「写个 bash 脚本」的根本原因。

---

## 4. 架构：给现有插件加 backend，而不是新起一套

```
dsh-plugin-train（取材层：复用，一行不改）
   hunt / fetch / toolchain / unpack / scan / plan / build / qc
        │  产出同一份 dataset_raw（与 DDSP 共用，不重复下载语料）
        ▼
   ┌──────────── backend 适配器（新增）────────────┐
   │  ddsp:  preprocess.py -c configs/x.yaml → train_reflow.py → 单 .pt   │
   │  rvc :  preprocess → f0 → hubert → filelist → train → index → pth+index │
   └───────────────────────────────────────────────┘
        ▼
   03_*/<角色>_RVC/         ← 注册层同构，dsh-plugin-cover 可直接发现
        ▼
dsh-plugin-cover
   convert.py（新增 rvc 分支）· mix_master.py（复用，需重标定）· verify_mix.py（复用）
```

### 4.1 工具映射（DDSP 19 → RVC 版）

| DDSP 工具 | RVC | 说明 |
|---|---|---|
| `train_doctor` / `train_setup` | 同名，加 backend 分支 | 校验 py312 / torch2.7.1+cu128 / hubert / rmvpe / 底模 / faiss |
| `train_acquire`/`hunt`/`unpack`/`fetch`/`toolchain` | **原样复用** | 取材与模型无关 |
| `train_build` / `train_qc` | 复用 + 加 RVC 判据 | 新增：总时长 ≥ 10 min、切片数、静音比、16k 转换成功率 |
| `train_preprocess` | 改名分支 | 命令换 ① |
| — | **新增** `rvc_f0` | ② 单独成工具（可断点续） |
| — | **新增** `rvc_feature` | ③ 单独成工具（可断点续） |
| — | **新增** `rvc_filelist` | ④ 必须有 |
| `train_run` | `rvc_train` | ⑤ 后台 detach + 日志落盘（`use_ddp=False`） |
| — | **新增** `rvc_index` | ⑥ |
| `train_register` | 同构扩展 | 拷 `assets/weights/<exp>_e*_s*.pth` + `added_*.index` 到音色库 |
| `train_infer` | `rvc_infer` | ⑦ |
| `train_status`/`stop`/`overview`/`plan`/`sources` | 复用 | 拓扑图由 6 段改成 7 段 |

净增 **4 个工具**（f0 / feature / filelist / index），其余复用或换命令。

### 4.2 目录布局

★ 实测改为 **E 盘**（D 盘只剩 28 G，97% 已用；整合包 7.6 G + 解压约 12 G 放不下）：

```
<ROOT>\                          # 独立根，不塞进 <AI_SONG>（跨盘但职责清晰）
├── repo\                           # RVC 2.3 源码（HEAD 81eed5e, 2026-08-04）
│   ├── assets\{hubert_base,rmvpe,pretrained_v2,weights,indices}
│   └── logs\<实验名>\               # 训练中间产物（硬约束：-e 拼 ./logs）
├── venv\                           # ★ 独立 py3.12.14 + torch2.7.1+cu128
├── dataset\<角色ASCII名>\            # 从 DDSP 数据集复制并改 ASCII 名
├── probe\{in,out}\                 # 换声验证
├── logs\                           # 各步骤日志
└── p1_run.py                       # 六段编排（带产物校验 / 断点续跑）

<AI_SONG>\03_音色模型_Xxx\<角色>_RVC\   # 注册后（沿用 DDSP 音色库惯例）
├── model.pth                       # 小模型（推理用）
├── added_*.index
└── .rvc-manifest.json              # 步数/采样率/v2/实验名/index 哈希
```

★ 音色库沿用 DDSP 的「一角色一目录」惯例，cover 插件按同一套发现规则即可，用户话术不变。
★ `logs/` 是硬约束（`-e` 拼相对路径），所以注册动作必须是「**拷出 + 清理 logs**」，否则第二个角色会撞名。

---

## 5. 验证计划（沿用本机已有的客观判据，不靠耳朵）

用**同一份**数据集（建议就用已达 24.8 分钟的那位角色）跑 A/B，判据全部来自 `ai-song-cover-pipeline`：

| 指标 | 方法 | DDSP 基线（实测） | RVC 预期 |
|---|---|---|---|
| 音色相似度 | **LTAS + 参考集相关性**（勿用单句频谱质心） | 目标 **+0.239** / 源 **+0.468** | **待测**，判定线：是否越过源 |
| 换声是否生效 | MFCC 欧氏距离 | 32.42 | 待测 |
| 音高保持 | F0 中位差 | 2.5 Hz（key 0） | 待测 |
| 换声后电平 | 有声段 RMS | 0.1811 → 0.1697（几乎无损） | **必须重测**（见坑 9） |
| 气口完整度 | `silent_ratio`（精确 0 采样占比） | 原唱 0.3% / DDSP **22.8%** | 待测；DDSP 靠后期混响补到 0.07% |
| 成品响度 | LUFS vs 目标 | −9 LUFS ±0.2 | 复用 |
| 推理耗时 | 4 min 干声 | **59 s** | 待测（RVC 分块推理，预计更慢） |

★ 做对比时**两版必须先锁到同一响度再听**（本机惯例 −9 LUFS），否则「谁更响谁赢」，听到的差异全是假的。

---

## 6. 资源与风险

**环境成本**：独立 venv（torch 2.7.1+cu128，约 3–4 GB）+ 底模（hubert ~360 MB / rmvpe ~180 MB / pretrained_v2 若干百 MB）+ 整合包。
走沙箱代理比直连快（实测 493 vs 145 KB/s），HF 可用 `hf-mirror.com`。

**8 GB 显存**：48k v2 训练偏紧，`-bs` 从 **2~4** 起步、`-c 0`；
推理若 OOM 调小 `configs/config.py` 的 `x_pad / x_query / x_center`。

**真正的风险不是技术，是判据**：
1. RVC 检索式的代价是 `index_rate` 高时**会「复读」训练集里的发音**，歌声听感可能比 DDSP 更「粘」不利索；
2. 因此 RVC **不该默认替换 DDSP**，而应作为「要像」时的第二条通路，由质检指标（LTAS 是否越过源）自动决定用哪版；
3. 2.3 是新版本（2026-07），社区资料几乎全是旧路径，**照抄教程的失败率极高**——这是本方案坚持「路径集中常量 + 版本锁定」的原因。

**移交人工**：无。整链可 CLI 化，不需要 GUI（本机 GUI 结构性起不来）。

---

## 7. 落地路线（三步，每步都有停止条件）

| 步 | 内容 | 通过条件 | 状态 |
|---|---|---|---|
| **P0 环境** | 源码 + 独立 venv（非整合包）→ 验 cu128 → 下 hubert/rmvpe/pretrained_v2 | 显卡被识别、`infer/cli.py --help` 可跑 | ✅ 已过 |
| **P1 单角色贯通** | 用现有 24.8 min 数据集跑完六段 → 50 epoch 出第一个可听模型 | 换声生效、F0 保持、无 OOM | ✅ 已过（§7b） |
| **P2 量化对拍** | 训练至收敛（200 轮 / rmvpe），与 DDSP 2 万步做同源同参考集对比 | 明确结论 + 选出可用模型 | ✅ 已过（§7c） |
| **P3 接进插件** | 给 `dsh-plugin-train` / `-cover` 加 `backend: ddsp\|rvc`，净增 4 个工具 | 一次调用完成训练→注册→换声 | ⏸ 未开工 |

**建议先做 P1**：它能在不写一行插件代码的前提下，回答「值不值得」这个前置问题。
P1 失败（例如 8 G 显存跑不动 48k v2），则整套方案降级为 40k v1，或直接回退 DDSP——**这个判断不该等到插件写完才发现**。

---

## 7b. ★ P1 实测结果（2026-10-25 12:45–13:56，已跑完）

**环境**：E 盘源码安装 + 独立 venv（py 3.12.14）；`torch 2.7.1+cu128`、CUDA 12.8、`RTX 5060 Laptop`、capability **(12,0)=sm_120**、8 GB 显存。**8 GB 完全够跑 48k v2 / bs 4**，全程未 OOM。

**数据**：芳乃 358 wav / 24.8 min（44.1 kHz）→ 切片 **445 个**（3.7 s/片）。

**六段实测耗时**：

| 段 | 命令要点 | 耗时 |
|---|---|---|
| ① 预处理 | `-m train.preprocess … 48000 8 logs/fangno_p1 False 3.7` | **24.5 s**（358 文件全成功） |
| ② F0 | `-m train.dataset.extract_f0 cpu logs/fangno_p1 8 pm` | **12.2 s** |
| ③ HuBERT | `-m train.dataset.extract_hubert_feature cuda:0 1 0 logs/fangno_p1 v2 True` | **77.0 s**（445/445） |
| ④ filelist | 自造（绝对路径 + 2 条静音） | <0.1 s |
| ⑤ 训练 | `-m train.train -e fangno_p1 -sr 48k -bs 4 -te 50 -se 10 …` | **≈47 min（57 s/epoch）** |
| ⑥ 索引 | `-m train.train_index fangno_p1 v2 <assets/indices> 8` | **11.1 s** |
| ⑦ 换声 | `-m infer.cli … --f0-method rmvpe`（GPU） | **18.6 s**（含加载；纯推理 特征0.42+F0 3.89+合成1.52 s） |

**客观指标（15 s 干声，《下等马》原唱，源≠芳乃）**：

| 指标 | 10 epoch | **50 epoch** | DDSP 基线 |
|---|---|---|---|
| 换声生效（MFCC L2） | 45.10 | **46.66** | 32.42 |
| F0 中位差 | 0.0 Hz | **−1.9 Hz** | 2.5 Hz |
| 峰值 | 0.734 | **0.797** | 0.871 |
| 电平变化 | −2.80 dB | **−2.84 dB** | 近无损（−0.14 dB） |
| `exact_zero_ratio` | 0.0 | **0.0** | **1.4 %** ★（原记 22.8 %，已修正，见 §7c） |
| 与目标 LTAS 相关（换声前） | — | +0.9477 | — |
| **与目标 LTAS 相关（换声后）** | — | **+0.9926** | 见 §8 天花板 |
| 目标本人自测（上限参考） | — | +0.9863 | +0.938 |

**结论（三条，都可复核）**：

1. **链路 100% 跑通**，从原始 wav 到换声成品无一处需要 GUI。八个脚本级坑已全部固化进 `p1_run.py`。
2. **音色确实朝目标移动**：LTAS 相关 +0.9477 → **+0.9926**（Δ +0.0449），且**高于目标本人自测的 0.9863**。这是 DDSP 做不到的方向性证据——但只是频谱平衡，不等于「听起来就是那个人」，且仅 1 条 15 s 样本，**不可外推**。
3. ★ **RVC 不产生「精确 0」的气口**：`exact_zero_ratio = 0.0`，而 DDSP 换声后有 **22.8%** 采样被归零、必须靠后期混响补救。这意味着 **RVC 产物可以跳过 DDSP 那条「补气口」工序**，混音链要相应简化。

**尚未做（P2 范围）**：多句盲测、与 DDSP 同响度 A/B、50 vs 200 epoch 收敛曲线、index_rate 扫描。当前 50 epoch 只是「能出声且方向对」，**不是可交付音质**。

---

## 7c. ★★ P2 实测结果（2026-10-25 15:01–19:23，200 轮跑完）

P1 用的是 `pm` 提音高（因为 rmvpe 当时下不动）。P2 把训练侧换成 **rmvpe**，从零重开一个实验 `fangno_p2`，跑满 **200 轮 / 22400 步 / 4 小时 21 分**。

### ① 训练规模与耗时

| 项 | P1 | P2 |
|---|---|---|
| F0 方法 | pm（parselmouth） | **rmvpe**（GPU，79.5 s / 445 片） |
| 轮次 | 50 | **200** |
| 每轮耗时 | 57 s | **65 s（1.09 min）** |
| 总耗时 | 47 min | **4 h 21 min** |
| 切片/特征 | 445 / 445 | 445 / 445（**与 P1 md5 完全一致**） |

> 特征与 `0_gt_wavs` 的 md5 在 P1/P2 之间逐字节相同，只有 `2a_f0/*.npy` 不同——这正是 pm→rmvpe 的唯一变量。**索引因此可以用同一批特征**，P2 的 `added_*.index` 与 P1 内容等价。

### ② 轮次曲线：**200 轮是白跑的**

同一句干声、`index_rate=0.75`、10 个检查点横评：

| epoch | 20 | 40 | 60 | 80 | 100 | 120 | 140 | 160 | 180 | 200 |
|---|---|---|---|---|---|---|---|---|---|---|
| r_lta | .9917 | .9914 | .9916 | .9921 | .9907 | **.9925** | .9909 | .9901 | .9916 | .9902 |
| Δr | .0440 | .0438 | .0439 | .0445 | .0430 | **.0449** | .0433 | .0424 | .0440 | .0426 |

**从第 20 轮起就进入平台区，之后 180 轮的全部波动（±0.002）都在噪声内。** 训练时间可以从 4 小时压到 40 分钟而不损失可测量的音色位移。

### ③ 二维网格：3 检查点 × 4 档 index_rate × 3 句干声（36 次推理）

| 检查点 | index_rate | r_lta | Δr | 朝目标 MFCC 改善 | 峰值 | 归零率 |
|---|---|---|---|---|---|---|
| e120 | 0.0 | .9909 | .0398 | 29.4 % | .872 | 0 |
| e120 | 0.5 | .9915 | .0404 | 24.1 % | .814 | 0 |
| **e120** | **0.75** | **.9920** | **.0409** | 20.3 % | .783 | **0** |
| e120 | 1.0 | **.9923** | **.0411** | 16.0 % | .773 | 0 |
| e200 | 0.0 | .9884 | .0373 | 31.4 % | .830 | 0 |
| e200 | 0.75 | .9885 | .0374 | 24.1 % | .758 | 0 |
| e20 | 0.75 | .9902 | .0391 | 15.5 % | .764 | 0 |

**两种指标在 index_rate 上打架，必须如实记下**：

- **LTAS 说**：index_rate 越高越像目标（.9909 → .9923）。
- **朝目标 MFCC 距离说**：index_rate 越高，换声幅度越小（29.4 % → 16.0 %）。

本鱼倾向相信 LTAS——它是「有声段长时平均谱」，对音素内容不敏感；而 15 s 单句的平均 MFCC 受咬字与音高影响极大，参考集也只有 25 条。**但这条分歧没有解决**，所以 `index_rate=0.75` 是按「折中 + 社区默认」定的，不是被数据证明的最优。

### ④ ★ 同源同参考集，DDSP 与 RVC 第一次正面对撞

同一 3 句干声、同一芳乃参考集（本人自测上限 r = **0.9863**），两边都用 rmvpe、都不变调：

| 系统 | r_lta | Δr（相对源） | 精确归零率 | F0 中位差 |
|---|---|---|---|---|
| 源干声（原唱泠鸢） | 0.9512 | — | — | — |
| **DDSP 20000 步** | 0.9323 | **−0.0189** ✗ | 1.4 % | −3.3 Hz |
| **RVC e120** | **0.9920** | **+0.0409** ✓ | **0.0 %** | −3.9 Hz |

**DDSP 在 3 句上一致地把频谱推离了目标**（负位移），RVC 一致地推近。这不是「谁更响」，是方向相反。

**必须同时说明三条限制**，否则这个结论会被用坏：

1. **LTAS 只测频谱平衡**，不测自然度、不测气息、不测「像不像」。RVC 可能在「听起来是芳乃」上仍然输——这要用耳朵判。
2. **DDSP 用的是默认 `method=auto` / `t_start=auto`**，没有为它调参。给它换 rk4、调 `infer_step`、换 `t_start` 可能翻盘。**这是「默认配置对比」，不是「引擎上限对比」。**
3. **样本仍是 3 句 15 s**，来自同一首歌同一段。跨歌、跨音域未测。

### ⑤ 修正一个 P1 的错误数字

P1 表里记的「DDSP `exact_zero_ratio` = 22.8 %」**在同口径复测下不成立**：本次同源实测是 **1.4 %（3 句均值）/ 1.75 %（单句）**。那个 22.8 % 来自另一套管线与文件，不可比。已在上表改正。

结论方向不变（RVC 确实是 0），但**幅度被我夸大了一个数量级**——这正是「换口径复测」的价值。

### ⑥ 交付物

- 模型：`<MODELS>\芳乃_RVC_步数120\`（`model.pth` 54.9 MB + `added_*.index` 166.8 MB + `.rvc-manifest.json`）
- 已用**绝对路径独立推理**验证（不依赖 `logs/`）：热启动 **5.35 s / 15 s 音频**（特征 0.27 + F0 3.17 + 合成 1.91）
- **对照页**：`换声对照页.html` —— 12 条音频**各自独立成文件、独立播放**，每条标清「是什么 / 哪个模型 / 什么参数」，
  并提供「原始输出 / 响度对齐」两档切换。声源、片段位置、指标口径全部写在页内。
  （生成器 `build_compare.py`，音频在 `audio/`，增益与峰值见 `audio_manifest.json`）
- **不再交付「把所有系统拼成一条 wav」的文件**。拼接会抹掉「哪一段是谁」，且响度未对齐时听感结论全被响度带偏。
  旧的拼接文件已移入 `_internal/`，仅作内部留档，不作为交付物。
- 脚本：`sweep.py`（checkpoint × index_rate 网格，支持多片段平均）、`compare_ddsp.py`（DDSP 同口径基线）、
  `build_compare.py`（独立对照页）、`ab_make.py`（内部拼接用）、`p2_launch.py`（WMI 脱离进程跑长训练）

### ⑦ P2 给插件方案的三个结论

1. **训练器不需要 200 轮**：默认给 100 轮，且必须暴露 `se`（每 N 轮存）——因为优劣在 20 轮后就看不出差别，存多了只是占盘。
2. **必须同时暴露 `f0_method` 与 `index_rate`**，且**索引是强依赖**：`index_rate>0` 时缺 `added_*.index` 直接崩，DDSP 那边没有这个概念。
3. **客观判据只留两条**：`Δr`（方向对不对）+ `exact_zero_ratio`（要不要补气口）。MFCC 类指标在本任务上不可靠，**已从插件验收判据里剔除**。

---

## 7d. ★ 台词换声：TTS+RVC vs 真人+RVC（2026-10-25 21:00–22:20）

### ① 实验设计（只留「谁说」一个变量）

台词 4 句（2.3 / 3.3 / 4.9 / 7.6 s）。做法是**先对真人录音做 ASR 转写，再用 TTS 按完全相同的文本合成**——
这样两条通道的差异只剩「谁说」，不掺「说了什么」。

| 通道 | 源 | 采样率 | 说明 |
|---|---|---|---|
| **T** | GPT-SoVITS **kurumi v3**（另一角色，日语） | 32 kHz | `text_split=cut0`，峰值归一 0.9 |
| **H** | **茉子 / 日立真子** 游戏内真人语音 | 44.1 kHz | 与芳乃是同一游戏、同一录音链路 |

两通道 × {e120, e200} × 4 句 = **16 次换声**（rmvpe / index_rate 0.75 / pitch 0）。

### ② 结果（4 句均值）

| 通道 | 系统 | r_lta | Δr | ASR 回读 CER | 时长比 | 精确归零 |
|---|---|---|---|---|---|---|
| T | 源（TTS） | .8655 | — | .0375 | 1.000 | .0674 |
| T | **e120** | **.9568** | **+.0912** | **.0375** | .995 | 0 |
| T | e200 | .9567 | +.0911 | .0581 | .995 | 0 |
| H | 源（真人） | .9629 | — | **.0000** | 1.000 | .0271 |
| H | e120 | .9632 | +.0003 | .0917 | .996 | 0 |
| H | e200 | .9605 | −.0024 | .1167 | .996 | 0 |

参考集自测上限 `r = 0.9863`。

### ③ 四个结论

1. **两条通道都能用**。TTS 通道换声后 CER 与源**完全一致（.0375 → .0375）**，一个字没多丢；
   真人通道反而从 0 升到 .091。原因：真人台词有更多气口、弱读、话尾，RVC 换音色时这些位置没有可靠 f0/内容，容易被抹平；
   TTS 源干净，反而更抗换声。**但别读成「TTS 更好」**——CER 只测「听得清」，测不出合成腔。

2. ★★ **`Δr` 被录音信道污染，不能跨信道比**。真人源一进来就有 `.9629`，几乎贴着上限 `.9863`——
   茉子和芳乃是同一个游戏、同一套录音链路的声优，**长时平均谱（LTAS）主要由信道决定，声纹只占很小一部分**。
   所以真人通道 Δr ≈ 0 **不代表换声失败**，而是这个指标在该通道上没有分辨力。
   ⇒ **§7c ④ 的解读必须加限定**：Δr 只能在**同一信道内部**比较（例如都是同一条原唱干声时比 RVC 与 DDSP），
   跨信道比（TTS vs 真声、分离人声 vs 游戏原声）会被信道差异放大，结论不可直接搬。

3. **e120 与 e200 在本任务里分不出高低**：Δr 差 0.0001，CER 反而 e120 更低（T .0375 vs .0581 / H .0917 vs .1167）。
   与主人听感（**e200 略好、但边际效率不高**）并不冲突——听的是自然度与音色质感，CER 听的是「能不能听懂」，
   两个指标**不重叠**。
   ⇒ **订正 §7c ②③ 的口径**：不该写成「e120 最优」，应写「**要质感取 e200，要性价比取 e120，差距不足以重训**」；
   新角色按 **100–150 轮封顶**，两个检查点各留一份。

4. **台词场景的验收判据要加一条 CER**。唱歌靠乐句兜住，说话不能：`Δr` 判「像不像」，
   `exact_zero_ratio` 判「要不要补气口」，**`CER` 判「听不听得懂」**。三条缺一个都可能交付残次品。

### ④ 两个新坑（都表现为「看起来完全不可用」，其实是管线 bug）

- ★ **GPT-SoVITS v2 会截断长句**：同一句 4.9 s 的台词，v2 只合成出 2.5 s，ASR 只听到「あ、申し遅れました」，
  后半句直接消失。**必须用 v3**（v3 对同一句给 7.1 s、CER 0.25）。
- ★★ **v3 返回的是 int16 量级数组（峰值 22352），而 `soundfile` 按 float 写 PCM_16 会硬削顶**：
  实测写出来 rms 0.95、频谱平坦度 0.15（正常语音 0.01–0.05），削波谐波把 HuBERT 特征彻底污染，
  换声后 CER 直接 **1.0**（完全不可懂）。归一化后立刻回到 **.0375**。
  ⇒ 跨工具链传音频，**写盘前必须查 dtype 与峰值**；看到「换声把人声搞崩了」，先怀疑输入电平，再怀疑模型。
- ★ **GPT-SoVITS 有自己的 `.venv`**（`<SOVITS>/.venv`，含 `pyopenjtalk` / `jieba_fast` / `peft`）。
  借用 AI-Song 的 DDSP env 会一路缺依赖（ffmpeg-python → peft → jieba → jieba_fast → pyopenjtalk），
  且 `jieba_fast` 在 Windows 没有 cp310 预编译包、源码编译失败。

### ⑤ 交付物

- **台词换声对照页**：`台词换声对照页.html`（24 条音频各自独立成文件、独立播放；
  每个台词块上下两栏 = TTS 通道 / 真人通道，逐条标注源/CER/Δr；提供「原始输出 / 统一 -20 dBFS」切换）
- 脚本：`dialogue_tts_ja.py`（TTS 合成）· `dialogue_conv.py`（两通道×两检查点换声）·
  `dialogue_metrics.py`（音色与时长）· `dialogue_cer.py`（ASR 回读 CER）· `build_dialogue_page.py`（对照页）
- **未做**：真人通道的 index_rate 扫描。
- 中文台词的跨语言验证 → 见 **§7e**（当晚补做）。

---

## 7e. ★★ 中文台词：日语训练的模型换中文（2026-10-25 22:47–23:20）

### ① 为什么必须单列一批

§7d 的源和训练集**同一种语言**，测不出「跨语言迁移」。而**芳乃的模型只用日语训过** ——
主人真正会用到它的场景却是中文台词。所以本批只换一个变量：把源从日语换成中文。

换 TTS 引擎的原因：本机 GPT-SoVITS 的两个音色（kurumi / yoshino）都是日语数据训的，
喂中文它们自己先念错，测出来是「TTS 不会中文」而不是「RVC 能不能换中文」。
⇒ 改用 **CosyVoice2-0.5B**（中文原生，zero-shot 克隆，Apache-2.0），参考音色用 `音色预设\DS娘_晓伊A3\reference.wav`（与芳乃不同人、不同链路）。

台词 6 句：**前 4 条是 §7d 四句的对应中译**（语义对齐，便于跨语言比较），
**后 2 条是中文特有难点**（语气词+数字+反问；多分句长句+连续齿音 sh/zh/ch）。

| 项 | 值 |
|---|---|
| 源 | CosyVoice2-0.5B · `cross_lingual` · 24 kHz · seed 20261025 |
| 换声 | 6 句 × {e120, e200} @ index_rate 0.75 + 6 句 × e120 @ {0.0, 0.50} = **24 次** |
| 参数 | rmvpe · **pitch 0** · protect 0.33 |

### ② 结果（6 句均值）

| 系统 | r_lta | Δr | 拼音 CER | 声调 CER | 声调轮廓 r | 半音 RMSE | 时长比 | 精确归零 |
|---|---|---|---|---|---|---|---|---|
| 源（CosyVoice2） | .7564 | — | .010 | .000 | 1.000 | .00 | 1.000 | 0 |
| **e120**（ir .75） | **.9705** | **+.2141** | **.017** | **.000** | **.985** | **.68** | .994 | 0 |
| e200（ir .75） | .9694 | +.2130 | .000 | .000 | .982 | .74 | .994 | 0 |
| e120（ir 0） | .9705 | +.2141 | .000 | .000 | .985 | .68 | .994 | 0 |
| e120（ir .50） | .9701 | +.2136 | .000 | .000 | .984 | .69 | .994 | 0 |

参考集自测上限 `r = 0.9863`；朝目标 MFCC 增益 `conv_gain`：e120 **+42.6%** / e200 +41.2%。
**最差的一条（拼音 CER .059，zh05「三遍了」被 ASR 听成「三遍啊」）在源上就已经存在**，换声没有让它更差。

### ③ 六个结论

1. **跨语言迁移成立：不掉字。** 拼音 CER 源 .010 → e120 .017 / e200 .000；声调 CER **全 0**
   （24 条换声输出里没有一个字的调被听错）。差的那一点点是 ASR 自己认错字，不是换声糊掉的。
   ⇒ 日语训练的 RVC 模型可以直接拿去换中文台词，**这是本批最重要的一条**。

2. ★★ **声调是被「工程约定」保住的，不是模型懂汉语。** `pitch 0` 的语义是「直接用源 f0 驱动」，
   f0 轮廓根本没经过模型决策 —— 所以四个调型原样过（轮廓 r .985 / 半音 RMSE .68）。
   推论：一旦为了变调去开 pitch shift（±N 半音），保的是轮廓、丢的是音区，**中文唱歌要单独验**。

3. ★★ **Δr +0.214 不能读成「中文换得更好」。** 日语 TTS 通道只有 +.091，差在**起点**：
   这次源是晓伊音色，与芳乃既不同人也不是同一套录音链路，LTAS 本来就远（.756 vs 日语批的 .866），
   所以有 0.21 的空间可挪。⇒ **Δr 是「挪了多远」，不是「像不像」的绝对值**，
   与 §7d ② 的纪律（跨信道不可比）是同一条，这里又拿到一次独立证据。

4. **index_rate 在中文上没有收益，甚至可以关掉。** ir 0 / .50 / .75 的 Δr 是
   +.2141 / +.2136 / +.2141，声调轮廓 r 是 .985 / .984 / .985，拼音 CER 是 .000 / .000 / .017 ——
   差异全在 1~2 个字的量级，属噪声。**索引库是日语帧，理论上会把中文特征拉向日语音系；实测没发生。**
   ⇒ 中文台词场景可以 `--index-rate 0`，省掉索引文件与这个理论风险（听感上索引通常让音色更贴，需耳朵复核）。

5. **e120 vs e200：第三次得到同一个答案。** Δr .2141 vs .2130、轮廓 r .985 vs .982、CER .017 vs .000，
   **客观指标不可分**。加上 §7c（唱歌）与 §7d（日语台词），"**新角色 100–150 轮封顶**"现在有三份独立证据。

6. **缺一环：真人中文源。** 本机**没有任何中文真人录音素材**（所有 GPT-SoVITS 角色、芳乃数据集都是日语），
   所以本批只有 TTS 通道。补录 3 句（16 kHz 以上、安静房间、正常语速）即可以同一套脚本复跑：
   存到 `<ROOT>\dialogue_zh\src_real\`，把脚本里的 `src_tts` 换成 `src_real`。

### ④ 判据升级：中文必须加"拼音层"CER

汉字层 CER 会把**同音字**判成错（实测 `真子 → 贞子`、`日立 → 日丽`，两者拼音完全相同），
把「ASR 认错字」误读成「RVC 掉字」。本批加了两层：

| 层 | 算的是什么 | 作用 |
|---|---|---|
| 汉字 CER | 字级编辑距离 | 沿用日语批，跨语言可对照 |
| **拼音 CER** | 汉字→带声调拼音后的编辑距离 | 同音字不罚，**同时把声调算进去** |
| **声调 CER** | 只比调号序列（妈=1 麻=2 马=3 骂=4） | 单独看「调有没有被换掉」 |

加上 **声调轮廓 r / 半音 RMSE**（源与出的 f0 曲线对齐后比），中文台词的可懂度判据才完整。

### ⑤ 三个新坑

- ★★ **CosyVoice 的 `instruct2` 会把指令当场词念出来。** 首轮 zh01 回读成
  「请只回答我的问题，**语速适中，句间自然停顿**」，CER 1.25。根因：instruct 必须以
  `You are a helpful assistant. ` 开头、以 `<|endofprompt|>` 结尾（见 `cosyvoice/utils/common.py`），
  少了就被当作正文继续念。补上前缀后长句仍会漏尾巴（zh04 多出「语速适中」）
  ⇒ 最终改用 `cross_lingual`（只取音色、无指令文本，没有可泄漏的东西）。
- ★★ **用 `pyin` 测声调会假报警。** 第一版算出 zh04 的轮廓 r=.44、RMSE 4 半音（等于换调型），
  但 ASR 的声调 CER 是 0 —— 两个判据不可能同时为真。查下去：换声后 `pyin` 的半音 std 从 2.94 跳到 **6.92**（八度跳变），
  是**估计器**坏了。改用换声器自身的 `rmvpe` 后 r=.99。
  ⇒ **两个判据打架时先怀疑测量工具**，别急着改结论。
- ★ **DTW 在短句上会把曲线拉错。** zh02 只有 1.2 s 浊音，`DTW(subseq)` 给 r=.64；
  而时长比只有 .990（差 1%），根本不需要弯曲。改**线性对齐**后同一批稳定在 .976–.990。

### ⑥ 交付物

- **中文台词对照页**：`中文台词对照页.html`（30 组音频各自独立成文件、独立播放；
  主区 = 源/e120/e200，另设 index_rate 三档对照区；逐条标注 Δr / 声调 r / CER / ASR 回读）
- 脚本（`<ROOT>\`）：`dialogue_tts_zh.py`（中文源）· `dialogue_conv_zh.py`（换声 + ir 扫描）·
  `dialogue_metrics_zh.py`（音色/声调）· `dialogue_cer_zh.py`（三层 CER）· `build_dialogue_zh_page.py`（对照页）
- **未做**：真人中文源、中文唱歌（pitch shift 下的声调保真）。

---

## 8. 与 DDSP 的能力边界对照（交付话术用）

| 维度 | DDSP-SVC 6.2 | RVC v2 48k |
|---|---|---|
| 相似度机制 | 学出来的音色向量，**难越源音色** | top1 检索替换，**理论上可越** |
| 咬字/节奏保持 | 极好（本机实测波形级贴近） | 好，但 `index_rate` 高时发音会被拉向训练集 |
| 情感 | 无接口，100% 继承 | 同 |
| 单角色数据量 | 越多越好（本机 24.8 min） | **≥ 10 min** 即可起步 |
| 单模型产物 | 1 个 `.pt`（自带 config） | **2 件套**：`.pth` + `.index`（必须配套） |
| 推理速度 | 快（4 min → 59 s） | 待测，预计更慢 |
| 调参面 | key / 步数 / 采样 | + `index_rate` / `protect` / `rms_mix_rate` / f0 方法 |
| 换声后电平 | 几乎无损 | **被预处理归一化改变，需重标定** |

---

## 9. 与本机 GPT-SoVITS 的分工（回答「都有 GPT-SoVITS 了，为什么还要 RVC」）

本机 `<SOVITS>` 已跑通 v5，实测零样本 cam++ 对真人质心 **0.8140**（真人同人留一法上限中位 0.7867），
热 RTF 0.42。既然它已经「比真人还像真人」，为什么翻唱还要 RVC？

**因为两者不是同一条链：**

| | GPT-SoVITS（TTS） | DDSP / RVC（SVC） |
|---|---|---|
| 输入 | **文本 + 参考音频** | **一段人声干声** |
| 谁决定旋律 | 模型自己（要说/唱的音高、时值由它生成） | **源音频**，一个音符都不改 |
| 咬字时值 / 颤音 / 气口 / 情绪 | 需要重新生成，容易「唱得平」 | **原封不动继承原唱** |
| 翻唱适用性 | 要从头「唱」一首，需额外做旋律与对齐 | 直接吃分离出的干声，天然跟拍 |
| 本机状态 | 已跑通，零样本可用 | DDSP 已通；RVC 未装 |

⇒ **RVC 替代的是 DDSP，不是 GPT-SoVITS。** 判断口诀：

- 有原唱、只想「换嗓子」→ **SVC（RVC / DDSP）**
- 没原唱、要「凭空说话或唱」→ **TTS（GPT-SoVITS）**

所以本机最终形态是**两条并行的音色通路**，不是二选一：
翻唱用 SVC 保表现力，配音/念白用 GPT-SoVITS 保自然度。RVC 的增量价值只有一个——**在保留原唱表现力的前提下，把「像不像」做上去**。

---

## 附：本次核查的信息来源

- 仓库结构与源码：`RVC-Project/Retrieval-based-Voice-Conversion-WebUI` @ `main`（`infer/cli.py`、`train/{preprocess,train,train_index,utils}.py`、`train/dataset/extract_hubert_feature.py`）
- 发布说明：Release **2.3.260718**（含 50 系整合包 `RVC20260718Nvidia50x0.7z`、`requirments_cu128_py312.txt`、单卡不走 DDP、断点续跑）
- 安全缺陷：issue #2866（checkpoint 提取页 `eval()` 读 `train.log`），2026-10-01 报告，`main` 未修
- 本机基线数据：`~/.workbuddy/skills/ai-song-cover-pipeline/SKILL.md`（§1 三坑、§5 验收、§7 性能、§8 天花板）
