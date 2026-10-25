# -*- coding: utf-8 -*-
"""程序化中文语料生成：字表 + 词表 + 模板句 → 声韵母覆盖均衡的句表。

给 RVC 训练集用的「不是随便抓一堆句子」的语料。三条设计约束：

  1. **音素覆盖**：按「声母+韵母」（不带调）算，贪心补齐没覆盖到的音节；
     再补一轮「音节 × 调号」，否则某些调从没出现过，换声会把调读平。
  2. **句长分布**：1–15 字都要有。全是长句 ⇒ 切片后气口太多；全是短句 ⇒
     模型学不到连读与句调（本机实测：短句占比过高时换声输出发「一顿一顿」）。
  3. **特殊读法**：数字、小数、年份、英文缩写、书名号、问号/感叹号/省略号，
     TTS 与 HuBERT 在这些地方最容易出错，必须显式出现在训练集里。

词表优先级：`--wordlist` 指定 > 本机 jieba dict.txt（含词频）> 内置小词表。
补覆盖用的「枚举句」形如 `苹果、铁路、豆浆、风筝。`：一句能塞 4 个音节，
读起来是正常列举，不会像生造句那样把 TTS 逼出怪语调。

★ 依赖 `pypinyin`（本机只有 CosyVoice env 里有；注意那个 python 是 3.10，
  `tomllib` 缺失 ⇒ rvcx.paths 读不到 rvcx.local.toml，所以这里不靠 paths.COSY）：

      <cosy-env>/Scripts/python.exe scripts/dataset/gen_corpus.py ...

产物（默认落在 `<WORK>/corpus/<name>/`）：
    text.txt        一行一句，喂给 synth_corpus.py
    corpus.jsonl    每行 {"id","text","bucket","note","syllables"}
    coverage.json   覆盖率报告（可复现、可比较）
"""
import sys as _sys
from pathlib import Path as _pth

_sys.path.insert(0, str(_pth(__file__).resolve().parents[2]))
from rvcx import paths as _paths  # noqa: E402

import argparse
import json
import random
import re
import statistics
from collections import Counter
from pathlib import Path

# --------------------------------------------------------------------------- #
# 内置词表：jieba dict 缺失时的兜底。都是高频、含常见声韵母的词。
# --------------------------------------------------------------------------- #
BUILTIN_WORDS = """
我们 你们 他们 什么 怎么 这样 那样 现在 时候 知道 觉得 可以 应该 已经 因为
所以 但是 如果 虽然 然后 或者 还有 没有 不是 就是 一起 自己 大家 东西 事情
今天 明天 昨天 早上 晚上 中午 时间 地方 问题 结果 意思 名字 声音 电话 电脑
中国 北京 上海 学校 老师 学生 同学 朋友 家里 路上 房间 桌子 椅子 窗户 门口
说话 听见 看见 起来 出来 回来 过来 下来 上去 打开 关上 放下 拿起 走到 站在
喜欢 讨厌 害怕 担心 高兴 难过 生气 着急 安静 热闹 干净 漂亮 好看 重要 简单
麻烦 复杂 清楚 明白 记住 忘记 认为 决定 准备 开始 结束 继续 停止 完成 改变
苹果 香蕉 西瓜 米饭 面条 鸡蛋 牛奶 咖啡 啤酒 火锅 零食 面包 汤面 意面 饼干
天气 下雨 下雪 刮风 太阳 月亮 星星 海边 山上 河边 树林 草地 花园 街道 城市
工作 学习 生活 旅行 游戏 电影 音乐 故事 计划 地图 消息 新闻 工具 机器 系统
第一 第二 第三 最后 另外 其实 大概 可能 一定 特别 非常 有点 稍微 完全 几乎
手指 眼睛 耳朵 鼻子 嘴巴 头发 肩膀 后背 膝盖 心跳 呼吸 力气 速度 方向 距离
"""

# 手写「自然句」：承担句调、语气、功能句型的多样性。tag 用于后续平衡抽样。
NATURAL = [
    # 1–3 字（极短，句调与语气词）
    ("好。", "short"), ("行吧。", "short"), ("可以。", "short"), ("不行。", "short"),
    ("知道了。", "short"), ("等一下。", "short"), ("真的吗？", "short"), ("别急。", "short"),
    ("走了。", "short"), ("没事。", "short"), ("谢谢。", "short"), ("对不起。", "short"),
    ("有点意思。", "short"), ("再说一遍。", "short"), ("你确定？", "short"),
    # 4–7 字（陈述、祈使、反问）
    ("我明白了。", "medium"), ("先别动。", "medium"), ("这不是重点。", "medium"),
    ("你听我说完。", "medium"), ("我们换个说法。", "medium"), ("把窗户关上。", "medium"),
    ("这盘棋还没下完。", "medium"), ("他刚才说什么？", "medium"), ("你可以再想想。", "medium"),
    ("别把顺序弄反了。", "medium"), ("这话说得太满。", "medium"),
    ("外面开始下雨了。", "medium"), ("桌上的水凉了。", "medium"),
    # 8–11 字（完整长句，含从句）
    ("我先确认一下你手上的资料。", "long"),
    ("这个问题比看上去要麻烦一点。", "long"),
    ("如果你现在改变主意，也不算晚。", "long"),
    ("他把整张地图摊开在桌子上。", "long"),
    ("我们先把结论放在一边，重新看证据。", "long"),
    ("走廊尽头那盏灯一直在闪。", "long"),
    ("她说话的时候习惯把手放在口袋里。", "long"),
    ("外面的风把窗帘吹得鼓起来。", "long"),
    # 12–15 字（多分句、连读）
    ("先确认我们看到的是什么，再决定下一步。", "long"),
    ("我已经把计划写得很完整了，现在轮到别人挑毛病。", "long"),
    ("这个结果并不理想，先停一下，把依据重新对齐。", "long"),
    ("猜得再热闹也没有用，还是得看这一局会怎么走。", "long"),
    ("厨房里飘出来一股味道，像是有人把汤煮糊了。", "long"),
    ("他把伞收起来抖了抖水，然后才推门进去。", "long"),
    # 数字 / 年份 / 小数 / 单位
    ("这次一共要三十七份。", "number"),
    ("第三排在左边第二个柜子里。", "number"),
    ("总共花了八百六十五块钱。", "number"),
    ("温度是零下四度，风不大。", "number"),
    ("误差只有百分之零点三。", "number"),
    ("会议定在下午两点半开始。", "number"),
    ("这一批大概有二百多件。", "number"),
    ("他的号码是零一三九开头。", "number"),
    # 英文混读
    ("这个文件是 PDF 格式的。", "english"),
    ("我们先把 API 的返回值对齐。", "english"),
    ("合同里写的是 a 和 b 两个版本。", "english"),
    ("这个 bug 我复现了三次。", "english"),
    ("麻烦把 log 和配置一起发我。", "english"),
    ("他管这个叫 GPU 占用。", "english"),
    # 标点 / 语气（问、叹、省略、破折、书名）
    ("你是认真的吗？", "punct"), ("这不可能吧！", "punct"),
    ("我……好像忘了带钥匙。", "punct"), ("等等——你先别说话。", "punct"),
    ("他读完了《城与墙》的最后一页。", "punct"),
    ("她说「明天再说」，然后就挂了电话。", "punct"),
    ("好，那就把地图展开。", "punct"),
    ("说真的，我有点后悔了。", "punct"),
    # 齿音/塞擦音密集（中文 TTS 最容易含糊的地方）
    ("四十四只石狮子。", "hard"), ("知道就是知道，不知道就是不知道。", "hard"),
    ("吃醋和吃饭是两回事。", "hard"), ("支持者只占少数。", "hard"),
    ("城墙上面晒着三件衬衫。", "hard"),
    # 轻声 / 儿化 / 变调
    ("孩子们在院子里玩。", "hard"), ("桌上面放着个小盒子。", "hard"),
    ("慢慢儿来，不着急。", "hard"), ("姑姑和叔叔都来了。", "hard"),
]

HAN = re.compile(r"[\u4e00-\u9fff]")
SEP = "、"
# 每行枚举几个词 → 决定句长分布。★ 实测 XiaoyiNeural(rate -5%) 约 4.4 字/秒，
# RVC 的切片窗口是 3.7 s ⇒ 想要 3–5 s 的句子，得 13–22 字，即 6–9 个双字词。
# 少于 3 词（约 1.4 s）的句子训不进模型，只能当句调样本留在句表里。
WORDS_PER_LINE = [5, 6, 6, 7, 7, 8, 8, 9, 10, 11]
PUNCT = set("，。、？！：；…—-《》「」“”")


# --------------------------------------------------------------------------- #
# 词表 / 字表
# --------------------------------------------------------------------------- #
def find_jieba_dict(explicit: str | None) -> Path | None:
    """jieba dict.txt 里带词频，是这机器上唯一现成的「词 + 频率」资源。

    ★ 不通过 paths.COSY 找：cosy env 的 python 是 3.10、没有 tomllib，
      rvcx.paths 会读不到 rvcx.local.toml（COSY=None）。直接问 jieba 自己最准。
    """
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.is_file() else None
    try:
        import jieba  # noqa: PLC0415
        p = Path(jieba.__file__).with_name("dict.txt")
        if p.is_file():
            return p
    except Exception:
        pass
    if _paths.COSY:
        for sp in ("Lib", "lib"):
            p = Path(_paths.COSY) / "env" / sp / "site-packages" / "jieba" / "dict.txt"
            if p.is_file():
                return p
    return None


def load_words(path: Path | None) -> list[tuple[str, int]]:
    """返回按词频降序的 (词, 频次)。只保留 2–4 字的纯汉字词。"""
    han = re.compile(r"^[\u4e00-\u9fff]{2,4}$")
    out: list[tuple[str, int]] = []
    if path:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 2:
                    continue
                w, f = parts[0], parts[1]
                if han.match(w):
                    try:
                        out.append((w, int(f)))
                    except ValueError:
                        continue
        out.sort(key=lambda x: -x[1])
    if len(out) < 200:                       # 没词表或词表坏了，用内置兜底
        out = [(w, 1000 - i) for i, w in enumerate(BUILTIN_WORDS.split())]
    return out


# --------------------------------------------------------------------------- #
# 拼音
# --------------------------------------------------------------------------- #
class Phone:
    """对 pypinyin 的一层薄封装：只关心「无调音节」和「调号」。"""

    def __init__(self) -> None:
        try:
            from pypinyin import Style, lazy_pinyin
        except ModuleNotFoundError:
            raise SystemExit(
                "[corpus] 缺 pypinyin —— 音素覆盖是这份脚本的唯一目的，不能降级跑。\n"
                f"         本机在 CosyVoice env 里：{_paths.COSY}\\env\\Scripts\\python.exe\n"
                "         或 pip install pypinyin 后重跑。"
            )
        self._lazy = lazy_pinyin
        self._Style = Style

    def syl(self, text: str) -> list[str]:
        """无调音节序列；非汉字原样返回（数字/字母也算一个占位）。"""
        return self._lazy(text, style=self._Style.NORMAL, errors=lambda s: list(s))

    def toned(self, text: str) -> list[str]:
        """带调音节（tone3：ni3）。统计「同一音节四个调是否都出现过」用。"""
        return self._lazy(text, style=self._Style.TONE3, errors=lambda s: list(s))


def char_syllables(chars: str, ph: Phone) -> dict[str, str]:
    """一次批量算完所有字的无调音节（逐字调用 pypinyin 要几十秒，批量只要一次）。"""
    uniq = sorted({c for c in chars if HAN.match(c)})
    syl = ph.syl("".join(uniq))
    return dict(zip(uniq, syl))


def char_tones(chars: str, ph: Phone) -> dict[str, str]:
    uniq = sorted({c for c in chars if HAN.match(c)})
    t = ph.toned("".join(uniq))
    return dict(zip(uniq, [x[-1] if x and x[-1] in "12345" else "0" for x in t]))


# --------------------------------------------------------------------------- #
# 组句
# --------------------------------------------------------------------------- #
def build(words, cs: dict[str, str], ct: dict[str, str], ph: Phone,
          want: int, per_line: int, rng: random.Random):
    """先铺手写句，再用「词条枚举句」两轮贪心：先补音节，再补调。"""
    rows: list[dict] = []
    covered: set[str] = set()
    tones: set[tuple[str, str]] = set()
    used: set[str] = set()

    def add(text: str, bucket: str, note: str = "") -> None:
        s = ph.syl(text)
        rows.append(dict(id=f"c{len(rows):04d}", text=text, bucket=bucket,
                         note=note, syllables=s))
        covered.update(s)
        for ch in text:
            if ch in cs:
                tones.add((cs[ch], ct.get(ch, "0")))

    for text, bucket in NATURAL:
        add(text, bucket, "natural")

    # ---- 第一轮：音节覆盖 ----
    inv = Counter()
    for w, f in words[:4000]:
        for ch in w:
            if ch in cs:
                inv[cs[ch]] += f
    idx: dict[str, list[str]] = {}
    for w, _f in words[:60000]:
        for ch in w:
            if ch in cs:
                idx.setdefault(cs[ch], []).append(w)

    def emit(buf: list[str], bucket: str, note: str) -> None:
        add(SEP.join(buf) + "。", bucket, note)

    buf: list[str] = []
    cap = rng.choice(WORDS_PER_LINE)   # ★ 每行的词数上限只抽一次；每轮都抽会退化成「取最小值」
    for s, _ in inv.most_common():
        if len(rows) >= want:
            break
        if s in covered:
            continue
        for w in idx.get(s, []):
            if w not in used:
                buf.append(w)
                used.add(w)
                covered.update(cs[ch] for ch in w if ch in cs)
                break
        # 每行词数随机 2–5：枚举句的字数直接决定句长分布，全固定 4 词会
        # 把 8 字句堆成一座山，模型学不到短句的句调。
        if len(buf) >= cap:
            emit(buf, "enum", "cover_syllable")
            buf, cap = [], rng.choice(WORDS_PER_LINE)
    if buf:
        emit(buf, "enum", "cover_syllable")
        buf = []

    # ---- 第二轮：调号覆盖（只补高频音节，否则会为了冷僻字写一堆怪词）----
    tone_idx: dict[tuple[str, str], list[str]] = {}
    for w, _f in words[:40000]:
        for ch in w:
            if ch in ct and ct[ch] != "0":
                tone_idx.setdefault((cs[ch], ct[ch]), []).append(w)
    hot = [s for s, _ in inv.most_common(400)]
    for s in hot:
        if len(rows) >= want:
            break
        for t in "1234":
            if (s, t) in tones:
                continue
            for w in tone_idx.get((s, t), []):
                if w not in used:
                    buf.append(w)
                    used.add(w)
                    tones.add((s, t))
                    for ch in w:
                        if ch in ct:
                            tones.add((cs[ch], ct[ch]))
                    break
            if len(buf) >= cap:
                emit(buf, "enum", "cover_tone")
                buf, cap = [], rng.choice(WORDS_PER_LINE)
    if buf:
        emit(buf, "enum", "cover_tone")
        buf = []

    # ---- 第三轮：凑够句数（覆盖已饱和，多出来的数据只为「时长/多样性」）----
    rest = [w for w, _f in words[1000:30000] if w not in used]
    while len(rows) < want and rest:
        take = rest[:rng.choice(WORDS_PER_LINE)]
        del rest[:len(take)]
        used.update(take)
        emit(take, "enum", "fill")

    rng.shuffle(rows)              # 打散，避免合成时长时间停在同类句型
    for i, r in enumerate(rows):
        r["id"] = f"c{i:04d}"
    return rows


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #
def report(rows, words, cs, ct, ph) -> dict:
    inv = Counter()
    for w, f in words[:4000]:
        for ch in w:
            if ch in cs:
                inv[cs[ch]] += f
    target = set(inv)

    got = Counter()
    tones: set[tuple[str, str]] = set()
    for r in rows:
        got.update(r["syllables"])
        for ch in r["text"]:
            if ch in cs:
                tones.add((cs[ch], ct.get(ch, "0")))
    hit = target & set(got)
    w_tot = sum(inv[s] for s in target)
    w_hit = sum(inv[s] for s in hit)
    lens = [len(HAN.findall(r["text"])) for r in rows]
    hot = [s for s, _ in inv.most_common(400)]
    tone4 = sum(1 for s in hot if all((s, t) in tones for t in "1234"))

    return dict(
        sentences=len(rows),
        chars=dict(n=len(lens), mean=round(statistics.mean(lens), 2),
                   p50=sorted(lens)[len(lens) // 2], min=min(lens), max=max(lens)),
        syllables_target=len(target),
        syllables_hit=len(hit),
        cover_unweighted=round(len(hit) / max(len(target), 1), 4),
        cover_freqweighted=round(w_hit / max(w_tot, 1), 4),
        hot_syllables_with_4_tones=f"{tone4}/{len(hot)}",
        buckets=dict(Counter(r["bucket"] for r in rows)),
        char_len_hist=dict(sorted(Counter(lens).items())),
    )


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_paths.WORK / "corpus" / "dsxiaoyi"))
    ap.add_argument("--sentences", type=int, default=300, help="目标句数")
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--wordlist", default=None, help="jieba dict.txt 路径（默认自动找）")
    ap.add_argument("--max-words-per-line", type=int, default=4,
                    help="枚举句每行塞几个词（4 词 ≈ 3.5–5 s，正好落在切片窗口）")
    ap.add_argument("--dry-run", action="store_true", help="只打印统计，不写文件")
    a = ap.parse_args()

    out = Path(a.out)
    wl = find_jieba_dict(a.wordlist)
    words = load_words(wl)
    ph = Phone()
    allchars = "".join(w for w, _ in words[:60000]) + "".join(t for t, _ in NATURAL)
    cs = char_syllables(allchars, ph)
    ct = char_tones(allchars, ph)

    rng = random.Random(a.seed)
    rows = build(words, cs, ct, ph, a.sentences, a.max_words_per_line, rng)
    rep = report(rows, words, cs, ct, ph)

    print(f"[corpus] 词表 = {wl if wl else '内置兜底'}")
    print(f"[corpus] 句数 {rep['sentences']}  字数 {rep['chars']}")
    print(f"[corpus] 音节覆盖 不加权 {rep['cover_unweighted']:.3f} / "
          f"按词频加权 {rep['cover_freqweighted']:.3f}  "
          f"({rep['syllables_hit']}/{rep['syllables_target']})")
    print(f"[corpus] 高频音节四声齐全 {rep['hot_syllables_with_4_tones']}  "
          f"句型分布 {rep['buckets']}")
    print(f"[corpus] 字数分布 {rep['char_len_hist']}")

    if a.dry_run:
        print("[corpus] --dry-run，未写文件")
        return

    out.mkdir(parents=True, exist_ok=True)
    (out / "corpus.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (out / "text.txt").write_text("".join(r["text"] + "\n" for r in rows), encoding="utf-8")
    (out / "coverage.json").write_text(
        json.dumps(dict(seed=a.seed, wordlist=str(wl) if wl else None, **rep),
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[corpus] 写出 -> {out}")


if __name__ == "__main__":
    main()
