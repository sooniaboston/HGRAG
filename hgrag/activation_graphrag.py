#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
 activation_graphrag.py  --  HGRAG 2.0 : Psychological Activation-Diffusion KG
===============================================================================

在 ``graphrag_retry.py`` (HGRAG v1) 基础上的知识图谱建立方法改写版。
**本文件不覆盖 graphrag_retry.py**，两者可并存。

-------------------------------------------------------------------------------
1. 理论依据
-------------------------------------------------------------------------------
(a) 心理语义网络激活扩散原理 (Collins & Loftus, 1975, *Psychological Review* 82(6))
    - "When a concept is processed, activation spreads out along the paths of the
      network in a decreasing gradient. The decrease is inversely proportional to
      the accessibility or strength of the links in the path."   (Local
      Processing Assumption #1)
    - **Fan effect**: 激活在节点的出边上被分摊 (activation is divided among the
      links emanating from each node) -> 本实现采用 *行归一化转移矩阵* M。
    - 激活随时间/中间活动衰减 -> 衰减因子 d (decay factor)。
    - Firing threshold: 未达阈值的节点不再传播 -> 本实现用阈值剪枝实现。
    - 语义网络的链接通常是双向的 ("These links are pointers, and usually go in
      both directions between two concepts") -> 最终网络为无向图。

    **本实现的关键数学性质**: 令 M 为行随机矩阵 (行和 = 1, 由 fan effect 导出),
    则 R = Σ_{t=1..K} d^t (M^T)^t 的每一行元素和 = Σ_t d^t = d(1-d^K)/(1-d) < 1
    (当 d<1)。因此 *任意* 扩散权重天然严格小于 1，而直接连边被显式赋权为 1。
    这在数学上精确满足 "直接连边权重=1, 间接连边权重<1" 的设计要求。

(b) 刘知远团队的经典 RAG 方法
    - Gao et al., "Retrieval-Augmented Generation for Large Language Models:
      A Survey" (arXiv:2312.10997) —— 团队对 RAG 的经典定义，其中知识图谱
      构建部分采用 LLM 驱动的实体/关系抽提。
    - KG-Infused RAG (THUNLP, arXiv:2506.09542) —— 团队最新的 spreading
      activation RAG 工作。其 spreading activation **没有数值权重** (靠 LLM
      逐轮挑选三元组来控制扩散规模), 仅有轮数上限 k=6 与种子数 k_e=3。
      **本工作的增量**: 把激活扩散从 "LLM 离散选边" 升级为 "Collins & Loftus
      数值扩散模型", 给出严格 <1 的间接边权重, 并与 embedding 客观关联度融合。
    - LLM-KG 构建 (Extract-Define-Canonicalize / CoT-based KG construction):
      实体发现 -> 关系发现 -> 本体对齐/规范化 的三步 CoT 流程。

(c) 实体归并 (Entity Resolution) 工业界最佳实践
    - Blocking (token / trigram / acronym / embedding-neighborhood 并集)
      -> 5 维 ScoreVector 打分 -> union-find 传递性合并 -> 不确定区间 LLM 仲裁。
    - 归并后保留 canonical name + 全部 aliases + merged_from + 证据 (可回溯)。

-------------------------------------------------------------------------------
2. 流水线
-------------------------------------------------------------------------------
    Stage 0  语料装载      : HotpotQA distractor validation -> 唯一段落库
    Stage 1  三元组抽提    : DeepSeek 官方 API, 三步 CoT (实体/关系/规范化)
    Stage 2  实体归并      : Blocking + ScoreVector + Union-Find (+LLM 仲裁)
    Stage 3  客观关联指标  : assoc = w_emb*cos + w_cooc*nPMI + w_kge*2-hop
    Stage 4  激活扩散      : 行归一化 M + 衰减 d + 阈值 -> 间接边权重 (<1)
    Stage 5  导出          : JSON / JSONL / GraphML / GEXF / embeddings
    Stage 6  检索评测      : 查询时激活扩散 + Recall@k 验证网络有效性

-------------------------------------------------------------------------------
3. 用法
-------------------------------------------------------------------------------
    python activation_graphrag.py --stage all --num-questions 200
    python activation_graphrag.py --stage extract --num-questions 200
    python activation_graphrag.py --stage build
    python activation_graphrag.py --stage evaluate

-------------------------------------------------------------------------------
4. 依赖
-------------------------------------------------------------------------------
    pip install openai datasets sentence-transformers scikit-learn networkx scipy
===============================================================================
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import os
import random
import re
import sys
import threading
import time
import unicodedata
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None

try:
    from scipy import sparse as sp
except ImportError:  # pragma: no cover
    sp = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:  # pragma: no cover
    TfidfVectorizer = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
LOG = logging.getLogger("activation_graphrag")


# ============================================================================
# 常量与默认配置
# ============================================================================

DEFAULT_DATA_ROOT = os.environ.get("RAG_DATA_ROOT", r"C:\data\zsk\RAG")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# 实体类型粗分类 (用于 type-agreement 打分与冲突检测)
TYPE_GROUPS: Dict[str, str] = {
    "person": "PERSON", "people": "PERSON", "human": "PERSON",
    "individual": "PERSON", "politician": "PERSON", "artist": "PERSON",
    "author": "PERSON", "actor": "PERSON", "musician": "PERSON",
    "scientist": "PERSON", "player": "PERSON", "director": "PERSON",
    "organization": "ORGANIZATION", "organisation": "ORGANIZATION",
    "company": "ORGANIZATION", "corporation": "ORGANIZATION",
    "firm": "ORGANIZATION", "institution": "ORGANIZATION",
    "university": "ORGANIZATION", "college": "ORGANIZATION",
    "school": "ORGANIZATION", "government": "ORGANIZATION",
    "agency": "ORGANIZATION", "team": "ORGANIZATION", "band": "ORGANIZATION",
    "group": "ORGANIZATION", "party": "ORGANIZATION", "military": "ORGANIZATION",
    "location": "LOCATION", "place": "LOCATION", "city": "LOCATION",
    "country": "LOCATION", "state": "LOCATION", "region": "LOCATION",
    "continent": "LOCATION", "geographic": "LOCATION",
    "work": "WORK", "film": "WORK", "movie": "WORK", "song": "WORK",
    "album": "WORK", "book": "WORK", "novel": "WORK", "magazine": "WORK",
    "newspaper": "WORK", "journal": "WORK", "series": "WORK",
    "tv show": "WORK", "show": "WORK", "play": "WORK", "game": "WORK",
    "software": "WORK", "publication": "WORK", "artwork": "WORK",
    "event": "EVENT", "war": "EVENT", "battle": "EVENT", "conflict": "EVENT",
    "conference": "EVENT", "election": "EVENT", "festival": "EVENT",
    "product": "PRODUCT", "brand": "PRODUCT", "model": "PRODUCT",
    "device": "PRODUCT", "drug": "PRODUCT", "engine": "PRODUCT",
    "date": "DATE", "year": "DATE", "time": "DATE", "period": "DATE",
    "century": "DATE", "decade": "DATE",
    "concept": "CONCEPT", "idea": "CONCEPT", "theory": "CONCEPT",
    "term": "CONCEPT", "principle": "CONCEPT", "field": "CONCEPT",
    "language": "LANGUAGE", "nationality": "NATIONALITY",
    "ethnicity": "NATIONALITY", "species": "SPECIES", "genus": "SPECIES",
    "family": "SPECIES",
}

# 同组类型互相兼容; 跨组不兼容
COMPATIBLE_PAIRS = {
    ("PERSON", "PERSON"), ("ORGANIZATION", "ORGANIZATION"),
    ("LOCATION", "LOCATION"), ("WORK", "WORK"), ("EVENT", "EVENT"),
    ("PRODUCT", "PRODUCT"), ("DATE", "DATE"), ("CONCEPT", "CONCEPT"),
    ("LANGUAGE", "LANGUAGE"), ("NATIONALITY", "NATIONALITY"),
    ("SPECIES", "SPECIES"),
}
# 概念类实体之间允许一定程度的交叉 (如 "machine learning" 与 "AI")
SOFT_COMPATIBLE = {("CONCEPT", "WORK"), ("CONCEPT", "PRODUCT"),
                   ("CONCEPT", "ORGANIZATION"), ("CONCEPT", "PERSON"),
                   ("WORK", "ORGANIZATION"), ("WORK", "PRODUCT"),
                   ("EVENT", "DATE"), ("DATE", "EVENT")}

# 公司/机构法律后缀 (归一化时剥离)
LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company",
    "ltd", "limited", "llc", "plc", "gmbh", "ag", "sa", "nv", "bv",
    "pty", "llp", "lp", "the", "and", "of", "group", "holdings",
    "foundation", "institute", "association", "society",
}

STOP_TOKENS = {
    "the", "a", "an", "of", "and", "or", "in", "on", "at", "to", "for",
    "with", "by", "from", "as", "is", "was", "were", "are", "be", "been",
    "that", "this", "it", "his", "her", "its", "their", "s",
}


# ============================================================================
# 工具函数
# ============================================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def normalize_text(s: str) -> str:
    """Unicode 规范化 + 小写 + 折叠空白。"""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def strip_paren(s: str) -> str:
    """去掉括号消歧义后缀, 如 'Jonathan Stark (tennis)' -> 'Jonathan Stark'。"""
    return re.sub(r"\s*\([^)]*\)\s*$", "", s).strip()


_LEGAL_NORMALIZE_CACHE: Dict[str, str] = {}


def legal_normalize(name: str) -> str:
    """剥离法律后缀与标点后的归一化键 (blocking key)。

    带缓存: 归并阶段会对同一名称反复调用 (百万级), 缓存后避免重复正则。
    """
    cached = _LEGAL_NORMALIZE_CACHE.get(name)
    if cached is not None:
        return cached
    s = strip_paren(normalize_text(name))
    s = re.sub(r"[^\w\s]", " ", s)
    tokens = [t for t in s.split() if t not in LEGAL_SUFFIXES and t not in STOP_TOKENS]
    if not tokens:
        tokens = [t for t in s.split() if t not in STOP_TOKENS]
    out = " ".join(tokens)
    if len(_LEGAL_NORMALIZE_CACHE) < 200000:
        _LEGAL_NORMALIZE_CACHE[name] = out
    return out


_SIG_TOKENS_CACHE: Dict[str, List[str]] = {}


def sig_tokens(name: str) -> List[str]:
    """显著词元 (用于 token blocking / 缩写还原)。"""
    cached = _SIG_TOKENS_CACHE.get(name)
    if cached is not None:
        return cached
    s = legal_normalize(name)
    out = [t for t in s.split() if t not in STOP_TOKENS and len(t) > 1]
    if len(_SIG_TOKENS_CACHE) < 200000:
        _SIG_TOKENS_CACHE[name] = out
    return out


_CHAR_NGRAMS_CACHE: Dict[str, Set[str]] = {}


def char_ngrams(name: str, n: int = 3) -> Set[str]:
    cached = _CHAR_NGRAMS_CACHE.get(name)
    if cached is not None:
        return cached
    s = f"  {legal_normalize(name)}  "
    out = {s[i:i + n] for i in range(max(1, len(s) - n + 1))}
    if len(_CHAR_NGRAMS_CACHE) < 200000:
        _CHAR_NGRAMS_CACHE[name] = out
    return out


_ACRONYM_OF_CACHE: Dict[str, str] = {}


def acronym_of(name: str) -> str:
    cached = _ACRONYM_OF_CACHE.get(name)
    if cached is not None:
        return cached
    toks = sig_tokens(name)
    out = "".join(t[0] for t in toks) if toks else ""
    if len(_ACRONYM_OF_CACHE) < 200000:
        _ACRONYM_OF_CACHE[name] = out
    return out


def is_acronym(name: str) -> bool:
    """
    判断名称是否为全大写缩写 (NBC / IBM / USA)。

    注意: 不能用 normalize_text (它会小写化), 否则永远返回 False。
    这里只做 Unicode 规范化与去点, 保留大小写信息。
    """
    s = unicodedata.normalize("NFKC", str(name or "")).replace(".", "").strip()
    s = re.sub(r"\s+", "", s)
    return 2 <= len(s) <= 8 and s.isalpha() and s.isupper()


def jaro_winkler(s1: str, s2: str) -> float:
    """Jaro-Winkler 相似度 (纯 Python 实现, 避免额外依赖)。"""
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    match_dist = max(len(s1), len(s2)) // 2 - 1
    if match_dist < 0:
        match_dist = 0
    s1_matches = [False] * len(s1)
    s2_matches = [False] * len(s2)
    matches = 0
    for i, c1 in enumerate(s1):
        start = max(0, i - match_dist)
        end = min(i + match_dist + 1, len(s2))
        for j in range(start, end):
            if s2_matches[j] or c1 != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break
    if matches == 0:
        return 0.0
    t1 = sum(1 for i, m in enumerate(s1_matches) if m and i % 2 == 0)
    t2 = sum(1 for j, m in enumerate(s2_matches) if m and j % 2 == 0)
    transpositions = t1 + t2
    jaro = (matches / len(s1) + matches / len(s2)
            + (matches - transpositions / 2) / matches) / 3.0
    prefix = 0
    for a, b in zip(s1[:4], s2[:4]):
        if a != b:
            break
        prefix += 1
    return jaro + prefix * 0.1 * (1 - jaro)


def stable_id(prefix: str, s: str) -> str:
    h = hashlib.sha1(normalize_text(s).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{h}"


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def read_json(path: str, default: Any = None) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:  # pragma: no cover
        LOG.warning("读取 %s 失败: %s", path, e)
        return default


def read_jsonl(path: str) -> List[Any]:
    """读取 JSONL, 自动跳过损坏行 (并发写入/进程中断可能留下半行)。"""
    out = []
    if not os.path.exists(path):
        return out
    n_bad = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                n_bad += 1
    if n_bad:
        LOG.warning("%s 中有 %d 行损坏已跳过", os.path.basename(path), n_bad)
    return out


_JSONL_WRITE_LOCK = threading.Lock()


def append_jsonl(path: str, obj: Any) -> None:
    """线程安全地追加一行 JSON (并发抽提时防止行交错)。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    with _JSONL_WRITE_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())


# ============================================================================
# Stage 0 : DeepSeek LLM 客户端
# ============================================================================

class DeepSeekClient:
    """DeepSeek 官方 API 客户端, 具备 JSON 健壮解析与指数退避重试。"""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEEPSEEK_BASE_URL,
        model: str = "deepseek-chat",
        max_retries: int = 8,
        timeout: float = 120.0,
    ):
        from openai import OpenAI  # 延迟导入

        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self._client = OpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0
        )
        self.n_calls = 0
        self.n_prompt_tokens = 0
        self.n_completion_tokens = 0

    # ------------------------------------------------------------------ #
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
                self.n_calls += 1
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    self.n_prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
                    self.n_completion_tokens += getattr(usage, "completion_tokens", 0) or 0
                return resp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001
                last_err = e
                msg = str(e).lower()
                # 上下文超长 -> 截断重试
                if "context length" in msg or "maximum context" in msg:
                    msgs = self._truncate_messages(messages)
                    messages = msgs
                    continue
                backoff = min(30.0, (2 ** attempt) * 1.5) * (0.6 + 0.8 * random.random())
                LOG.warning("LLM 调用失败 (%d/%d): %s -> %.1fs 后重试",
                            attempt + 1, self.max_retries, str(e)[:160], backoff)
                time.sleep(backoff)
        raise RuntimeError(f"LLM 调用最终失败: {last_err}")

    @staticmethod
    def _truncate_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """上下文超长时, 截断最长的 user 消息。"""
        out = list(messages)
        for i, m in enumerate(out):
            if m.get("role") == "user" and len(m.get("content", "")) > 3000:
                out[i] = {**m, "content": m["content"][:1500] + "\n...[TRUNCATED]"}
        return out

    # ------------------------------------------------------------------ #
    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 2048,
        max_repair: int = 2,
    ) -> Optional[Dict[str, Any]]:
        """返回解析后的 JSON 对象; 失败则尝试修复重试。"""
        for r in range(max_repair + 1):
            try:
                raw = self.chat(messages, temperature=temperature, max_tokens=max_tokens)
                parsed = parse_json_robust(raw)
                if parsed is not None:
                    return parsed
                LOG.warning("JSON 解析失败 (repair %d/%d)", r, max_repair)
            except Exception as e:  # noqa: BLE001
                LOG.warning("chat_json 异常: %s", str(e)[:200])
                if r == max_repair:
                    break
                time.sleep(2.0 * (r + 1))
        return None

    def stats(self) -> Dict[str, Any]:
        return {
            "calls": self.n_calls,
            "prompt_tokens": self.n_prompt_tokens,
            "completion_tokens": self.n_completion_tokens,
            "model": self.model,
        }


def parse_json_robust(raw: str) -> Optional[Dict[str, Any]]:
    """健壮 JSON 解析: 去 think 块 / markdown 围栏 / 括号匹配 / 常见错误修复。"""
    if not raw:
        return None
    s = raw
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL)
    s = re.sub(r"```json\s*", "", s)
    s = re.sub(r"```\s*", "", s)
    s = s.strip()

    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

    # 括号匹配扫描所有候选 JSON 对象
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                cand = s[start:i + 1]
                for fixed in (cand,
                              re.sub(r",\s*}", "}", cand),
                              re.sub(r",\s*]", "]", cand),
                              re.sub(r",\s*}", "}", re.sub(r",\s*]", "]", cand))):
                    try:
                        obj = json.loads(fixed)
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        continue
                start = -1
    return None


# ============================================================================
# Stage 0 : 语料装载 (HotpotQA)
# ============================================================================

@dataclass
class Passage:
    """知识库中的最小检索单元。"""
    pid: str
    title: str
    text: str
    sentences: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"pid": self.pid, "title": self.title,
                "text": self.text, "sentences": self.sentences}


def load_hotpotqa_corpus(
    data_root: str,
    num_questions: int = 200,
    split: str = "validation",
    max_chars: int = 6000,
) -> Tuple[List[Passage], List[Dict[str, Any]]]:
    """
    装载 HotpotQA distractor 语料。

    HotpotQA (Yang et al., EMNLP 2018) 是 RAG 领域最通用的多跳 QA 基准之一,
    也是 GraphRAG / HippoRAG / KG-Infused RAG / LightRAG 的标准评测集之一。

    Returns:
        passages : 去重后的段落库 (按 title 去重, 同 title 的句子取并集)
        questions: 问题列表 (含 supporting_facts 标注, 用于评测)
    """
    from datasets import load_from_disk

    ds_dir = os.path.join(data_root, "datasets", "hotpotqa_distractor_validation")
    if not os.path.isdir(ds_dir):
        raise FileNotFoundError(
            f"未找到 HotpotQA 数据集: {ds_dir}\n"
            f"请先运行下载脚本 (见 README 或 --stage download)。"
        )
    ds = load_from_disk(ds_dir)
    # load_from_disk 可能返回 Dataset (单 split) 或 DatasetDict (多 split)
    if hasattr(ds, "column_names") and "question" in getattr(ds, "column_names", []):
        data = ds
    else:
        data = ds[split] if split in ds else ds[list(ds.keys())[0]]
    n = min(num_questions, len(data))
    data = data.select(range(n))

    # title -> {sentences: ordered list}
    title_store: Dict[str, Dict[str, Any]] = {}
    questions: List[Dict[str, Any]] = []

    for ex in data:
        ctx = ex["context"]
        titles = ctx["title"]
        sents = ctx["sentences"]
        for title, s_list in zip(titles, sents):
            title = (title or "").strip()
            if not title:
                continue
            st = title_store.setdefault(title, {"sents": [], "seen": set()})
            for s in s_list:
                s = (s or "").strip()
                if s and s not in st["seen"]:
                    st["seen"].add(s)
                    st["sents"].append(s)
        questions.append({
            "id": ex["id"],
            "question": ex["question"],
            "answer": ex["answer"],
            "type": ex["type"],
            "level": ex["level"],
            "gold_titles": sorted(set(ex["supporting_facts"]["title"])),
        })

    passages: List[Passage] = []
    for title, st in title_store.items():
        text = " ".join(st["sents"])
        if len(text) > max_chars:
            text = text[:max_chars]
        passages.append(Passage(
            pid=stable_id("p", title),
            title=title,
            text=text,
            sentences=st["sents"],
        ))

    LOG.info("HotpotQA 语料: %d 个问题, %d 个唯一段落", len(questions), len(passages))
    return passages, questions


# ============================================================================
# Stage 1 : 三元组抽提 (刘知远团队经典 RAG 方法 / 三步 CoT)
# ============================================================================

EXTRACTION_SYSTEM_PROMPT = """You are a precise knowledge-graph extraction expert.
You follow a strict three-step Chain-of-Thought procedure and then output ONLY JSON.

Procedure:
Step 1 (Entity Discovery): identify all salient entities mentioned in the passage.
        For each entity give: name (surface form as written), type, a short factual
        description (<= 25 words), and aliases (other surface forms / full names /
        abbreviations referring to the SAME entity, e.g. full name + acronym).
Step 2 (Relation Discovery): state the explicit relation between entities that are
        both present in the passage. Relation types should be short predicate
        phrases such as "founded by", "located in", "member of", "spouse of",
        "released by", "employer", "capital of", "successor of", "award received".
Step 3 (Canonicalization): pick ONE canonical name per entity (prefer the most
        complete/canonical form; for persons use the full name) and make sure the
        aliases list contains every other surface form.

Output format (JSON only, no markdown, no explanation):
{
  "entities": [
    {"name": "...", "type": "...", "description": "...", "aliases": ["..."]}
  ],
  "relations": [
    {"source": "...", "target": "...", "type": "...", "evidence": "..."}
  ]
}
The "source" and "target" of every relation MUST exactly match a "name" field of
some entity in the same JSON. Keep the entity list focused on salient entities
(roughly 5-15 per passage); do not include generic nouns.
"""

CANONICALIZATION_SYSTEM_PROMPT = """You are an entity-resolution expert.
Given two entity mentions, decide whether they refer to THE SAME real-world entity
(including full name vs. abbreviation vs. acronym vs. spelling variants).

Answer with JSON only:
{"same": true|false, "confidence": 0.0-1.0, "canonical_name": "...",
 "reason": "<= 20 words"}

Be conservative: if the two mentions could plausibly be different entities
(e.g. two people with the same name, a city vs. a building), answer same=false.
"""


class TripleExtractor:
    """基于 DeepSeek 的实体/关系抽提器 (含断点续传)。"""

    def __init__(self, client: DeepSeekClient, output_dir: str,
                 max_chars: int = 6000, temperature: float = 0.1):
        self.client = client
        self.output_dir = output_dir
        self.max_chars = max_chars
        self.temperature = temperature
        self.raw_path = os.path.join(output_dir, "raw_extractions.jsonl")
        self.done = {r["pid"] for r in read_jsonl(self.raw_path)}

    # ------------------------------------------------------------------ #
    def extract_one(self, passage: Passage) -> Optional[Dict[str, Any]]:
        if passage.pid in self.done:
            return None
        user = (
            f"Passage title: {passage.title}\n"
            f"Passage text:\n{passage.text[: self.max_chars]}\n\n"
            "Perform Step 1 (Entity Discovery), Step 2 (Relation Discovery), "
            "Step 3 (Canonicalization) and return the JSON."
        )
        obj = self.client.chat_json(
            [
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
            max_tokens=2048,
        )
        if obj is None:
            LOG.warning("段落 %s (%s) 抽提失败", passage.pid, passage.title[:40])
            return None

        entities = []
        for e in (obj.get("entities") or [])[:40]:
            if not isinstance(e, dict):
                continue
            name = str(e.get("name") or "").strip()
            if not name or len(name) > 180:
                continue
            aliases = e.get("aliases") or []
            if isinstance(aliases, str):
                aliases = [aliases]
            aliases = [str(a).strip() for a in aliases
                       if isinstance(a, (str, int, float)) and str(a).strip()]
            entities.append({
                "name": name,
                "type": str(e.get("type") or "concept").strip().lower(),
                "description": str(e.get("description") or "").strip()[:400],
                "aliases": list(dict.fromkeys([name] + aliases)),
                "source_pid": passage.pid,
                "source_title": passage.title,
            })

        rels = []
        for r in (obj.get("relations") or [])[:80]:
            if not isinstance(r, dict):
                continue
            s = str(r.get("source") or "").strip()
            t = str(r.get("target") or "").strip()
            rt = str(r.get("type") or "related to").strip()[:80]
            if not s or not t or s == t:
                continue
            rels.append({
                "source": s, "target": t, "type": rt,
                "evidence": str(r.get("evidence") or "").strip()[:200],
                "source_pid": passage.pid,
                "source_title": passage.title,
            })

        rec = {
            "pid": passage.pid,
            "title": passage.title,
            "entities": entities,
            "relations": rels,
        }
        append_jsonl(self.raw_path, rec)
        self.done.add(passage.pid)
        return rec

    # ------------------------------------------------------------------ #
    def run(self, passages: List[Passage], workers: int = 8) -> List[Dict[str, Any]]:
        todo = [p for p in passages if p.pid not in self.done]
        LOG.info("待抽提段落: %d / %d (已完成 %d)", len(todo), len(passages), len(self.done))
        records: List[Dict[str, Any]] = []
        if not todo:
            return read_jsonl(self.raw_path)

        t0 = time.time()
        ok = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(self.extract_one, p): p for p in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                p = futs[fut]
                try:
                    rec = fut.result()
                    if rec is not None:
                        ok += 1
                except Exception as e:  # noqa: BLE001
                    LOG.warning("抽提异常 %s: %s", p.pid, str(e)[:160])
                if i % 50 == 0 or i == len(futs):
                    el = time.time() - t0
                    rate = i / max(el, 1e-6)
                    LOG.info("抽提进度 %d/%d (成功 %d) %.2f doc/s, 预计剩余 %.0fs",
                             i, len(futs), ok, rate, (len(futs) - i) / max(rate, 1e-9))
        LOG.info("抽提完成: 成功 %d / %d, 耗时 %.0fs", ok, len(futs), time.time() - t0)
        return read_jsonl(self.raw_path)


# ============================================================================
# Stage 2 : 实体归并 (Entity Resolution)
# ============================================================================

@dataclass
class EntityMention:
    """一个被提及的实体表面形式 (mention)。"""
    mid: str
    name: str
    etype: str
    description: str
    aliases: List[str]
    source_pids: Set[str] = field(default_factory=set)
    source_titles: Set[str] = field(default_factory=set)

    @property
    def all_names(self) -> List[str]:
        return list(dict.fromkeys([self.name] + list(self.aliases)))


class UnionFind:
    def __init__(self):
        self.parent: Dict[str, str] = {}
        self.rank: Dict[str, int] = {}

    def add(self, x: str) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x: str) -> str:
        self.add(x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:      # 路径压缩
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> str:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return ra


@dataclass
class MergeEvidence:
    """一次归并的证据 (可审计)。"""
    a: str
    b: str
    score: float
    components: Dict[str, float]
    strategy: str


class EntityResolver:
    """
    实体归并: Blocking -> ScoreVector -> Union-Find -> (可选) LLM 仲裁。

    归并结果保留 canonical name + 全部 aliases + merged_from + 证据,
    保证可回溯 (可逆合并)。
    """

    def __init__(
        self,
        embedder: "Embedder",
        client: Optional[DeepSeekClient] = None,
        merge_threshold: float = 0.75,
        ambiguous_low: float = 0.65,
        use_llm_arbitration: bool = True,
        max_llm_arbitrations: int = 400,
        name_gate: float = 0.78,
    ):
        self.embedder = embedder
        self.client = client
        self.merge_threshold = merge_threshold
        self.ambiguous_low = ambiguous_low
        self.use_llm_arbitration = use_llm_arbitration
        self.max_llm_arbitrations = max_llm_arbitrations
        # 名称证据门控: name_similarity 低于该值时禁止 rule-based 合并
        self.name_gate = name_gate
        self.evidence: List[MergeEvidence] = []
        self.veto_counts: Dict[str, int] = defaultdict(int)
        # mention 级预计算缓存 (由 _prepare_mention_cache 填充)
        self._mc_distinctive: Dict[Tuple[str, str, Tuple[str, ...]], Set[str]] = {}
        self._mc_numbers: Dict[Tuple[str, str, Tuple[str, ...]], Set[str]] = {}
        self._mc_acr_sig: Dict[Tuple[str, str, Tuple[str, ...]], Set[str]] = {}
        self._mc_multi: Dict[Tuple[str, str, Tuple[str, ...]], bool] = {}
        self._mc_names: Dict[Tuple[str, str, Tuple[str, ...]], List[str]] = {}

    # ------------------------------------------------------------------ #
    @staticmethod
    def normalize_type(t: str) -> str:
        t = (t or "").strip().lower()
        if not t:
            return "CONCEPT"
        if t in TYPE_GROUPS:
            return TYPE_GROUPS[t]
        for k, v in TYPE_GROUPS.items():
            if k in t:
                return v
        return "CONCEPT"

    @staticmethod
    def type_agreement(t1: str, t2: str) -> float:
        if t1 == t2:
            return 1.0
        if t1 == "CONCEPT" or t2 == "CONCEPT":
            return 0.5
        if (t1, t2) in COMPATIBLE_PAIRS or (t2, t1) in COMPATIBLE_PAIRS:
            return 1.0
        if (t1, t2) in SOFT_COMPATIBLE or (t2, t1) in SOFT_COMPATIBLE:
            return 0.75
        return 0.0

    # ------------------------------------------------------------------ #
    def _blocking(self, mentions: List[EntityMention]) -> Set[Tuple[str, str]]:
        """
        多路 blocking 的并集。每种 blocking 有不同的盲区, 并集的召回率才是关键。

        性能说明: 朴素做法在 ~2000 mention 上会产生 40 万+ 候选对, 逐对打分
        代价过高。这里对 *通用* blocking 键 (单个字符 3-gram、单个人名词元)
        设更严格的上限, 只让 *有区分度* 的键 (归一化全名、首字母缩写、
        稀有词元) 参与配对, 从而在保持召回的同时把候选对压到万级。
        """
        pairs: Set[Tuple[str, str]] = set()
        buckets: Dict[str, List[str]] = defaultdict(list)

        # 键 -> 允许的最大桶大小 (越小 = 越严格)
        # "N:" 归一化全名  : 几乎唯一, 桶大
        # "T:" 词元       : 可能很泛 (如 "university"), 桶中
        # "G:" 字符3-gram : 极易泛化 (如 "ang"), 桶小
        # "A:" 缩写       : 唯一性高, 桶大
        BUCKET_CAP = {"N:": 400, "T:": 60, "G:": 25, "A:": 400}
        MAX_PAIR_BUCKET = 120      # 桶内成员超过此数则整桶跳过

        def add_bucket(key: str, mid: str) -> None:
            cap = BUCKET_CAP.get(key[:2], 60)
            if len(buckets[key]) < cap:
                buckets[key].append(mid)

        for m in mentions:
            names = m.all_names
            for nm in names:
                # (1) 归一化精确键
                add_bucket("N:" + legal_normalize(nm), m.mid)
                # (2) 词元 blocking
                for t in sig_tokens(nm):
                    add_bucket("T:" + t, m.mid)
                # (3) 字符 n-gram blocking (仅较长名称, 短名无区分度)
                if len(nm) >= 6:
                    for g in char_ngrams(nm):
                        add_bucket("G:" + g, m.mid)
                # (4) 缩写 blocking
                if is_acronym(nm):
                    add_bucket("A:" + nm.replace(".", "").upper(), m.mid)
                else:
                    ac = acronym_of(nm)
                    if 2 <= len(ac) <= 8:
                        add_bucket("A:" + ac.upper(), m.mid)

        for key, members in buckets.items():
            if len(members) < 2:
                continue
            uniq = list(dict.fromkeys(members))
            if len(uniq) > MAX_PAIR_BUCKET:
                continue
            for i in range(len(uniq)):
                for j in range(i + 1, len(uniq)):
                    pairs.add((uniq[i], uniq[j]))

        # (5) embedding 邻域 blocking
        try:
            emb = self.embedder.encode([m.name for m in mentions], normalize=True)
            sims = emb @ emb.T
            np.fill_diagonal(sims, -1.0)
            k = min(8, len(mentions) - 1)
            if k > 0:
                idx = np.argpartition(-sims, kth=k, axis=1)[:, :k]
                for i in range(len(mentions)):
                    for j in idx[i]:
                        if j <= i:
                            continue
                        if sims[i, j] >= 0.80:
                            pairs.add((mentions[i].mid, mentions[int(j)].mid))
        except Exception as e:  # noqa: BLE001
            LOG.warning("embedding blocking 失败: %s", str(e)[:160])

        LOG.info("blocking 候选对: %d (mentions=%d)", len(pairs), len(mentions))
        return pairs

    # ------------------------------------------------------------------ #
    def _neighbor_sets(self, mentions: List[EntityMention],
                       relations: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
        """1-hop 邻居名集合 (用于 neighbor_overlap 打分)。"""
        by_name: Dict[str, str] = {}
        for m in mentions:
            for nm in m.all_names:
                by_name.setdefault(nm.lower(), m.mid)
        nbrs: Dict[str, Set[str]] = defaultdict(set)
        for r in relations:
            s = by_name.get(str(r["source"]).lower())
            t = by_name.get(str(r["target"]).lower())
            if not s or not t or s == t:
                continue
            nbrs[s].add(t)
            nbrs[t].add(s)
        return {k: set(v) for k, v in nbrs.items()}

    @staticmethod
    def _soft_jaccard(a: Set[str], b: Set[str]) -> float:
        if not a and not b:
            return 0.0        # 证据缺失 != 匹配证据 (absence of evidence is not evidence)
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union else 0.0

    # ------------------------------------------------------------------ #
    # 过度合并防护 (over-merge guards)
    #
    # 工业界经验: "Bad merges are just as harmful" —— 错误合并会污染整个图,
    # 且不可逆。因此 name_similarity 不能只是加权项之一, 必须作为 *门控*:
    # 名称证据不足时, 语义相似 (semantic) 不能单独把两个实体推过阈值。
    # ------------------------------------------------------------------ #

    # 过于泛化、不足以区分实体的 "头名词" (head noun)
    GENERIC_HEADS = {
        "university", "college", "school", "institute", "company", "corporation",
        "inc", "ltd", "group", "team", "club", "band", "church", "party",
        "government", "department", "agency", "council", "committee", "society",
        "association", "foundation", "hospital", "museum", "library", "theatre",
        "theater", "studio", "records", "network", "systems", "technologies",
        "film", "movie", "show", "series", "book", "album", "song", "game",
        "city", "town", "village", "county", "state", "province", "district",
        "river", "mountain", "island", "lake", "park", "street", "avenue",
        "award", "prize", "championship", "tournament", "festival",
        "newspaper", "magazine", "journal", "channel", "program", "project",
        "war", "battle", "election", "convention", "summit", "treaty",
    }

    def _distinctive_tokens(self, names: List[str]) -> Set[str]:
        """提取有区分度的词元 (剔除通用头名词与停用词)。"""
        toks: Set[str] = set()
        for nm in names:
            for t in sig_tokens(nm):
                if t in self.GENERIC_HEADS or t in STOP_TOKENS:
                    continue
                toks.add(t)
        return toks

    # ------------------------------------------------------------------ #
    # 归并热路径的预计算缓存
    #   归并阶段有数十万候选对, 而 _distinctive_tokens / _numbers_in /
    #   _acronym_signatures 只依赖单个 mention, 与配对无关, 因此预计算。
    #
    #   缓存键包含名称与别名, 而不仅是 mid: 同一个 mid 在不同调用中可能
    #   对应不同 mention (例如测试中复用 mid), 只用 mid 会读到脏数据。
    # ------------------------------------------------------------------ #
    @staticmethod
    def _mc_key(m: EntityMention) -> Tuple[str, str, Tuple[str, ...]]:
        return (m.mid, m.name, tuple(m.aliases))

    def _prepare_mention_cache(self, mentions: List[EntityMention]) -> None:
        self._mc_distinctive = {}
        self._mc_numbers = {}
        self._mc_acr_sig = {}
        self._mc_names = {}
        for m in mentions:
            k = self._mc_key(m)
            self._mc_distinctive[k] = self._distinctive_tokens(m.all_names)
            nums: Set[str] = set()
            for nm in m.all_names:
                nums |= self._numbers_in(nm)
            self._mc_numbers[k] = nums
            self._mc_acr_sig[k] = self._acronym_signatures(m.all_names)
            self._mc_multi[k] = any(len(sig_tokens(nm)) >= 2 for nm in m.all_names)
            self._mc_names[k] = [legal_normalize(x) for x in m.all_names
                                 if legal_normalize(x)]

    def _mc_distinctive_of(self, m: EntityMention) -> Set[str]:
        k = self._mc_key(m)
        c = self._mc_distinctive.get(k)
        if c is None:
            c = self._distinctive_tokens(m.all_names)
            self._mc_distinctive[k] = c
        return c

    def _mc_numbers_of(self, m: EntityMention) -> Set[str]:
        k = self._mc_key(m)
        c = self._mc_numbers.get(k)
        if c is None:
            c = set()
            for nm in m.all_names:
                c |= self._numbers_in(nm)
            self._mc_numbers[k] = c
        return c

    def _mc_acr_sig_of(self, m: EntityMention) -> Set[str]:
        k = self._mc_key(m)
        c = self._mc_acr_sig.get(k)
        if c is None:
            c = self._acronym_signatures(m.all_names)
            self._mc_acr_sig[k] = c
        return c

    def _mc_names_of(self, m: EntityMention) -> List[str]:
        k = self._mc_key(m)
        c = self._mc_names.get(k)
        if c is None:
            c = [legal_normalize(x) for x in m.all_names if legal_normalize(x)]
            self._mc_names[k] = c
        return c

    def _mc_multi_of(self, m: EntityMention) -> bool:
        """名称集合中是否存在多词元名称 (用于收紧缩写前缀匹配)。"""
        k = self._mc_key(m)
        c = self._mc_multi.get(k)
        if c is None:
            c = any(len(sig_tokens(nm)) >= 2 for nm in m.all_names)
            self._mc_multi[k] = c
        return c
    @staticmethod
    def _numbers_in(name: str) -> Set[str]:
        return set(re.findall(r"\d+", name or ""))

    @staticmethod
    def _acronym_signatures(names: List[str]) -> Set[str]:
        """
        为一个名称集合生成 "缩写签名"。

        三类签名:
          (a) 名称本身就是缩写 (NBC -> "NBC")
          (b) 名称的首字母缩写 (International Business Machines -> "IBM")
          (c) 名称词元的直接拼接 (United States -> "UNITEDSTATES")

        (c) 很关键: 它让 "USA" 能通过 "US" 这一前缀与 "United States" 匹配。
        """
        sigs: Set[str] = set()
        for nm in names:
            if not nm:
                continue
            if is_acronym(nm):
                sigs.add(nm.replace(".", "").upper())
            toks = sig_tokens(nm)
            if toks:
                # 首字母缩写: 仅当 token 数 >= 2 时才有区分意义。
                # 单 token (如 "UCLA") 的首字母 "U" 太弱, 会让 UCLA 与 USC
                # 被误判为相关 (共享 "U"), 因此单独处理。
                if len(toks) >= 2:
                    sigs.add("".join(t[0] for t in toks).upper())
                sigs.add("".join(toks).upper())
        return sigs

    @classmethod
    def _acronym_matches_fullname(cls, acr: str, full: str) -> bool:
        """
        判断缩写 acr 是否对应全称 full。

        规则:
          1. 全称首字母缩写完全相等 (IBM <-> International Business Machines)
          2. 全称首字母缩写以缩写开头 (US <-> United States)
          3. 缩写以全称首字母缩写开头 (USA <-> United States)
          4. 缩写是全称首字母的 *子序列* (USA <-> United States of America),
             要求缩写至少 3 个字母, 且是按序子序列。
        """
        acr_clean = acr.replace(".", "").upper()
        toks = sig_tokens(full)
        if not toks or len(acr_clean) < 2:
            return False
        ini = "".join(t[0] for t in toks).upper()
        if ini == acr_clean:
            return True
        if ini.startswith(acr_clean):
            return True
        if acr_clean.startswith(ini):
            return True
        if len(acr_clean) >= 3:
            it = iter(ini)
            if all(ch in it for ch in acr_clean):
                return True
        return False

    @classmethod
    def _acronym_relates(cls, a_names: List[str], b_names: List[str]) -> bool:
        """
        判断 a、b 之间是否存在 "缩写 <-> 全称" 关系。

        直接在 *名称* 层面判定 (而不是仅比较签名集合), 这样才能正确区分:
            "USA"  <-> "United States of America"   (真别名, 应合并)
            "UCLA" <-> "USC"                          (不同实体, 不应合并)
        判定方式: 一方的某个名称是缩写, 且该缩写匹配另一方的某个全称。
        """
        acr_a = [nm for nm in a_names if is_acronym(nm)]
        acr_b = [nm for nm in b_names if is_acronym(nm)]
        full_a = [nm for nm in a_names if not is_acronym(nm)]
        full_b = [nm for nm in b_names if not is_acronym(nm)]

        # a 的缩写匹配 b 的全称
        for x in acr_a:
            for y in full_b:
                if cls._acronym_matches_fullname(x, y):
                    return True
        # b 的缩写匹配 a 的全称
        for y in acr_b:
            for x in full_a:
                if cls._acronym_matches_fullname(y, x):
                    return True
        # 两侧都是缩写 => 只有完全相同 (或前缀关系且较短者是较长者的缩写
        # 形式) 才算别名; "UCLA" 与 "USC" 无公共前缀, 因此不会误配。
        for x in acr_a:
            for y in acr_b:
                if x.replace(".", "").upper() == y.replace(".", "").upper():
                    return True
        return False

    @classmethod
    def _acronym_relates_cached(cls, a_names: List[str], b_names: List[str]) -> bool:
        """缓存包装: 保留旧签名接口, 转调 _acronym_relates。"""
        return cls._acronym_relates(a_names, b_names)

    def _merge_veto(self, a: EntityMention, b: EntityMention,
                    comps: Dict[str, float]) -> Optional[str]:
        """
        返回否决原因; None 表示不否决。

        否决规则 (任一命中即否决 rule-based 合并, 只能由 LLM 仲裁决定):
          V1 名词冲突      : 两名称含不同数字 (Apple Watch Series 2 vs 3)
          V2 无共同区分词  : 区分度词元交集为空, 且非缩写/包含关系
          V3 人名冲突      : PERSON 类实体姓氏不同
          V4 名称证据不足  : name_similarity 低于门控阈值
        """
        names_a = self._mc_names_of(a)
        names_b = self._mc_names_of(b)

        # V1 数字冲突
        num_a = self._mc_numbers_of(a)
        num_b = self._mc_numbers_of(b)
        if num_a and num_b and num_a != num_b:
            return "V1_number_conflict"

        # V2 无共同区分词 (且非包含/缩写关系)
        dt_a = self._mc_distinctive_of(a)
        dt_b = self._mc_distinctive_of(b)
        containment = any(x in y or y in x for x in names_a for y in names_b)
        acronym_rel = self._acronym_relates(a.all_names, b.all_names)
        if dt_a and dt_b and not (dt_a & dt_b) and not containment and not acronym_rel:
            return "V2_no_shared_distinctive_token"

        # V3 人名冲突 (姓氏/尾词不同) —— 比 V4 更精确, 故先判定
        #   人名的区分性在 *尾词* (姓): "Robert Downey" 与 "Robert Redford"
        #   共享名 "Robert" 但姓不同 => 两个不同的人。
        #   因此判定条件是 "尾词冲突", 而不是 "任意词元不同"。
        if a.etype == "PERSON" and b.etype == "PERSON":
            toks_a = sig_tokens(a.name)
            toks_b = sig_tokens(b.name)
            if toks_a and toks_b:
                # 姓氏 (最后一个词元) 不一致, 且不存在包含关系
                if toks_a[-1] != toks_b[-1] and not containment:
                    return "V3_person_name_conflict"

        # V4 名称证据门控: 名称相似度过低时, 语义相似不足以单独支撑合并。
        # 例外: 存在 "缩写 <-> 全称" 或包含关系时, 名称证据本身已经足够强
        #       (International Business Machines <-> IBM 必须能合并)。
        if (comps.get("name_similarity", 0.0) < self.name_gate
                and not acronym_rel and not containment):
            return "V4_insufficient_name_evidence"

        # V2b "同类不同体" 冲突: 两侧的区分词不完全相同, 且各自都带有
        #     *排他性*修饰词 (非共享的区分词), 例如
        #       University of California at Los Angeles  vs  University of Southern California
        #     共享 "university"/"california", 但 "los angeles" 与 "southern"
        #     是各自独有的限定语 => 两个不同的大学, 不能合并。
        #     判定方式: 存在两侧各自独有、且长度 >= 3 的区分词 (排除
        #     通用头名词与所有共享词)。
        #     放在 V3/V4 之后, 因为人名冲突与名称证据不足是更精确的诊断。
        if dt_a and dt_b and dt_a != dt_b and not containment and not acronym_rel:
            only_a = {t for t in dt_a - dt_b if len(t) >= 3}
            only_b = {t for t in dt_b - dt_a if len(t) >= 3}
            if only_a and only_b:
                # 两侧的排他词互不相同 => 不同的具体实体
                return "V2b_exclusive_modifier_conflict"

        # V5 "头词-修饰词" 上下位关系: 一方的区分词元集合是另一方的真子集,
        #    且多出来的部分 *全部* 是通用头名词 (不含区分信息), 例如
        #       New York         vs  New York City      (差集 = {city})
        #       New York         vs  State of New York  (差集 = {state})
        #    这类是 *上下位* 而非 *等同*: 上位词吞掉下位词会在 KG 中
        #    制造事实错误 (把纽约州当成纽约市)。
        #    要求 acronym_rel=False, 以保证 "International Business
        #    Machines" 与 "IBM" 这类真别名不受影响。
        if dt_a and dt_b and not acronym_rel:
            only_a = dt_a - dt_b
            only_b = dt_b - dt_a
            if only_a and not only_b and not (only_a - self.GENERIC_HEADS):
                # a 更具体, b 只剩通用头名词 => b 是 a 的上位类别
                return "V5_head_word_hyponym_conflict"
            if only_b and not only_a and not (only_b - self.GENERIC_HEADS):
                # b 更具体, a 只剩通用头名词 => a 是 b 的上位类别
                return "V5_head_word_hyponym_conflict"

        return None

    # ------------------------------------------------------------------ #
    def _score_pair(self, a: EntityMention, b: EntityMention,
                    nbrs: Dict[str, Set[str]],
                    emb_a: Optional[np.ndarray],
                    emb_b: Optional[np.ndarray],
                    desc_emb: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, float]:
        names_a = self._mc_names_of(a)
        names_b = self._mc_names_of(b)
        name_sim = 0.0
        for x in names_a:
            for y in names_b:
                if x == y:
                    name_sim = 1.0
                    break
                name_sim = max(name_sim, jaro_winkler(x, y))
            if name_sim >= 0.97:
                break
        if name_sim < 0.9:
            raw_a, raw_b = a.name.lower(), b.name.lower()
            name_sim = max(name_sim, jaro_winkler(raw_a, raw_b))

        # 语义相似度 (名称 + 描述)
        sem_sim = 0.0
        if emb_a is not None and emb_b is not None:
            sem_sim = float(np.dot(emb_a, emb_b))

        type_ag = self.type_agreement(a.etype, b.etype)

        nbr_ovl = self._soft_jaccard(nbrs.get(a.mid, set()), nbrs.get(b.mid, set()))

        # 描述相似度: 使用预先批量计算的 embedding (避免逐对推理)
        desc_sim = 0.0
        if desc_emb and a.description and b.description:
            da = desc_emb.get(a.mid)
            db = desc_emb.get(b.mid)
            if da is not None and db is not None:
                desc_sim = float(np.dot(da, db))

        comps = {
            "name_similarity": round(name_sim, 4),
            "semantic_similarity": round(sem_sim, 4),
            "type_agreement": round(type_ag, 4),
            "neighbor_overlap": round(nbr_ovl, 4),
            "description_similarity": round(desc_sim, 4),
        }
        score = (0.30 * name_sim + 0.25 * sem_sim + 0.20 * type_ag
                 + 0.20 * nbr_ovl + 0.05 * desc_sim)

        # 归一化名称完全一致 => 必合并 (surface-form identity)。
        # 这是实体消解中最强的证据: 两个 mention 的任一名称/别名经归一化后
        # 完全相同, 它们就是同一实体的不同表述 (全称/缩写/标点差异)。
        # 纯加权打分在这种情况下仍可能低于阈值 (例如 embedding 恰好把它们
        # 推远), 导致真别名漏合并 —— 因此这里显式置为满分。
        if name_sim >= 0.999:
            score = max(score, 1.0)
            comps["exact_name_match"] = 1.0
        else:
            comps["exact_name_match"] = 0.0

        comps["score"] = round(score, 4)
        return comps

    # ------------------------------------------------------------------ #
    def _llm_arbitrate(self, a: EntityMention, b: EntityMention) -> Optional[bool]:
        """对不确定区间的候选对调用 LLM 做二元仲裁。"""
        if self.client is None:
            return None
        user = (
            f"Mention A:\n- name: {a.name}\n- type: {a.etype}\n"
            f"- description: {a.description[:200]}\n"
            f"- aliases: {a.aliases[:6]}\n\n"
            f"Mention B:\n- name: {b.name}\n- type: {b.etype}\n"
            f"- description: {b.description[:200]}\n"
            f"- aliases: {b.aliases[:6]}\n\n"
            "Do these two mentions refer to the same real-world entity?"
        )
        obj = self.client.chat_json(
            [
                {"role": "system", "content": CANONICALIZATION_SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            temperature=0.0, max_tokens=300,
        )
        if not obj:
            return None
        same = obj.get("same")
        if isinstance(same, str):
            same = same.strip().lower() in ("true", "yes", "1")
        if not isinstance(same, bool):
            return None
        return same

    # ------------------------------------------------------------------ #
    def resolve(self, mentions: List[EntityMention],
                relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        执行归并, 返回 canonical entity 列表。

        每个 canonical entity:
            eid, name, type, description, aliases[], merged_from[],
            source_pids[], source_titles[], degree
        """
        by_mid = {m.mid: m for m in mentions}
        nbrs = self._neighbor_sets(mentions, relations)
        candidates = self._blocking(mentions)

        # 预计算与配对无关的 mention 级属性 (热路径加速)
        self._prepare_mention_cache(mentions)

        # 预先计算 embedding 以加速打分 (一次性批量推理, 避免逐对模型调用)
        emb_map: Dict[str, np.ndarray] = {}
        desc_emb: Dict[str, np.ndarray] = {}
        try:
            mat = self.embedder.encode(
                [f"{m.name}. {m.description}" for m in mentions], normalize=True)
            for i, m in enumerate(mentions):
                emb_map[m.mid] = mat[i]
        except Exception as e:  # noqa: BLE001
            LOG.warning("mention embedding 失败: %s", str(e)[:160])
        try:
            desc_idx = [i for i, m in enumerate(mentions) if m.description]
            if desc_idx:
                dmat = self.embedder.encode(
                    [mentions[i].description for i in desc_idx], normalize=True)
                for k, i in enumerate(desc_idx):
                    desc_emb[mentions[i].mid] = dmat[k]
        except Exception as e:  # noqa: BLE001
            LOG.warning("description embedding 失败: %s", str(e)[:160])

        uf = UnionFind()
        for m in mentions:
            uf.add(m.mid)

        scored: List[Tuple[float, str, str, Dict[str, float]]] = []
        for a, b in candidates:
            ma, mb = by_mid[a], by_mid[b]
            comps = self._score_pair(ma, mb, nbrs, emb_map.get(a), emb_map.get(b),
                                     desc_emb=desc_emb)
            scored.append((comps["score"], a, b, comps))
        scored.sort(key=lambda x: -x[0])

        llm_used = 0
        merged_pairs = 0
        vetoed_pairs = 0
        for score, a, b, comps in scored:
            if uf.find(a) == uf.find(b):
                continue
            decision = None
            strategy = "rule_based"
            ma, mb = by_mid[a], by_mid[b]

            # --- 过度合并防护 (veto) -----------------------------------
            # 先做否决检查: 命中硬性冲突规则时, 即使综合分超过阈值也不合并。
            # 只有 LLM 仲裁可以推翻 veto (LLM 有完整上下文, 规则难以覆盖的
            # 长尾情形交由模型判断)。
            veto_reason = self._merge_veto(ma, mb, comps)
            if veto_reason:
                self.veto_counts[veto_reason] += 1
                vetoed_pairs += 1
                if not (self.use_llm_arbitration
                        and score >= self.ambiguous_low):
                    self.evidence.append(
                        MergeEvidence(a, b, score, comps,
                                      f"vetoed:{veto_reason}"))
                    continue
                # 允许 LLM 仲裁推翻 veto
                if llm_used < self.max_llm_arbitrations:
                    llm_used += 1
                    strategy = "llm_arbitrated"
                    decision = self._llm_arbitrate(ma, mb)
                else:
                    self.evidence.append(
                        MergeEvidence(a, b, score, comps,
                                      f"vetoed:{veto_reason}"))
                    continue
            elif score >= self.merge_threshold:
                decision = True
            elif self.use_llm_arbitration and self.ambiguous_low <= score < self.merge_threshold:
                if llm_used < self.max_llm_arbitrations:
                    llm_used += 1
                    strategy = "llm_arbitrated"
                    decision = self._llm_arbitrate(ma, mb)
            if decision:
                uf.union(a, b)
                merged_pairs += 1
                self.evidence.append(MergeEvidence(a, b, score, comps, strategy))
            elif decision is False:
                self.evidence.append(MergeEvidence(a, b, score, comps, "llm_rejected"))

        LOG.info("实体归并: %d mentions -> 候选对 %d, 合并 %d 对 (LLM 仲裁 %d 次, 否决 %d 对)",
                 len(mentions), len(scored), merged_pairs, llm_used, vetoed_pairs)
        if self.veto_counts:
            LOG.info("否决规则命中统计: %s", dict(self.veto_counts))

        # 聚合为 canonical entities
        groups: Dict[str, List[EntityMention]] = defaultdict(list)
        for m in mentions:
            groups[uf.find(m.mid)].append(m)

        # 关系重映射到 canonical
        name_to_eid: Dict[str, str] = {}
        for m in mentions:
            for nm in m.all_names:
                name_to_eid.setdefault(nm.lower(), m.mid)

        canonical: List[Dict[str, Any]] = []
        eid_by_root: Dict[str, str] = {}
        for root, group in groups.items():
            # canonical name: 优先 LLM 指定的规范名, 否则取最长显著形式
            all_names: List[str] = []
            for m in group:
                all_names.extend(m.all_names)
            name_count: Dict[str, int] = defaultdict(int)
            for nm in all_names:
                name_count[nm] += 1
            best = max(all_names, key=lambda x: (name_count[x], len(x)))
            eid = stable_id("e", best)
            eid_by_root[root] = eid

            types = [self.normalize_type(m.etype) for m in group]
            type_counter: Dict[str, int] = defaultdict(int)
            for t in types:
                type_counter[t] += 1
            etype = max(type_counter.items(), key=lambda kv: kv[1])[0]

            descs = [m.description for m in group if m.description]
            desc = max(descs, key=len) if descs else ""
            pids: Set[str] = set()
            titles: Set[str] = set()
            for m in group:
                pids |= m.source_pids
                titles |= m.source_titles
            aliases = sorted({a for a in all_names if a != best},
                             key=lambda s: (-len(s), s))
            canonical.append({
                "eid": eid,
                "name": best,
                "type": etype,
                "description": desc[:400],
                "aliases": aliases,
                "merged_from": sorted(m.mid for m in group),
                "source_pids": sorted(pids),
                "source_titles": sorted(titles),
                "mention_count": len(group),
            })

        # 关系重映射
        canon_rels: List[Dict[str, Any]] = []
        seen_rel: Set[Tuple[str, str, str]] = set()
        for r in relations:
            s_mid = name_to_eid.get(str(r["source"]).lower())
            t_mid = name_to_eid.get(str(r["target"]).lower())
            if not s_mid or not t_mid:
                continue
            s_eid = eid_by_root.get(uf.find(s_mid))
            t_eid = eid_by_root.get(uf.find(t_mid))
            if not s_eid or not t_eid or s_eid == t_eid:
                continue
            key = (s_eid, t_eid, r["type"].lower())
            if key in seen_rel:
                continue
            seen_rel.add(key)
            canon_rels.append({
                "source": s_eid, "target": t_eid,
                "source_name": r["source"], "target_name": r["target"],
                "type": r["type"],
                "evidence": r.get("evidence", ""),
                "source_pid": r.get("source_pid", ""),
                "source_title": r.get("source_title", ""),
            })

        for e in canonical:
            e["degree"] = sum(1 for r in canon_rels
                              if r["source"] == e["eid"] or r["target"] == e["eid"])

        LOG.info("归并完成: %d 个 canonical 实体, %d 条规范三元组",
                 len(canonical), len(canon_rels))
        return canonical, canon_rels


# ============================================================================
# Stage 3 : 客观关联指标
# ============================================================================

class Embedder:
    """sentence-transformers 封装 (带本地缓存)。"""

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL,
                 cache_dir: Optional[str] = None, device: Optional[str] = None,
                 mem_cache: bool = True):
        from sentence_transformers import SentenceTransformer

        os.environ.setdefault("HF_HOME", cache_dir or os.path.join(DEFAULT_DATA_ROOT, "hf_cache"))
        LOG.info("加载 embedding 模型: %s", model_name)
        self.model = SentenceTransformer(model_name, device=device, cache_folder=cache_dir)
        self.model_name = model_name
        self.dim = self.model.get_sentence_embedding_dimension()
        # 内存级 embedding 缓存: 参数扫描时同一批文本会被反复编码
        # (归并 -> 关联 -> 检索, 每次重建都要编码数千条), 缓存可省 90%+ 时间。
        self._mem: Dict[str, np.ndarray] = {} if mem_cache else None
        self._mem_hits = 0
        self._mem_misses = 0

    def encode(self, texts: Sequence[str], batch_size: int = 64,
               normalize: bool = True) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        items = list(texts)

        if self._mem is None:
            arr = self.model.encode(
                items, batch_size=batch_size,
                convert_to_numpy=True, normalize_embeddings=normalize,
                show_progress_bar=False,
            )
            return np.asarray(arr, dtype=np.float32)

        # ---- 缓存路径 ----
        out: List[Optional[np.ndarray]] = [None] * len(items)
        missing: List[str] = []
        seen_missing: Set[str] = set()
        for i, t in enumerate(items):
            hit = self._mem.get(t)
            if hit is not None:
                out[i] = hit
                self._mem_hits += 1
            else:
                if t not in seen_missing:
                    seen_missing.add(t)
                    missing.append(t)
                out[i] = None
        if missing:
            arr = self.model.encode(
                missing, batch_size=batch_size,
                convert_to_numpy=True, normalize_embeddings=normalize,
                show_progress_bar=False,
            )
            arr = np.asarray(arr, dtype=np.float32)
            self._mem_misses += len(missing)
            for t, row in zip(missing, arr):
                self._mem[t] = row
            for i, t in enumerate(items):
                if out[i] is None:
                    out[i] = self._mem[t]
        return np.stack(out, axis=0)

    def cache_stats(self) -> Dict[str, int]:
        return {"hits": self._mem_hits, "misses": self._mem_misses,
                "size": len(self._mem or {})}


class AssociationMetric:
    """
    客观的实体关联度指标。

    assoc(i,j) = w_emb * cos(emb_i, emb_j)
                + w_cooc * nPMI_norm(i,j)          (段落共现的归一化 PMI)
                + w_kge  * kg2hop(i,j)              (知识图谱 2-hop 关联, 带衰减)

    三路信号分别对应: 分布式语义、语料统计、图结构证据。
    """

    def __init__(self, embedder: Embedder,
                 w_emb: float = 0.50, w_cooc: float = 0.30, w_kge: float = 0.20,
                 cooc_decay: float = 0.35, emb_topk: int = 16):
        self.embedder = embedder
        self.w_emb = w_emb
        self.w_cooc = w_cooc
        self.w_kge = w_kge
        self.cooc_decay = cooc_decay
        self.emb_topk = emb_topk
    # ------------------------------------------------------------------ #
    @staticmethod
    def _n_pmi(count_ij: float, count_i: float, count_j: float,
               total: float) -> float:
        if count_ij <= 0 or count_i <= 0 or count_j <= 0 or total <= 0:
            return 0.0
        p_ij = count_ij / total
        p_i = count_i / total
        p_j = count_j / total
        pmi = math.log(p_ij / (p_i * p_j))
        denom = -math.log(p_ij)
        if denom <= 1e-12:
            return 0.0
        return max(-1.0, min(1.0, pmi / denom))

    # ------------------------------------------------------------------ #
    def compute(
        self,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        passages: List[Passage],
        top_k: int = 8,
        min_assoc: float = 0.35,
        kg_hops: int = 2,
    ) -> Tuple[Dict[Tuple[str, str], float], Dict[str, Any]]:
        """
        返回:
            assoc_map : {(eid_i, eid_j): assoc} 仅含 > min_assoc 的候选对
            stats     : 诊断信息
        """
        signals, stats = self.compute_signals(entities, relations, passages,
                                             kg_hops=kg_hops)
        assoc_map = self.fuse(signals, top_k=top_k, min_assoc=min_assoc)
        stats["weights"] = {"emb": self.w_emb, "cooc": self.w_cooc, "kge": self.w_kge}
        LOG.info("关联指标: %d 实体, 候选对 %d, 剪枝后 %d",
                 stats.get("n_entities", 0), stats.get("n_assoc_candidates", 0),
                 len(assoc_map))
        return assoc_map, stats

    # ------------------------------------------------------------------ #
    def compute_signals(
        self,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        passages: List[Passage],
        kg_hops: int = 2,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        计算与权重无关的 "原始信号"。

        返回 signals, 内含三路稀疏信号:
            emb  : {(i,j): cos}      embedding 相似度
            cooc : {(i,j): nPMI}     段落共现归一化 PMI
            kg   : {(i,j): decay}    KG 2-hop 结构邻近

        分离原始信号与权重融合的目的: 调参时只需重新融合 (O(#候选对)),
        无需重算 nPMI/BFS/embedding, 使权重网格搜索成本降低约两个数量级。
        """
        eids = [e["eid"] for e in entities]
        idx = {eid: i for i, eid in enumerate(eids)}
        n = len(eids)
        if n == 0:
            return {}, {"n_entities": 0}

        # ---------- (1) embedding 相似度 ----------
        texts = [f"{e['name']}. {e.get('description','')}" for e in entities]
        emb = self.embedder.encode(texts, normalize=True)

        # ---------- (2) 段落共现 nPMI (稀疏) ----------
        pid_index: Dict[str, int] = {}
        entity_pids: Dict[str, List[int]] = defaultdict(list)
        for e in entities:
            for pid in e.get("source_pids", [])[:80]:
                if pid not in pid_index:
                    pid_index[pid] = len(pid_index)
                entity_pids[e["eid"]].append(pid_index[pid])

        # cooc_map[(i,j)] = 归一化 PMI 缩放到 [0,1] (1 = 最大正相关)
        cooc_map: Dict[Tuple[int, int], float] = {}
        if sp is not None and pid_index:
            n_pid = len(pid_index)
            rows: List[int] = []
            cols: List[int] = []
            for eid, pids in entity_pids.items():
                for p in set(pids):
                    rows.append(idx[eid])
                    cols.append(p)
            data = np.ones(len(rows), dtype=np.float32)
            M = sp.csr_matrix((data, (rows, cols)), shape=(n, n_pid))
            C = (M @ M.T).tocsr()
            C.setdiag(0)
            C.eliminate_zeros()
            marg = np.asarray(M.sum(axis=1)).ravel()
            total = float(n_pid)
            Ccoo = C.tocoo()
            for i, j, v in zip(Ccoo.row, Ccoo.col, Ccoo.data):
                cooc_map[(int(i), int(j))] = \
                    (self._n_pmi(v, marg[i], marg[j], total) + 1.0) / 2.0
            del C, M

        # ---------- (3) KG 2-hop 关联 (稀疏, 邻接表 BFS) ----------
        adj: Dict[int, Set[int]] = defaultdict(set)
        for r in relations:
            s, t = idx.get(r["source"]), idx.get(r["target"])
            if s is None or t is None or s == t:
                continue
            adj[s].add(t)
            adj[t].add(s)
        kg2_map: Dict[Tuple[int, int], float] = {}
        if kg_hops > 0:
            # 逐源点 BFS, 只保留最短路径上的权重 (避免稠密 n x n)
            for src in range(n):
                if src not in adj:
                    continue
                visited = {src}
                frontier = [src]
                for hop in range(1, kg_hops + 1):
                    decay_w = self.cooc_decay ** (hop - 1)
                    nxt: List[int] = []
                    for u in frontier:
                        for x in adj.get(u, ()):  # noqa: B007
                            if x in visited:
                                continue
                            visited.add(x)
                            nxt.append(x)
                            kg2_map[(src, x)] = decay_w
                    if not nxt:
                        break
                    frontier = nxt

        # ---------- 融合 (稀疏) ----------
        # 候选对 = embedding 近邻 ∪ 共现对 ∪ KG 2-hop 对
        # 使用分块矩阵乘法求 embedding top-k, 避免 O(n^2) Python 循环
        cand: Set[Tuple[int, int]] = set()
        block = 1024
        emb_top: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        for start in range(0, n, block):
            end = min(start + block, n)
            sim = emb[start:end] @ emb.T
            for ii in range(end - start):
                gi = start + ii
                row = sim[ii]
                k = min(self.emb_topk, n - 1)
                if k <= 0:
                    continue
                idxs = np.argpartition(-row, kth=k - 1)[:k]
                for gj in idxs:
                    gj = int(gj)
                    if gj == gi:
                        continue
                    emb_top[gi].append((gj, float(row[gj])))
        for gi, lst in emb_top.items():
            for gj, s in lst:
                cand.add((gi, gj))
        cand.update(cooc_map.keys())
        cand.update(kg2_map.keys())

        # ---------- 打包原始信号 ----------
        # emb_sig 只保留 embedding top-k 内的对 (更远的对视为无 embedding 证据)
        emb_sig: Dict[Tuple[int, int], float] = {}
        for gi, lst in emb_top.items():
            for gj, sv in lst:
                if gi == gj:
                    continue
                emb_sig[(gi, gj)] = float(np.clip(sv, 0.0, 1.0))

        signals = {
            "eids": eids,
            "emb": emb_sig,
            "cooc": {k: float(np.clip(v, 0.0, 1.0))
                     for k, v in cooc_map.items()},
            "kg": {k: float(np.clip(v, 0.0, 1.0))
                   for k, v in kg2_map.items()},
            "cand": cand,
        }
        stats = {
            "n_entities": n,
            "n_passages_for_cooc": len(pid_index),
            "n_candidates": len(cand),
            "n_emb_pairs": len(emb_sig),
            "n_cooc_pairs": len(cooc_map),
            "n_kg_pairs": len(kg2_map),
        }
        return signals, stats

    # ------------------------------------------------------------------ #
    def fuse(self, signals: Dict[str, Any],
             top_k: int = 8,
             min_assoc: float = 0.35,
             w_emb: Optional[float] = None,
             w_cooc: Optional[float] = None,
             w_kge: Optional[float] = None) -> Dict[Tuple[str, str], float]:
        """
        用给定权重融合原始信号, 返回剪枝后的 assoc_map。

        权重可覆盖实例默认值, 以便在不重算信号的前提下做网格搜索。
        """
        eids: List[str] = signals["eids"]
        emb_sig: Dict[Tuple[int, int], float] = signals["emb"]
        cooc_sig: Dict[Tuple[int, int], float] = signals["cooc"]
        kg_sig: Dict[Tuple[int, int], float] = signals["kg"]
        cand: Set[Tuple[int, int]] = signals["cand"]

        w_e = self.w_emb if w_emb is None else w_emb
        w_c = self.w_cooc if w_cooc is None else w_cooc
        w_k = self.w_kge if w_kge is None else w_kge

        assoc_map: Dict[Tuple[str, str], float] = {}
        for (i, j) in cand:
            if i == j:
                continue
            s_e = emb_sig.get((i, j), 0.0)
            c_e = cooc_sig.get((i, j), 0.0)
            k_e = kg_sig.get((i, j), 0.0)
            val = w_e * s_e + w_c * c_e + w_k * k_e
            if val >= min_assoc:
                assoc_map[(eids[i], eids[j])] = round(val, 5)

        # 稀疏化: 每实体只保留 top_k
        per_entity: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for (a, b), v in assoc_map.items():
            per_entity[a].append((b, v))
        pruned: Dict[Tuple[str, str], float] = {}
        for a, lst in per_entity.items():
            lst.sort(key=lambda x: -x[1])
            for b, v in lst[:top_k]:
                pruned[(a, b)] = v
        return pruned


# ============================================================================
# Stage 4 : 激活扩散网络构建
# ============================================================================

@dataclass
class AACConfig:
    """激活扩散网络的超参数。"""
    decay: float = 0.5           # Collins & Loftus 的时间/距离衰减因子 d
    hops: int = 3               # 扩散跳数 K (行和 = Σ d^t < 1)
    prune_per_row: int = 60     # 每行保留的扩散质量上限 (稀疏化)
    indirect_top_k: int = 12    # 每个节点保留的间接边数量
    indirect_min: float = 0.02  # 间接边权重下限
    firing_threshold: float = 0.005  # firing threshold: 低于该激活不再传播
    direct_weight: float = 1.0  # 直接连边权重 (按设计固定为 1)

    def theory_bound(self) -> float:
        """
        累积扩散流的行和上界 Σ_{t=1..K} d^t。

        因为 M 行随机 => M^t 亦行随机 => R 每行元素和 = Σ_t d^t。
        只要该值 < 1, 就保证 *任何* 间接边权重严格小于直接边的 1.0。
        """
        return sum(self.decay ** t for t in range(1, self.hops + 1))

    def is_theoretically_sound(self) -> bool:
        """理论约束: Σ d^t < 1 (否则间接边可能超过直接边, 语义颠倒)。"""
        return self.theory_bound() < 1.0


class ActivationDiffusion:
    """
    Collins & Loftus (1975) 数值化激活扩散。

    传播通道:
        直接通道  : KG 三元组边, 权重 1.0
        关联通道  : Stage-3 的客观关联度 assoc(i,j) ∈ (0,1)

    传播算子:
        M[u,v] = w(u,v) / Σ_{v'} w(u,v')      (fan effect: 激活按出边分摊)
        A^(t+1) = d * M^T A^(t)               (衰减 d)
        R       = Σ_{t=1..K} d^t (M^T)^t      (累积扩散流)

    关键性质: M 行随机 => R 每行元素和 = Σ_t d^t < 1 (当 d<1),
    故 **任意间接边权重严格小于 1**, 而直接边显式为 1.0。
    """

    def __init__(self, config: Optional[AACConfig] = None):
        self.cfg = config or AACConfig()
        if sp is None:
            raise ImportError("需要 scipy 以进行稀疏扩散计算")

    # ------------------------------------------------------------------ #
    def build_transition(
        self,
        eids: List[str],
        direct_edges: List[Tuple[int, int]],
        assoc_edges: List[Tuple[int, int, float]],
    ) -> sp.csr_matrix:
        """构造行归一化转移矩阵 M (含 fan effect)。"""
        n = len(eids)
        rows: List[int] = []
        cols: List[int] = []
        vals: List[float] = []

        # 直接通道权重 = 1.0
        direct_set: Set[Tuple[int, int]] = set()
        for (u, v) in direct_edges:
            if u == v:
                continue
            direct_set.add((u, v))
            direct_set.add((v, u))     # 语义网络链接双向 (Collins & Loftus)
            rows.append(u); cols.append(v); vals.append(self.cfg.direct_weight)
            rows.append(v); cols.append(u); vals.append(self.cfg.direct_weight)

        # 关联通道 (排除已有直接边的对)
        for (u, v, w) in assoc_edges:
            if u == v or w <= 0:
                continue
            if (u, v) in direct_set:
                continue
            rows.append(u); cols.append(v); vals.append(w)
            rows.append(v); cols.append(u); vals.append(w)

        M = sp.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)
        M.sum_duplicates()

        # fan effect: 行归一化
        deg = np.asarray(M.sum(axis=1)).ravel()
        deg[deg == 0] = 1.0
        inv = sp.diags(1.0 / deg)
        return (inv @ M).tocsr()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _prune_rows(mat: sp.spmatrix, top_k: int, min_val: float) -> sp.csr_matrix:
        """每行仅保留 top_k 个最大元素 (稀疏化, 同时实现 firing threshold)。"""
        mat = mat.tocsr()
        keep_r: List[int] = []
        keep_c: List[int] = []
        keep_v: List[float] = []
        for i in range(mat.shape[0]):
            s, e = mat.indptr[i], mat.indptr[i + 1]
            if s == e:
                continue
            vals = mat.data[s:e]
            cols = mat.indices[s:e]
            if len(vals) > top_k:
                idx = np.argpartition(-vals, top_k)[:top_k]
            else:
                idx = np.arange(len(vals))
            for k in idx:
                if vals[k] >= min_val:
                    keep_r.append(i); keep_c.append(int(cols[k])); keep_v.append(float(vals[k]))
        return sp.csr_matrix((keep_v, (keep_r, keep_c)), shape=mat.shape, dtype=np.float32)

    # ------------------------------------------------------------------ #
    def _first_hit_hops(self, M: sp.csr_matrix,
                        cfg: Optional["AACConfig"] = None) -> Dict[Tuple[int, int], int]:
        """
        计算每个节点对的 "首次到达跳数"。

        累积扩散流 R = Σ_{t=1..K} d^t M^t 把不同跳数的贡献混在一起, 无法
        从 R 反推某条边是几跳扩散的结果。这里同步迭代 M^t, 记录每个
        (i,j) 第一次在 M^t 中出现时的 t。

        返回 {(i,j): t}, t ∈ [1, K]。仅在直接图上可达的节点对才会出现;
        纯关联边 (两节点在 KG 中无关系) 不在 M 的可达集内, 因此不出现。
        """
        cfg = cfg or self.cfg
        first: Dict[Tuple[int, int], int] = {}
        cur = sp.identity(M.shape[0], dtype=M.dtype, format="csr")
        for t in range(1, cfg.hops + 1):
            cur = (cur @ M).tocsr()
            cur = self._prune_rows(cur, cfg.prune_per_row, cfg.firing_threshold)
            coo = cur.tocoo()
            for i, j in zip(coo.row, coo.col):
                key = (int(i), int(j))
                if key not in first:
                    first[key] = t
        return first

    # ------------------------------------------------------------------ #
    def diffuse(self, M: sp.spmatrix) -> sp.csr_matrix:
        """
        执行 K 跳激活扩散, 返回累积扩散流 R = Σ_{t=1..K} d^t M^t。

        方向约定: M 是行随机矩阵, M[i,j] 表示源节点 i 的激活中有多大比例
        流向目标节点 j (Collins & Loftus 的 fan effect)。因此激活更新为

            A^{(t+1)} = d * M A^{(t)},     A^{(0)} = e_i (单位激活在源点 i)

        则 R[i,j] = 从 i 出发的单位激活经 t 跳衰减后到达 j 的累积量。

        由于 M 行随机, M^t 亦行随机, 故 R 每一行的元素和 = Σ_{t=1..K} d^t < 1
        (当 d < 1), 这在数学上保证 *任意* 间接连边权重严格小于 1。
        """
        cfg = self.cfg
        R: Optional[sp.csr_matrix] = None
        # A^(0) = I, 迭代得到 cur = M^t (t = 1..K)
        # 注意: 必须从单位阵起步。若从 M 起步再乘, 得到的是 M^(t+1),
        #       实际扩散了 K+1 跳, 与文档化的 R = Σ_{t=1..K} d^t M^t 不符。
        cur = sp.identity(M.shape[0], dtype=M.dtype, format="csr")
        for t in range(1, cfg.hops + 1):
            cur = (cur @ M).tocsr()
            cur = self._prune_rows(cur, cfg.prune_per_row, cfg.firing_threshold)
            term = cur * (cfg.decay ** t)
            R = term if R is None else (R + term).tocsr()
        assert R is not None
        return R

    # ------------------------------------------------------------------ #
    def build_network(
        self,
        entities: List[Dict[str, Any]],
        direct_relations: List[Dict[str, Any]],
        assoc_map: Dict[Tuple[str, str], float],
        passages: List[Passage],
    ) -> Dict[str, Any]:
        """组装最终激活扩散网络。"""
        eids = [e["eid"] for e in entities]
        idx = {eid: i for i, eid in enumerate(eids)}

        direct_edges: List[Tuple[int, int]] = []
        for r in direct_relations:
            u, v = idx.get(r["source"]), idx.get(r["target"])
            if u is None or v is None or u == v:
                continue
            direct_edges.append((u, v))

        assoc_edges = [(idx[a], idx[b], w) for (a, b), w in assoc_map.items()
                       if a in idx and b in idx]

        M = self.build_transition(eids, direct_edges, assoc_edges)
        LOG.info("扩散开始: |V|=%d, 直接边=%d, 关联边=%d, d=%.2f, K=%d",
                 len(eids), len(direct_edges), len(assoc_edges),
                 self.cfg.decay, self.cfg.hops)
        R = self.diffuse(M)
        LOG.info("扩散完成: R 非零元 %d", R.nnz)

        # ---------- 直接边 (权重 = 1.0) ----------
        direct_adjacency: Set[Tuple[int, int]] = set()
        edges: List[Dict[str, Any]] = []
        # 按无序对聚合关系, 避免同一条边被写入两次
        rel_by_pair: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
        for r in direct_relations:
            u, v = idx.get(r["source"]), idx.get(r["target"])
            if u is None or v is None or u == v:
                continue
            key = (min(u, v), max(u, v))
            rel_by_pair[key].append(r)
            direct_adjacency.add((u, v))
            direct_adjacency.add((v, u))

        for (u, v), rels in rel_by_pair.items():
            types = sorted({x["type"] for x in rels})
            edges.append({
                "source": eids[u],
                "target": eids[v],
                "kind": "direct",
                "weight": self.cfg.direct_weight,
                "relation": "; ".join(types[:4]),
                "relations": types[:8],
                "n_evidence": len(rels),
                "hop": 1,
            })

        # ---------- 间接边 (权重 < 1) ----------
        Rcoo = R.tocoo()
        cand: Dict[Tuple[int, int], float] = {}
        for i, j, v in zip(Rcoo.row, Rcoo.col, Rcoo.data):
            if i == j or v <= 0:
                continue
            key = (min(i, j), max(i, j))
            if key in direct_adjacency:
                continue
            if key in cand and cand[key] >= v:
                continue
            cand[key] = float(v)

        # 对称化: 几何平均 (双向语义网络)
        sym: Dict[Tuple[int, int], float] = {}
        for (i, j), v in cand.items():
            fwd = float(R[i, j]) if R[i, j] != 0 else v
            bwd = float(R[j, i]) if R[j, i] != 0 else v
            w = math.sqrt(max(fwd, 0.0) * max(bwd, 0.0))
            sym[(i, j)] = w

        # 每节点 top-k + 权重下限
        per_node: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        for (i, j), w in sym.items():
            if w < self.cfg.indirect_min:
                continue
            per_node[i].append((j, w))
            per_node[j].append((i, w))
        selected: Dict[Tuple[int, int], float] = {}
        for i, lst in per_node.items():
            lst.sort(key=lambda x: -x[1])
            for j, w in lst[: self.cfg.indirect_top_k]:
                key = (min(i, j), max(i, j))
                selected[key] = max(selected.get(key, 0.0), w)

        # 预计算最短跳数 (一次 BFS 求所有源点的距离, 避免逐边 BFS)
        # hop_dist 是 *直接图* 上的 BFS 距离, 仅用于诊断信息。
        hop_dist = self._all_pairs_hops(direct_adjacency, len(eids), max_depth=4)

        # 实际扩散跳数: 记录每个节点对首次出现在哪一跳的扩散项中。
        # 累积流 R = Σ_t d^t M^t 把所有跳数混在一起, 无法反推单跳归属,
        # 因此在扩散过程中同步记录 "首次到达跳数"。
        first_hop = self._first_hit_hops(M, cfg=self.cfg)

        for (i, j), w in selected.items():
            # hop 语义:
            #   1..K  = 该节点对在第几跳的扩散项中首次出现 (真实扩散跳数)
            #   0     = 纯关联边: 两节点在扩散矩阵 M 中不可达 (KG 中无路径连接),
            #           其权重完全来自 Stage-3 的客观关联度, 不经过任何扩散跳
            #   -1    = 直接边
            hop = first_hop.get((i, j), 0)
            edges.append({
                "source": eids[i],
                "target": eids[j],
                "kind": "indirect",
                "weight": round(float(w), 6),
                "relation": "semantically_associated",
                "relations": [],
                "n_evidence": 0,
                "hop": hop,
                "hop_kind": ("diffusion" if hop >= 1 else "association_only"),
                "direct_graph_hop": hop_dist.get((i, j), 0),
            })

        LOG.info("网络组装: 直接边 %d, 间接边 %d",
                 sum(1 for e in edges if e["kind"] == "direct"),
                 sum(1 for e in edges if e["kind"] == "indirect"))

        # ---------- 节点属性 ----------
        by_eid = {e["eid"]: e for e in entities}
        title_to_pid: Dict[str, str] = {}
        pid_to_title: Dict[str, str] = {}
        for p in passages:
            pid_to_title[p.pid] = p.title
            title_to_pid.setdefault(p.title, p.pid)

        deg = defaultdict(int)
        wdeg = defaultdict(float)
        for e in edges:
            deg[e["source"]] += 1
            deg[e["target"]] += 1
            wdeg[e["source"]] += e["weight"]
            wdeg[e["target"]] += e["weight"]

        # 文档映射: 实体 -> 段落 (供检索)
        ent_to_pids: Dict[str, Set[str]] = defaultdict(set)
        for e in entities:
            for pid in e.get("source_pids", []):
                ent_to_pids[e["eid"]].add(pid)

        nodes: List[Dict[str, Any]] = []
        for e in entities:
            eid = e["eid"]
            node = {
                "eid": eid,
                "name": e["name"],
                "type": e.get("type", "CONCEPT"),
                "description": e.get("description", ""),
                "aliases": e.get("aliases", [])[:20],
                "degree": deg.get(eid, 0),
                "weighted_degree": round(wdeg.get(eid, 0.0), 4),
                "mention_count": e.get("mention_count", 1),
                "source_titles": e.get("source_titles", [])[:20],
                "source_pids": sorted(ent_to_pids.get(eid, set())),
            }
            nodes.append(node)

        network = {
            "nodes": nodes,
            "edges": edges,
            "meta": {
                "n_nodes": len(nodes),
                "n_direct_edges": sum(1 for e in edges if e["kind"] == "direct"),
                "n_indirect_edges": sum(1 for e in edges if e["kind"] == "indirect"),
                "decay": self.cfg.decay,
                "hops": self.cfg.hops,
                "direct_weight": self.cfg.direct_weight,
                "max_indirect_weight": max((e["weight"] for e in edges
                                           if e["kind"] == "indirect"), default=0.0),
                "theory_bound_sum_d_t": sum(self.cfg.decay ** t
                                            for t in range(1, self.cfg.hops + 1)),
                "theory_sound": self.cfg.is_theoretically_sound(),
                "diffusion_hop_range": [1, self.cfg.hops],
            },
        }
        if not self.cfg.is_theoretically_sound():
            LOG.warning(
                "理论约束被违反: Σ_{t=1..%d} d^t = %.4f >= 1 (d=%.2f)。"
                "此时间接边权重可能超过直接边的 1.0, 语义颠倒。"
                "请降低 decay 或 hops。",
                self.cfg.hops, self.cfg.theory_bound(), self.cfg.decay)
        return network

    # ------------------------------------------------------------------ #
    @staticmethod
    def _all_pairs_hops(direct_adjacency: Set[Tuple[int, int]], n: int,
                        max_depth: int = 4) -> Dict[Tuple[int, int], int]:
        """
        在直接图上一次性 BFS 求所有节点对的最短跳数 (上限 max_depth)。

        返回 {(i,j): dist} (i<j)。未连通返回 0。
        """
        adj: Dict[int, List[int]] = defaultdict(list)
        for (u, v) in direct_adjacency:
            adj[u].append(v)
        dist: Dict[Tuple[int, int], int] = {}
        for src in range(n):
            seen = {src}
            frontier = [src]
            for d in range(1, max_depth + 1):
                nxt: List[int] = []
                for u in frontier:
                    for v in adj.get(u, ()):  # noqa: B007
                        if v in seen:
                            continue
                        seen.add(v)
                        nxt.append(v)
                        key = (min(src, v), max(src, v))
                        dist[key] = d
                if not nxt:
                    break
                frontier = nxt
        return dist


# ============================================================================
# Stage 6 : 查询时激活扩散 + 检索评测
# ============================================================================

class ActivationRetriever:
    """
    基于激活扩散网络的检索器 (查询时激活扩散)。

    三处相对基线的增强:

    1. **混合种子 (hybrid seeding)**
       基线只用 dense embedding 取 top-k_e 种子, 但 HotpotQA 的问题几乎总是
       *字面* 包含关键实体名 ("Which magazine was started first, X or Y?")。
       因此种子分数 = α·dense_sim + β·lexical_match, 其中 lexical_match
       衡量 query 与实体名/别名的字符级重合 (含精确匹配加成)。

    2. **IDF 加权 (entity IDF)**
       基线按 sqrt(段落内实体数) 归一化, 但 "United States" 这类高频实体会
       在大量段落中出现并主导排序。按实体在段落库中的出现频率取 IDF 后,
       低频高区分度实体才会主导得分。

    3. **稠密回退 (dense fallback)**
       若某段落完全没有被激活实体覆盖, 用 query-passage dense 相似度兜底,
       避免冷门段落被完全排除。
    """

    def __init__(self, network: Dict[str, Any], embedder: Embedder,
                 decay: float = 0.5, max_hops: int = 3,
                 seed_entities: int = 3, firing_threshold: float = 0.01,
                 passages: Optional[List[Passage]] = None,
                 seed_alpha: float = 0.6,
                 lexical_weight: float = 0.4,
                 use_idf: bool = True,
                 idf_power: float = 1.0,
                 norm_beta: float = 1.0,
                 dense_fallback: bool = True,
                 dense_weight: float = 0.3,
                 title_bonus: float = 0.0):
        self.network = network
        self.embedder = embedder
        self.decay = decay
        self.max_hops = max_hops
        self.k_e = seed_entities
        self.threshold = firing_threshold
        self.seed_alpha = seed_alpha
        self.lexical_weight = lexical_weight
        self.use_idf = use_idf
        self.idf_power = idf_power
        self.norm_beta = norm_beta
        self.dense_fallback = dense_fallback
        self.dense_weight = dense_weight
        self.title_bonus = title_bonus

        self.nodes: Dict[str, Dict[str, Any]] = {n["eid"]: n for n in network["nodes"]}
        self.adj: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for e in network["edges"]:
            self.adj[e["source"]].append((e["target"], e["weight"]))
            self.adj[e["target"]].append((e["source"], e["weight"]))

        # 实体 -> 段落
        self.ent_pids: Dict[str, Set[str]] = defaultdict(set)
        for n in network["nodes"]:
            for pid in n.get("source_pids", []):
                self.ent_pids[n["eid"]].add(pid)

        # 实体描述 embedding (查询种子初始化)
        texts = [f"{n['name']}. {n.get('description','')}" for n in network["nodes"]]
        self.node_ids = [n["eid"] for n in network["nodes"]]
        self.node_emb = embedder.encode(texts, normalize=True)

        # 实体名/别名 -> eid (词面匹配用)
        self.name_index: Dict[str, str] = {}
        for n in network["nodes"]:
            for nm in [n["name"]] + list(n.get("aliases", [])):
                key = legal_normalize(nm)
                if key and key not in self.name_index:
                    self.name_index[key] = n["eid"]

        # ---------- 实体 IDF ----------
        # df(e) = 包含该实体的段落数; idf = log((1+N)/(1+df)) + 1
        self.ent_idf: Dict[str, float] = {}
        if use_idf:
            n_pass = max(1, len({pid for s in self.ent_pids.values() for pid in s}))
            for eid, pids in self.ent_pids.items():
                df = len(pids)
                self.ent_idf[eid] = (math.log((1.0 + n_pass) / (1.0 + df)) + 1.0) ** idf_power

        # ---------- 段落侧索引 ----------
        self.passages: List[Passage] = passages or []
        self.pid_to_text: Dict[str, str] = {}
        self.pid_emb: Optional[np.ndarray] = None
        if self.passages:
            for p in self.passages:
                self.pid_to_text[p.pid] = f"{p.title}. {p.text}"
            if dense_fallback:
                try:
                    self.pid_emb = embedder.encode(
                        [self.pid_to_text[p.pid] for p in self.passages],
                        normalize=True)
                except Exception as e:  # noqa: BLE001
                    LOG.warning("段落 embedding 失败, 关闭 dense 回退: %s", str(e)[:120])
                    self.pid_emb = None
        self.pid_index: Dict[str, int] = {p.pid: i for i, p in enumerate(self.passages)}

        # 段落 -> 激活实体数 (用于归一化, 预先算好避免 O(E*P) 扫描)
        self.pid_ent_count: Dict[str, int] = defaultdict(int)
        for eid, pids in self.ent_pids.items():
            for pid in pids:
                self.pid_ent_count[pid] += 1

    # ------------------------------------------------------------------ #
    def _lexical_seed_score(self, query: str) -> Dict[str, float]:
        """
        词面种子分数: query 与实体名/别名的重合度。

        exact  : query 完整包含实体名 -> 1.0
        partial: 实体名是 query 的子串 -> 按长度比例衰减
        token  : 与 query 有共同显著词元 -> 按交集大小 / 并集大小
        """
        q = legal_normalize(query)
        q_tokens = set(sig_tokens(query))
        scores: Dict[str, float] = defaultdict(float)
        for name_key, eid in self.name_index.items():
            if not name_key:
                continue
            if name_key in q:
                # 精确包含: 越长的名字越可信 (但不超过 1)
                scores[eid] = max(scores[eid],
                                  min(1.0, 0.75 + 0.25 * len(name_key) / max(len(q), 1)))
            else:
                ent_tokens = set(name_key.split())
                if ent_tokens and q_tokens:
                    inter = ent_tokens & q_tokens
                    if inter:
                        jac = len(inter) / len(ent_tokens | q_tokens)
                        scores[eid] = max(scores[eid], jac * 0.5)
        return scores

    # ------------------------------------------------------------------ #
    def seed(self, query: str) -> List[Tuple[str, float]]:
        """
        种子实体: 混合 dense 与词面匹配 (对齐 KG-Infused RAG 的 k_e 种子机制)。
        """
        q = self.embedder.encode([query], normalize=True)[0]
        sims = self.node_emb @ q
        lex = self._lexical_seed_score(query)

        # 融合分数
        fused = self.seed_alpha * sims
        if lex:
            lex_arr = np.zeros_like(sims)
            for i, eid in enumerate(self.node_ids):
                if eid in lex:
                    lex_arr[i] = lex[eid]
            fused = fused + self.lexical_weight * lex_arr

        k = min(self.k_e, len(fused))
        if k <= 0:
            return []
        top = np.argpartition(-fused, kth=k - 1)[:k]
        top = sorted(top, key=lambda i: -fused[i])
        return [(self.node_ids[int(i)], float(max(fused[i], 0.0))) for i in top]

    # ------------------------------------------------------------------ #
    def spread(self, seeds: List[Tuple[str, float]]) -> Dict[str, float]:
        """查询时激活扩散 (Collins & Loftus 式)。"""
        act: Dict[str, float] = defaultdict(float)
        for eid, s in seeds:
            act[eid] = max(act[eid], max(s, 0.0))
        if not act:
            return {}
        for _ in range(self.max_hops):
            nxt: Dict[str, float] = defaultdict(float)
            for u, au in act.items():
                if au < self.threshold:
                    continue
                nbrs = self.adj.get(u, [])
                if not nbrs:
                    continue
                total = sum(w for _, w in nbrs)
                if total <= 0:
                    continue
                for v, w in nbrs:
                    nxt[v] += au * (w / total) * self.decay
            for v, val in nxt.items():
                act[v] = max(act[v], val)
        return {k: v for k, v in act.items() if v > 0}

    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        返回 top-k 段落。

        得分 = [Σ_{e∈p} IDF(e)·act(e)] / (段落内激活实体数)^{norm_beta/2}
               + dense_weight · dense_sim(p)          (仅当段落未被激活实体覆盖时)
               + title_bonus · [标题词面命中]
        """
        seeds = self.seed(query)
        act = self.spread(seeds)
        if not act and not self.dense_fallback:
            return []

        pid_score: Dict[str, float] = defaultdict(float)
        for eid, a in act.items():
            w = a * (self.ent_idf.get(eid, 1.0) if self.use_idf else 1.0)
            for pid in self.ent_pids.get(eid, ()):  # noqa: B007
                pid_score[pid] += w

        # 归一化: 抑制 "实体多" 的段落主导排序
        if self.norm_beta > 0:
            for pid in list(pid_score.keys()):
                cnt = max(1, self.pid_ent_count.get(pid, 1))
                pid_score[pid] /= (cnt ** (self.norm_beta / 2.0))

        # 标题词面加成
        if self.title_bonus > 0 and self.passages:
            q = legal_normalize(query)
            q_tokens = set(sig_tokens(query))
            for p in self.passages:
                t_tokens = set(sig_tokens(p.title))
                if t_tokens and q_tokens and (t_tokens & q_tokens):
                    pid_score[p.pid] += self.title_bonus

        # dense 回退: 未被激活实体覆盖的段落
        if self.dense_fallback and self.pid_emb is not None:
            qv = self.embedder.encode([query], normalize=True)[0]
            dsims = self.pid_emb @ qv
            for i, p in enumerate(self.passages):
                if p.pid not in pid_score:
                    pid_score[p.pid] = self.dense_weight * float(dsims[i])

        if not pid_score:
            return []
        ranked = sorted(pid_score.items(), key=lambda kv: -kv[1])[:top_k]
        return [{"pid": pid, "score": round(s, 5)} for pid, s in ranked]

    # ------------------------------------------------------------------ #
    def evaluate(self, questions: List[Dict[str, Any]],
                 passages: List[Passage], top_k: int = 10) -> Dict[str, Any]:
        """在 HotpotQA 上评测 Recall@k。"""
        pid_to_title = {p.pid: p.title for p in passages}
        title_to_pid = {p.title: p.pid for p in passages}
        recalls = {1: [], 3: [], 5: [], 10: []}
        for qi, q in enumerate(questions):
            res = self.retrieve(q["question"], top_k=max(top_k, 10))
            titles = [pid_to_title.get(r["pid"], "") for r in res]
            gold = set(q["gold_titles"])
            for k in recalls:
                hit = len(gold & set(titles[:k]))
                recalls[k].append(hit / max(len(gold), 1))
            if (qi + 1) % 100 == 0:
                LOG.info("评测进度 %d/%d", qi + 1, len(questions))
        out = {f"recall@{k}": round(float(np.mean(v)), 4) for k, v in recalls.items()}
        out["n_questions"] = len(questions)
        LOG.info("激活扩散检索评测: %s", out)
        return out


# ============================================================================
# 主流程
# ============================================================================

class ActivationGraphRAGPipeline:
    """HGRAG 2.0 知识图谱构建流水线。"""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.data_root = args.data_root
        self.out_dir = args.output_dir or os.path.join(self.data_root, "kg")
        os.makedirs(self.out_dir, exist_ok=True)
        set_seed(args.seed)

        self.client: Optional[DeepSeekClient] = None
        if args.deepseek_api_key:
            self.client = DeepSeekClient(
                api_key=args.deepseek_api_key,
                base_url=args.deepseek_base_url,
                model=args.deepseek_model,
                max_retries=args.max_retries,
            )
        self.embedder: Optional[Embedder] = None

    # ------------------------------------------------------------------ #
    def _embedder(self) -> Embedder:
        if self.embedder is None:
            self.embedder = Embedder(
                self.args.embed_model,
                cache_dir=os.path.join(self.data_root, "hf_cache"),
            )
        return self.embedder

    # ------------------------------------------------------------------ #
    def stage_corpus(self) -> Tuple[List[Passage], List[Dict[str, Any]]]:
        passages, questions = load_hotpotqa_corpus(
            self.data_root,
            num_questions=self.args.num_questions,
            split=self.args.split,
        )
        write_json(os.path.join(self.out_dir, "questions.json"), questions)
        write_json(os.path.join(self.out_dir, "passages.json"),
                   [p.to_dict() for p in passages])
        return passages, questions

    # ------------------------------------------------------------------ #
    def stage_extract(self, passages: List[Passage]) -> List[Dict[str, Any]]:
        if self.client is None:
            raise RuntimeError("Stage 1 需要 --deepseek-api-key")
        extractor = TripleExtractor(self.client, self.out_dir,
                                    max_chars=self.args.max_passage_chars)
        records = extractor.run(passages, workers=self.args.workers)
        LOG.info("LLM 调用统计: %s", self.client.stats())
        return records

    # ------------------------------------------------------------------ #
    @staticmethod
    def _records_to_mentions(records: List[Dict[str, Any]]) -> List[EntityMention]:
        """把抽提结果整理为 mention 列表 (同一 surface form 跨文档合并)。"""
        store: Dict[str, EntityMention] = {}
        for rec in records:
            pid = rec.get("pid", "")
            title = rec.get("title", "")
            for e in rec.get("entities", []):
                name = e.get("name", "").strip()
                if not name:
                    continue
                key = legal_normalize(name)
                if not key:
                    continue
                mid = stable_id("m", key)
                m = store.get(mid)
                if m is None:
                    m = EntityMention(
                        mid=mid, name=name,
                        etype=EntityResolver.normalize_type(e.get("type", "")),
                        description=e.get("description", "")[:400],
                        aliases=list(e.get("aliases", [])),
                    )
                    store[mid] = m
                else:
                    for a in e.get("aliases", []):
                        if a not in m.aliases:
                            m.aliases.append(a)
                    if len(e.get("description", "")) > len(m.description):
                        m.description = e["description"][:400]
                if pid:
                    m.source_pids.add(pid)
                    m.source_titles.add(title)
        return list(store.values())

    # ------------------------------------------------------------------ #
    def stage_build(self, passages: List[Passage], questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        embedder = self._embedder()

        # ---------- Stage 1 结果 ----------
        records = read_jsonl(os.path.join(self.out_dir, "raw_extractions.jsonl"))
        if not records:
            raise RuntimeError("未找到抽提结果, 请先运行 --stage extract")
        LOG.info("载入抽提记录 %d 条", len(records))

        mentions = self._records_to_mentions(records)
        LOG.info("mention 总数: %d", len(mentions))

        raw_relations: List[Dict[str, Any]] = []
        for rec in records:
            raw_relations.extend(rec.get("relations", []))
        LOG.info("原始三元组: %d", len(raw_relations))

        # ---------- Stage 2 实体归并 ----------
        resolver = EntityResolver(
            embedder=embedder,
            client=self.client if self.args.llm_arbitration else None,
            merge_threshold=self.args.merge_threshold,
            ambiguous_low=self.args.ambiguous_low,
            use_llm_arbitration=self.args.llm_arbitration,
            max_llm_arbitrations=self.args.max_llm_arbitrations,
            name_gate=self.args.name_gate,
        )
        entities, relations = resolver.resolve(mentions, raw_relations)
        write_json(os.path.join(self.out_dir, "entities.json"), entities)
        write_json(os.path.join(self.out_dir, "relations.json"), relations)
        write_json(os.path.join(self.out_dir, "merge_evidence.json"),
                   [asdict(e) for e in resolver.evidence])

        # ---------- Stage 3 关联指标 ----------
        metric = AssociationMetric(
            embedder,
            w_emb=self.args.w_emb, w_cooc=self.args.w_cooc, w_kge=self.args.w_kge,
            cooc_decay=self.args.cooc_decay,
        )
        assoc_map, assoc_stats = metric.compute(
            entities, relations, passages,
            top_k=self.args.assoc_top_k,
            min_assoc=self.args.min_assoc,
            kg_hops=self.args.kg_hops,
        )

        # ---------- Stage 4 激活扩散 ----------
        cfg = AACConfig(
            decay=self.args.decay,
            hops=self.args.hops,
            prune_per_row=self.args.prune_per_row,
            indirect_top_k=self.args.indirect_top_k,
            indirect_min=self.args.indirect_min,
            firing_threshold=self.args.firing_threshold,
        )
        diffuser = ActivationDiffusion(cfg)
        network = diffuser.build_network(entities, relations, assoc_map, passages)
        network["meta"]["assoc_stats"] = assoc_stats
        network["meta"]["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        network["meta"]["source_dataset"] = "HotpotQA distractor / validation"

        # ---------- Stage 5 导出 ----------
        net_path = os.path.join(self.out_dir, "activation_network.json")
        write_json(net_path, network)
        self._export_graphml(network, os.path.join(self.out_dir, "activation_network.graphml"))
        self._export_gexf(network, os.path.join(self.out_dir, "activation_network.gexf"))
        with open(os.path.join(self.out_dir, "triples.jsonl"), "w", encoding="utf-8") as f:
            for r in relations:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # 实体 embedding 缓存 (供下游 dense path 复用)
        emb = embedder.encode([f"{e['name']}. {e.get('description','')}"
                               for e in entities], normalize=True)
        np.save(os.path.join(self.out_dir, "entity_embeddings.npy"), emb)
        with open(os.path.join(self.out_dir, "entity_embedding_meta.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"eids": [e["eid"] for e in entities]}, f)

        LOG.info("网络已保存: %s", net_path)
        LOG.info("网络统计: %s", json.dumps(network["meta"], ensure_ascii=False)[:400])

        # ---------- Stage 6 评测 ----------
        if self.args.evaluate:
            retriever = ActivationRetriever(
                network, embedder,
                decay=self.args.decay, max_hops=self.args.hops,
                seed_entities=self.args.seed_entities,
                firing_threshold=self.args.firing_threshold,
                passages=passages,
                seed_alpha=self.args.seed_alpha,
                lexical_weight=self.args.lexical_weight,
                use_idf=self.args.use_idf,
                idf_power=self.args.idf_power,
                norm_beta=self.args.norm_beta,
                dense_fallback=self.args.dense_fallback,
                dense_weight=self.args.dense_weight,
                title_bonus=self.args.title_bonus,
            )
            eval_res = retriever.evaluate(questions, passages, top_k=10)
            write_json(os.path.join(self.out_dir, "retrieval_eval.json"), eval_res)

        return network

    # ------------------------------------------------------------------ #
    @staticmethod
    def _export_graphml(network: Dict[str, Any], path: str) -> None:
        if nx is None:
            return
        G = nx.Graph()
        for n in network["nodes"]:
            G.add_node(n["eid"], name=n["name"], type=n["type"],
                       description=n["description"][:300],
                       aliases="|".join(n["aliases"][:10]),
                       degree=n["degree"], weighted_degree=n["weighted_degree"])
        for e in network["edges"]:
            G.add_edge(e["source"], e["target"], weight=e["weight"],
                       kind=e["kind"], relation=e["relation"], hop=e["hop"])
        nx.write_graphml(G, path)

    @staticmethod
    def _export_gexf(network: Dict[str, Any], path: str) -> None:
        if nx is None:
            return
        G = nx.Graph()
        for n in network["nodes"]:
            G.add_node(n["eid"], name=n["name"], type=n["type"],
                       description=n["description"][:300])
        for e in network["edges"]:
            G.add_edge(e["source"], e["target"], weight=e["weight"],
                       kind=e["kind"], relation=e["relation"], hop=e["hop"])
        try:
            nx.write_gexf(G, path)
        except Exception as e:  # noqa: BLE001
            LOG.warning("GEXF 导出失败: %s", str(e)[:160])

    # ------------------------------------------------------------------ #
    def stage_evaluate(self, passages: List[Passage], questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        net_path = os.path.join(self.out_dir, "activation_network.json")
        network = read_json(net_path)
        if not network:
            raise RuntimeError(f"未找到网络文件 {net_path}, 请先运行 --stage build")
        retriever = ActivationRetriever(
            network, self._embedder(),
            decay=self.args.decay, max_hops=self.args.hops,
            seed_entities=self.args.seed_entities,
            firing_threshold=self.args.firing_threshold,
            passages=passages,
            seed_alpha=self.args.seed_alpha,
            lexical_weight=self.args.lexical_weight,
            use_idf=self.args.use_idf,
            idf_power=self.args.idf_power,
            norm_beta=self.args.norm_beta,
            dense_fallback=self.args.dense_fallback,
            dense_weight=self.args.dense_weight,
            title_bonus=self.args.title_bonus,
        )
        res = retriever.evaluate(questions, passages, top_k=10)
        write_json(os.path.join(self.out_dir, "retrieval_eval.json"), res)
        return res


# ============================================================================
# CLI
# ============================================================================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="HGRAG 2.0 : 基于心理语义网络激活扩散的知识图谱构建",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 路径
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                   help="数据根目录 (数据集/模型/输出)")
    p.add_argument("--output-dir", default=None,
                   help="输出目录 (默认 <data-root>/kg)")
    p.add_argument("--stage", default="all",
                   choices=["all", "corpus", "extract", "build", "evaluate"],
                   help="运行的阶段")

    # 数据
    p.add_argument("--num-questions", type=int, default=200,
                   help="使用的前 N 个 HotpotQA 问题 (决定语料规模)")
    p.add_argument("--split", default="validation", help="数据集 split")

    # LLM
    p.add_argument("--deepseek-api-key",
                   default=os.environ.get("DEEPSEEK_API_KEY"),
                   help="DeepSeek API key (也可用环境变量 DEEPSEEK_API_KEY)")
    p.add_argument("--deepseek-base-url", default=DEEPSEEK_BASE_URL)
    p.add_argument("--deepseek-model", default="deepseek-chat")
    p.add_argument("--max-retries", type=int, default=8)
    p.add_argument("--workers", type=int, default=8, help="LLM 并发数")
    p.add_argument("--max-passage-chars", type=int, default=6000)

    # Embedding
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)

    # 实体归并
    p.add_argument("--merge-threshold", type=float, default=0.82,
                   help="归并判定阈值 (调优后最优值)")
    p.add_argument("--ambiguous-low", type=float, default=0.65,
                   help="LLM 仲裁区间下界")
    p.add_argument("--llm-arbitration", action="store_true", default=True,
                   help="对不确定区间的候选对调用 LLM 仲裁")
    p.add_argument("--no-llm-arbitration", dest="llm_arbitration",
                   action="store_false")
    p.add_argument("--max-llm-arbitrations", type=int, default=400)
    p.add_argument("--name-gate", type=float, default=0.65,
                   help="名称证据门控: name_similarity 低于该值禁止 rule-based 合并 "
                        "(缩写<->全称 与包含关系除外)")

    # 关联指标
    p.add_argument("--w-emb", type=float, default=0.70)
    p.add_argument("--w-cooc", type=float, default=0.20)
    p.add_argument("--w-kge", type=float, default=0.10)
    p.add_argument("--cooc-decay", type=float, default=0.35)
    p.add_argument("--assoc-top-k", type=int, default=8)
    p.add_argument("--min-assoc", type=float, default=0.35)
    p.add_argument("--kg-hops", type=int, default=2)

    # 激活扩散
    p.add_argument("--decay", type=float, default=0.6,
                   help="激活衰减因子 d (直接边权重固定为 1)")
    p.add_argument("--hops", type=int, default=2,
                   help="扩散跳数 K (理论权重上界 = Σ d^t)")
    p.add_argument("--prune-per-row", type=int, default=60)
    p.add_argument("--indirect-top-k", type=int, default=12)
    p.add_argument("--indirect-min", type=float, default=0.02)
    p.add_argument("--firing-threshold", type=float, default=0.001)
    p.add_argument("--strict-theory", action="store_true",
                   help="若 Σ d^t >= 1 (理论不成立) 则直接报错退出")
    p.add_argument("--seed-entities", type=int, default=3,
                   help="查询时种子实体数 k_e")
    p.add_argument("--seed-alpha", type=float, default=0.9,
                   help="种子融合中 dense 相似度的权重 (1-alpha 为词面匹配权重)")
    p.add_argument("--lexical-weight", type=float, default=0.3,
                   help="种子融合中词面匹配的权重")
    p.add_argument("--no-idf", dest="use_idf", action="store_false", default=True,
                   help="关闭实体 IDF 加权")
    p.add_argument("--idf-power", type=float, default=1.0,
                   help="实体 IDF 的幂次 (0 = 无 IDF)")
    p.add_argument("--norm-beta", type=float, default=0.5,
                   help="段落归一化指数 (0 = 不归一化, 1 = sqrt 归一化)")
    p.add_argument("--no-dense-fallback", dest="dense_fallback",
                   action="store_false", default=True,
                   help="关闭未被激活实体覆盖段落的 dense 回退")
    p.add_argument("--dense-weight", type=float, default=0.1,
                   help="dense 回退权重")
    p.add_argument("--title-bonus", type=float, default=0.1,
                   help="标题词面命中的分数加成")
    p.add_argument("--evaluate", action="store_true", default=True,
                   help="构建后自动评测 Recall@k")
    p.add_argument("--no-evaluate", dest="evaluate", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    # ---------- 理论约束前置检查 ----------
    # M 行随机 => R 行和 = Σ_{t=1..K} d^t。若该值 >= 1, 间接边权重可能
    # 超过直接边的 1.0, "直接 KG 证据 > 语义关联" 的前提失效。
    _bound = sum(args.decay ** t for t in range(1, args.hops + 1))
    if _bound >= 1.0:
        LOG.warning("理论约束警告: Σ_{t=1..%d} d^t = %.4f >= 1 (d=%.2f, K=%d)。"
                    "间接边权重可能超过直接边, 语义颠倒。",
                    args.hops, _bound, args.decay, args.hops)
        if args.strict_theory:
            LOG.error("--strict-theory 已启用, 终止执行。")
            return 2
    else:
        LOG.info("理论约束满足: Σ_{t=1..%d} d^t = %.4f < 1 (间接边必小于直接边)",
                 args.hops, _bound)

    pipeline = ActivationGraphRAGPipeline(args)
    stage = args.stage

    LOG.info("=" * 70)
    LOG.info("HGRAG 2.0 激活扩散知识图谱构建 | stage=%s", stage)
    LOG.info("data-root=%s", pipeline.data_root)
    LOG.info("=" * 70)

    passages, questions = pipeline.stage_corpus()

    if stage == "corpus":
        LOG.info("语料准备完成: %d 段落", len(passages))
        return 0

    if stage == "extract":
        pipeline.stage_extract(passages)
        return 0

    if stage in ("build", "all"):
        if stage == "all":
            pipeline.stage_extract(passages)
        pipeline.stage_build(passages, questions)
        return 0

    if stage == "evaluate":
        pipeline.stage_evaluate(passages, questions)
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
