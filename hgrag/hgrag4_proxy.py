#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HGRAG —— LLM-agnostic API 代理 (可发布实现)
=========================================

本模块实现论文中所称的 "lightweight, LLM-agnostic API proxy":

  * 暴露 OpenAI 兼容的 ``POST /v1/chat/completions`` 端点;
  * 代理层位于客户端与下游推理服务 (vLLM / SGLang / OpenAI / ...) 之间;
  * **下游推理服务无需任何改动** —— 代理只按 OpenAI 协议转发请求,
    代理本身完成检索、上下文拼装与消息改写。

这是审稿人 #2 concern #3 所要求的可验证性支撑: "no backend modification"
这句话, 只有在代理的实现被发布出来时才可被读者检验。

四个步骤 (与论文正文一致):
  1. 查询抽取   —— 取最后一条 role=user 消息作为查询 q;
  2. 混合检索   —— 调用 R(q, K) 取 top-k 文档与检索统计;
  3. 上下文构造 —— 把命中文档格式化为结构化知识块, 拼成知识前缀 C;
  4. 消息改写   —— 把最后一条 user 消息替换为 [C || 分隔符 || q],
                   其余对话历史原样保留 (维持多轮一致性)。

每请求可覆盖的扩展字段 (放在请求体的 ``rag`` 对象里):
    enable_rag    (bool)   是否注入检索上下文, 默认 True
    rag_top_k     (int)    检索条数, 默认 10
    rag_keywords  (list)   显式关键词约束 K, 缺省则由问句自动抽取
    rag_max_chars (int)    单个知识块的正文截断长度, 默认 800

仅依赖标准库 (http.server / urllib), 便于读者直接运行。

用法:
    python hgrag4_proxy.py --kg-dir C:\\data\\zsk\\RAG\\kgv4 \\
        --port 8000 --backend-url http://127.0.0.1:8001/v1

    # 不接下游推理服务, 只看代理注入的上下文 (便于离线检查):
    python hgrag4_proxy.py --kg-dir ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import activation_graphrag as A  # noqa: E402
import hgrag4_experiments as E4  # noqa: E402
import hgrag4_splits as SPL  # noqa: E402

LOG = logging.getLogger("hgrag4.proxy")

DEFAULT_SEP = "\n\n---\n\n"
MAX_BODY_BYTES = 32 * 1024 * 1024


# ============================================================================
# 上下文构造
# ============================================================================
def _strip_paren(s: str) -> str:
    return re.sub(r"\s*\([^)]*\)", " ", s).strip()


def extract_query(messages: List[Dict[str, Any]]) -> Optional[str]:
    """取最后一条 role=user 消息作为查询 q。"""
    for m in reversed(messages or []):
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str) and c.strip():
                return c.strip()
            if isinstance(c, list):  # 多模态 content 数组
                parts = [p.get("text", "") for p in c
                         if isinstance(p, dict) and p.get("type") == "text"]
                if parts:
                    return "\n".join(parts).strip()
    return None


def build_context_block(docs: List[Dict[str, Any]],
                        corpus: E4.Corpus,
                        max_chars: int = 800) -> str:
    """
    把检索命中格式化为结构化知识块。

    每个块的格式为::

        [文档 <i>] <title>
        <body>

    块之间用空行分隔; 整个前缀以一个分隔符与原问句相连。
    """
    chunks: List[str] = []
    for i, d in enumerate(docs, start=1):
        pid = d.get("pid", "")
        title = corpus.pid_to_title.get(pid, d.get("title", "") or "")
        text = corpus.pid_text.get(pid, d.get("text", "") or "")
        if not text:
            text = d.get("text", "") or ""
        body = text.strip()
        if max_chars > 0 and len(body) > max_chars:
            body = body[:max_chars].rstrip() + " ..."
        chunks.append(f"[Document {i}] {title}\n{body}")
    return "\n\n".join(chunks)


def augment_messages(messages: List[Dict[str, Any]],
                      context: str,
                      separator: str = DEFAULT_SEP
                      ) -> Tuple[List[Dict[str, Any]], bool]:
    """
    消息改写: 把最后一条 user 消息替换为 [C || separator || q],
    其余历史原样保留。

    返回 (新消息列表, 是否改写了)。
    """
    out = [dict(m) for m in (messages or [])]
    idx = None
    for i in range(len(out) - 1, -1, -1):
        m = out[i]
        if isinstance(m, dict) and m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str) and c.strip():
                idx = i
                break
            if isinstance(c, list):
                parts = [p.get("text", "") for p in c
                         if isinstance(p, dict) and p.get("type") == "text"]
                if parts:
                    out[i] = dict(m)
                    out[i]["content"] = (context + separator
                                         + "\n".join(parts)).strip()
                    return out, True
    if idx is None:
        return out, False
    out[idx] = dict(out[idx])
    out[idx]["content"] = (context + separator + out[idx]["content"]).strip()
    return out, True


# ============================================================================
# 代理核心 (与 HTTP 层解耦, 便于单元测试)
# ============================================================================
class HGRAGProxy:
    """
    代理核心: 持有检索器, 把一次 chat/completions 请求变成
    (改写后的请求, 检索统计)。
    """

    def __init__(self, corpus: E4.Corpus,
                 retriever: Any,
                 default_top_k: int = 10,
                 max_chars: int = 800,
                 separator: str = DEFAULT_SEP):
        self.corpus = corpus
        self.retriever = retriever
        self.default_top_k = int(default_top_k)
        self.max_chars = int(max_chars)
        self.separator = separator
        self.n_requests = 0
        self.n_injected = 0
        self._lock = threading.Lock()

    # ---------------- 检索 ---------------- #
    def retrieve(self, query: str,
                 keywords: Optional[List[str]] = None,
                 top_k: Optional[int] = None) -> Tuple[List[Dict[str, Any]],
                                                      Dict[str, Any]]:
        top_k = int(top_k or self.default_top_k)
        kws = keywords
        if kws is None:
            # 自动抽取: 与论文中的评测完全相同的无泄漏抽取器
            kws = E4.extract_keywords_from_question({"question": query},
                                                    self.corpus)
        docs, stats = self.retriever.retrieve(query, top_k=top_k, keywords=kws)
        return docs, stats

    # ---------------- 单次请求处理 ---------------- #
    def handle_chat_completion(self, body: Dict[str, Any]
                               ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        处理一个 chat/completions 请求体。

        返回 (转发给下游的请求体, 检索统计)。
        """
        body = dict(body or {})
        messages = body.get("messages") or []
        opts = body.get("rag") or {}
        if not isinstance(opts, dict):
            opts = {}
        # 代理自己的扩展字段绝不转发给下游推理服务 —— 下游看到的请求体
        # 与客户端发出的请求体逐字段相同 (除 messages 外)。
        body.pop("rag", None)

        enable = bool(opts.get("enable_rag", True))
        top_k = opts.get("rag_top_k", self.default_top_k)
        kws = opts.get("rag_keywords")
        if kws is not None and not isinstance(kws, list):
            kws = None
        max_chars = int(opts.get("rag_max_chars", self.max_chars))

        query = extract_query(messages)
        t0 = time.perf_counter()
        docs: List[Dict[str, Any]] = []
        stats: Dict[str, Any] = {}
        if enable and query:
            docs, stats = self.retrieve(query, keywords=kws, top_k=top_k)
            context = build_context_block(docs, self.corpus, max_chars)
            new_messages, changed = augment_messages(messages, context,
                                                     self.separator)
            if changed:
                body["messages"] = new_messages
        else:
            changed = False

        latency_ms = (time.perf_counter() - t0) * 1000.0
        meta = {
            "enable_rag": enable,
            "injected": bool(changed),
            "n_docs": len(docs),
            "top_k": top_k,
            "n_keywords": (len(kws) if kws is not None
                           else len(E4.extract_keywords_from_question(
                               {"question": query or ""}, self.corpus))),
            "titles": [self.corpus.pid_to_title.get(d.get("pid", ""), "")
                       for d in docs],
            "retrieval_latency_ms": round(latency_ms, 3),
            "retrieval_stats": {
                k: stats.get(k) for k in
                ("t_total", "t_milvus", "t_kg", "t_fusion",
                 "milvus_ranked", "kg_ranked")
                if k in stats
            },
        }
        with self._lock:
            self.n_requests += 1
            if changed:
                self.n_injected += 1
        return body, meta


# ============================================================================
# HTTP 层
# ============================================================================
class _Handler(BaseHTTPRequestHandler):
    server_version = "HGRAGProxy/1.0"
    proxy: HGRAGProxy = None          # type: ignore[assignment]
    backend_url: str = ""
    backend_token: str = ""
    dry_run: bool = False

    # -------- 工具 -------- #
    def _json(self, code: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> Dict[str, Any]:
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return {}
        if n > MAX_BODY_BYTES:
            raise ValueError("request body too large")
        raw = self.rfile.read(n)
        return json.loads(raw.decode("utf-8"))

    def log_message(self, fmt: str, *args: Any) -> None:
        LOG.info("%s - %s", self.address_string(), fmt % args)

    # -------- 路由 -------- #
    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") in ("/healthz", "/health"):
            p = self.proxy
            self._json(200, {
                "status": "ok",
                "n_requests": p.n_requests,
                "n_injected": p.n_injected,
                "backend": None if self.dry_run else self.backend_url,
                "dry_run": self.dry_run,
            })
            return
        self._json(404, {"error": {"message": "not found",
                                    "type": "invalid_request_error"}})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._json(404, {"error": {"message": "not found",
                                        "type": "invalid_request_error"}})
            return
        try:
            body = self._read_body()
        except Exception as exc:  # noqa: BLE001
            self._json(400, {"error": {"message": f"bad request: {exc}",
                                        "type": "invalid_request_error"}})
            return

        try:
            fwd, meta = self.proxy.handle_chat_completion(body)
        except Exception as exc:  # noqa: BLE001
            LOG.exception("retrieval failed")
            self._json(500, {"error": {"message": f"retrieval failed: {exc}",
                                        "type": "server_error"}})
            return

        if self.dry_run:
            # 离线模式: 不转发, 直接把改写后的请求与检索统计返回, 便于检查
            self._json(200, {
                "id": "chatcmpl-dryrun",
                "object": "chat.completion",
                "model": body.get("model", "hgrag-dryrun"),
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant",
                                "content": "(dry-run: no backend call)"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                          "total_tokens": 0},
                "hgrag": meta,
                "forwarded_request": fwd,
            })
            return

        # -------- 透明转发到下游推理服务 -------- #
        url = self.backend_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url = url + "/chat/completions"
        data = json.dumps(fwd, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        if self.backend_token:
            req.add_header("Authorization", f"Bearer {self.backend_token}")
        # 透传客户端的鉴权头 (若下游使用自有 scheme)
        auth = self.headers.get("Authorization")
        if auth and not self.backend_token:
            req.add_header("Authorization", auth)

        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                raw = resp.read()
                code = resp.getcode()
                ctype = resp.headers.get("Content-Type",
                                         "application/json")
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            code = exc.code
            ctype = exc.headers.get("Content-Type", "application/json")
        except Exception as exc:  # noqa: BLE001
            self._json(502, {"error": {"message": f"backend error: {exc}",
                                        "type": "server_error"}})
            return

        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


# ============================================================================
# 装配
# ============================================================================
def build_proxy(kg_dir: str, embed_model: str, cache_dir: str,
                milvus_backend: str = "exact",
                es_index: Optional[Any] = None,
                default_top_k: int = 10,
                max_chars: int = 800) -> HGRAGProxy:
    """载入语料与图谱, 构造检索器, 返回代理核心。"""
    corpus = E4.Corpus(kg_dir, embed_model, cache_dir, milvus_backend)
    if es_index is not None:
        corpus.shared_es = es_index
    # 部署配置 = 论文中的出厂配置, 与 hgrag4_run.py 的 DEFAULT_PARAMS
    # 逐字段一致 (即在 DEV (44 题) 上经坐标下降 + 5-fold CV + 1-SE 规则
    # 选出的最终配置):
    #   w_v=0.4, w_g=0.5, w_kw=0.2, eta=0.0,
    #   保证机制关闭 (g=0) —— 该机制在消融中被证明有害, 已从部署配置移除,
    #   norm_mode="rank", lexical="bm25", w_bm25=0.7, lex_norm="max"。
    retriever = E4.build_retriever(
        corpus,
        w_v=0.4, w_g=0.5, w_kw=0.2, eta=0.0,
        guarantee_k=0, use_guarantee=False, guarantee_mode="off",
        kw_pos=0.35, kw_count=0.35, kw_assoc=0.30,
        norm_mode="rank", kw_scope="cand", design="sum",
        w_title=0.0, lexical="bm25", w_bm25=0.7, lex_norm="max",
        es_index=es_index,
    )
    return HGRAGProxy(corpus, retriever, default_top_k=default_top_k,
                      max_chars=max_chars)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="HGRAG LLM-agnostic API proxy")
    p.add_argument("--kg-dir", default=r"C:\data\zsk\RAG\kgv4")
    p.add_argument("--embed-model", default=A.DEFAULT_EMBED_MODEL)
    p.add_argument("--cache-dir", default=r"C:\data\zsk\RAG\hf_cache")
    p.add_argument("--milvus-backend", default="exact",
                   choices=["milvus-lite", "exact"])
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--backend-url", default="",
                   help="下游推理服务, 例如 http://127.0.0.1:8001/v1")
    p.add_argument("--backend-token", default=os.environ.get(
        "HGRAG_BACKEND_TOKEN", ""))
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--max-chars", type=int, default=800)
    p.add_argument("--dry-run", action="store_true",
                   help="不转发到下游, 直接返回改写后的请求与检索统计")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    LOG.info("载入语料与图谱: %s", args.kg_dir)
    proxy = build_proxy(args.kg_dir, args.embed_model, args.cache_dir,
                        args.milvus_backend,
                        default_top_k=args.top_k,
                        max_chars=args.max_chars)

    handler = type("_BoundHandler", (_Handler,), {
        "proxy": proxy,
        "backend_url": args.backend_url,
        "backend_token": args.backend_token,
        "dry_run": args.dry_run,
    })

    srv = ThreadingHTTPServer((args.host, args.port), handler)
    LOG.info("代理已启动: http://%s:%d/v1/chat/completions", args.host,
             args.port)
    if args.dry_run:
        LOG.info("dry-run 模式: 不转发到下游推理服务")
    elif args.backend_url:
        LOG.info("下游推理服务: %s", args.backend_url)
    else:
        LOG.warning("未指定 --backend-url, 将以 dry-run 方式运行")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        LOG.info("停止代理")
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
