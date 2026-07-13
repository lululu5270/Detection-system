from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

from kama_claude.core.rag.indexer import DEFAULT_INDEX_PATH
from kama_claude.core.rag.model import RagSearchHit
from kama_claude.core.rag.store import RagStore
from kama_claude.core.rag.text import tokenize


class RagRetriever:
    # 初始化本地 BM25 风格检索器，默认读取项目级索引
    def __init__(self, index_path: Path = DEFAULT_INDEX_PATH) -> None:
        self._index_path = index_path

    # 针对 query 检索 top_k 个相关 chunk
    def search(self, query: str, *, top_k: int = 5) -> list[RagSearchHit]:
        chunks = RagStore(self._index_path).read_chunks()
        query_terms = tokenize(query)
        if not chunks or not query_terms:
            return []

        chunk_terms = [tokenize(c.text) for c in chunks]
        doc_freq: Counter[str] = Counter()
        for terms in chunk_terms:
            doc_freq.update(set(terms))

        avg_len = sum(len(t) for t in chunk_terms) / max(1, len(chunk_terms))
        scored: list[RagSearchHit] = []
        for chunk, terms in zip(chunks, chunk_terms, strict=True):
            score = _bm25_score(query_terms, terms, doc_freq, len(chunks), avg_len)
            if score > 0:
                scored.append(RagSearchHit(chunk=chunk, score=score))

        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_k]


# 计算单个 chunk 对 query 的 BM25 风格相关性分数
def _bm25_score(
    query_terms: list[str],
    doc_terms: list[str],
    doc_freq: Counter[str],
    doc_count: int,
    avg_len: float,
) -> float:
    tf = Counter(doc_terms)
    doc_len = len(doc_terms)
    k1 = 1.5
    b = 0.75
    score = 0.0
    for term in set(query_terms):
        freq = tf.get(term, 0)
        if freq <= 0:
            continue
        df = doc_freq.get(term, 0)
        idf = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
        denom = freq + k1 * (1 - b + b * doc_len / max(avg_len, 1.0))
        score += idf * (freq * (k1 + 1) / denom)
    return score
