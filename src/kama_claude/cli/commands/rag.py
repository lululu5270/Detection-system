from __future__ import annotations

from pathlib import Path

from kama_claude.core.rag.indexer import DEFAULT_INDEX_PATH, DEFAULT_SOURCE_DIR, build_index
from kama_claude.core.rag.retriever import RagRetriever


# 执行 kama rag index，重建项目级知识库索引
def cmd_rag_index(source: str, index_path: str, chunk_size: int) -> None:
    report = build_index(
        Path(source),
        index_path=Path(index_path),
        chunk_size=chunk_size,
    )
    print(
        f"indexed documents={report.documents} chunks={report.chunks} "
        f"index={report.index_path}"
    )


# 执行 kama rag search，在本地知识库索引中检索片段
def cmd_rag_search(query: str, index_path: str, top_k: int) -> None:
    hits = RagRetriever(Path(index_path)).search(query, top_k=top_k)
    if not hits:
        print("no results")
        return
    for i, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        loc = ""
        if chunk.start_line is not None and chunk.end_line is not None:
            loc = f":{chunk.start_line}-{chunk.end_line}"
        print(f"[{i}] {chunk.source}{loc} score={hit.score:.3f}")
        print(chunk.text)
        print()


DEFAULT_RAG_SOURCE = str(DEFAULT_SOURCE_DIR)
DEFAULT_RAG_INDEX = str(DEFAULT_INDEX_PATH)
