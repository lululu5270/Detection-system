from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from kama_claude.core.rag.chunker import chunk_text
from kama_claude.core.rag.model import RagChunk
from kama_claude.core.rag.store import RagStore
from kama_claude.core.rag.text import extract_text, is_supported_document

DEFAULT_SOURCE_DIR = Path(".kama/rag/documents")
DEFAULT_INDEX_PATH = Path(".kama/rag/index.jsonl")
DEFAULT_META_PATH = Path(".kama/rag/meta.json")


@dataclass(frozen=True)
class RagIndexReport:
    documents: int
    chunks: int
    index_path: Path


# 返回当前 UTC 时间的 ISO 8601 字符串
def _now() -> str:
    return datetime.now(UTC).isoformat()


# 将路径转换成相对当前工作目录的稳定来源字符串
def _source_name(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


# 枚举单个文件或目录下所有支持索引的文档
def iter_documents(source: Path) -> list[Path]:
    source = source.expanduser()
    if source.is_file():
        return [source] if is_supported_document(source) else []
    if not source.exists():
        return []
    return sorted(p for p in source.rglob("*") if p.is_file() and is_supported_document(p))


# 从指定文档源重建项目级 RAG 索引
def build_index(
    source: Path = DEFAULT_SOURCE_DIR,
    *,
    index_path: Path = DEFAULT_INDEX_PATH,
    meta_path: Path = DEFAULT_META_PATH,
    chunk_size: int = 900,
) -> RagIndexReport:
    chunks: list[RagChunk] = []
    documents = iter_documents(source)
    for doc in documents:
        text = extract_text(doc)
        if text.strip():
            chunks.extend(chunk_text(_source_name(doc), text, chunk_size=chunk_size))

    RagStore(index_path).write_chunks(chunks)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(
            {
                "created_at": _now(),
                "source": _source_name(source),
                "documents": len(documents),
                "chunks": len(chunks),
                "chunk_size": chunk_size,
                "retrieval": "local_bm25",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return RagIndexReport(documents=len(documents), chunks=len(chunks), index_path=index_path)
