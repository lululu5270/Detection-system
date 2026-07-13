from __future__ import annotations

import json
from pathlib import Path

from kama_claude.core.rag.model import RagChunk


class RagStore:
    # 初始化基于 JSONL 的本地 RAG 索引存储
    def __init__(self, index_path: Path) -> None:
        self._index_path = index_path

    # 将 chunk 列表完整写入索引文件
    def write_chunks(self, chunks: list[RagChunk]) -> None:
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        with self._index_path.open("w", encoding="utf-8") as f:
            for chunk in chunks:
                row = {
                    "id": chunk.id,
                    "source": chunk.source,
                    "text": chunk.text,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 从索引文件读取全部 chunk，文件不存在时返回空列表
    def read_chunks(self) -> list[RagChunk]:
        if not self._index_path.exists():
            return []

        chunks: list[RagChunk] = []
        for line in self._index_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            chunks.append(
                RagChunk(
                    id=str(row["id"]),
                    source=str(row["source"]),
                    text=str(row["text"]),
                    start_line=row.get("start_line"),
                    end_line=row.get("end_line"),
                )
            )
        return chunks
