from __future__ import annotations

import hashlib

from kama_claude.core.rag.model import RagChunk


# 根据来源和块序号生成稳定 chunk ID
def _chunk_id(source: str, index: int, text: str) -> str:
    digest = hashlib.sha1(f"{source}\0{index}\0{text}".encode()).hexdigest()
    return digest[:16]


# 将文档文本按行聚合成适合检索返回的片段
def chunk_text(source: str, text: str, *, chunk_size: int = 900) -> list[RagChunk]:
    lines = text.splitlines()
    chunks: list[RagChunk] = []
    buf: list[str] = []
    start_line: int | None = None

    def flush(end_line: int) -> None:
        nonlocal buf, start_line
        body = "\n".join(buf).strip()
        if body and start_line is not None:
            idx = len(chunks)
            chunks.append(
                RagChunk(
                    id=_chunk_id(source, idx, body),
                    source=source,
                    text=body,
                    start_line=start_line,
                    end_line=end_line,
                )
            )
        buf = []
        start_line = None

    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            if buf:
                flush(line_no - 1)
            continue
        if start_line is None:
            start_line = line_no
        projected = len("\n".join([*buf, stripped]))
        if buf and projected > chunk_size:
            flush(line_no - 1)
            start_line = line_no
        buf.append(stripped)

    if buf:
        flush(len(lines))
    return chunks
