from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RagChunk:
    id: str
    source: str
    text: str
    start_line: int | None = None
    end_line: int | None = None


@dataclass(frozen=True)
class RagSearchHit:
    chunk: RagChunk
    score: float
