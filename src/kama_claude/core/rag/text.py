from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

_SUPPORTED_SUFFIXES = frozenset({".md", ".txt", ".docx"})
_WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


# 判断指定路径是否是当前 RAG 索引器支持的文本文档
def is_supported_document(path: Path) -> bool:
    return path.suffix.lower() in _SUPPORTED_SUFFIXES


# 从 md、txt 或 docx 文档中提取可索引文本
def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".docx":
        return _extract_docx_text(path)
    raise ValueError(f"unsupported RAG document type: {path.suffix}")


# 从 docx 的 word/document.xml 中提取段落文本
def _extract_docx_text(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as zf:
            raw = zf.read("word/document.xml")
    except (KeyError, zipfile.BadZipFile) as exc:
        raise ValueError(f"invalid docx file: {path}") from exc

    root = ElementTree.fromstring(raw)
    lines: list[str] = []
    for para in root.iter(f"{_WORD_NS}p"):
        parts = [node.text or "" for node in para.iter(f"{_WORD_NS}t")]
        line = "".join(parts).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


# 将文本归一化为用于关键词检索的 token 列表，中文同时保留单字和二元组
def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for match in re.finditer(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", text.lower()):
        part = match.group(0)
        if re.fullmatch(r"[A-Za-z0-9_]+", part):
            tokens.append(part)
        else:
            chars = list(part)
            tokens.extend(chars)
            tokens.extend("".join(chars[i : i + 2]) for i in range(len(chars) - 1))
    return tokens
