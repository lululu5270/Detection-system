from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from kama_claude.core.rag.indexer import build_index
from kama_claude.core.rag.retriever import RagRetriever
from kama_claude.core.rag.text import extract_text
from kama_claude.core.tools.builtin.rag_search import RagSearchTool


# 创建只包含 document.xml 的最小 docx 文件，供提取逻辑测试使用
def _write_docx(path: Path, paragraphs: list[str]) -> None:
    body = "".join(
        f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>" for text in paragraphs
    )
    xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}</w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("word/document.xml", xml)


# 功能：验证 docx 文档中的中文段落可以被提取为纯文本
# 设计：构造最小 docx zip，避免依赖 Word 或 python-docx，同时覆盖真实 document.xml 路径
def test_extract_docx_text_reads_paragraphs(tmp_path: Path) -> None:
    doc = tmp_path / "radar.docx"
    _write_docx(doc, ["雷达干扰异常案例", "建议先检查天线链路"])

    text = extract_text(doc)

    assert "雷达干扰异常案例" in text
    assert "建议先检查天线链路" in text


# 功能：验证索引构建后中文查询可以召回相关知识片段
# 设计：用一个包含干扰排查信息的 txt 文档建索引，断言 top1 来源和内容都匹配预期
def test_rag_index_and_search_chinese_query(tmp_path: Path) -> None:
    source = tmp_path / "documents"
    source.mkdir()
    (source / "case.txt").write_text(
        "雷达受到压制式干扰时，先检查频谱占用，再调整工作频点。\n",
        encoding="utf-8",
    )
    index = tmp_path / "index.jsonl"

    report = build_index(source, index_path=index, meta_path=tmp_path / "meta.json")
    hits = RagRetriever(index).search("雷达干扰怎么排查", top_k=1)

    assert report.documents == 1
    assert report.chunks == 1
    assert hits
    assert "调整工作频点" in hits[0].chunk.text


# 功能：验证 rag_search 工具能返回带来源和分数的检索结果
# 设计：直接调用工具 invoke，覆盖 Agent 实际调用时使用的格式化输出路径
@pytest.mark.asyncio
async def test_rag_search_tool_formats_hits(tmp_path: Path) -> None:
    source = tmp_path / "documents"
    source.mkdir()
    (source / "manual.md").write_text(
        "## 排障流程\n雷达回波异常时，需要核对干扰源方向和设备日志。",
        encoding="utf-8",
    )
    index = tmp_path / "index.jsonl"
    build_index(source, index_path=index, meta_path=tmp_path / "meta.json")

    result = await RagSearchTool(index).invoke({"query": "回波异常 干扰源", "top_k": 3})

    assert not result.is_error
    assert "source=" in result.content
    assert "设备日志" in result.content
