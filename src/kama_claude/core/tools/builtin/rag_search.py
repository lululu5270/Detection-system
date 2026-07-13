from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from kama_claude.core.rag.indexer import DEFAULT_INDEX_PATH
from kama_claude.core.rag.retriever import RagRetriever
from kama_claude.core.tools.base import BaseTool, ToolResult


class RagSearchParams(BaseModel):
    model_config = ConfigDict(extra="ignore")
    query: str
    top_k: int = Field(default=5, ge=1, le=10)


class RagSearchTool(BaseTool):
    params_model = RagSearchParams
    name = "rag_search"
    description = (
        "Search the local project RAG knowledge base when the answer needs domain facts, "
        "historical cases, manuals, troubleshooting procedures, or experiment records. "
        "Use the returned snippets as cited evidence."
    )
    input_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language search query for the local RAG knowledge base.",
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum number of snippets to return, from 1 to 10.",
                "minimum": 1,
                "maximum": 10,
                "default": 5,
            },
        },
        "required": ["query"],
    }

    # 绑定索引路径，默认使用项目级 RAG 索引
    def __init__(self, index_path: Path = DEFAULT_INDEX_PATH) -> None:
        self._retriever = RagRetriever(index_path)

    # 执行本地知识库检索并格式化为可被模型引用的片段
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        parsed = RagSearchParams.model_validate(params)
        hits = self._retriever.search(parsed.query, top_k=parsed.top_k)
        if not hits:
            return ToolResult(
                content=(
                    "No RAG results found. Run `kama rag index` after placing documents "
                    "under `.kama/rag/documents`."
                )
            )

        blocks: list[str] = []
        for i, hit in enumerate(hits, start=1):
            chunk = hit.chunk
            loc = ""
            if chunk.start_line is not None and chunk.end_line is not None:
                loc = f":{chunk.start_line}-{chunk.end_line}"
            blocks.append(
                f"[{i}] source={chunk.source}{loc} score={hit.score:.3f}\n{chunk.text}"
            )
        return ToolResult(content="\n\n".join(blocks))
