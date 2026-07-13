from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from kama_claude.core.bus.events import StepFinishedEvent, StepStartedEvent
from kama_claude.core.context import ExecutionContext
from kama_claude.core.events.bus import EventBus
from kama_claude.core.llm.base import LLMProvider
from kama_claude.core.loop import AgentLoop
from kama_claude.core.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from kama_claude.core.compact.compactor import Compactor
    from kama_claude.core.permissions.manager import PermissionManager

log = logging.getLogger(__name__)

_PLAN_SYSTEM = (
    "Before acting, create a concise execution plan for the user's goal. "
    "Do not call tools in this planning step. Return only a numbered plan with "
    "the evidence to gather, commands or files likely needed, and success criteria. "
    "The next phase will execute the plan with tools."
)

_EXECUTE_INSTRUCTION = (
    "Execute the plan above. Use tools when needed, update the plan if evidence "
    "contradicts it, and finish with a concise final answer when the goal is complete."
)


# 杩斿洖褰撳墠 UTC 鏃堕棿鐨?ISO 8601 瀛楃涓?
def _now() -> str:
    return datetime.now(UTC).isoformat()


class PlanExecuteLoop:
    # 初始化计划执行循环，内部复用现有 AgentLoop 执行工具阶段
    def __init__(
        self,
        provider: LLMProvider,
        registry: ToolRegistry,
        bus: EventBus,
        *,
        permission_manager: PermissionManager | None = None,
        compactor: Compactor | None = None,
        compact_threshold: float = 0.80,
        session_id: str = "",
    ) -> None:
        self._provider = provider
        self._registry = registry
        self._bus = bus
        self._inner = AgentLoop(
            provider,
            registry,
            bus,
            permission_manager=permission_manager,
            compactor=compactor,
            compact_threshold=compact_threshold,
            session_id=session_id,
        )

    # 执行计划生成阶段，然后进入常规工具执行循环
    async def run(self, context: ExecutionContext) -> None:
        await self._bus.publish(
            StepStartedEvent(run_id=context.run_id, step=0, ts=_now())
        )
        try:
            response = await self._provider.chat(
                messages=context.messages,
                tool_schemas=[],
                bus=self._bus,
                run_id=context.run_id,
                step=0,
                system=context.system_prompt(_PLAN_SYSTEM),
            )
        except asyncio.CancelledError:
            context.mark_failed("cancelled")
            raise
        except Exception:
            log.exception("plan generation failed run_id=%s", context.run_id)
            context.mark_failed("plan_llm_error")
            await self._bus.publish(
                StepFinishedEvent(run_id=context.run_id, step=0, ts=_now())
            )
            return

        plan = response.text.strip()
        if not plan:
            plan = "1. Inspect the task context.\n2. Use available tools as needed.\n3. Finish."
        context.add_assistant_message([{"type": "text", "text": "Execution plan:\n" + plan}])
        context.messages.append({"role": "user", "content": _EXECUTE_INSTRUCTION})
        await self._bus.publish(
            StepFinishedEvent(run_id=context.run_id, step=0, ts=_now())
        )

        if not context.is_done():
            await self._inner.run(context)
