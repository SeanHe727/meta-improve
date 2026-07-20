"""Multi-Agent orchestrator (Planner / Worker / Reviewer).

Reuses what we already built:
  - Planner (Phase 5) to decompose the goal into a task DAG,
  - query() (the ReAct loop) as the Worker that executes each task,
  - Reviewer (Phase 6 Part 1) as a quality gate.

The new bit vs Plan-and-Execute is the REVIEW LOOP: each task's result is
reviewed; if rejected, the worker retries with the reviewer's feedback (bounded).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from ..llm.base import LlmClient
from ..plan.models import ExecutionPlan, Task
from ..plan.planner import Planner
from ..prompt.assembler import PromptAssembler
from ..tools.registry import ToolRegistry
from .query import query
from .reviewer import Reviewer


async def multi_agent(
    *,
    client: LlmClient,
    registry: ToolRegistry,
    goal: str,
    cwd: str,
    memory: Any = None,
    code_index: Any = None,
    max_task_turns: int = 8,
    max_retries: int = 2,
) -> AsyncIterator[dict[str, Any]]:
    # 1. PLANNER decomposes the goal into a task DAG (reused from Phase 5).
    yield {"type": "text_delta", "text": f"[planner] planning: {goal}\n"}
    try:
        plan = await Planner(client).create_plan(goal)
    except Exception as exc:  # noqa: BLE001
        yield {"type": "error", "error": exc}
        return
    yield {"type": "text_delta", "text": plan.summarize() + "\n\n"}

    reviewer = Reviewer(client)
    system_prompt = PromptAssembler(
        cwd=cwd,
        model=client.model_name,
        provider=client.provider_name,
        tool_names=registry.list_names(),
    ).build()

    # 2. WORKERS execute tasks in dependency order (parallel batches), each
    #    result gated by the REVIEWER with bounded retries.
    while True:
        batch = plan.executable_tasks()
        if not batch:
            break
        results = await asyncio.gather(
            *(
                _run_reviewed_task(
                    client,
                    registry,
                    reviewer,
                    plan,
                    task,
                    cwd,
                    memory,
                    code_index,
                    system_prompt,
                    max_task_turns,
                    max_retries,
                )
                for task in batch
            )
        )
        for task, events in results:
            for event in events:
                yield event
            yield {
                "type": "tool_result",
                "name": f"task:{task.id}",
                "content": task.result,
                "is_error": task.status.value == "failed",
            }

    # 3. SUMMARIZE.
    yield {"type": "text_delta", "text": "\n" + _final_summary(plan)}
    yield {"type": "done", "total_turns": 0, "total_tokens": 0, "messages": []}


async def _run_reviewed_task(
    client: LlmClient,
    registry: ToolRegistry,
    reviewer: Reviewer,
    plan: ExecutionPlan,
    task: Task,
    cwd: str,
    memory: Any,
    code_index: Any,
    system_prompt: str,
    max_task_turns: int,
    max_retries: int,
) -> tuple[Task, list[dict[str, Any]]]:
    # returns (task, events-to-report) so the caller can yield them in order.
    task.mark_running()
    events: list[dict[str, Any]] = []
    feedback = ""
    result = ""

    for attempt in range(max_retries + 1):
        # WORKER executes the task (with prior feedback on retries).
        result = await _worker_execute(
            client,
            registry,
            plan,
            task,
            cwd,
            memory,
            code_index,
            system_prompt,
            max_task_turns,
            feedback,
        )
        # REVIEWER checks the result.
        review = await reviewer.review(task.description, result)
        if review.approved:
            events.append({"type": "text_delta", "text": f"[reviewer] approved {task.id}\n"})
            task.mark_completed(result)
            return task, events
        # rejected -> feed the feedback back for the next attempt.
        feedback = review.feedback
        events.append(
            {
                "type": "text_delta",
                "text": f"[reviewer] rejected {task.id} (attempt {attempt + 1}): {feedback}\n",
            }
        )

    # out of retries: keep the last result but mark it failed (reviewer never satisfied).
    task.mark_failed(f"{result}\n[not approved after {max_retries + 1} attempts]")
    return task, events


async def _worker_execute(
    client: LlmClient,
    registry: ToolRegistry,
    plan: ExecutionPlan,
    task: Task,
    cwd: str,
    memory: Any,
    code_index: Any,
    system_prompt: str,
    max_task_turns: int,
    feedback: str,
) -> str:
    text = ""
    async for event in query(
        client=client,
        registry=registry,
        system_prompt=system_prompt,
        user_message=_worker_message(plan, task, feedback),
        cwd=cwd,
        memory=memory,
        code_index=code_index,
        max_turns=max_task_turns,
    ):
        if event.get("type") == "text_delta":
            text += str(event.get("text") or "")
        elif event.get("type") == "error":
            return f"Error: {event['error']}"
    return text.strip() or "(no output)"


def _worker_message(plan: ExecutionPlan, task: Task, feedback: str) -> str:
    lines = [f"Overall goal: {plan.goal}", f"Your task: {task.description}"]
    if task.dependencies:
        lines.append("\nResults from prerequisite tasks:")
        for dep_id in task.dependencies:
            dep = plan.get(dep_id)
            if dep:
                lines.append(f"- [{dep.id}] {dep.description}: {dep.result}")
    if feedback:
        lines.append(f"\nA reviewer rejected your previous attempt. Fix this: {feedback}")
    lines.append("\nComplete this task concretely, using tools when needed.")
    return "\n".join(lines)


def _final_summary(plan: ExecutionPlan) -> str:
    lines = ["Team run complete. Task results:"]
    for task in plan.tasks.values():
        lines.append(f"- [{task.id}] {task.status.value}: {task.description}")
    return "\n".join(lines)
