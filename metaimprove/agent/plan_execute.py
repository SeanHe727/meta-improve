"""Plan-and-Execute agent.

A layer ABOVE the ReAct loop: it does NOT modify query(). Instead it
  1. asks a Planner to decompose the goal into a task DAG,
  2. executes tasks in dependency order (parallelizing independent ones),
     running EACH task through the existing ReAct loop (query()),
  3. summarizes the results.
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


async def plan_execute(
    *,
    client: LlmClient,
    registry: ToolRegistry,
    goal: str,
    cwd: str,
    memory: Any = None,
    code_index: Any = None,
    max_task_turns: int = 8,
) -> AsyncIterator[dict[str, Any]]:
    # 1. PLAN: ask the Planner to turn `goal` into a task DAG.
    yield {"type": "text_delta", "text": f"Planning: {goal}\n"}
    try:
        plan = await Planner(client).create_plan(goal)
    except Exception as exc:  # noqa: BLE001 - surface planning failure as an event
        yield {"type": "error", "error": exc}
        return
    yield {"type": "text_delta", "text": plan.summarize() + "\n\n"}

    # one shared system prompt for every task's ReAct run.
    system_prompt = PromptAssembler(
        cwd=cwd,
        model=client.model_name,
        provider=client.provider_name,
        tool_names=registry.list_names(),
    ).build()

    # 2. EXECUTE: loop until no task is runnable.
    while True:
        batch = plan.executable_tasks()
        if not batch:
            break
        if len(batch) > 1:
            yield {
                "type": "text_delta",
                "text": f"Running in parallel: {', '.join(t.id for t in batch)}\n",
            }
        # run the whole batch concurrently; independent tasks don't block each other.
        results = await asyncio.gather(
            *(
                _run_task(
                    client,
                    registry,
                    plan,
                    task,
                    cwd,
                    memory,
                    code_index,
                    system_prompt,
                    max_task_turns,
                )
                for task in batch
            )
        )
        for task in results:
            status = "failed" if task.status.value == "failed" else "done"
            yield {
                "type": "tool_result",
                "name": f"task:{task.id}",
                "content": task.result,
                "is_error": status == "failed",
            }

    # 3. SUMMARIZE: combine the completed task results.
    yield {"type": "text_delta", "text": "\n" + _final_summary(plan)}
    yield {"type": "done", "total_turns": 0, "total_tokens": 0, "messages": []}


async def _run_task(
    client: LlmClient,
    registry: ToolRegistry,
    plan: ExecutionPlan,
    task: Task,
    cwd: str,
    memory: Any,
    code_index: Any,
    system_prompt: str,
    max_task_turns: int,
) -> Task:
    task.mark_running()
    text = ""
    try:
        async for event in query(
            client=client,
            registry=registry,
            system_prompt=system_prompt,
            user_message=_task_message(plan, task),
            cwd=cwd,
            memory=memory,
            code_index=code_index,
            max_turns=max_task_turns,
        ):
            if event.get("type") == "text_delta":
                text += str(event.get("text") or "")
            elif event.get("type") == "error":
                raise event["error"]
        task.mark_completed(text.strip() or "(no output)")
    except Exception as exc:  # noqa: BLE001 - a task failure shouldn't crash the plan
        task.mark_failed(str(exc))
    return task


def _task_message(plan: ExecutionPlan, task: Task) -> str:
    # give the task its goal context plus the results of its dependencies.
    lines = [f"Overall goal: {plan.goal}", f"Your task: {task.description}"]
    if task.dependencies:
        lines.append("\nResults from prerequisite tasks:")
        for dep_id in task.dependencies:
            dep = plan.get(dep_id)
            if dep:
                lines.append(f"- [{dep.id}] {dep.description}: {dep.result}")
    lines.append("\nComplete this task concretely, using tools when needed.")
    return "\n".join(lines)


def _final_summary(plan: ExecutionPlan) -> str:
    lines = ["Plan complete. Task results:"]
    for task in plan.tasks.values():
        lines.append(f"- [{task.id}] {task.status.value}: {task.description}")
    return "\n".join(lines)
