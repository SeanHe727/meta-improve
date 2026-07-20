"""Improvement Orchestrator: the analyst that produces an ImprovementProposal.

Reuses the ReAct loop (query) with a READ-ONLY tool subset so the model can
investigate the actual source, grounded by the execution trajectory and the
user's intent. It does NOT write code (that's the Writer). Its final message is
the ImprovementProposal as JSON, validated via Pydantic — with one self-correcting
repair retry if the first JSON doesn't satisfy the schema.

Kept separate from Planner on purpose: Planner decomposes a plain goal for the
common /plan and /team execution modes; the Orchestrator is the richer,
evidence-grounded planner for the self-improvement pipeline.
"""

from __future__ import annotations

import json
from typing import Any

from ..agent.query import query
from ..llm.base import LlmClient
from ..llm.collect import collect_text
from ..llm.parse import parse_json_model
from ..tools.registry import ToolRegistry
from ..types import Message
from .models import ImprovementProposal

ORCHESTRATOR_PROMPT = """You are the improvement Orchestrator: an analyst, not an implementer.

Given a user's improvement intent, an execution trajectory (evidence of how the agent
actually behaved), and READ-ONLY access to the codebase, investigate and produce an
evidence-grounded ImprovementProposal.

Rules:
- Use the read-only tools (read_file, search_code, grep, list_dir, glob) to inspect the
  REAL source before proposing. Do not guess.
- Ground every claim in inspectable evidence: a code location, a trajectory record, or a test.
- You must NOT write or modify any code.
- Score benefit/risk/effort 1-5 and confidence 0.0-1.0. Propose a decision
  (proceed / abstain / needs_human); a deterministic policy gate makes the final call.
- Prefer abstain when evidence is weak or expected value is low.

When done, output ONLY the proposal as ONE JSON object in a ```json code block. Field types:
- evidence[]: {source_type: trajectory|test|benchmark|code|log, reference: str, observation: str}
- tasks[]: {id: str, description: str, dependencies: [str], acceptance_criteria_ids: [str]}
- benefit/risk/effort: int 1-5; confidence: float 0-1
- decision: proceed|abstain|needs_human
- acceptance_criteria[]: {id, description, mode: red_green|invariant|metric_improvement|
  non_regression|manual, verification_method: existing_test|generated_test|static_check|
  benchmark|manual, required: bool}
- evaluation_plan: {primary_metric: null OR {name, direction: increase|decrease, min_delta: float},
  guardrail_metrics: [Metric], scenarios: [str], run_count: int}
- also: summary, problem_statement, goals[], non_goals[], affected_components[],
  dependencies[], decision_reason, rollback_plan, alternatives_considered[]
Use null for primary_metric and [] for guardrail_metrics unless the change is metric-based."""


class Orchestrator:
    def __init__(self, client: LlmClient, registry: ToolRegistry, cwd: str):
        self.client = client
        self.cwd = cwd
        # analyst gets read-only tools ONLY — it must never modify code.
        self.registry = _read_only_registry(registry)

    async def analyze(
        self, intent: str, trajectory: list[dict[str, Any]], max_turns: int = 12
    ) -> ImprovementProposal:
        # 1. run the ReAct loop; collect its final text (the proposal JSON).
        text = await self._investigate(intent, trajectory, max_turns)
        # 2. validate; on schema failure, do ONE self-correcting repair pass.
        try:
            return parse_json_model(text, ImprovementProposal)
        except ValueError as exc:
            fixed = await self._repair(text, str(exc))
            return parse_json_model(fixed, ImprovementProposal)

    async def _investigate(
        self, intent: str, trajectory: list[dict[str, Any]], max_turns: int
    ) -> str:
        text = ""
        async for event in query(
            client=self.client,
            registry=self.registry,
            system_prompt=ORCHESTRATOR_PROMPT,
            user_message=_build_request(intent, trajectory),
            cwd=self.cwd,
            max_turns=max_turns,
        ):
            if event.get("type") == "text_delta":
                text += str(event.get("text") or "")
            elif event.get("type") == "error":
                raise event["error"]
        return text

    async def _repair(self, bad_output: str, error: str) -> str:
        # feed the model its own output + the validation error; ask for corrected JSON.
        msg = (
            f"Your JSON did not satisfy the schema:\n{error}\n\n"
            f"Your previous output was:\n{bad_output}\n\n"
            "Output ONLY the corrected proposal as a single ```json block that matches "
            "the schema exactly (fix the field types shown in the error)."
        )
        return await collect_text(
            self.client, [Message(role="user", content=msg)], system_prompt=ORCHESTRATOR_PROMPT
        )


def _read_only_registry(registry: ToolRegistry) -> ToolRegistry:
    read_only = ToolRegistry()
    read_only.register_all(
        [t for name in registry.list_names() if (t := registry.get(name)) and t.is_read_only]
    )
    return read_only


def _build_request(intent: str, trajectory: list[dict[str, Any]]) -> str:
    lines = [f"Improvement intent:\n{intent}", "", "Execution trajectory (evidence records):"]
    if trajectory:
        lines += [f"- {json.dumps(rec, ensure_ascii=False)}" for rec in trajectory[:100]]
    else:
        lines.append("- (no trajectory provided)")
    lines.append(
        "\nInvestigate the codebase with the read-only tools, then output the "
        "ImprovementProposal JSON."
    )
    return "\n".join(lines)
