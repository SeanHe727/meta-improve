# meta-improve

A **reliability-first, spec-driven, human-in-the-loop self-improving coding agent**.

meta-improve is a terminal coding agent (ReAct tool loop, memory, RAG, planning,
multi-agent, MCP, safety, a Runtime API) that can also **improve its own codebase
through a controlled, evidence-grounded pipeline** — not by unrestricted recursive
self-modification, but by turning source code, execution trajectories, evaluation
results, and user intent into a reviewable, measurable, reversible improvement.

## Why it's different

Most self-improving-agent work optimizes for *autonomy*. meta-improve optimizes for
**control and reliability**:

- **Evidence-grounded proposals** — every change is justified by an inspectable
  trajectory record, code location, or test, not by ungrounded reflection.
- **Evaluation defined *before* implementation** — acceptance criteria and an
  evaluation plan are frozen up front (test-driven), so "did it improve?" reduces to
  "do the pre-approved checks pass?".
- **A deterministic policy gate** — benefit/risk/effort/confidence and abstention are
  decided by fixed, explainable rules; the LLM's judgment is *evidence*, not the
  final authority.
- **Principled abstention** — the agent knows when *not* to change itself (weak
  evidence or low expected value → abstain).
- **Isolation + rollback** — changes are made in a Git worktree, gated by a command
  blacklist, path sandbox, HITL approval, and content-addressed snapshots.
- **Human review at the *spec* stage** — reviewers approve the *intent* (cheap)
  before any code is written.

See [`docs/self-improving-agent-design.md`](docs/self-improving-agent-design.md) for
the full pipeline design.

## The improvement pipeline

```
trajectory + source + evaluation + intent
        → Orchestrator (read-only analyst) → ImprovementProposal (RFC + evidence)
        → deterministic Policy Gate        → proceed / abstain / needs_human
        → human/agent review + freeze (hash)
        → Test Agent generates tests from acceptance criteria → baseline
        → Git worktree isolation
        → Writer implements + Critic reviews (bounded)
        → Evaluation Runner (before/after) + Verifier + Policy Gate
        → report; accept only if all required checks pass, else roll back
```

## The underlying agent

- OpenAI-compatible streaming LLM client with a unified event model.
- ReAct tool loop; built-in tools (read/write file, list/glob/grep, bash, code search).
- Long-term memory (SQLite), local code index (RAG), project instructions (`PAI.md`).
- Plan-and-Execute and Multi-Agent (Planner/Worker/Reviewer) modes.
- MCP client (connect external tool servers) + defence-in-depth safety
  (command guard, path sandbox, HITL, JSONL audit log).
- A FastAPI Runtime API (threads / turns / events).

## Quick start

```bash
uv sync --extra dev
export OPENAI_API_KEY=...        # or a provider-specific key
uv run python -m metaimprove.entrypoints.cli -p "read README.md and summarize it"
```

## Status

The base agent and the first half of the improvement pipeline (data contract,
trajectory persistence, read-only Orchestrator, deterministic policy gate,
content-addressed snapshots) are implemented. Git-worktree isolation, the
Writer-Critic loop, the Test Agent, and the evaluation runner are in progress.

## License

MIT.
