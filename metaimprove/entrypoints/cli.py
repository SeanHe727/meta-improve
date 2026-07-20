"""
Interface between CLI and client
CLI -> Create Client -> Use Client
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Annotated

import typer

from ..agent.orchestrator import multi_agent
from ..agent.plan_execute import plan_execute
from ..agent.query import query
from ..config import load_config
from ..llm.openai_compatible import OpenAICompatibleClient
from ..mcp.client import McpClientManager
from ..memory.manager import MemoryManager
from ..prompt.assembler import PromptAssembler
from ..rag.code_index import CodeIndex
from ..tools.builtins import get_builtin_tools
from ..tools.registry import ToolRegistry

# Long-term memory and the code index live in SQLite files under the user's home.
_MEMORY_DB = Path.home() / ".meta-improve" / "memory.db"
_CODE_INDEX_DB = Path.home() / ".meta-improve" / "code_index.db"

app = typer.Typer(
    name="meta-improve",
    help="meta-improve - Terminal AI Agent in Python",
    invoke_without_command=True,
    no_args_is_help=False,
)


# register the main entrypoints
@app.callback()
def main(
    ctx: typer.Context,
    prompt: Annotated[
        str | None, typer.Option("-p", "--prompt", help="use -p/--prompt to start")
    ] = None,
    model: Annotated[
        str | None, typer.Option("-m", "--model", help="use -m/--model to start")
    ] = None,
    provider: Annotated[str | None, typer.Option("--provider", help="Override provider")] = None,
    cwd: Annotated[Path | None, typer.Option("--cwd", help="Working directory")] = None,
    plan: Annotated[
        bool, typer.Option("--plan", help="Use Plan-and-Execute mode instead of ReAct")
    ] = False,
    team: Annotated[
        bool, typer.Option("--team", help="Use Multi-Agent (Planner/Worker/Reviewer) mode")
    ] = False,
    yes: Annotated[
        bool, typer.Option("--yes", help="Auto-approve dangerous tools (skip HITL prompts)")
    ] = False,
) -> None:
    # if a subcommand (e.g. `serve`) was invoked, let it handle everything.
    if ctx.invoked_subcommand is not None:
        return
    # resolve working directory and load the merged config (CLI overrides applied).
    root = str((cwd or Path.cwd()).resolve())
    config = load_config(cli_provider=provider, cli_model=model)

    # generate response, calling run_prompt() in one-shot mode. No REPL yet.
    if prompt is not None:
        asyncio.run(run_prompt(prompt, config, root, use_plan=plan, use_team=team, auto_yes=yes))
    else:
        typer.echo('Interactive REPL is not implemented yet. Use -p "your prompt".')


def _make_approval_callback(auto_yes: bool):
    # HITL: ask the user to approve a dangerous tool before it runs.
    def approve(info: dict) -> bool:
        if auto_yes:
            return True
        typer.echo(f"\n Agent wants to run '{info['name']}' with: {info['args']}", err=True)
        answer = input("Approve? [y/N] ").strip().lower()
        return answer in ("y", "yes")

    return approve


async def run_prompt(
    prompt: str,
    config,
    cwd: str,
    use_plan: bool = False,
    use_team: bool = False,
    auto_yes: bool = False,
) -> None:
    # fail fast if there is no API key.
    if not config.api_key:
        typer.echo("Fatal error: API key is not configured.", err=True)
        raise typer.Exit(1)

    # build client from config.
    client = OpenAICompatibleClient(
        provider_name=config.provider,
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
    )

    # register built-in tools so the agent can act, not just chat.
    registry = ToolRegistry()
    registry.register_all(get_builtin_tools())

    # discover + register external MCP tools from .meta-improve/mcp.json (if any).
    mcp_manager = McpClientManager(cwd)
    registry.register_all(await mcp_manager.load_tools())
    for name, err in mcp_manager.last_errors.items():
        typer.echo(f"[mcp] server '{name}' failed to load: {err}", err=True)

    # long-term memory, scoped to this project (cwd).
    memory = MemoryManager(_MEMORY_DB, scope=cwd)

    # code index for this project; rebuild at startup so search_code is fresh.
    # (real tools rebuild incrementally / on demand; full rebuild is fine here.)
    code_index = CodeIndex(root=cwd, db_path=_CODE_INDEX_DB)
    code_index.rebuild()

    # build structured prompt; tool names + recalled memories go in for grounding.
    system_prompt = PromptAssembler(
        cwd=cwd,
        model=config.model,
        provider=config.provider,
        tool_names=registry.list_names(),
        memories=[m.content for m in memory.list_memory()],  # inject memory into system_prompt
    ).build()

    # pick the mode: Multi-Agent (--team), Plan-and-Execute (--plan), or ReAct.
    if use_team:
        events = multi_agent(
            client=client,
            registry=registry,
            goal=prompt,
            cwd=cwd,
            memory=memory,
            code_index=code_index,
        )
    elif use_plan:
        events = plan_execute(
            client=client,
            registry=registry,
            goal=prompt,
            cwd=cwd,
            memory=memory,
            code_index=code_index,
        )
    else:
        events = query(
            client=client,
            registry=registry,
            system_prompt=system_prompt,
            user_message=prompt,
            cwd=cwd,
            memory=memory,
            code_index=code_index,
            approval_callback=_make_approval_callback(auto_yes),
        )

    # consume the event stream: stream text, note tool/task use, report errors.
    async for event in events:
        etype = event["type"]
        if etype == "text_delta":
            print(event["text"], end="", flush=True)
        elif etype == "tool_result":
            status = "error" if event["is_error"] else "ok"
            typer.echo(f"\n[{event['name']} -> {status}]", err=True)
        elif etype == "error":
            typer.echo(f"\nFatal error: {event['error']}", err=True)
            raise typer.Exit(1)
    print()  # final newline after the streamed reply


@app.command("serve")
def serve(
    port: Annotated[int, typer.Option("--port", help="HTTP port")] = 8080,
    host: Annotated[str, typer.Option("--host", help="Bind host")] = "127.0.0.1",
    cwd: Annotated[Path | None, typer.Option("--cwd", help="Working directory")] = None,
) -> None:
    """Start the Runtime API HTTP server."""
    import uvicorn

    from ..runtime.api import create_app
    from ..runtime.state import RuntimeState

    root = str((cwd or Path.cwd()).resolve())  # cwd
    config = load_config()
    if not config.api_key:
        typer.echo("Fatal error: model API key is not configured.", err=True)
        raise typer.Exit(1)
    # the Runtime API's own key (clients must present this), separate from the model key.
    api_key = os.getenv("METAIMPROVE_RUNTIME_API_KEY", "dev-key")

    state = RuntimeState(config, root, api_key)
    fastapi_app = create_app(state)  # create application

    typer.echo(f"meta-improve Runtime API on http://{host}:{port} (x-api-key: {api_key})")
    uvicorn.run(fastapi_app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    app()
