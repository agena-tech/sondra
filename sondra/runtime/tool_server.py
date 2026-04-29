from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib
import os
import signal
import sys
from types import SimpleNamespace
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ValidationError


SANDBOX_MODE = os.getenv("SONDRA_SANDBOX_MODE", "false").lower() == "true"
if not SANDBOX_MODE:
    raise RuntimeError("Tool server should only run in sandbox mode (SONDRA_SANDBOX_MODE=true)")

parser = argparse.ArgumentParser(description="Start Sondra tool server")
parser.add_argument("--token", required=True, help="Authentication token")
parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")  # nosec
parser.add_argument("--port", type=int, required=True, help="Port to bind to")
parser.add_argument(
    "--timeout",
    type=int,
    default=120,
    help="Hard timeout in seconds for each request execution (default: 120)",
)

args = parser.parse_args()
EXPECTED_TOKEN = args.token
REQUEST_TIMEOUT = args.timeout

app = FastAPI()
security = HTTPBearer()
security_dependency = Depends(security)

agent_tasks: dict[str, asyncio.Task[Any]] = {}


def verify_token(credentials: HTTPAuthorizationCredentials) -> str:
    if not credentials or credentials.scheme != "Bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication scheme. Bearer token required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if credentials.credentials != EXPECTED_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return credentials.credentials


class ToolExecutionRequest(BaseModel):
    agent_id: str
    tool_name: str
    kwargs: dict[str, Any]


class ToolExecutionResponse(BaseModel):
    result: Any | None = None
    error: str | None = None


async def _run_tool(agent_id: str, tool_name: str, kwargs: dict[str, Any]) -> Any:
    # Ensure tool package side-effects run in sandbox process so registry is populated.
    import sondra.tools  # noqa: F401
    from sondra.tools.argument_parser import convert_arguments
    from sondra.tools.context import set_current_agent_id
    from sondra.tools.registry import get_tool_by_name, needs_agent_state

    set_current_agent_id(agent_id)
    tool_name = (tool_name or "").strip()

    tool_func = get_tool_by_name(tool_name)
    if not tool_func:
        # Lazy-load known modules in case registry side-effects were skipped.
        fallback_modules = {
            "analyze_image": "sondra.tools.screen_vision.screen_vision_actions",
        }
        module_name = fallback_modules.get(tool_name)
        if module_name:
            with contextlib.suppress(Exception):
                importlib.import_module(module_name)
            tool_func = get_tool_by_name(tool_name)

    if not tool_func and tool_name == "analyze_image":
        # Final hard fallback: call screen_vision action directly
        with contextlib.suppress(Exception):
            from sondra.tools.screen_vision.screen_vision_actions import analyze_image as direct_tool

            tool_func = direct_tool

    if not tool_func:
        raise ValueError(f"Tool '{tool_name}' not found")

    converted_kwargs = convert_arguments(tool_func, kwargs)
    if needs_agent_state(tool_name):
        # Sandbox request payload doesn't carry full agent_state; provide minimal context.
        minimal_agent_state = SimpleNamespace(
            agent_id=agent_id,
            context={"scan_mode": os.getenv("SONDRA_SCAN_MODE", "general")},
        )
        return await asyncio.to_thread(tool_func, minimal_agent_state, **converted_kwargs)

    return await asyncio.to_thread(tool_func, **converted_kwargs)


@app.post("/execute", response_model=ToolExecutionResponse)
async def execute_tool(
    request: ToolExecutionRequest, credentials: HTTPAuthorizationCredentials = security_dependency
) -> ToolExecutionResponse:
    verify_token(credentials)

    agent_id = request.agent_id

    if agent_id in agent_tasks:
        old_task = agent_tasks[agent_id]
        if not old_task.done():
            old_task.cancel()

    task = asyncio.create_task(
        asyncio.wait_for(
            _run_tool(agent_id, request.tool_name, request.kwargs), timeout=REQUEST_TIMEOUT
        )
    )
    agent_tasks[agent_id] = task

    try:
        result = await task
        return ToolExecutionResponse(result=result)

    except asyncio.CancelledError:
        return ToolExecutionResponse(error="Cancelled by newer request")

    except TimeoutError:
        return ToolExecutionResponse(error=f"Tool timed out after {REQUEST_TIMEOUT}s")

    except ValidationError as e:
        return ToolExecutionResponse(error=f"Invalid arguments: {e}")

    except (ValueError, RuntimeError, ImportError) as e:
        return ToolExecutionResponse(error=f"Tool execution error: {e}")

    except Exception as e:  # noqa: BLE001
        return ToolExecutionResponse(error=f"Unexpected error: {e}")

    finally:
        if agent_tasks.get(agent_id) is task:
            del agent_tasks[agent_id]


@app.post("/register_agent")
async def register_agent(
    agent_id: str, credentials: HTTPAuthorizationCredentials = security_dependency
) -> dict[str, str]:
    verify_token(credentials)
    return {"status": "registered", "agent_id": agent_id}


@app.get("/health")
async def health_check() -> dict[str, Any]:
    return {
        "status": "healthy",
        "sandbox_mode": str(SANDBOX_MODE),
        "environment": "sandbox" if SANDBOX_MODE else "main",
        "auth_configured": "true" if EXPECTED_TOKEN else "false",
        "active_agents": len(agent_tasks),
        "agents": list(agent_tasks.keys()),
    }


def signal_handler(_signum: int, _frame: Any) -> None:
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_IGN)
    for task in agent_tasks.values():
        task.cancel()
    sys.exit(0)


if hasattr(signal, "SIGPIPE"):
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
