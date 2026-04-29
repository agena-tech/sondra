import asyncio
import contextlib
import json
import logging
import random
import re
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional
from urllib.parse import quote_plus, urlparse

if TYPE_CHECKING:
    from sondra.telemetry.tracer import Tracer

from jinja2 import (
    Environment,
    FileSystemLoader,
    select_autoescape,
)
from litellm import acompletion

from sondra.llm import LLM, LLMConfig, LLMRequestFailedError
from sondra.llm.utils import clean_content
from sondra.cognition.decision_engine import DecisionEngine
from sondra.cognition.metacognition_lite import MetacognitionLite
from sondra.memory import MemoryIndexManager, MemoryRuntime, PersistentMemoryStore
from sondra.memory.signal_catalog import get_memory_signal_catalog, normalize_signal_text
from sondra.memory.profile_manager import extract_profile_fact as extract_profile_fact_impl
from sondra.memory.semantic_processor import compute_initial_importance as compute_initial_importance_impl
from sondra.memory.semantic_processor import extract_semantic as extract_semantic_impl
from sondra.memory.semantic_processor import is_valid_semantic_item as is_valid_semantic_item_impl
from sondra.memory.task_manager import update_task_state as update_task_state_impl
from sondra.memory.conversation_memory import (
    build_memory_status_lines as build_memory_status_lines_impl,
    extract_memory_source_id as extract_memory_source_id_impl,
    fallback_memory_rows_from_recent as fallback_memory_rows_from_recent_impl,
    memory_get as memory_get_impl,
    memory_search as memory_search_impl,
    persist_general_messages_to_disk as persist_general_messages_to_disk_impl,
    search_memory_rows as search_memory_rows_impl,
)
from sondra.runtime import SandboxInitializationError
from sondra.tools import get_tool_by_name, get_tool_names, process_tool_invocations
from sondra.utils.resource_paths import get_sondra_resource_path

from .state import AgentState


logger = logging.getLogger(__name__)


class AgentMeta(type):
    agent_name: str
    jinja_env: Environment

    def __new__(cls, name: str, bases: tuple[type, ...], attrs: dict[str, Any]) -> type:
        new_cls = super().__new__(cls, name, bases, attrs)

        if name == "BaseAgent":
            return new_cls

        prompt_dir = get_sondra_resource_path("agents", name)

        new_cls.agent_name = name
        new_cls.jinja_env = Environment(
            loader=FileSystemLoader(prompt_dir),
            autoescape=select_autoescape(enabled_extensions=(), default_for_string=False),
        )

        return new_cls


class BaseAgent(metaclass=AgentMeta):
    max_iterations = 300
    agent_name: str = ""
    jinja_env: Environment
    default_llm_config: LLMConfig | None = None
    MEMORY_CONTEXT_MARKER: str = "[MEMORY CONTEXT]"
    MEMORY_CONTEXT_END_MARKER: str = "[END MEMORY CONTEXT]"
    DECISION_CONTEXT_MARKER: str = "[DECISION CONTEXT]"
    LEGACY_MEMORY_CONTEXT_HEADER: str = "--- MEMORY CONTEXT ---"
    VISUAL_TOOL_NAMES: tuple[str, ...] = ("analyze_image",)
    SCHEDULE_WAITING_MESSAGE: str = (
        "\no waiting\n"
        "Waiting for the scheduled time."
    )
    PERSISTENT_MEMORY_DELETE_COMMAND: str = "DELETE_PERSISTENT_MEM"
    PERSISTENT_MEMORY_DELETING_MESSAGE: str = "💿 Persistent memory is being deleted..."
    PERSISTENT_MEMORY_RESET_MESSAGE: str = "💿 Persistent memory has been reset!"
    PERSISTENT_MEMORY_READING_MESSAGE: str = "💿 LLM reading memory ..."
    RESPONSE_NOISE_SNIPPETS: tuple[str, ...] = (
        "Execution was cancelled. I'm now waiting for new instructions.",
        "Execution was cancelled.",
        "I'm now waiting for new instructions.",
        "<tool_response><eos>",
        "<tool_response>",
        "<eos>",
        "<channel|>",
    )
    MAX_MEMORY_CALLS_PER_TURN: int = 2
    MAX_MEMORY_RESULT_CHARS: int = 500
    MEMORY_TOOL_TIMEOUT_SEC: int = 2
    MEMORY_UPDATE_DUPLICATE_WINDOW_SEC: float = 1.5
    LOCAL_HISTORY_PROMPT_BUDGET_CHARS: int = 3000
    DEFAULT_HISTORY_PROMPT_BUDGET_CHARS: int = 12000
    AGENT_EMOTION_DEFAULTS: dict[str, float] = {
        "happiness": 12.0,
        "sadness": 4.0,
        "stress": 6.0,
        "neutral": 50.0,
    }
    AGENT_EMOTION_BASELINE: dict[str, float] = {
        "happiness": 0.0,
        "sadness": 0.0,
        "stress": 0.0,
        "neutral": 50.0,
    }
    AGENT_EMOTION_DECAY: float = 3.5
    AGENT_EMOTION_INERTIA: float = 0.88
    EPISODIC_IMPORTANT_ACTIONS: tuple[str, ...] = (
        "memory_search",
        "memory_get",
        "process_tool_invocations",
    )

    OLLAMA_FORCED_TOOL_RETRY_LIMIT: int = 2
    RECURRING_HINTS: tuple[str, ...] = (
        "surekli",
        "sürekli",
        "devamli",
        "devamlı",
        "tekrar tekrar",
        "tekrarla",
        "tekrarlı",
        "durmadan",
        "repeat",
        "repeatedly",
        "repeatly",
        "again and again",
        "continuously",
        "continually",
        "keep repeating",
        "keep doing",
        "ongoing",
        "recurring",
    )
    RECURRING_TIME_PATTERN: re.Pattern[str] = re.compile(
        r"\b(?:(?:saat\s*)?her|every(?: day)?(?: at)?)\s*(?P<hour>\d{1,2})[:.](?P<minute>\d{2})(?:\s*[']?(?:te|de|da|ta))?\b",
        flags=re.IGNORECASE,
    )
    RELATIVE_DELAY_PATTERN: re.Pattern[str] = re.compile(
        r"\b(?P<amount>\d{1,4})\s*(?P<unit>saniye|sn|second|seconds|sec|dakika|dk|minute|minutes)\s+sonra\b",
        flags=re.IGNORECASE,
    )
    RECURRING_INTERVAL_PATTERN: re.Pattern[str] = re.compile(
        r"\b(?:her|every)\s*(?P<amount>\d{1,4})\s*(?P<unit>saniye(?:de)?|sn|second|seconds|sec|dakika(?:da)?|dk|minute|minutes)(?:\s*(?:da|de))?\s*(?:bir)?\b",
        flags=re.IGNORECASE,
    )
    SCHEDULE_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(
            r"\b(?P<day>yarın|yarin|tomorrow)\s*(?:saat\s*)?(?P<hour>\d{1,2})[:.](?P<minute>\d{2})(?:\s*[']?(?:te|de|da|ta))?\b",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"\b(?P<day>bugün|bugun|today)\s*(?:saat\s*)?(?P<hour>\d{1,2})[:.](?P<minute>\d{2})(?:\s*[']?(?:te|de|da|ta))?\b",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"\bsaat\s*(?P<hour>\d{1,2})[:.](?P<minute>\d{2})(?:\s*[']?(?:te|de|da|ta))?\b",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"\b(?P<hour>\d{1,2})[:.](?P<minute>\d{2})\s*[']?(?P<suffix>te|de|da|ta)\b",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"\b(?P<day>today|tomorrow)\s+at\s*(?P<hour>\d{1,2})[:.](?P<minute>\d{2})\b",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"\bat\s*(?P<hour>\d{1,2})[:.](?P<minute>\d{2})\b",
            flags=re.IGNORECASE,
        ),
    )

    def _build_runtime_tools(self) -> dict[str, Any]:
        tools_map: dict[str, Any] = {}
        for tool_name in get_tool_names():
            name = str(tool_name or "").strip()
            if not name:
                continue
            tool_func = get_tool_by_name(name)
            if callable(tool_func):
                tools_map[name] = tool_func
        return tools_map

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.signal_catalog = get_memory_signal_catalog()

        self.local_sources = config.get("local_sources", [])
        self.non_interactive = config.get("non_interactive", False)

        if "max_iterations" in config:
            self.max_iterations = config["max_iterations"]

        self.llm_config_name = config.get("llm_config_name", "default")
        self.llm_config = config.get("llm_config", self.default_llm_config)
        if self.llm_config is None:
            raise ValueError("llm_config is required but not provided")
        state_from_config = config.get("state")
        if state_from_config is not None:
            self.state = state_from_config
        else:
            self.state = AgentState(
                agent_name="Root Agent",
                max_iterations=self.max_iterations,
            )

        self.llm = LLM(self.llm_config, agent_name=self.agent_name)
        self.state.update_context("scan_mode", str(getattr(self.llm_config, "scan_mode", "")))
        self.memory_runtime = MemoryRuntime()
        self.memory_store: PersistentMemoryStore | None = None
        self.memory_index_manager: MemoryIndexManager | None = None
        self.episodic = self.memory_runtime.episodic
        if self._is_general_root_agent():
            try:
                self.memory_runtime.initialize(
                    model=getattr(self.llm_config, "litellm_model", None),
                    sync_limit=300,
                )
                self._sync_memory_runtime_handles()
                self.state.update_context("memory_last_persisted_index", 0)
                self.state.update_context(
                    "conversation_session_id",
                    str(self.state.context.get("session_id", self.state.agent_id)),
                )
                self._ensure_general_prepare_hook()
                self._restore_last_emotion_snapshot()
            except Exception as e:
                logger.exception("General memory/identity initialization failed: %s", e)
                self.memory_runtime.clear_persistent_handles()
                self._sync_memory_runtime_handles()

        try:
            self.llm.set_agent_identity(self.state.agent_name, self.state.agent_id)
        except Exception as e:
            logger.exception("General memory/identity initialization failed: %s", e)
            self.memory_runtime.clear_persistent_handles()
            self._sync_memory_runtime_handles()
        self._current_task: asyncio.Task[Any] | None = None
        self._task_runner_task: asyncio.Task[Any] | None = None
        self._force_stop = False
        self._reported_scheduled_agent_ids: set[str] = set()
        self._scheduled_task_name_seq: int = 0
        self._scheduled_task_name_map: dict[int, int] = {}
        self._scheduled_task_worker_map: dict[str, int] = {}
        self._memory_update_task: asyncio.Task[Any] | None = None
        self._pending_memory_update_inputs: list[str] = []
        self._active_memory_update_input: str = ""
        self._last_memory_update_signature: str = ""
        self._last_memory_update_at: float = 0.0
        self._memory_calls_this_turn: int = 0
        self._memory_queries_this_turn: set[str] = set()
        self._memory_sync_turn_key: str = ""
        self._last_memory_tool_query: str = ""
        self._seen_memory_for_turn: set[str] = set()
        self._last_memory_hits: list[dict[str, Any]] = []
        self._last_memory_query: str = ""
        self.decision_engine = DecisionEngine()
        self.metacognition_lite = MetacognitionLite()
        self.TOOLS: dict[str, Any] = self._build_runtime_tools()
        if self._is_general_root_agent():
            self.TOOLS["memory_search"] = self.memory_search
            self.TOOLS["memory_get"] = self.memory_get

        from sondra.telemetry.tracer import get_global_tracer

        tracer = get_global_tracer()
        if tracer:
            tracer.log_agent_creation(
                agent_id=self.state.agent_id,
                name=self.state.agent_name,
                task=self.state.task,
                parent_id=self.state.parent_id,
            )
            if self.state.parent_id is None:
                scan_config = tracer.scan_config or {}
                exec_id = tracer.log_tool_execution_start(
                    agent_id=self.state.agent_id,
                    tool_name="scan_start_info",
                    args=scan_config,
                )
                tracer.update_tool_execution(execution_id=exec_id, status="completed", result={})

            else:
                exec_id = tracer.log_tool_execution_start(
                    agent_id=self.state.agent_id,
                    tool_name="subagent_start_info",
                    args={
                        "name": self.state.agent_name,
                        "task": self.state.task,
                        "parent_id": self.state.parent_id,
                    },
                )
                tracer.update_tool_execution(execution_id=exec_id, status="completed", result={})

        self._add_to_agents_graph()

    def _signal_list(self, file_name: str, *path: str) -> list[Any]:
        return self.signal_catalog.get_list(file_name, *path)

    def _signal_map(self, file_name: str, *path: str) -> dict[str, Any]:
        return self.signal_catalog.get_mapping(file_name, *path)

    def _signal_value(self, file_name: str, *path: str, default: Any = None) -> Any:
        return self.signal_catalog.get_value(file_name, *path, default=default)

    def _add_to_agents_graph(self) -> None:
        from sondra.tools.agents_graph import agents_graph_actions

        node = {
            "id": self.state.agent_id,
            "name": self.state.agent_name,
            "task": self.state.task,
            "status": "running",
            "parent_id": self.state.parent_id,
            "created_at": self.state.start_time,
            "finished_at": None,
            "result": None,
            "llm_config": self.llm_config_name,
            "agent_type": self.__class__.__name__,
            "state": self.state.model_dump(),
        }
        agents_graph_actions._agent_graph["nodes"][self.state.agent_id] = node

        agents_graph_actions._agent_instances[self.state.agent_id] = self
        agents_graph_actions._agent_states[self.state.agent_id] = self.state

        if self.state.parent_id:
            agents_graph_actions._agent_graph["edges"].append(
                {"from": self.state.parent_id, "to": self.state.agent_id, "type": "delegation"}
            )

        if self.state.agent_id not in agents_graph_actions._agent_messages:
            agents_graph_actions._agent_messages[self.state.agent_id] = []

        if self.state.parent_id is None and agents_graph_actions._root_agent_id is None:
            agents_graph_actions._root_agent_id = self.state.agent_id

    async def agent_loop(self, task: str) -> dict[str, Any]:  # noqa: PLR0912, PLR0915
        from sondra.telemetry.tracer import get_global_tracer

        tracer = get_global_tracer()

        try:
            await self._initialize_sandbox_and_state(task)
        except SandboxInitializationError as e:
            return self._handle_sandbox_error(e, tracer)

        while True:
            if self._force_stop:
                self._force_stop = False
                await self._enter_waiting_state(tracer, was_cancelled=True)
                continue

            await self._check_agent_messages(self.state)

            if self.state.is_waiting_for_input():
                await self._wait_for_input()
                continue

            if self.state.should_stop():
                if self.non_interactive:
                    return self.state.final_result or {}
                await self._enter_waiting_state(tracer)
                continue

            if self.state.llm_failed:
                await self._wait_for_input()
                continue

            self.state.increment_iteration()

            if (
                self.state.is_approaching_max_iterations()
                and not self.state.max_iterations_warning_sent
            ):
                self.state.max_iterations_warning_sent = True
                remaining = self.state.max_iterations - self.state.iteration
                warning_msg = (
                    f"URGENT: You are approaching the maximum iteration limit. "
                    f"Current: {self.state.iteration}/{self.state.max_iterations} "
                    f"({remaining} iterations remaining). "
                    f"Please prioritize completing your required task(s) and calling "
                    f"the appropriate finish tool (finish_scan for root agent, "
                    f"agent_finish for sub-agents) as soon as possible."
                )
                self.state.add_message("user", warning_msg)

            if self.state.iteration == self.state.max_iterations - 3:
                final_warning_msg = (
                    "CRITICAL: You have only 3 iterations left! "
                    "Your next message MUST be the tool call to the appropriate "
                    "finish tool: finish_scan if you are the root agent, or "
                    "agent_finish if you are a sub-agent. "
                    "No other actions should be taken except finishing your work "
                    "immediately."
                )
                self.state.add_message("user", final_warning_msg)

            try:
                iteration_task = asyncio.create_task(self._process_iteration(tracer))
                self._current_task = iteration_task
                should_finish = await iteration_task
                self._current_task = None
                self._reinforce_used_semantic_memory()
                self._episodic_to_semantic()

                if should_finish is None and not self.non_interactive:
                    await self._enter_waiting_state(tracer, text_response=True)
                    continue

                if should_finish:
                    if self.non_interactive:
                        self.state.set_completed({"success": True})
                        if tracer:
                            tracer.update_agent_status(self.state.agent_id, "completed")
                        return self.state.final_result or {}
                    await self._enter_waiting_state(tracer, task_completed=True)
                    continue

            except asyncio.CancelledError:
                self._current_task = None
                if tracer:
                    partial_content = tracer.finalize_streaming_as_interrupted(self.state.agent_id)
                    if partial_content and partial_content.strip():
                        self.state.add_message(
                            "assistant", f"{partial_content}\n\n[ABORTED BY USER]"
                        )
                if self.non_interactive:
                    raise
                await self._enter_waiting_state(tracer, error_occurred=False, was_cancelled=True)
                continue

            except LLMRequestFailedError as e:
                result = self._handle_llm_error(e, tracer)
                if result is not None:
                    return result
                continue

            except (RuntimeError, ValueError, TypeError) as e:
                if not await self._handle_iteration_error(e, tracer):
                    if self.non_interactive:
                        self.state.set_completed({"success": False, "error": str(e)})
                        if tracer:
                            tracer.update_agent_status(self.state.agent_id, "failed")
                        raise
                    await self._enter_waiting_state(tracer, error_occurred=True)
                    continue

    async def _wait_for_input(self) -> None:
        if self._force_stop:
            return

        if self.state.has_waiting_timeout():
            self.state.resume_from_waiting()
            self.state.add_message("user", "Waiting timeout reached. Resuming execution.")

            from sondra.telemetry.tracer import get_global_tracer

            tracer = get_global_tracer()
            if tracer:
                tracer.update_agent_status(self.state.agent_id, "running")

            try:
                from sondra.tools.agents_graph.agents_graph_actions import _agent_graph

                if self.state.agent_id in _agent_graph["nodes"]:
                    _agent_graph["nodes"][self.state.agent_id]["status"] = "running"
            except (ImportError, KeyError):
                pass

            return

        await asyncio.sleep(0.5)

    async def _enter_waiting_state(
        self,
        tracer: Optional["Tracer"],
        task_completed: bool = False,
        error_occurred: bool = False,
        was_cancelled: bool = False,
        text_response: bool = False,
    ) -> None:
        self.state.enter_waiting_state()

        if tracer:
            if text_response:
                tracer.update_agent_status(self.state.agent_id, "waiting_for_input")
            elif task_completed:
                tracer.update_agent_status(self.state.agent_id, "completed")
            elif error_occurred:
                tracer.update_agent_status(self.state.agent_id, "error")
            elif was_cancelled:
                tracer.update_agent_status(self.state.agent_id, "stopped")
            else:
                tracer.update_agent_status(self.state.agent_id, "stopped")

        if text_response:
            return

        if task_completed:
            self.state.add_message(
                "assistant",
                "Task completed. I'm now waiting for follow-up instructions or new tasks.",
            )
        elif error_occurred:
            self.state.add_message(
                "assistant", "An error occurred. I'm now waiting for new instructions."
            )
        elif was_cancelled:
            self.state.add_message(
                "assistant", "Execution was cancelled. I'm now waiting for new instructions."
            )
        else:
            self.state.add_message(
                "assistant",
                "Execution paused. I'm now waiting for new instructions or any updates.",
            )

    async def _initialize_sandbox_and_state(self, task: str) -> None:
        import os

        sandbox_mode = os.getenv("SONDRA_SANDBOX_MODE", "false").lower() == "true"
        if not sandbox_mode and self.state.sandbox_id is None:
            from sondra.runtime import get_runtime

            try:
                runtime = get_runtime()
                sandbox_info = await runtime.create_sandbox(
                    self.state.agent_id, self.state.sandbox_token, self.local_sources
                )
                self.state.sandbox_id = sandbox_info["workspace_id"]
                self.state.sandbox_token = sandbox_info["auth_token"]
                self.state.sandbox_info = sandbox_info

                if "agent_id" in sandbox_info:
                    self.state.sandbox_info["agent_id"] = sandbox_info["agent_id"]

                caido_port = sandbox_info.get("caido_port")
                if caido_port:
                    from sondra.telemetry.tracer import get_global_tracer

                    tracer = get_global_tracer()
                    if tracer:
                        tracer.caido_url = f"localhost:{caido_port}"
            except Exception as e:
                from sondra.telemetry import posthog

                posthog.error("sandbox_init_error", str(e))
                raise

        is_general_idle_boot = self.llm_config.scan_mode == "general" and (
            not task.strip() or task.strip() == "__GENERAL_IDLE__"
        )

        if is_general_idle_boot:
            if not self.state.context.get("general_ready_shown", False):
                model_name = self._format_ready_model_name()
                self.state.add_message("assistant", f"Initializing ...\n{model_name} is ready.")
                self.state.update_context("general_ready_shown", True)
            self.state.enter_waiting_state()
            with contextlib.suppress(Exception):
                from sondra.telemetry.tracer import get_global_tracer

                tracer = get_global_tracer()
                if tracer:
                    tracer.update_agent_status(self.state.agent_id, "waiting")
            with contextlib.suppress(Exception):
                from sondra.tools.agents_graph.agents_graph_actions import _agent_graph

                if self.state.agent_id in _agent_graph["nodes"]:
                    _agent_graph["nodes"][self.state.agent_id]["status"] = "waiting"
            self._ensure_task_runner_started()
            return

        if not self.state.task:
            self.state.task = task

        if self._is_general_root_agent() and self._is_persistent_memory_reset_command(task):
            if self._handle_persistent_memory_reset_command():
                self.state.enter_waiting_state()
                return

        self.state.add_message("user", task)
        self.state.update_context("last_user_turn_raw", task)
        self.state.update_context(
            "last_user_message_event_id",
            f"init_msg::{self.state.agent_id}::{self.state.start_time}",
        )
        self.state.update_context("memory_reply_sent_for_turn", "")
        self.state.update_context("visual_screenshot_attempted_for_user", "")
        self.state.update_context("context_overflow_active", False)
        self.state.update_context("context_overflow_retry_count", 0)
        self.state.update_context("forced_tool_name", "")
        self.state.update_context("forced_tool_retry_count", 0)
        self._memory_calls_this_turn = 0
        self._memory_queries_this_turn.clear()
        self._memory_sync_turn_key = ""
        self._last_memory_tool_query = ""
        self._seen_memory_for_turn.clear()
        self._last_memory_hits = []
        self._last_memory_query = ""
        self.state.update_context("social_tone_retry_count", 0)
        self.state.update_context("low_quality_reply_retry_count", 0)
        self._clear_memory_result_messages()
        self._clear_memory_prompt_directive_messages()
        self._persist_general_messages_to_disk()
        if self._is_general_root_agent():
            self._extract_profile_fact(task)
            self._inject_auto_memory_context(task)
            self._schedule_semantic_and_task_updates(task)
        self._ensure_task_runner_started()

    def _format_ready_model_name(self) -> str:
        raw = str(getattr(self.llm_config, "litellm_model", "") or "").strip()
        if "/" in raw:
            raw = raw.split("/")[-1]
        raw = raw.replace("_", "-")
        return raw.upper() if raw else "MODEL"

    def _memory_session_id(self) -> str:
        return str(self.state.context.get("conversation_session_id", self.state.agent_id) or "").strip()

    def _last_emotion_store_path(self) -> Path:
        return get_sondra_resource_path("memory", "memory_signals", "last_emotion.json")

    def _build_last_emotion_snapshot(self) -> dict[str, Any]:
        context = self.state.context
        return {
            "saved_at": datetime.now(UTC).isoformat(),
            "happiness": round(self._coerce_emotion_percent(context.get("emotion_happiness", 0.0), 0.0), 4),
            "sadness": round(self._coerce_emotion_percent(context.get("emotion_sadness", 0.0), 0.0), 4),
            "stress": round(self._coerce_emotion_percent(context.get("emotion_stress", 0.0), 0.0), 4),
            "neutral": round(self._coerce_emotion_percent(context.get("emotion_neutral", 50.0), 50.0), 4),
            "confidence": round(self._coerce_emotion_percent(context.get("emotion_confidence", 35.0), 35.0), 4),
            "curiosity": round(self._coerce_emotion_percent(context.get("emotion_curiosity", 50.0), 50.0), 4),
            "tone": str(context.get("emotion_tone", "") or "").strip(),
            "signal_category": str(context.get("emotion_signal_category", "") or "").strip(),
            "signal_strength": round(float(context.get("emotion_signal_strength", 0.0) or 0.0), 6),
        }

    def persist_last_emotion_snapshot(self) -> bool:
        if not self._is_general_root_agent():
            return False
        try:
            path = self._last_emotion_store_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = self._build_last_emotion_snapshot()
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to persist last emotion snapshot: %s", exc)
            return False

    def _restore_last_emotion_snapshot(self) -> None:
        if not self._is_general_root_agent():
            return
        try:
            path = self._last_emotion_store_path()
            if not path.exists():
                return
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to restore last emotion snapshot: %s", exc)
            return

        numeric_keys = (
            ("happiness", "emotion_happiness", 0.0),
            ("sadness", "emotion_sadness", 0.0),
            ("stress", "emotion_stress", 0.0),
            ("neutral", "emotion_neutral", 50.0),
            ("confidence", "emotion_confidence", 35.0),
            ("curiosity", "emotion_curiosity", 50.0),
        )
        restored = False
        for payload_key, context_key, default in numeric_keys:
            if payload_key not in payload:
                continue
            restored = True
            self.state.update_context(
                context_key,
                self._coerce_emotion_percent(payload.get(payload_key), default),
            )

        tone = str(payload.get("tone", "") or "").strip()
        if tone:
            restored = True
            self.state.update_context("emotion_tone", tone)
        signal_category = str(payload.get("signal_category", "") or "").strip()
        if signal_category:
            restored = True
            self.state.update_context("emotion_signal_category", signal_category)
        with contextlib.suppress(Exception):
            signal_strength = float(payload.get("signal_strength", 0.0) or 0.0)
            self.state.update_context("emotion_signal_strength", signal_strength)
            restored = True

        if not restored:
            return

        self.state.update_context("boot_last_emotion_pending", True)
        self.state.update_context("boot_last_emotion_saved_at", str(payload.get("saved_at", "") or "").strip())
        self.state.update_context("boot_last_emotion_category", signal_category)

    def _consume_last_emotion_boot_lines(self) -> list[str]:
        if not bool(self.state.context.get("boot_last_emotion_pending", False)):
            return []

        happiness = self._coerce_emotion_percent(self.state.context.get("emotion_happiness", 0.0), 0.0)
        sadness = self._coerce_emotion_percent(self.state.context.get("emotion_sadness", 0.0), 0.0)
        stress = self._coerce_emotion_percent(self.state.context.get("emotion_stress", 0.0), 0.0)
        neutral = self._coerce_emotion_percent(self.state.context.get("emotion_neutral", 50.0), 50.0)
        category = str(self.state.context.get("boot_last_emotion_category", "") or "").strip().lower()
        saved_at = str(self.state.context.get("boot_last_emotion_saved_at", "") or "").strip()

        self.state.update_context("boot_last_emotion_pending", False)

        lines = [
            (
                "Startup carryover for this turn only: the user's last known emotion profile "
                f"before the previous shutdown was happiness={happiness:.0f}/100, "
                f"sadness={sadness:.0f}/100, stress={stress:.0f}/100, neutral={neutral:.0f}/100."
            )
        ]
        if category:
            lines.append(f"Previous session ended with user emotion category: {category}.")
        if saved_at:
            lines.append(f"Snapshot saved at: {saved_at}.")
        lines.append("Use this only as soft startup context, then return to the normal live emotion flow.")
        return lines

    async def _process_iteration(self, tracer: Optional["Tracer"]) -> bool | None:
        if self._is_general_root_agent():
            self._apply_decision_layer(self._original_user_turn_raw())
            self.state.update_context("memory_feedback_success", True)
            self.state.update_context("memory_feedback_reason", "success")
        self._ensure_memory_every_iteration()
        user_turn_raw = self._original_user_turn_raw()
        ollama_tool_guard = self._is_general_root_agent() and self._is_ollama_tool_guard_enabled()
        forced_tool = self._route_obvious_tool_request(user_turn_raw) if ollama_tool_guard else ""
        prefer_direct_conversation = (
            self._is_general_root_agent() and self._should_prefer_direct_conversational_reply(user_turn_raw)
        )
        social_turn_directive = (
            self._build_social_turn_reply_directive(user_turn_raw) if prefer_direct_conversation else ""
        )

        if ollama_tool_guard and forced_tool == "__tool_inventory__":
            reply = self._build_tool_inventory_reply()
            self._emit_assistant_message(reply)
            self._persist_general_messages_to_disk()
            if tracer:
                tracer.clear_streaming_content(self.state.agent_id)
                tracer.log_chat_message(
                    content=clean_content(reply),
                    role="assistant",
                    agent_id=self.state.agent_id,
                )
            if self._should_use_dynamic_general_routing():
                self.state.enter_waiting_state()
                if tracer:
                    tracer.update_agent_status(self.state.agent_id, "waiting")
                self._mark_waiting_in_tracer_and_graph()
            return False

        if ollama_tool_guard:
            self.state.update_context(
                "forced_tool_name",
                forced_tool if forced_tool and forced_tool != "__tool_inventory__" else "",
            )

        if self._is_general_root_agent():
            self._inject_prompt_memory_directive(user_turn_raw)
            self._maybe_force_prompt_memory_search(user_turn_raw)
        if self._is_general_root_agent():
            with contextlib.suppress(Exception):
                await self._maybe_rerank_memory_context(user_turn_raw)
        control_reply_only = False
        if self._is_general_root_agent():
            pending_control_prompt = str(self.state.context.get("pending_control_reply_prompt", "") or "").strip()
            if pending_control_prompt:
                control_reply_only = True
            self._dispatch_due_tasks_now()
            if self._handle_delete_indexed_task_request(user_turn_raw):
                control_reply_only = True
            elif self._handle_delete_all_tasks_request(user_turn_raw):
                control_reply_only = True
            if control_reply_only:
                control_prompt = str(self.state.context.get("pending_control_reply_prompt", "") or "").strip()
                if control_prompt:
                    self.state.add_message("user", control_prompt)
        final_response = None
        stream_speculative_text = not bool(ollama_tool_guard and forced_tool and forced_tool != "__tool_inventory__")

        conversation_history = self.state.get_conversation_history()
        if social_turn_directive:
            conversation_history = [
                *conversation_history,
                {"role": "system", "content": social_turn_directive},
            ]
        if ollama_tool_guard and forced_tool and forced_tool != "__tool_inventory__":
            forced_tool_directive = self._build_forced_tool_system_directive(forced_tool)
            if forced_tool_directive:
                conversation_history = [
                    *conversation_history,
                    {"role": "system", "content": forced_tool_directive},
                ]

        async for response in self.llm.generate(conversation_history):
            final_response = response
            if tracer and response.content and stream_speculative_text:
                raw_stream_content = str(response.content)
                sanitized_stream = self._strip_internal_metadata_blocks(raw_stream_content)
                if self._looks_like_tool_payload_prefix(raw_stream_content):
                    sanitized_stream = ""
                sanitized_stream = self._sanitize_model_output_for_user(sanitized_stream)
                if sanitized_stream and self._looks_like_tool_or_command_output_reply(sanitized_stream):
                    sanitized_stream = ""
                if prefer_direct_conversation and self._looks_like_tool_shorthand_reply(sanitized_stream):
                    sanitized_stream = ""
                if (
                    sanitized_stream
                    and not self._looks_like_internal_reasoning_prefix(raw_stream_content)
                    and not self._looks_like_tool_payload_prefix(raw_stream_content)
                ):
                    tracer.update_streaming_content(self.state.agent_id, sanitized_stream)

        if final_response is None:
            if tracer:
                tracer.clear_streaming_content(self.state.agent_id)
            self.state.update_context("memory_feedback_success", False)
            self.state.update_context("memory_feedback_reason", "generation_failure")
            return False

        raw_model_content = str(final_response.content or "")
        content_stripped = self._strip_internal_metadata_blocks(raw_model_content).strip()
        content_stripped = self._sanitize_model_output_for_user(content_stripped)
        final_response.content = content_stripped

        if (
            prefer_direct_conversation
            and content_stripped
            and self._looks_like_low_quality_direct_reply(user_turn_raw, content_stripped)
        ):
            if self._queue_low_quality_reply_regeneration(user_turn_raw, content_stripped, tracer):
                return False

        if ollama_tool_guard and forced_tool and self._looks_like_fake_tool_usage_reply(content_stripped):
            if tracer:
                tracer.clear_streaming_content(self.state.agent_id)
            self.state.update_context("last_user_turn_raw", self._original_user_turn_raw())
            self.state.add_message(
                "user",
                "Do not describe or simulate tool usage. Use exactly one real tool call now."
            )
            self.state.update_context("memory_feedback_success", False)
            self.state.update_context("memory_feedback_reason", "reasoning_error")
            return False

        if not content_stripped and self._looks_like_internal_reasoning_payload(raw_model_content):
            if tracer:
                tracer.clear_streaming_content(self.state.agent_id)
            if not self._is_model_regeneration_retry_enabled():
                return False
            memory_result = self._latest_memory_result_text()
            if memory_result:
                self.state.add_message(
                    "system",
                    (
                        "[MEMORY RESULT]\n"
                        f"{memory_result}\n\n"
                        "Reply again in plain text only.\n"
                        "Use the memory result naturally in your answer.\n"
                        "Do NOT output internal reasoning.\n"
                        "Do NOT call any tools."
                    ),
                )
            else:
                self.state.add_message(
                    "user",
                    "Reply again in plain text only. Do NOT output internal reasoning. Do NOT call any tools.",
                )
            self.state.update_context("memory_feedback_success", False)
            self.state.update_context("memory_feedback_reason", "reasoning_error")
            return False
        if not content_stripped and self._contains_agent_metadata_tokens(raw_model_content):
            if tracer:
                tracer.clear_streaming_content(self.state.agent_id)
            if not self._is_model_regeneration_retry_enabled():
                return False
            self.state.add_message(
                "user",
                (
                    "Do not output internal agent metadata tags. "
                    "Answer naturally in plain text for the user."
                ),
            )
            self.state.update_context("memory_feedback_success", False)
            self.state.update_context("memory_feedback_reason", "reasoning_error")
            return False
        if control_reply_only:
            if not content_stripped:
                if not self._is_model_regeneration_retry_enabled():
                    return False
                self.state.add_message(
                    "user",
                    "Provide one short natural confirmation sentence for the control action. Do not call tools.",
                )
                return False

            final_response.content = content_stripped

        if not content_stripped:
            if tracer:
                tracer.clear_streaming_content(self.state.agent_id)
            if prefer_direct_conversation and self._is_model_regeneration_retry_enabled():
                generated_reply = await self._generate_direct_reply_without_tools(tracer)
                if generated_reply:
                    if self._looks_like_low_quality_direct_reply(user_turn_raw, generated_reply):
                        if self._queue_low_quality_reply_regeneration(user_turn_raw, generated_reply, tracer):
                            return False

                    self.state.update_context("social_tone_retry_count", 0)
                    self.state.update_context("low_quality_reply_retry_count", 0)
                    self.state.add_message("assistant", generated_reply)
                    self._persist_general_messages_to_disk()
                    if tracer:
                        tracer.clear_streaming_content(self.state.agent_id)
                        tracer.log_chat_message(
                            content=clean_content(generated_reply),
                            role="assistant",
                            agent_id=self.state.agent_id,
                        )
                    if self._should_use_dynamic_general_routing():
                        self.state.enter_waiting_state()
                        if tracer:
                            tracer.update_agent_status(self.state.agent_id, "waiting")
                        self._mark_waiting_in_tracer_and_graph()
                    return False
            if not self._is_model_regeneration_retry_enabled():
                return False
            corrective_message = (
                "You MUST NOT respond with empty messages. "
                "If you currently have nothing to do or say, use an appropriate tool instead:\n"
                "- Use agents_graph_actions.wait_for_message to wait for messages "
                "from user or other agents\n"
                "- Use agents_graph_actions.agent_finish if you are a sub-agent "
                "and your task is complete\n"
                "- Use finish_actions.finish_scan if you are the root/main agent "
                "and the scan is complete"
            )
            self.state.add_message("user", corrective_message)
            self.state.update_context("memory_feedback_success", False)
            self.state.update_context("memory_feedback_reason", "reasoning_error")
            return False

        actions = (
            final_response.tool_invocations
            if hasattr(final_response, "tool_invocations") and final_response.tool_invocations
            else []
        )
        actions = self._filter_valid_tool_actions(actions)
        forced_tool_name = str(self.state.context.get("forced_tool_name", "") or "").strip()
        forced_tool_retry_count = int(self.state.context.get("forced_tool_retry_count", 0) or 0)

        if ollama_tool_guard and forced_tool_name and actions:
            matching_actions = [
                action
                for action in actions
                if str(action.get("toolName", "") or "").strip() == forced_tool_name
            ]

            if not matching_actions:
                if tracer:
                    tracer.clear_streaming_content(self.state.agent_id)

                if forced_tool_retry_count < self.OLLAMA_FORCED_TOOL_RETRY_LIMIT:
                    self.state.update_context("last_user_turn_raw", self._original_user_turn_raw())
                    self.state.add_message("user", self._build_forced_tool_guidance(forced_tool_name))
                    self.state.update_context("forced_tool_retry_count", forced_tool_retry_count + 1)
                    self.state.update_context("memory_feedback_success", False)
                    self.state.update_context("memory_feedback_reason", "reasoning_error")
                    return False

                fallback_action = self._build_forced_tool_fallback_action(
                    forced_tool_name,
                    self._original_user_turn_raw(),
                )
                if fallback_action:
                    return await self._execute_actions([fallback_action], tracer)

                generated_reply = await self._generate_direct_reply_without_tools(tracer)
                if generated_reply:
                    self._emit_assistant_message(generated_reply)
                    self._persist_general_messages_to_disk()
                    if tracer:
                        tracer.log_chat_message(
                            content=clean_content(generated_reply),
                            role="assistant",
                            agent_id=self.state.agent_id,
                        )

                if self._should_use_dynamic_general_routing():
                    self.state.enter_waiting_state()
                    if tracer:
                        tracer.update_agent_status(self.state.agent_id, "waiting")
                    self._mark_waiting_in_tracer_and_graph()

                return False

            actions = matching_actions[:1]

        if ollama_tool_guard and forced_tool_name and not actions:
            if tracer:
                tracer.clear_streaming_content(self.state.agent_id)

            if forced_tool_retry_count < self.OLLAMA_FORCED_TOOL_RETRY_LIMIT:
                self.state.update_context("last_user_turn_raw", self._original_user_turn_raw())
                self.state.add_message("user", self._build_forced_tool_guidance(forced_tool_name))
                self.state.update_context("forced_tool_retry_count", forced_tool_retry_count + 1)
                self.state.update_context("memory_feedback_success", False)
                self.state.update_context("memory_feedback_reason", "reasoning_error")
                return False

            fallback_action = self._build_forced_tool_fallback_action(
                forced_tool_name,
                self._original_user_turn_raw(),
            )
            if fallback_action:
                return await self._execute_actions([fallback_action], tracer)

            generated_reply = await self._generate_direct_reply_without_tools(tracer)
            if generated_reply:
                self._emit_assistant_message(generated_reply)
                self._persist_general_messages_to_disk()
                if tracer:
                    tracer.log_chat_message(
                        content=clean_content(generated_reply),
                        role="assistant",
                        agent_id=self.state.agent_id,
                    )
            if self._should_use_dynamic_general_routing():
                self.state.enter_waiting_state()
                if tracer:
                    tracer.update_agent_status(self.state.agent_id, "waiting")
                self._mark_waiting_in_tracer_and_graph()
            return False

        if actions and self._is_general_root_agent():
            filtered_actions: list[Any] = []
            visual_already_attempted = self._visual_screenshot_attempted_for_current_turn()
            visual_in_this_response = False
            for action in actions:
                tool_name = str(action.get("toolName", "")).strip()
                if tool_name in self.VISUAL_TOOL_NAMES:
                    if visual_already_attempted or visual_in_this_response:
                        continue
                    visual_in_this_response = True
                    self._mark_visual_screenshot_attempted_for_current_turn()
                filtered_actions.append(action)
            actions = filtered_actions
        if ollama_tool_guard and actions:
            self.state.update_context("forced_tool_name", "")
            self.state.update_context("forced_tool_retry_count", 0)

        display_content = clean_content(str(final_response.content or "")).strip()

        if prefer_direct_conversation and actions:
            actions = []
            if self._queue_social_turn_regeneration(
                user_turn_raw,
                "social_turn_tool_suppressed",
                tracer,
            ):
                return False

        if (
            prefer_direct_conversation
            and display_content
            and self._looks_like_tool_or_command_output_reply(display_content)
        ):
            if self._queue_social_turn_regeneration(
                user_turn_raw,
                "social_turn_tool_text_suppressed",
                tracer,
            ):
                return False

        if (
            display_content
            and self._is_how_are_you_turn(user_turn_raw)
            and self._response_needs_social_tone_rewrite(user_turn_raw, display_content)
        ):
            if self._queue_social_turn_regeneration(
                user_turn_raw,
                "social_turn_rewrite_requested",
                tracer,
            ):
                return False

        if actions:
            display_content = ""

        if (
            prefer_direct_conversation
            and not display_content
            and not actions
            and self._is_model_regeneration_retry_enabled()
        ):
            generated_reply = await self._generate_direct_reply_without_tools(tracer)
            if generated_reply:
                if self._looks_like_low_quality_direct_reply(user_turn_raw, generated_reply):
                    if self._queue_low_quality_reply_regeneration(user_turn_raw, generated_reply, tracer):
                        return False

                self.state.update_context("social_tone_retry_count", 0)
                self.state.update_context("low_quality_reply_retry_count", 0)
                self.state.add_message("assistant", generated_reply)
                self._persist_general_messages_to_disk()
                if tracer:
                    tracer.clear_streaming_content(self.state.agent_id)
                    tracer.log_chat_message(
                        content=clean_content(generated_reply),
                        role="assistant",
                        agent_id=self.state.agent_id,
                    )
                if self._should_use_dynamic_general_routing():
                    self.state.enter_waiting_state()
                    if tracer:
                        tracer.update_agent_status(self.state.agent_id, "waiting")
                    self._mark_waiting_in_tracer_and_graph()
                return False

        thinking_blocks = getattr(final_response, "thinking_blocks", None)
        if display_content:
            self.state.update_context("social_tone_retry_count", 0)
            self.state.update_context("low_quality_reply_retry_count", 0)
            self.state.add_message("assistant", display_content, thinking_blocks=thinking_blocks)
            self._persist_general_messages_to_disk()
            if tracer:
                tracer.clear_streaming_content(self.state.agent_id)
                tracer.log_chat_message(
                    content=clean_content(display_content),
                    role="assistant",
                    agent_id=self.state.agent_id,
                )

        if control_reply_only and actions:
            self.state.add_message(
                "user",
                "Do not call tools for this control response. Reply with one short natural confirmation sentence only.",
            )
            actions = []

        if control_reply_only:
            pending_control_type = str(self.state.context.get("pending_control_type", "") or "").strip().lower()
            self.state.update_context("pending_control_reply_prompt", "")
            self.state.update_context("pending_control_type", "")
            if pending_control_type == "add":
                self._emit_assistant_message(self.SCHEDULE_WAITING_MESSAGE)
            self._persist_general_messages_to_disk()
            if pending_control_type == "add":
                self.state.enter_waiting_state()
                if tracer:
                    tracer.update_agent_status(self.state.agent_id, "waiting")
                self._mark_waiting_in_tracer_and_graph()
            return False

        if actions:
            if tracer:
                tracer.clear_streaming_content(self.state.agent_id)
            return await self._execute_actions(actions, tracer)

        if self._should_use_dynamic_general_routing():
            self.state.enter_waiting_state()
            if tracer:
                tracer.update_agent_status(self.state.agent_id, "waiting")

        return False

    def _is_ollama_tool_guard_enabled(self) -> bool:
        from sondra.config import Config

        env_model = str(Config.get("sondra_llm") or "").strip().lower()
        return env_model == "ollama" or env_model.startswith("ollama/")

    def _is_model_regeneration_retry_enabled(self) -> bool:
        return self._is_ollama_tool_guard_enabled()

    def _normalize_user_text(self, text: str) -> str:
        return self._normalize_match_text(text)

    def _normalize_match_text(self, text: str) -> str:
        return normalize_signal_text(text)

    def _normalized_signal_list(self, file_name: str, *path: str) -> list[str]:
        values = self._signal_list(file_name, *path)
        result: list[str] = []
        for value in values:
            normalized = self._normalize_user_text(str(value or ""))
            if normalized:
                result.append(normalized)
        return result

    def _is_how_are_you_turn(self, text: str) -> bool:
        normalized = self._normalize_user_text(text)
        if not normalized:
            return False
        phrases = self._signal_list("message_check", "conversation", "how_are_you_phrases")
        return any(phrase in normalized for phrase in phrases)

    def _looks_like_social_or_emotional_turn(self, text: str) -> bool:
        normalized = self._normalize_user_text(text)
        if not normalized:
            return False
        if self._is_how_are_you_turn(text):
            return True
        social_phrases = self._signal_list("message_check", "conversation", "social_phrases")
        if any(phrase in normalized for phrase in social_phrases):
            return True
        if self._looks_like_self_feeling_claim(text):
            return True

        emotion_weights = self._signal_map("emotion_signals", "weights")
        if not emotion_weights:
            return False

        personal_cues = self._normalized_signal_list("message_check", "conversation", "personal_cues")
        has_personal_cue = any(cue in normalized for cue in personal_cues)
        for category in ("happiness", "sadness", "frustration", "anger", "insult"):
            markers = emotion_weights.get(category, {})
            if not isinstance(markers, dict):
                continue
            for phrase in markers:
                if not phrase or len(phrase) < 4:
                    continue
                if phrase in normalized and (has_personal_cue or normalized.startswith(phrase)):
                    return True
        return False

    def _is_brief_social_turn(self, text: str) -> bool:
        normalized = self._normalize_user_text(text)
        if not normalized:
            return False

        brief_social_phrases = set(
            self._normalized_signal_list("message_check", "conversation", "brief_social_phrases")
        )

        if normalized in brief_social_phrases:
            return True

        words = normalized.split()
        return len(words) <= 3 and normalized in brief_social_phrases

    def _should_prefer_direct_conversational_reply(self, text: str) -> bool:
        normalized = self._normalize_user_text(text)
        if not normalized:
            return False

        if self._looks_like_correction_only_turn(text):
            return True
        if self._looks_like_memory_fact_statement(text):
            return True
        if self._is_how_are_you_turn(text):
            return True
        if self._looks_like_meta_conversation_turn(text):
            return True
        if self._looks_like_social_or_emotional_turn(text):
            return True
        if self._is_tool_inventory_request(text) or self._is_explicit_memory_request(text):
            return False
        if self._contains_url_or_domain(text) or self._contains_file_path(text):
            return False
        if self._looks_like_terminal_request(text) or self._looks_like_python_request(text):
            return False
        if self._looks_like_live_web_request(text):
            return False

        if self._is_brief_social_turn(text):
            social_external_cues = self._normalized_signal_list(
                "message_check", "conversation", "social_external_cues"
            )
            if not any(cue in normalized for cue in social_external_cues):
                return True

        return False

    def _looks_like_meta_conversation_turn(self, text: str) -> bool:
        normalized = self._normalize_user_text(text)
        if not normalized:
            return False
        markers = self._normalized_signal_list("message_check", "conversation", "meta_conversation_markers")
        return any(marker in normalized for marker in markers)

    def _looks_like_self_feeling_claim(self, text: str) -> bool:
        normalized = self._normalize_user_text(text)
        if not normalized:
            return False
        claim_phrases = self._signal_list("message_check", "conversation", "self_feeling_claim_phrases")
        return any(phrase in normalized for phrase in claim_phrases)

    def _looks_like_correction_only_turn(self, text: str) -> bool:
        normalized = self._normalize_user_text(text)
        if not normalized:
            return False
        correction_markers = self._signal_list("message_check", "conversation", "correction_markers")
        recall_markers = self._signal_list("message_check", "conversation", "recall_markers")
        has_correction = any(marker in normalized for marker in correction_markers)
        has_recall = any(marker in normalized for marker in recall_markers)
        return has_correction and not has_recall

    def _looks_like_tool_shorthand_reply(self, text: str) -> bool:
        raw = str(text or "").strip()
        if not raw:
            return False
        first_line = raw.splitlines()[0].strip().lower()
        shorthand_prefixes = self._signal_list("message_check", "conversation", "tool_shorthand_prefixes")
        return any(first_line.startswith(prefix) for prefix in shorthand_prefixes)

    def _looks_like_tool_or_command_output_reply(self, text: str) -> bool:
        raw = str(text or "").strip()
        if not raw:
            return False
        lowered = raw.lower()
        normalized = self._normalize_user_text(raw)
        if self._looks_like_tool_shorthand_reply(raw):
            return True
        command_output_markers = self._normalized_signal_list(
            "output_filters", "tool_output", "command_output_markers"
        )
        if normalized and any(marker in normalized for marker in command_output_markers):
            return True
        first_lines = [line.strip().lower() for line in raw.splitlines()[:3] if line.strip()]
        shell_prefixes = ("> $", "$ ", "❯ ", "➜ ", "powershell>", "cmd>")
        return any(line.startswith(shell_prefixes) for line in first_lines)

    def _is_likely_turkish_turn(self, text: str) -> bool:
        raw = str(text or "")
        normalized = self._normalize_user_text(text)
        if any(ch in raw for ch in "çğıöşüÇĞİÖŞÜ"):
            return True
        turkish_markers = self._signal_list("message_check", "conversation", "turkish_markers")
        return any(marker in normalized for marker in turkish_markers)

    def _recent_visible_user_messages(self, limit: int = 4) -> list[str]:
        results: list[str] = []
        for msg in reversed(self.state.messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
            content = self._strip_internal_metadata_blocks(content).strip()
            if not content:
                continue
            if "<inter_agent_message>" in content:
                continue
            if content.startswith("Tool Results:") or "<tool_result>" in content:
                continue
            if self._is_memory_context_text(content):
                continue
            if self._is_internal_metadata_text(content):
                continue
            if self._is_internal_control_user_message(content):
                continue
            results.append(content)
            if len(results) >= limit:
                break
        return list(reversed(results))


    def _latest_visible_assistant_message(self) -> str:
        for msg in reversed(self.state.messages):
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
            content = self._strip_internal_metadata_blocks(content).strip()
            if not content:
                continue
            if self._is_memory_result_text(content):
                continue
            if content == self.PERSISTENT_MEMORY_READING_MESSAGE:
                continue
            return content
        return ""


    def _has_excessive_line_repetition(self, text: str) -> bool:
        lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
        if len(lines) < 2:
            return False

        repeat_run = 1
        last_norm = ""
        for line in lines:
            current_norm = self._normalize_match_text(line)
            if current_norm and current_norm == last_norm:
                repeat_run += 1
                if repeat_run >= 3:
                    return True
            else:
                repeat_run = 1
                last_norm = current_norm
        return False


    def _contains_suspicious_foreign_script(self, text: str) -> bool:
        raw = str(text or "")
        if not raw:
            return False
        return bool(re.search(r"[\u0E00-\u0E7F]", raw))


    def _looks_like_low_quality_direct_reply(self, user_text: str, response_text: str) -> bool:
        response_norm = self._normalize_user_text(response_text)
        user_norm = self._normalize_user_text(user_text)

        if not response_norm:
            return True

        if response_norm == user_norm and len(response_norm.split()) <= 12:
            return True

        recent_user_norms = {
            self._normalize_user_text(item)
            for item in self._recent_visible_user_messages(limit=4)
            if str(item or "").strip()
        }
        if response_norm in recent_user_norms and len(response_norm.split()) <= 12:
            return True

        last_assistant_norm = self._normalize_user_text(self._latest_visible_assistant_message())
        if last_assistant_norm and response_norm == last_assistant_norm:
            return True

        if self._has_excessive_line_repetition(response_text):
            return True

        if self._is_likely_turkish_turn(user_text) and self._contains_suspicious_foreign_script(response_text):
            return True

        return False

    def _build_low_quality_reply_recovery_prompt(self, user_text: str, bad_reply: str) -> str:
        language_line = (
            "Reply in Turkish."
            if self._is_likely_turkish_turn(user_text)
            else "Reply in the user's language."
        )
        return (
            "Rewrite your last reply.\n"
            "Do NOT repeat the user's words verbatim.\n"
            "Do NOT repeat an earlier user message.\n"
            "Do NOT repeat your previous answer.\n"
            "Do NOT output junk, random foreign-script text, or repeated lines.\n"
            "Keep it short, natural, and directly relevant.\n"
            f"{language_line}\n"
            "Plain text only. Do NOT call tools."
        )


    def _queue_low_quality_reply_regeneration(
        self,
        user_text: str,
        bad_reply: str,
        tracer: Optional["Tracer"] = None,
    ) -> bool:
        if not self._is_model_regeneration_retry_enabled():
            return False
        retry_count = int(self.state.context.get("low_quality_reply_retry_count", 0) or 0)
        if retry_count >= 2:
            return False

        if tracer:
            tracer.clear_streaming_content(self.state.agent_id)

        self.state.add_message(
            "user",
            self._build_low_quality_reply_recovery_prompt(user_text, bad_reply),
        )
        self.state.update_context("low_quality_reply_retry_count", retry_count + 1)
        self.state.update_context("memory_feedback_success", False)
        self.state.update_context("memory_feedback_reason", "low_quality_direct_reply")
        return True

    def _build_social_turn_reply_directive(self, user_text: str) -> str:
        if self._looks_like_correction_only_turn(user_text):
            return (
                "The user is correcting your previous answer.\n"
                "Reply in plain text only.\n"
                "Do NOT call any tools.\n"
                "Acknowledge the correction briefly and stay on the corrected point.\n"
                "Do NOT ask unrelated follow-up questions."
            )
        if self._looks_like_memory_fact_statement(user_text):
            return (
                "The user is sharing a personal fact, preference, goal, or current step.\n"
                "Reply in plain text only.\n"
                "Do NOT call any tools.\n"
                "Acknowledge the fact naturally in 1-2 short sentences.\n"
                "Do NOT greet the user again if you already greeted earlier in the conversation.\n"
                "Do NOT repeat the user's name unless it is necessary.\n"
                "Prefer a direct acknowledgment over a greeting.\n"
                "Do NOT switch to unrelated explanations.\n"
                "Do NOT ask broad follow-up questions unless the user asked for help."
            )
        if self._is_how_are_you_turn(user_text):
            return (
                "This is a social check-in turn.\n"
                "Reply in plain text only.\n"
                "Do NOT call any tools.\n"
                "Do NOT claim a literal internal emotional state such as 'I'm great', 'I'm fine', 'iyiyim', or similar.\n"
                "Do NOT say the user changed your mood, energy, or happiness.\n"
                "Answer in an availability-oriented way and offer help naturally."
            )
        return (
            "This is a casual social or emotional user turn.\n"
            "Reply in plain text only.\n"
            "Do NOT call any tools.\n"
            "Do NOT describe your own feelings, energy, or mood changes.\n"
            "Acknowledge the user's tone naturally and respond briefly."
        )

    def _response_needs_social_tone_rewrite(self, user_text: str, response_text: str) -> bool:
        normalized = self._normalize_user_text(response_text)
        if not normalized:
            return False
        if self._looks_like_self_feeling_claim(response_text):
            return True

        tone = str(self.state.context.get("emotion_tone", "") or "").strip().lower()
        category = str(self.state.context.get("emotion_signal_category", "") or "").strip().lower()
        playful_markers = self._signal_list("message_check", "conversation", "playful_markers")
        how_are_you_disallowed_self_claims = self._normalized_signal_list(
            "message_check", "conversation", "how_are_you_disallowed_self_claims"
        )

        if self._is_how_are_you_turn(user_text) and (
            any(marker in normalized for marker in how_are_you_disallowed_self_claims)
        ):
            return True

        if tone in {"stabilizing", "empathetic"} or category in {"critical", "aggressive"}:
            if any(marker in normalized for marker in playful_markers):
                return True

        return False

    def _build_social_turn_regeneration_prompt(self, user_text: str) -> str:
        tone = str(self.state.context.get("emotion_tone", "") or "").strip().lower()
        category = str(self.state.context.get("emotion_signal_category", "") or "").strip().lower()
        lines = [
            "Rewrite your last reply naturally in plain text only.",
            "Do NOT call any tools.",
            "Do NOT describe your own feelings, mood changes, happiness, or energy shifts.",
            "Do NOT say the user changed your internal state.",
            "Do NOT output analysis, chain-of-thought, or planner notes such as 'We are in ...', 'The user ...', or 'I will ...'.",
        ]
        if self._is_how_are_you_turn(user_text):
            lines.append(
                "For this social check-in, answer in an availability-oriented way instead of saying you feel good, great, happy, or similar."
            )
        else:
            lines.append("Acknowledge the user's tone naturally and keep the reply concise.")

        if tone in {"stabilizing", "empathetic"} or category in {"critical", "aggressive"}:
            lines.append("Avoid playful wording.")
            lines.append("Keep the tone calm, steady, and reassuring.")
        elif tone == "warm" or category == "positive":
            lines.append("You may sound warm and encouraging, but still avoid claiming your own emotions.")
        else:
            lines.append("Keep the tone natural and balanced.")

        return "\n".join(lines)

    def _queue_social_turn_regeneration(
        self,
        user_text: str,
        reason: str,
        tracer: Optional["Tracer"] = None,
    ) -> bool:
        if not self._is_model_regeneration_retry_enabled():
            return False
        retry_count = int(self.state.context.get("social_tone_retry_count", 0) or 0)
        if retry_count >= 2:
            return False
        if tracer:
            tracer.clear_streaming_content(self.state.agent_id)
        self.state.add_message("user", self._build_social_turn_regeneration_prompt(user_text))
        self.state.update_context("social_tone_retry_count", retry_count + 1)
        self.state.update_context("memory_feedback_success", False)
        self.state.update_context("memory_feedback_reason", reason)
        return True

    def _queue_direct_turn_tool_regeneration(
        self,
        user_text: str,
        reason: str,
        tracer: Optional["Tracer"] = None,
    ) -> bool:
        if not self._is_model_regeneration_retry_enabled():
            return False
        retry_count = int(self.state.context.get("social_tone_retry_count", 0) or 0)
        if retry_count >= 2:
            return False
        if tracer:
            tracer.clear_streaming_content(self.state.agent_id)
        language_line = (
            "Reply in Turkish."
            if self._is_likely_turkish_turn(user_text)
            else "Reply in the user's language."
        )
        self.state.add_message(
            "user",
            (
                "Rewrite your last reply as a direct conversational answer.\n"
                "Do NOT call any tools.\n"
                "Do NOT output terminal commands, tool names, code execution, or command output.\n"
                "If this is a greeting or check-in, answer briefly and naturally.\n"
                "If this is a user fact, acknowledge it briefly.\n"
                "If the user asks why you wrote something, apologize briefly and explain that it was an incorrect tool-like response.\n"
                f"{language_line}\n"
                "Plain text only."
            ),
        )
        self.state.update_context("social_tone_retry_count", retry_count + 1)
        self.state.update_context("memory_feedback_success", False)
        self.state.update_context("memory_feedback_reason", reason)
        return True

    def _sync_memory_runtime_handles(self) -> None:
        self.memory_store = self.memory_runtime.store
        self.memory_index_manager = self.memory_runtime.index_manager
        self.episodic = self.memory_runtime.episodic

    def _is_tool_inventory_request(self, text: str) -> bool:
        normalized = self._normalize_user_text(text)
        if not normalized:
            return False
        tool_inventory_keywords = self._signal_list("message_check", "tool_inventory_keywords")
        return any(keyword in normalized for keyword in tool_inventory_keywords)

    def _contains_url_or_domain(self, text: str) -> bool:
        raw = str(text or "").strip().lower()
        if not raw:
            return False
        if any(x in raw for x in ("http://", "https://", "www.")):
            return True
        return bool(re.search(r"\b[a-z0-9-]+\.(com|net|org|io|dev|app|ai|co|gov|edu)\b", raw))

    def _extract_browser_target_url(self, text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""

        explicit_match = re.search(r"(?i)\b(?:https?|file)://[^\s<>'\"\])]+", raw)
        if explicit_match:
            return explicit_match.group(0).rstrip(".,;:!?)]}")

        domain_match = re.search(
            r"(?i)\b(?:www\.)?[a-z0-9-]+(?:\.[a-z0-9-]+)+(?:/[^\s<>'\"\])]+)?",
            raw,
        )
        if not domain_match:
            return ""

        candidate = domain_match.group(0).rstrip(".,;:!?)]}")
        if not candidate:
            return ""
        if candidate.lower().startswith(("http://", "https://", "file://")):
            return candidate
        return f"https://{candidate}"

    def _extract_browser_search_query(self, text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""

        for pattern in (r"\"([^\"]+)\"", r"'([^']+)'", r"â€œ([^â€]+)â€", r"â€˜([^â€™]+)â€™"):
            match = re.search(pattern, raw)
            if match:
                value = str(match.group(1) or "").strip()
                if value:
                    return value

        cleaned = re.sub(r"(?i)\b(?:https?|file)://[^\s<>'\"\])]+", " ", raw)
        cleaned = re.sub(
            r"(?i)\b(?:www\.)?[a-z0-9-]+(?:\.[a-z0-9-]+)+(?:/[^\s<>'\"\])]+)?",
            " ",
            cleaned,
        )
        normalized = self._normalize_user_text(cleaned)
        cleanup_phrases = self._normalized_signal_list(
            "message_check", "routing", "browser_search_cleanup_phrases"
        )
        for phrase in cleanup_phrases:
            normalized = normalized.replace(phrase, " ")
        normalized = re.sub(r"\s+", " ", normalized).strip(" .,:;!?-")
        return normalized

    def _browser_search_url_for_host(self, host: str, query: str) -> str:
        clean_query = str(query or "").strip()
        if not clean_query:
            return ""
        encoded = quote_plus(clean_query)
        lowered_host = str(host or "").strip().lower()
        if "bing." in lowered_host:
            return f"https://www.bing.com/search?q={encoded}"
        if "duckduckgo." in lowered_host or "ddg.gg" in lowered_host:
            return f"https://duckduckgo.com/?q={encoded}"
        return f"https://www.google.com/search?q={encoded}"

    def _extract_terminal_command(self, text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""

        for pattern in (r"`([^`]+)`", r"\"([^\"]+)\"", r"'([^']+)'"):
            match = re.search(pattern, raw)
            if match:
                command = str(match.group(1) or "").strip()
                if command:
                    return command

        normalized = self._normalize_user_text(raw)
        patterns = (
            r"terminalde\s+(.+?)\s+komutunu\s+calistir",
            r"terminalde\s+(.+?)\s+calistir",
            r"(.+?)\s+komutunu\s+calistir",
            r"run\s+(.+?)\s+in\s+the\s+terminal",
            r"execute\s+(.+?)\s+in\s+the\s+terminal",
        )
        for pattern in patterns:
            match = re.search(pattern, normalized)
            if not match:
                continue
            command = str(match.group(1) or "").strip(" .,:;!?")
            if command:
                return command

        for token in ("pwd", "ls", "dir", "whoami", "env", "printenv"):
            if re.search(rf"\b{re.escape(token)}\b", normalized):
                return token

        return ""

    def _extract_list_files_path(self, text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            return "/workspace"

        path_match = re.search(r"(/workspace|/[A-Za-z0-9._/\-]+|\b[A-Za-z]:\\[^\s]+)", raw)
        if path_match:
            candidate = str(path_match.group(1) or "").strip()
            if candidate:
                return candidate
        return "/workspace"

    def _extract_search_files_regex(self, text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""

        for pattern in (r"\"([^\"]+)\"", r"'([^']+)'", r"`([^`]+)`"):
            match = re.search(pattern, raw)
            if match:
                value = str(match.group(1) or "").strip()
                if value:
                    return value

        cleaned = re.sub(r"(/workspace|/[A-Za-z0-9._/\-]+|\b[A-Za-z]:\\[^\s]+)", " ", raw)
        normalized = self._normalize_user_text(cleaned)
        file_search_cues = self._normalized_signal_list(
            "message_check", "routing", "file_search_cues"
        )
        for phrase in file_search_cues:
            normalized = normalized.replace(phrase, " ")
        normalized = re.sub(r"\s+", " ", normalized).strip(" .,:;!?-")
        return normalized

    def _extract_python_code(self, text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""

        fenced_match = re.search(r"```(?:python)?\s*(.*?)```", raw, re.DOTALL | re.IGNORECASE)
        if fenced_match:
            code = str(fenced_match.group(1) or "").strip()
            if code:
                return code

        for pattern in (
            r"`([^`]+)`",
            r"\"([^\"]+)\"",
            r"'([^']+)'",
        ):
            match = re.search(pattern, raw, re.DOTALL)
            if not match:
                continue
            code = str(match.group(1) or "").strip()
            if code and any(token in code for token in ("print(", "import ", "def ", "for ", "=")):
                return code

        normalized = self._normalize_user_text(raw)
        inline_patterns = (
            r"python\s+calistir[:\s]+(.+)",
            r"python\s+execute[:\s]+(.+)",
            r"run python[:\s]+(.+)",
            r"execute python[:\s]+(.+)",
            r"python kodunu calistir[:\s]+(.+)",
            r"(.+?)\s+python\s+calistir$",
            r"(.+?)\s+python\s+execute$",
            r"(.+?)\s+python kodunu calistir$",
        )
        for pattern in inline_patterns:
            match = re.search(pattern, normalized, re.DOTALL)
            if not match:
                continue
            code = str(match.group(1) or "").strip(" .,:;!?")
            if code:
                return code

        return ""

    def _build_forced_tool_system_directive(self, tool_name: str) -> str:
        requested_tool = str(tool_name or "").strip()
        if not requested_tool:
            return ""
        return (
            "This user turn requires a real tool invocation.\n"
            f"You must call exactly one tool: {requested_tool}.\n"
            "Do not explain what you will do.\n"
            "Do not say you are about to launch, open, inspect, run, execute, or search.\n"
            "Do not answer in plain text first.\n"
            "Output only the real tool call."
        )

    def _build_forced_tool_fallback_action(
        self,
        tool_name: str,
        user_text: str,
    ) -> dict[str, Any] | None:
        requested_tool = str(tool_name or "").strip()
        source_text = str(user_text or "").strip()
        if not requested_tool or not source_text:
            return None

        if requested_tool == "browser_action":
            target_url = self._extract_browser_target_url(source_text)
            if not target_url:
                return None
            return {
                "toolName": "browser_action",
                "args": {"action": "launch", "url": target_url},
            }

        if requested_tool == "terminal_execute":
            command = self._extract_terminal_command(source_text)
            if not command:
                return None
            return {
                "toolName": "terminal_execute",
                "args": {"command": command},
            }

        if requested_tool == "list_files":
            return {
                "toolName": "list_files",
                "args": {"path": self._extract_list_files_path(source_text)},
            }

        if requested_tool == "search_files":
            regex = self._extract_search_files_regex(source_text)
            if not regex:
                return None
            return {
                "toolName": "search_files",
                "args": {
                    "path": self._extract_list_files_path(source_text),
                    "regex": regex,
                    "file_pattern": "*",
                },
            }

        if requested_tool == "web_search":
            query = self._extract_browser_search_query(source_text) or source_text
            query = str(query or "").strip()
            if not query:
                return None
            return {
                "toolName": "web_search",
                "args": {"query": query},
            }

        if requested_tool == "python_action":
            code = self._extract_python_code(source_text)
            if code:
                return {
                    "toolName": "python_action",
                    "args": {"action": "execute", "code": code},
                }
            return {
                "toolName": "python_action",
                "args": {"action": "new_session"},
            }

        return None

    def _latest_tool_result_text(
        self,
        tool_name: str,
        conversation_history: list[dict[str, Any]],
    ) -> str:
        marker = f"<tool_name>{tool_name}</tool_name>"
        for message in reversed(conversation_history):
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content", "")
            if isinstance(content, str):
                text = content
            else:
                text = str(content)
                with contextlib.suppress(Exception):
                    text = json.dumps(content, ensure_ascii=False)
            if marker in text:
                return text
        return ""

    def _tool_result_indicates_failure(
        self,
        tool_name: str,
        conversation_history: list[dict[str, Any]],
    ) -> bool:
        tool_result_text = self._latest_tool_result_text(tool_name, conversation_history)
        if not tool_result_text:
            return False

        lowered = tool_result_text.lower()
        generic_error_markers = self._signal_list("message_check", "tool_failures", "generic_error_markers")
        if any(marker in lowered for marker in generic_error_markers):
            return True

        if tool_name == "web_search":
            failure_markers = self._signal_list("message_check", "tool_failures", "web_search_failure_markers")
            return any(marker in lowered for marker in failure_markers)

        return False

    def _build_web_search_browser_fallback_action(
        self,
        actions: list[Any],
        conversation_history: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not (self._is_general_root_agent() and self._is_ollama_tool_guard_enabled()):
            return None
        if self._matched_routed_tool_name(actions) != "web_search":
            return None
        if "browser_action" not in self.TOOLS:
            return None
        if not self._tool_result_indicates_failure("web_search", conversation_history):
            return None

        current_turn = str(self.state.context.get("last_user_message_event_id", "") or "").strip()
        if not current_turn:
            current_turn = self._current_user_turn_key()
        handled_turn = str(
            self.state.context.get("web_search_browser_fallback_turn", "") or ""
        ).strip()
        if current_turn and handled_turn == current_turn:
            return None

        raw_user_text = self._original_user_turn_raw()
        target_url = self._extract_browser_target_url(raw_user_text)
        search_query = self._extract_browser_search_query(raw_user_text)
        fallback_url = ""

        if target_url:
            parsed = urlparse(target_url)
            host = (parsed.netloc or parsed.path or "").lower()
            if search_query and any(x in host for x in ("google.", "bing.", "duckduckgo.", "ddg.gg")):
                fallback_url = self._browser_search_url_for_host(host, search_query)
            else:
                fallback_url = target_url
        elif search_query:
            fallback_url = self._browser_search_url_for_host("", search_query)

        if not fallback_url:
            return None

        if current_turn:
            self.state.update_context("web_search_browser_fallback_turn", current_turn)

        return {
            "toolName": "browser_action",
            "args": {
                "action": "launch",
                "url": fallback_url,
            },
        }

    def _contains_file_path(self, text: str) -> bool:
        raw = str(text or "").strip()
        if not raw:
            return False
        return bool(re.search(r"(/workspace|/[A-Za-z0-9._/\-]+|\b[A-Za-z]:\\)", raw))

    def _looks_like_terminal_request(self, text: str) -> bool:
        normalized = self._normalize_user_text(text)
        if not normalized:
            return False
        if self._looks_like_code_writing_request(text):
            return False
        terminal_cues = self._normalized_signal_list("message_check", "routing", "terminal_cues")
        token_values = set(re.findall(r"[a-z0-9_./:-]+", normalized))
        for cue in terminal_cues:
            marker = str(cue or "").strip()
            if not marker:
                continue
            if len(marker) <= 3 and " " not in marker:
                if marker in token_values:
                    return True
                continue
            if marker in normalized:
                return True
        return False

    def _looks_like_python_request(self, text: str) -> bool:
        normalized = self._normalize_user_text(text)
        if not normalized:
            return False
        if self._looks_like_code_writing_request(text):
            return False
        python_cues = self._normalized_signal_list("message_check", "routing", "python_cues")
        if any(cue in normalized for cue in python_cues):
            return True
        return "```python" in str(text or "").lower()

    def _looks_like_code_writing_request(self, text: str) -> bool:
        normalized = self._normalize_user_text(text)
        if not normalized:
            return False
        if not any(token in normalized for token in ("python", "kod", "code", "script", "betik")):
            return False
        execute_terms = self._normalized_signal_list(
            "message_check", "routing", "code_execution_terms"
        )
        if any(term in normalized for term in execute_terms):
            return False
        write_terms = self._normalized_signal_list(
            "message_check", "routing", "code_writing_terms"
        )
        return any(term in normalized for term in write_terms)

    def _looks_like_live_web_request(self, text: str) -> bool:
        normalized = self._normalize_user_text(text)
        if not normalized:
            return False
        live_phrase_cues = self._normalized_signal_list("message_check", "routing", "live_phrase_cues")
        live_token_cues = self._normalized_signal_list(
            "message_check", "routing", "live_token_cues"
        )
        if any(cue in normalized for cue in live_phrase_cues):
            return True
        token_values = set(re.findall(r"[a-z0-9']+", normalized))
        return any(token in token_values for token in live_token_cues)

    def _route_obvious_tool_request(self, text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            raw = self._original_user_turn_raw()
        normalized = self._normalize_user_text(raw)
        if not normalized:
            return ""

        if self._is_tool_inventory_request(raw):
            return "__tool_inventory__"

        if self._is_explicit_memory_request(raw) and "memory_search" in self.TOOLS:
            return "memory_search"

        browser_route_cues = self._normalized_signal_list("message_check", "routing", "browser_route_cues")
        file_search_cues = self._normalized_signal_list("message_check", "routing", "file_search_cues")

        if self._should_prefer_direct_conversational_reply(raw):
            return ""

        if self._contains_url_or_domain(raw):
            if any(cue in normalized for cue in browser_route_cues):
                if "browser_action" in self.TOOLS:
                    return "browser_action"
            if "web_search" in self.TOOLS:
                return "web_search"

        if self._contains_file_path(raw):
            if any(x in normalized for x in file_search_cues) and "search_files" in self.TOOLS:
                return "search_files"
            if "list_files" in self.TOOLS:
                return "list_files"

        if self._looks_like_terminal_request(raw) and "terminal_execute" in self.TOOLS:
            return "terminal_execute"

        if self._looks_like_python_request(raw) and "python_action" in self.TOOLS:
            return "python_action"

        if self._looks_like_live_web_request(raw) and "web_search" in self.TOOLS:
            return "web_search"

        return ""

    def _build_tool_inventory_reply(self) -> str:
        tool_names = sorted(str(name).strip() for name in self.TOOLS.keys() if str(name).strip())
        if not tool_names:
            return "Su anda erisilebilir arac gorunmuyor."
        return "Su anda kullanabildigim araclar:\n" + "\n".join(f"- {name}" for name in tool_names)

    def _matched_routed_tool_name(self, actions: list[Any]) -> str:
        if not (self._is_general_root_agent() and self._is_ollama_tool_guard_enabled()):
            return ""
        routed_tool = self._route_obvious_tool_request(
            self._original_user_turn_raw()
        )
        if not routed_tool or routed_tool == "__tool_inventory__":
            return ""
        if len(actions) != 1:
            return ""
        action = actions[0] if isinstance(actions[0], dict) else {}
        tool_name = str(action.get("toolName", "") or "").strip()
        return tool_name if tool_name == routed_tool else ""

    def _should_pause_after_browser_open(
        self,
        actions: list[Any],
        operation_success: bool,
    ) -> bool:
        if not operation_success:
            return False
        if not (self._is_general_root_agent() and self._is_ollama_tool_guard_enabled()):
            return False

        for action in actions:
            if not isinstance(action, dict):
                continue
            if str(action.get("toolName", "") or "").strip() != "browser_action":
                continue

            args = action.get("args", {})
            if not isinstance(args, dict):
                args = {}
            browser_action = str(args.get("action", "") or "").strip().lower()
            if browser_action in {"launch", "open", "goto", "navigate"}:
                return True
            if str(args.get("url", "") or "").strip():
                return True

        return False

    def _build_routed_tool_followup_directive(self, tool_name: str) -> str:
        return (
            "[TOOL RESULT]\n"
            f"The required tool {tool_name} already ran for this user turn.\n\n"
            "Use the existing Tool Results to answer directly.\n"
            f"Do NOT call {tool_name} again for this turn."
        )

    async def _maybe_finish_routed_tool_turn(
        self,
        actions: list[Any],
        tracer: Optional["Tracer"],
        operation_success: bool,
    ) -> bool:
        tool_name = self._matched_routed_tool_name(actions)
        if not tool_name:
            return False

        self.state.add_message("system", self._build_routed_tool_followup_directive(tool_name))
        followup_reply = await self._generate_direct_reply_without_tools(tracer)
        if followup_reply:
            self.state.add_message("assistant", followup_reply)
            self._persist_general_messages_to_disk()
            if tracer:
                tracer.clear_streaming_content(self.state.agent_id)
                tracer.log_chat_message(
                    content=clean_content(followup_reply),
                    role="assistant",
                    agent_id=self.state.agent_id,
                )

        if self._should_use_dynamic_general_routing():
            self.state.enter_waiting_state()
            if tracer:
                tracer.update_agent_status(self.state.agent_id, "waiting")
            self._mark_waiting_in_tracer_and_graph()

        self.state.update_context("memory_feedback_success", bool(operation_success))
        return True

    def _build_forced_tool_guidance(self, tool_name: str) -> str:
        if tool_name == "memory_search":
            query = self._original_user_turn_raw() or "benim adim neydi?"
            return (
                "Memory recall is required.\n"
                "Do not answer from guesswork.\n"
                "Use exactly one valid tool call now.\n"
                "Use memory_search.\n"
                "Use the exact XML tool-call format below.\n"
                "<function=memory_search>\n"
                f"<parameter=query>{query}</parameter>\n"
                "</function>"
            )
        if tool_name == "web_search":
            return (
                "External lookup required.\n"
                "Do not answer from memory.\n"
                "Use exactly one valid tool call now.\n"
                "Use web_search.\n"
                "Use the exact XML tool-call format below.\n"
                "<function=web_search>\n"
                "<parameter=query>your search query</parameter>\n"
                "</function>"
            )
        if tool_name == "browser_action":
            target_url = self._extract_browser_target_url(self._original_user_turn_raw())
            if not target_url:
                target_url = "https://example.com"
            return (
                "External page interaction required.\n"
                "Do not answer from memory.\n"
                "Use exactly one valid tool call now.\n"
                "Use browser_action.\n"
                "Use the exact XML tool-call format below.\n"
                "Do not call memory_search, web_search, think, or any other tool first.\n"
                "Do not output shell-style commands such as 'list /url', 'open url', or 'goto url'.\n"
                "<function=browser_action>\n"
                "<parameter=action>launch</parameter>\n"
                f"<parameter=url>{target_url}</parameter>\n"
                "</function>"
            )
        if tool_name == "search_files":
            return (
                "File search is required.\n"
                "Do not answer from memory.\n"
                "Use exactly one valid tool call now.\n"
                "Use search_files.\n"
                "Use the exact XML tool-call format below.\n"
                "<function=search_files>\n"
                "<parameter=path>/workspace</parameter>\n"
                "<parameter=regex>TODO</parameter>\n"
                "<parameter=file_pattern>*.py</parameter>\n"
                "</function>"
            )
        if tool_name == "list_files":
            return (
                "Directory inspection is required.\n"
                "Do not answer from memory.\n"
                "Use exactly one valid tool call now.\n"
                "Use list_files.\n"
                "Use the exact XML tool-call format below.\n"
                "<function=list_files>\n"
                "<parameter=path>/workspace</parameter>\n"
                "</function>"
            )
        if tool_name == "python_action":
            python_code = self._extract_python_code(self._original_user_turn_raw())
            if python_code:
                return (
                    "Python execution is required.\n"
                    "Do not answer from memory.\n"
                    "Use exactly one valid tool call now.\n"
                    "Use python_action.\n"
                    "Use the exact XML tool-call format below.\n"
                    "<function=python_action>\n"
                    "<parameter=action>execute</parameter>\n"
                    f"<parameter=code>{python_code}</parameter>\n"
                    "</function>"
                )
            return (
                "Python execution is required.\n"
                "Do not answer from memory.\n"
                "Use exactly one valid tool call now.\n"
                "Use python_action.\n"
                "Use the exact XML tool-call format below.\n"
                "<function=python_action>\n"
                "<parameter=action>new_session</parameter>\n"
                "</function>"
            )
        if tool_name == "terminal_execute":
            return (
                "Terminal execution is required.\n"
                "Do not answer from memory.\n"
                "Use exactly one valid tool call now.\n"
                "Use terminal_execute.\n"
                "Use the exact XML tool-call format below.\n"
                "<function=terminal_execute>\n"
                "<parameter=command>pwd</parameter>\n"
                "</function>"
            )
        return (
            "An external action is required.\n"
            "Do not answer from memory.\n"
            "Use exactly one valid tool call now."
        )

    def _looks_like_fake_tool_usage_reply(self, text: str) -> bool:
        normalized = self._normalize_user_text(text)
        if not normalized:
            return False
        fake_patterns = self._normalized_signal_list(
            "message_check", "conversation", "fake_tool_usage_patterns"
        )
        return any(p in normalized for p in fake_patterns)
    async def _execute_actions(self, actions: list[Any], tracer: Optional["Tracer"]) -> bool:
        """Execute actions and return True if agent should finish."""
        original_user_turn = self._original_user_turn_raw()
        if self._is_general_root_agent() and self._should_prefer_direct_conversational_reply(
            original_user_turn
        ):
            if tracer:
                tracer.clear_streaming_content(self.state.agent_id)
            if not self._is_model_regeneration_retry_enabled():
                if self._should_use_dynamic_general_routing():
                    self.state.enter_waiting_state()
                    if tracer:
                        tracer.update_agent_status(self.state.agent_id, "waiting")
                    self._mark_waiting_in_tracer_and_graph()
                return False
            if self._queue_direct_turn_tool_regeneration(
                original_user_turn,
                "direct_turn_tool_execution_suppressed",
                tracer,
            ):
                return False
            generated_reply = await self._generate_direct_reply_without_tools(tracer)
            if generated_reply:
                self._emit_assistant_message(generated_reply)
                self._persist_general_messages_to_disk()
            if self._should_use_dynamic_general_routing():
                self.state.enter_waiting_state()
                if tracer:
                    tracer.update_agent_status(self.state.agent_id, "waiting")
                self._mark_waiting_in_tracer_and_graph()
            return False

        for action in actions:
            self.state.add_action(action)

        operation_success = True
        remaining_actions: list[Any] = []
        memory_actions: list[Any] = []
        last_memory_rendered_result: str = ""
        memory_tool_executed_for_turn = False
        memory_tool_skipped_for_turn = False
        for action in actions:
            tool_name = str(action.get("toolName", "")).strip()
            is_memory_tool = tool_name in {"memory_search", "memory_get"}
            if self._is_general_root_agent() and is_memory_tool and tool_name in self.TOOLS:
                memory_actions.append(action)
            else:
                remaining_actions.append(action)

        for action in memory_actions:
            tool_name = str(action.get("toolName", "")).strip()
            args = action.get("args", {})
            if not isinstance(args, dict):
                args = {}
            query = str(args.get("query", "") or "").strip()
            citation = str(args.get("citation", "") or "").strip()
            memory_id = str(args.get("id", "") or "").strip()
            source = str(args.get("source", "") or "").strip()
            index = str(args.get("index", "") or args.get("rank", "") or "").strip()
            if tool_name == "memory_search":
                prompt_signal = self._prompt_memory_signal(self._latest_user_message_raw())
                if not query:
                    query = str(prompt_signal.get("suggested_query", "") or self._latest_user_message_raw()).strip()
                if not self._is_explicit_memory_request(self._latest_user_message_raw()):
                    memory_tool_skipped_for_turn = True
                    self.state.add_message(
                        "system",
                        (
                            "[MEMORY RESULT]\n"
                            "Memory lookup skipped for this turn.\n\n"
                            "Prefer MEMORY CONTEXT and answer directly.\n"
                            "Do NOT call memory_search again."
                        ),
                    )
                    self._record_episodic_event(
                        action=tool_name,
                        result="skipped: not explicit memory request",
                        success=False,
                        metadata={"reason": "not_explicit_memory_request"},
                    )
                    continue
                query_key = query.strip().lower()
                if self._memory_calls_this_turn >= self.MAX_MEMORY_CALLS_PER_TURN:
                    memory_tool_skipped_for_turn = True
                    self.state.add_message(
                        "system",
                        (
                            "[MEMORY RESULT]\n"
                            "Memory is already retrieved for this user turn.\n\n"
                            "Answer directly now.\n"
                            "Do NOT call memory_search again."
                        ),
                    )
                    self._record_episodic_event(
                        action=tool_name,
                        result="skipped: memory already retrieved this turn",
                        success=False,
                        metadata={"reason": "max_memory_calls_per_turn"},
                    )
                    continue
                if query_key and self._last_memory_tool_query == query_key:
                    memory_tool_skipped_for_turn = True
                    self.state.add_message(
                        "system",
                        (
                            "[MEMORY RESULT]\n"
                            "This memory query is already handled in this user turn.\n\n"
                            "Answer directly now.\n"
                            "Do NOT call memory_search again."
                        ),
                    )
                    self._record_episodic_event(
                        action=tool_name,
                        result="skipped: duplicate memory query in current turn",
                        success=False,
                        metadata={"reason": "duplicate_query_last"},
                    )
                    continue
                if query_key and query_key in self._memory_queries_this_turn:
                    memory_tool_skipped_for_turn = True
                    self.state.add_message(
                        "system",
                        (
                            "[MEMORY RESULT]\n"
                            "Memory for this query is already retrieved in this user turn.\n\n"
                            "Answer directly now.\n"
                            "Do NOT call memory_search again."
                        ),
                    )
                    self._record_episodic_event(
                        action=tool_name,
                        result="skipped: memory query already handled in current turn",
                        success=False,
                        metadata={"reason": "duplicate_query_set"},
                    )
                    continue
                self._last_memory_tool_query = query_key
                if query_key:
                    self._memory_queries_this_turn.add(query_key)
                self._emit_assistant_message(self.PERSISTENT_MEMORY_READING_MESSAGE)
            memory_tool_executed_for_turn = True
            execution_id = None
            if tracer and tool_name not in {"memory_search", "memory_get"}:
                execution_id = tracer.log_tool_execution_start(
                    self.state.agent_id,
                    tool_name,
                    dict(args),
                )
            try:
                start_time = time.monotonic()
                tool_func = self.TOOLS.get(tool_name)
                if tool_name == "memory_search":
                    tool_result = await asyncio.wait_for(
                        asyncio.to_thread(self._safe_memory_tool_call, tool_func, query=query),
                        timeout=float(self.MEMORY_TOOL_TIMEOUT_SEC),
                    )
                elif tool_name == "memory_get":
                    tool_result = await asyncio.wait_for(
                        asyncio.to_thread(
                            self._safe_memory_tool_call,
                            tool_func,
                            citation=citation,
                            id=memory_id,
                            source=source,
                            query=query,
                            index=index,
                        ),
                        timeout=float(self.MEMORY_TOOL_TIMEOUT_SEC),
                    )
                else:
                    if not tool_func:
                        raise ValueError(f"Tool '{tool_name}' is not available")
                    tool_result = tool_func(**args)
                elapsed = time.monotonic() - start_time
                if tool_name in {"memory_search", "memory_get"} and elapsed > 3.0:
                    logger.warning("[MEMORY] slow tool call: %s took %.2fs", tool_name, elapsed)
                if tracer and execution_id:
                    tracer.update_tool_execution(execution_id, "completed", tool_result)
            except asyncio.TimeoutError:
                logger.warning("[MEMORY] tool timeout: %s", tool_name)
                operation_success = False
                tool_result = [
                    "[MEMORY STATUS]",
                    "status: timeout",
                    "reason: memory_tool_timeout",
                    "action: answer directly without additional memory retrieval",
                ]
                self._record_episodic_event(
                    action=tool_name,
                    result="failed: timeout",
                    success=False,
                    metadata={"reason": "timeout"},
                )
                if tracer and execution_id:
                    tracer.update_tool_execution(execution_id, "error", "memory_tool_timeout")
            except Exception as e:  # noqa: BLE001
                operation_success = False
                tool_result = f"Error: {e}"
                self._record_episodic_event(
                    action=tool_name,
                    result=f"failed: {e}",
                    success=False,
                    metadata={"reason": "exception"},
                )
                if tracer and execution_id:
                    tracer.update_tool_execution(execution_id, "error", str(e))
            finally:
                if tool_name == "memory_search":
                    self._memory_calls_this_turn += 1

            if isinstance(tool_result, list):
                rendered_lines = self._prepare_memory_result_lines(tool_result)
                rendered_result = "\n".join([f"- {line}" for line in rendered_lines])
                if not rendered_result:
                    rendered_result = "(no memory found)"
            else:
                rendered_result = str(tool_result)
            rendered_result = self._truncate_memory_result(rendered_result)
            last_memory_rendered_result = rendered_result
            tool_success = self._infer_tool_success(rendered_result)
            self._record_episodic_event(
                action=tool_name,
                result=rendered_result,
                success=tool_success,
                metadata={"query": query, "citation": citation, "id": memory_id, "source": source, "index": index},
            )
            if not tool_success:
                operation_success = False

            self._clear_memory_result_messages()
            self._clear_memory_prompt_directive_messages()
            self.state.add_message(
                "system",
                (
                    f"[MEMORY RESULT]\n{rendered_result}\n\n"
                    "Use this information to continue your response.\n"
                    "You now have the memory.\n"
                    "Answer directly.\n"
                    "Do NOT call memory_search again for this user turn."
                ),
            )

        conversation_history = self.state.get_conversation_history()
        should_agent_finish = False
        if remaining_actions:
            errors_before = len(self.state.errors)
            tool_task = asyncio.create_task(
                process_tool_invocations(remaining_actions, conversation_history, self.state)
            )
            self._current_task = tool_task

            try:
                should_agent_finish = await tool_task
                self._current_task = None
                tool_invocations_success = len(self.state.errors) == errors_before
                if not tool_invocations_success:
                    operation_success = False
                self._record_episodic_event(
                    action="process_tool_invocations",
                    result=", ".join(
                        sorted(
                            {
                                str(a.get("toolName", "")).strip()
                                for a in remaining_actions
                                if isinstance(a, dict) and str(a.get("toolName", "")).strip()
                            }
                        )
                    ),
                    success=tool_invocations_success,
                    metadata={"count": len(remaining_actions)},
                )
            except asyncio.CancelledError:
                self._current_task = None
                self.state.add_error("Tool execution cancelled by user")
                self.state.update_context("memory_feedback_success", False)
                self._record_episodic_event(
                    action="process_tool_invocations",
                    result="failed: cancelled",
                    success=False,
                    metadata={"count": len(remaining_actions)},
                )
                raise

        web_search_browser_fallback = self._build_web_search_browser_fallback_action(
            remaining_actions,
            conversation_history,
        )
        if web_search_browser_fallback:
            fallback_task = asyncio.create_task(
                process_tool_invocations([web_search_browser_fallback], conversation_history, self.state)
            )
            self._current_task = fallback_task
            fallback_failed = False
            try:
                await fallback_task
                self._current_task = None
                fallback_failed = self._tool_result_indicates_failure(
                    "browser_action",
                    conversation_history,
                )
                if fallback_failed:
                    operation_success = False
                self._record_episodic_event(
                    action="web_search_browser_fallback",
                    result=str(
                        (web_search_browser_fallback.get("args", {}) or {}).get("url", "")
                    ),
                    success=not fallback_failed,
                    metadata={"source": "web_search"},
                )
            except asyncio.CancelledError:
                self._current_task = None
                self.state.add_error("Browser fallback cancelled by user")
                self.state.update_context("memory_feedback_success", False)
                self._record_episodic_event(
                    action="web_search_browser_fallback",
                    result="failed: cancelled",
                    success=False,
                    metadata={"source": "web_search"},
                )
                raise

            self.state.messages = conversation_history
            if fallback_failed:
                self._emit_assistant_message(
                    "Web search failed, and I could not open the browser fallback."
                )
                self._persist_general_messages_to_disk()
                self.state.update_context("memory_feedback_reason", "tool_failure")
            else:
                self.state.add_message(
                    "system",
                    self._build_routed_tool_followup_directive("browser_action"),
                )
                followup_reply = await self._generate_direct_reply_without_tools(tracer)
                if followup_reply:
                    self.state.add_message("assistant", followup_reply)
                    self._persist_general_messages_to_disk()
                    if tracer:
                        tracer.clear_streaming_content(self.state.agent_id)
                        tracer.log_chat_message(
                            content=clean_content(followup_reply),
                            role="assistant",
                            agent_id=self.state.agent_id,
                        )
            if self._should_use_dynamic_general_routing():
                self.state.enter_waiting_state()
                if tracer:
                    tracer.update_agent_status(self.state.agent_id, "waiting")
                self._mark_waiting_in_tracer_and_graph()
            self.state.update_context("memory_feedback_success", not fallback_failed)
            return False

        self.state.messages = conversation_history
        if memory_actions and not remaining_actions:
            turn_key = self._current_user_turn_key()
            already_sent_for_turn = (
                str(self.state.context.get("memory_reply_sent_for_turn", "") or "").strip() == turn_key
            )
            if not already_sent_for_turn:
                if memory_tool_skipped_for_turn:
                    self.state.add_message(
                        "system",
                        (
                            "[MEMORY RESULT]\n"
                            "Memory tool call was skipped for this user turn.\n\n"
                            "Use existing MEMORY CONTEXT and answer directly.\n"
                            "Do NOT call memory_search again for this turn."
                        ),
                    )
                elif memory_tool_executed_for_turn:
                    self.state.add_message(
                        "system",
                        (
                            "[MEMORY RESULT]\n"
                            "Memory retrieval completed.\n\n"
                            "Answer the user directly now.\n"
                            "Do NOT call memory_search again for this turn."
                        ),
                    )

                followup_reply = await self._generate_direct_reply_without_tools(tracer)
                if followup_reply:
                    self.state.add_message("assistant", followup_reply)
                    self._persist_general_messages_to_disk()
                    if tracer:
                        tracer.clear_streaming_content(self.state.agent_id)
                        tracer.log_chat_message(
                            content=clean_content(followup_reply),
                            role="assistant",
                            agent_id=self.state.agent_id,
                        )
                elif (
                    memory_tool_executed_for_turn
                    and last_memory_rendered_result
                    and self._is_model_regeneration_retry_enabled()
                ):
                    self.state.add_message(
                        "system",
                        (
                            "[MEMORY RESULT]\n"
                            f"{last_memory_rendered_result}\n\n"
                            "Reply again in plain text only.\n"
                            "Use the memory result naturally in your answer.\n"
                            "Do NOT call memory_search again for this turn.\n"
                            "Do NOT call any other tools."
                        ),
                    )
                    self.state.update_context("memory_feedback_success", False)
                    self.state.update_context("memory_feedback_reason", "reasoning_error")
                    return False
                self.state.update_context("memory_reply_sent_for_turn", turn_key)
            if self._should_use_dynamic_general_routing():
                self.state.enter_waiting_state()
                if tracer:
                    tracer.update_agent_status(self.state.agent_id, "waiting")
                self._mark_waiting_in_tracer_and_graph()
            if not bool(operation_success):
                current_reason = str(self.state.context.get("memory_feedback_reason", "") or "").strip().lower()
                if current_reason in {"", "success"}:
                    self.state.update_context("memory_feedback_reason", "tool_failure")
            self.state.update_context("memory_feedback_success", bool(operation_success))
            return False

        if await self._maybe_finish_routed_tool_turn(remaining_actions, tracer, bool(operation_success)):
            return False

        if should_agent_finish:
            if not bool(operation_success):
                current_reason = str(self.state.context.get("memory_feedback_reason", "") or "").strip().lower()
                if current_reason in {"", "success"}:
                    self.state.update_context("memory_feedback_reason", "tool_failure")
            self.state.update_context("memory_feedback_success", bool(operation_success))
            self.state.set_completed({"success": True})
            if tracer:
                tracer.update_agent_status(self.state.agent_id, "completed")
            if self.non_interactive and self.state.parent_id is None:
                return True
            return True

        if not bool(operation_success):
            current_reason = str(self.state.context.get("memory_feedback_reason", "") or "").strip().lower()
            if current_reason in {"", "success"}:
                self.state.update_context("memory_feedback_reason", "tool_failure")
        self.state.update_context("memory_feedback_success", bool(operation_success))
        return False

    async def _check_agent_messages(self, state: AgentState) -> None:  # noqa: PLR0912
        try:
            from sondra.tools.agents_graph.agents_graph_actions import _agent_graph, _agent_messages

            agent_id = state.agent_id
            if not agent_id or agent_id not in _agent_messages:
                return

            messages = _agent_messages[agent_id]
            if messages:
                has_new_messages = False
                control_command_handled = False
                for message in messages:
                    if not message.get("read", False):
                        sender_id = message.get("from")

                        if state.is_waiting_for_input():
                            if state.llm_failed:
                                if sender_id == "user":
                                    state.resume_from_waiting()
                                    has_new_messages = True

                                    from sondra.telemetry.tracer import get_global_tracer

                                    tracer = get_global_tracer()
                                    if tracer:
                                        tracer.update_agent_status(state.agent_id, "running")
                            else:
                                state.resume_from_waiting()
                                has_new_messages = True

                                from sondra.telemetry.tracer import get_global_tracer

                                tracer = get_global_tracer()
                                if tracer:
                                    tracer.update_agent_status(state.agent_id, "running")

                        if sender_id == "user":
                            sender_name = "User"
                            user_content = str(message.get("content", ""))
                            if self._is_general_root_agent() and self._is_persistent_memory_reset_command(user_content):
                                if self._handle_persistent_memory_reset_command():
                                    has_new_messages = True
                                    message["read"] = True
                                    control_command_handled = True
                                    break
                            state.update_context("visual_screenshot_attempted_for_user", "")
                            state.update_context("last_user_turn_raw", user_content)
                            state.update_context("last_user_message_event_id", str(message.get("id", "") or "").strip())
                            state.update_context("memory_reply_sent_for_turn", "")
                            state.update_context("context_overflow_active", False)
                            state.update_context("context_overflow_retry_count", 0)
                            state.update_context("forced_tool_name", "")
                            state.update_context("forced_tool_retry_count", 0)
                            self._memory_calls_this_turn = 0
                            self._memory_queries_this_turn.clear()
                            self._memory_sync_turn_key = ""
                            self._last_memory_tool_query = ""
                            self._seen_memory_for_turn.clear()
                            self._last_memory_hits = []
                            self._last_memory_query = ""
                            self._clear_memory_result_messages()
                            self._clear_memory_prompt_directive_messages()
                            state.add_message("user", user_content)
                            self._persist_general_messages_to_disk()
                            if self._is_general_root_agent():
                                self._ensure_task_runner_started()
                                if self._handle_scheduled_task_creation_request(user_content):
                                    has_new_messages = True
                                    message["read"] = True
                                    control_command_handled = True
                                    break
                            if self._is_general_root_agent():
                                self._extract_profile_fact(user_content)
                                self._inject_auto_memory_context(user_content)
                            if self._is_general_root_agent():
                                self._schedule_semantic_and_task_updates(user_content)
                            if self._is_general_root_agent():
                                if self._handle_list_scheduled_tasks_request(user_content):
                                    has_new_messages = True
                                    message["read"] = True
                                    control_command_handled = True
                                    break
                                if self._handle_delete_indexed_task_request(user_content):
                                    has_new_messages = True
                                    message["read"] = True
                                    control_command_handled = True
                                    break
                                if self._handle_delete_all_tasks_request(user_content):
                                    has_new_messages = True
                                    message["read"] = True
                                    control_command_handled = True
                                    break
                        else:
                            if sender_id and sender_id in _agent_graph.get("nodes", {}):
                                sender_name = _agent_graph["nodes"][sender_id]["name"]

                            message_content = f"""<inter_agent_message>
    <delivery_notice>
        <important>You have received a message from another agent. You should acknowledge
        this message and respond appropriately based on its content. However, DO NOT echo
        back or repeat the entire message structure in your response. Simply process the
        content and respond naturally as/if needed.</important>
    </delivery_notice>
    <sender>
        <agent_name>{sender_name}</agent_name>
        <agent_id>{sender_id}</agent_id>
    </sender>
    <message_metadata>
        <type>{message.get("message_type", "information")}</type>
        <priority>{message.get("priority", "normal")}</priority>
        <timestamp>{message.get("timestamp", "")}</timestamp>
    </message_metadata>
    <content>
{message.get("content", "")}
    </content>
    <delivery_info>
        <note>This message was delivered during your task execution.
        Please acknowledge and respond if needed.</note>
    </delivery_info>
</inter_agent_message>"""
                            state.add_message("user", message_content.strip())

                        message["read"] = True

                if control_command_handled:
                    return

                if has_new_messages and not state.is_waiting_for_input():
                    from sondra.telemetry.tracer import get_global_tracer

                    tracer = get_global_tracer()
                    if tracer:
                        tracer.update_agent_status(agent_id, "running")

        except Exception as e:  # noqa: BLE001
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Error checking agent messages: {e}")
            return

    def _is_general_root_agent(self) -> bool:
        llm_config = getattr(self, "llm_config", None)
        state = getattr(self, "state", None)
        return (
            getattr(llm_config, "scan_mode", None) == "general"
            and getattr(state, "parent_id", None) is None
        )

    def _ensure_general_prepare_hook(self) -> None:
        if not self._is_general_root_agent():
            return
        if getattr(self.llm, "_general_prepare_hook_installed", False):
            return

        self.llm.system_prompt = str(self.llm.system_prompt or "")

        def _prepare_messages_with_memory(
            llm_self: Any,
            conversation_history: list[dict[str, Any]],
        ) -> list[dict[str, Any]]:
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": str(llm_self.system_prompt or "")}
            ]

            memory_messages = [
                msg for msg in self.state.messages if self._message_contains_memory_context(msg)
            ]
            if memory_messages:
                messages.append(memory_messages[-1])

            overflow_active = bool(self.state.context.get("context_overflow_active", False))
            max_chars = self._history_prompt_budget_chars()
            system_history = [msg for msg in conversation_history if msg.get("role") == "system"]
            normal_history = [msg for msg in conversation_history if msg.get("role") != "system"]

            selected_system = system_history[-llm_self.memory_compressor.keep_system_messages :]
            selected_normal_reversed: list[dict[str, Any]] = []
            total_chars = 0
            max_tail = 20 if overflow_active else 120

            for msg in reversed(normal_history):
                size = llm_self.memory_compressor._message_size(msg)
                if selected_normal_reversed and (total_chars + size) > max_chars:
                    break
                selected_normal_reversed.append(msg)
                total_chars += size
                if len(selected_normal_reversed) >= max_tail:
                    break

            selected_normal = list(reversed(selected_normal_reversed))
            selected = selected_system + selected_normal

            if overflow_active:
                self.state.update_context("context_overflow_active", False)
                self.state.update_context("context_overflow_retry_count", 0)

            for msg in selected:
                if self._message_contains_memory_context(msg):
                    continue
                messages.append(msg)

            return messages

        self.llm._prepare_messages = MethodType(  # type: ignore[method-assign]
            _prepare_messages_with_memory, self.llm
        )
        self.llm._general_prepare_hook_installed = True  # type: ignore[attr-defined]

    def _latest_user_message_raw(self) -> str:
        for msg in reversed(self.state.messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
            content = self._strip_internal_metadata_blocks(content).strip()
            if not content:
                continue
            if "<inter_agent_message>" in content:
                continue
            if content.startswith("Tool Results:") or "<tool_result>" in content:
                continue
            if self._is_memory_context_text(content):
                continue
            if self._is_internal_metadata_text(content):
                continue
            if self._is_internal_control_user_message(content):
                continue
            return content
        return ""

    def _original_user_turn_raw(self) -> str:
        cached = str(self.state.context.get("last_user_turn_raw", "") or "").strip()
        if cached:
            return cached
        return self._latest_user_message_raw().strip()

    def _current_user_turn_key(self) -> str:
        return self._original_user_turn_raw()

    def _visual_screenshot_attempted_for_current_turn(self) -> bool:
        marker = str(self.state.context.get("visual_screenshot_attempted_for_user", "") or "").strip()
        current_user = self._current_user_turn_key()
        return bool(marker and current_user and marker == current_user)

    def _mark_visual_screenshot_attempted_for_current_turn(self) -> None:
        current_user = self._current_user_turn_key()
        if current_user:
            self.state.update_context("visual_screenshot_attempted_for_user", current_user)

    def _is_memory_context_text(self, text: str) -> bool:
        content = str(text or "").strip()
        if not content:
            return False
        return (
            content.startswith(self.MEMORY_CONTEXT_MARKER)
            or content.startswith(self.LEGACY_MEMORY_CONTEXT_HEADER)
        )

    def _is_memory_result_text(self, text: str) -> bool:
        return str(text or "").strip().startswith("[MEMORY RESULT]")

    def _message_contains_memory_result(self, msg: dict[str, Any]) -> bool:
        content = msg.get("content", "")
        if isinstance(content, str):
            return self._is_memory_result_text(content)
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") != "text":
                    continue
                if self._is_memory_result_text(str(item.get("text", ""))):
                    return True
        return False

    def _clear_memory_result_messages(self) -> None:
        self.state.messages = [
            msg for msg in self.state.messages if not self._message_contains_memory_result(msg)
        ]
        if self._is_general_root_agent():
            persisted_idx = int(self.state.context.get("memory_last_persisted_index", 0))
            if persisted_idx > len(self.state.messages):
                self.state.update_context("memory_last_persisted_index", len(self.state.messages))

    def _clear_memory_prompt_directive_messages(self) -> None:
        filtered: list[dict[str, Any]] = []
        for msg in self.state.messages:
            meta = msg.get("meta", {})
            if isinstance(meta, dict) and str(meta.get("type", "")).strip() == "memory_prompt_directive":
                continue
            filtered.append(msg)
        self.state.messages = filtered

    def _is_local_llm_backend(self) -> bool:
        api_base = str(getattr(self.llm_config, "api_base", "") or "").lower()
        if not api_base:
            return False
        local_markers = self._signal_list("message_check", "routing", "local_backend_markers")
        return any(marker in api_base for marker in local_markers)

    def _history_prompt_budget_chars(self) -> int:
        if self._is_local_llm_backend():
            return self.LOCAL_HISTORY_PROMPT_BUDGET_CHARS
        return self.DEFAULT_HISTORY_PROMPT_BUDGET_CHARS

    def _clean_memory_result_text(self, text: str) -> str:
        return str(text or "").strip()

    def _prompt_memory_signal(self, text: str) -> dict[str, Any]:
        if not self._is_general_root_agent():
            return {}

        query = str(text or "").strip()
        if not query:
            return {}

        if self._looks_like_memory_fact_statement(query):
            return {
                "signal": "fact",
                "force_memory_search": False,
                "allow_memory_search": False,
                "suggested_query": "",
                "route": "prompt_signal",
            }

        if self._looks_like_correction_only_turn(query):
            return {
                "signal": "none",
                "force_memory_search": False,
                "allow_memory_search": False,
                "suggested_query": "",
                "route": "prompt_signal",
            }

        if self._looks_like_social_or_emotional_turn(query):
            return {
                "signal": "none",
                "force_memory_search": False,
                "allow_memory_search": False,
                "suggested_query": "",
                "route": "prompt_signal",
            }

        if self.memory_store:
            with contextlib.suppress(Exception):
                signal = self.memory_store.analyze_prompt_signal(query)
                if isinstance(signal, dict):
                    return dict(signal)

        lowered = query.lower()
        force_terms = self._signal_list("prompt_memory", "prompt_signal", "recall_phrases")
        force = any(phrase in lowered for phrase in force_terms)

        return {
            "signal": "fallback" if force else "none",
            "force_memory_search": bool(force),
            "allow_memory_search": bool(force),
            "suggested_query": query if force else "",
            "route": "prompt_signal",
        }

    def _is_explicit_memory_request(self, text: str) -> bool:
        signal = self._prompt_memory_signal(text)
        return bool(signal.get("allow_memory_search") or signal.get("force_memory_search"))

    def _should_skip_memory_learning(self, text: str) -> bool:
        cleaned = str(text or "").strip()
        if not cleaned:
            return True

        if self._looks_like_memory_fact_statement(cleaned):
            return False

        signal = self._prompt_memory_signal(cleaned)
        if str(signal.get("signal", "") or "").strip().lower() in {"recall", "correction"}:
            return True

        if "?" in cleaned:
            return True

        return False

    def _looks_like_memory_fact_statement(self, text: str) -> bool:
        cleaned = str(text or "").strip()
        if not cleaned:
            return False
        if "?" in cleaned:
            return False
        normalized = self._normalize_user_text(cleaned)
        if not normalized:
            return False
        fact_cues = self._signal_list("semantic_memory", "fact_statement_cues")
        if any(cue in normalized for cue in fact_cues):
            return True
        current_step_patterns = self._signal_list("semantic_memory", "current_step_patterns")
        return any(re.search(pattern, normalized, re.IGNORECASE) for pattern in current_step_patterns)

    def _is_profile_semantic_duplicate(self, semantic_item: str) -> bool:
        normalized = self._normalize_user_text(semantic_item)
        return normalized.startswith("user s name is ") or normalized.startswith("users name is ")

    def _inject_prompt_memory_directive(self, user_content: str) -> None:
        if not self._is_general_root_agent():
            return
        self._clear_memory_prompt_directive_messages()
        signal = self._prompt_memory_signal(user_content)
        if not bool(signal.get("force_memory_search")):
            return
        suggested_query = str(signal.get("suggested_query", "") or user_content or "").strip()
        lines = [
            "[MEMORY SIGNAL]",
            f"signal: {str(signal.get('signal', 'none') or 'none')}",
            "action: call memory_search once before answering",
        ]
        if suggested_query:
            lines.append(f"query: {suggested_query}")
        self.state.add_message(
            "system",
            "\n".join(lines),
            meta={"type": "memory_prompt_directive"},
        )

    def _maybe_force_prompt_memory_search(self, user_content: str) -> None:
        if not self._is_general_root_agent() or not self.memory_store:
            return
        if any(self._message_contains_memory_result(msg) for msg in self.state.messages):
            return
        signal = self._prompt_memory_signal(user_content)
        if not bool(signal.get("force_memory_search")):
            return
        query = str(signal.get("suggested_query", "") or user_content or "").strip()
        if not query:
            return
        query_key = query.strip().lower()
        if query_key in self._memory_queries_this_turn:
            return
        with contextlib.suppress(Exception):
            self._emit_assistant_message(self.PERSISTENT_MEMORY_READING_MESSAGE)
        with contextlib.suppress(Exception):
            tool_result = self.memory_search(query)
            self._memory_calls_this_turn += 1
            self._memory_queries_this_turn.add(query_key)
            if isinstance(tool_result, list):
                rendered_lines = self._prepare_memory_result_lines(tool_result)
                rendered_result = "\n".join([f"- {line}" for line in rendered_lines]) or "(no memory found)"
            else:
                rendered_result = str(tool_result or "(no memory found)")
            rendered_result = self._truncate_memory_result(rendered_result)
            self._clear_memory_result_messages()
            self.state.add_message(
                "system",
                (
                    f"[MEMORY RESULT]\n{rendered_result}\n\n"
                    "Use this information to continue your response.\n"
                    "You now have the memory.\n"
                    "Answer directly.\n"
                    "Do NOT call memory_search again for this user turn."
                ),
            )

    async def _generate_direct_reply_without_tools(self, tracer: Optional["Tracer"]) -> str:
        """Generate one direct assistant reply with tools disabled."""
        final_response = None
        conversation_history = list(self.state.get_conversation_history())
        conversation_history.append(
            {
                "role": "system",
                "content": (
                    "Reply in plain text only. "
                    "Do NOT call any tools. "
                    "Do NOT output internal reasoning, chain-of-thought, planning notes, or analysis."
                ),
            }
        )
        response_stream = self.llm.generate(conversation_history)

        async for response in response_stream:
            final_response = response
            if response.content and tracer:
                raw_stream_content = str(response.content)
                sanitized_stream = self._strip_internal_metadata_blocks(raw_stream_content)
                if self._looks_like_tool_payload_prefix(raw_stream_content):
                    sanitized_stream = ""
                sanitized_stream = self._sanitize_model_output_for_user(sanitized_stream)
                if sanitized_stream and self._looks_like_tool_or_command_output_reply(sanitized_stream):
                    sanitized_stream = ""
                if (
                    sanitized_stream
                    and not self._looks_like_internal_reasoning_prefix(raw_stream_content)
                    and not self._looks_like_tool_payload_prefix(raw_stream_content)
                ):
                    tracer.update_streaming_content(self.state.agent_id, sanitized_stream)

        if final_response is None:
            return ""

        raw_model_content = str(final_response.content or "")
        content_stripped = self._strip_internal_metadata_blocks(raw_model_content).strip()
        content_stripped = self._sanitize_model_output_for_user(content_stripped)
        if not content_stripped:
            return ""
        if self._looks_like_tool_or_command_output_reply(content_stripped):
            return ""
        return content_stripped

    def _sanitize_model_output_for_user(self, text: str) -> str:
        cleaned = str(text or "").strip()
        if not cleaned:
            return ""
        cleaned = self._strip_internal_reasoning_payload(cleaned)
        if not cleaned:
            return ""
        cleaned = self._strip_plaintext_reasoning_preamble(cleaned)
        if not cleaned:
            return ""
        if cleaned.startswith(self.PERSISTENT_MEMORY_READING_MESSAGE):
            cleaned = cleaned[len(self.PERSISTENT_MEMORY_READING_MESSAGE):].lstrip(" \n\r\t.:")
        for snippet in self.RESPONSE_NOISE_SNIPPETS:
            if snippet and snippet in cleaned:
                cleaned = cleaned.replace(snippet, "").strip()
        cleaned = self._strip_reasoning_headers(cleaned)
        return cleaned

    def _strip_plaintext_reasoning_preamble(self, text: str) -> str:
        cleaned = str(text or "").strip()
        if not cleaned:
            return ""
        lines = cleaned.splitlines()
        reasoning_markers = self._signal_list("output_filters", "reasoning", "plaintext_markers")
        special_plaintext_tokens = set(self._signal_list("output_filters", "reasoning", "special_plaintext_tokens"))
        saw_reasoning = False
        kept_lines: list[str] = []
        for line in lines:
            stripped = line.strip()
            normalized = self._normalize_match_text(stripped)
            if not kept_lines:
                if normalized in special_plaintext_tokens:
                    saw_reasoning = True
                    continue
                if any(marker in normalized for marker in reasoning_markers):
                    saw_reasoning = True
                    continue
                if saw_reasoning and not stripped:
                    continue
                if saw_reasoning and stripped:
                    kept_lines.append(line)
                    continue
            kept_lines.append(line)
        return "\n".join(kept_lines).strip()

    def _strip_reasoning_headers(self, text: str) -> str:
        cleaned = str(text or "").strip()
        if not cleaned:
            return ""

        marker_lines = set(self._signal_list("output_filters", "reasoning", "header_lines"))
        header_prefixes = tuple(self._signal_list("output_filters", "reasoning", "header_prefixes"))

        kept_lines = []
        for line in cleaned.splitlines():
            candidate = line.strip().lower()
            if candidate in marker_lines:
                continue
            if any(candidate.startswith(prefix) for prefix in header_prefixes):
                continue
            kept_lines.append(line)

        cleaned = "\n".join(kept_lines).strip()
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned

    def _is_internal_reasoning_dict(self, payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False

        internal_keys = set(self._signal_list("output_filters", "reasoning", "internal_keys"))
        visible_output_keys = set(self._signal_list("output_filters", "reasoning", "visible_output_keys"))
        keys = {str(key or "").strip().lower() for key in payload.keys()}
        has_internal = bool(keys & internal_keys)
        if not has_internal:
            return False
        if not (keys & visible_output_keys):
            return True

        for key in ("answer", "response", "final", "content", "output"):
            if str(payload.get(key, "") or "").strip():
                return False

        message_payload = payload.get("message")
        if isinstance(message_payload, dict):
            if str(message_payload.get("content", "") or "").strip():
                return False
        elif isinstance(message_payload, str) and message_payload.strip():
            return False

        return True

    def _looks_like_tool_payload_prefix(self, text: str) -> bool:
        raw = str(text or "").strip()
        if not raw:
            return False
        lowered = raw.lower().lstrip("` \n\t")
        prefixes = self._signal_list("output_filters", "tool_payload", "prefixes")
        return any(lowered.startswith(prefix) for prefix in prefixes)

    def _looks_like_internal_reasoning_prefix(self, text: str) -> bool:
        raw = str(text or "").strip()
        if not raw:
            return False
        lowered = raw.lower().lstrip("` \n\t")
        contains_tokens = self._signal_list("output_filters", "reasoning", "contains_tokens")
        if any(token in lowered for token in contains_tokens):
            return True
        if self._looks_like_tool_payload_prefix(lowered):
            return False
        prefixes = self._signal_list("output_filters", "reasoning", "json_prefixes")
        return any(lowered.startswith(prefix) for prefix in prefixes)

    def _strip_internal_reasoning_payload(self, text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""

        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.IGNORECASE | re.DOTALL).strip()
        if not raw:
            return ""

        payload_text = raw
        if raw.startswith("```") and raw.endswith("```"):
            lines = raw.splitlines()
            if len(lines) >= 3 and lines[-1].strip().startswith("```"):
                payload_text = "\n".join(lines[1:-1]).strip()

        try:
            payload = json.loads(payload_text)
        except Exception:
            if self._looks_like_internal_reasoning_prefix(payload_text):
                return ""
            if self._looks_like_tool_payload_prefix(payload_text):
                return ""
            return raw

        if isinstance(payload, dict):
            action_name = str(
                payload.get("action")
                or payload.get("tool")
                or payload.get("toolname")
                or payload.get("toolName")
                or payload.get("name")
                or ""
            ).strip().lower()
            if action_name == "think":
                return ""
            message_payload = payload.get("message")
            if isinstance(message_payload, dict):
                content_value = str(message_payload.get("content", "") or "").strip()
                if content_value:
                    return content_value
            for key in ("answer", "response", "final", "content", "output"):
                value = str(payload.get(key, "") or "").strip()
                if value:
                    return value
            if self._is_internal_reasoning_dict(payload):
                return ""

        return raw

    def _looks_like_internal_reasoning_payload(self, text: str) -> bool:
        raw = str(text or "").strip()
        if not raw:
            return False
        if "<think>" in raw.lower() and "</think>" in raw.lower():
            return True

        payload_text = raw
        if raw.startswith("```") and raw.endswith("```"):
            lines = raw.splitlines()
            if len(lines) >= 3 and lines[-1].strip().startswith("```"):
                payload_text = "\n".join(lines[1:-1]).strip()

        try:
            payload = json.loads(payload_text)
        except Exception:
            return self._looks_like_internal_reasoning_prefix(payload_text)

        if not isinstance(payload, dict):
            return False

        return self._is_internal_reasoning_dict(payload)

    def _latest_memory_result_text(self) -> str:
        for msg in reversed(self.state.messages):
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
            raw = content.strip()
            if not raw.startswith("[MEMORY RESULT]"):
                continue
            payload = raw[len("[MEMORY RESULT]") :].strip()
            payload = payload.split("\n\nUse this information", 1)[0].strip()
            if payload:
                return payload
        return ""

    def _filter_valid_tool_actions(self, actions: Any) -> list[dict[str, Any]]:
        if not isinstance(actions, list):
            return []
        valid_actions: list[dict[str, Any]] = []
        for action in actions:
            if not isinstance(action, dict):
                continue
            tool_name = str(action.get("toolName", "") or "").strip()
            if not tool_name:
                continue
            if tool_name not in self.TOOLS:
                continue
            args = action.get("args", {})
            if not isinstance(args, dict):
                args = {}
            valid_actions.append({"toolName": tool_name, "args": args})
        return valid_actions


    def _safe_memory_tool_call(self, fn: Any, **kwargs: Any) -> Any:
        if not callable(fn):
            return []
        try:
            return fn(**kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Memory tool call failed: %s", exc)
            return []

    def _json_payload_contains_tool_call(self, payload: Any) -> bool:
        if isinstance(payload, list):
            for item in payload:
                if self._json_payload_contains_tool_call(item):
                    return True
            return False

        if not isinstance(payload, dict):
            return False

        for key in ("tool_calls", "toolcalls", "tools", "calls", "invocations"):
            value = payload.get(key)
            if self._json_payload_contains_tool_call(value):
                return True

        return self._looks_like_direct_tool_action(payload)

    def _prepare_memory_result_lines(self, rows: list[Any]) -> list[str]:
        lines: list[str] = []
        for row in rows:
            text = self._clean_memory_result_text(str(row or ""))
            if not text:
                continue
            key = text.lower()
            if key in self._seen_memory_for_turn:
                continue
            self._seen_memory_for_turn.add(key)
            lines.append(text)
            if len(lines) >= 5:
                break
        return lines

    def _truncate_memory_result(self, text: str) -> str:
        value = str(text or "").strip()
        if len(value) <= self.MAX_MEMORY_RESULT_CHARS:
            return value
        return value[: self.MAX_MEMORY_RESULT_CHARS - 3] + "..."

    def _is_internal_metadata_text(self, text: str) -> bool:
        content = str(text or "")
        if not content:
            return False
        lowered = content.lower()
        if "<inter_agent_message>" in lowered:
            return False
        if self._contains_agent_metadata_tokens(lowered):
            return True
        return (
            "internal metadata: do not echo or reference" in lowered
            or "internal metadata: do not echo or" in lowered
        )

    def _strip_internal_metadata_blocks(self, text: str) -> str:
        raw = str(text or "")
        if not raw:
            return ""
        cleaned = raw
        cleaned = self._strip_tag_block(cleaned, "<agent_identity>", "</agent_identity>")
        cleaned = self._strip_tag_block(cleaned, "<agentidentity>", "</agentidentity>")
        cleaned = self._strip_tag_block(cleaned, "<meta>", "</meta>")
        lines: list[str] = []
        for line in cleaned.splitlines():
            lowered_line = line.lower()
            if self._is_internal_metadata_text(line):
                continue
            if self._contains_agent_metadata_tokens(lowered_line):
                continue
            lines.append(line)
        cleaned = self._collapse_blank_lines("\n".join(lines))
        return cleaned.strip()

    def _contains_agent_metadata_tokens(self, text: str) -> bool:
        lowered = str(text or "").lower()
        metadata_tokens = self._signal_list("output_filters", "metadata", "agent_metadata_tokens")
        return any(token in lowered for token in metadata_tokens)

    def _strip_tag_block(self, text: str, start_tag: str, end_tag: str) -> str:
        source = str(text or "")
        start_lower = start_tag.lower()
        end_lower = end_tag.lower()
        while True:
            lowered = source.lower()
            start_idx = lowered.find(start_lower)
            if start_idx < 0:
                return source
            end_idx = lowered.find(end_lower, start_idx + len(start_lower))
            if end_idx < 0:
                source = source[:start_idx]
                continue
            source = source[:start_idx] + source[end_idx + len(end_tag) :]

    def _collapse_blank_lines(self, text: str) -> str:
        lines = str(text or "").splitlines()
        collapsed: list[str] = []
        blank_run = 0
        for line in lines:
            if line.strip():
                blank_run = 0
                collapsed.append(line)
                continue
            blank_run += 1
            if blank_run <= 1:
                collapsed.append("")
        return "\n".join(collapsed)

    def _is_internal_control_user_message(self, text: str) -> bool:
        lowered = str(text or "").strip().lower()
        if not lowered:
            return False
        internal_control_user_prefixes = self._signal_list("message_check", "internal_control_user_prefixes")
        if any(lowered.startswith(prefix) for prefix in internal_control_user_prefixes):
            return True
        return False

    def _ensure_memory_every_iteration(self) -> None:
        if not self._is_general_root_agent():
            return
        query = str(self.state.context.get("last_user_turn_raw", "") or "").strip()
        if not query:
            query = self._current_user_turn_key()
        if not query:
            return
        self._inject_auto_memory_context(query)

    def _clear_decision_context_messages(self) -> None:
        filtered: list[dict[str, Any]] = []
        for msg in self.state.messages:
            meta = msg.get("meta", {})
            if isinstance(meta, dict) and str(meta.get("type", "")).strip() == "decision_context":
                continue
            filtered.append(msg)
        self.state.messages = filtered

    def _decision_memory_hits_count(self) -> int:
        count = 0
        for msg in self.state.messages:
            meta = msg.get("meta", {})
            if not (isinstance(meta, dict) and str(meta.get("type", "")).strip() == "memory_context"):
                continue
            content = str(msg.get("content", "") or "")
            for line in content.splitlines():
                item = line.strip()
                if not item:
                    continue
                if item in {self.MEMORY_CONTEXT_MARKER, self.MEMORY_CONTEXT_END_MARKER}:
                    continue
                if item.startswith("[") and "]" in item:
                    count += 1
        return count

    def _decision_emotion_state(self) -> dict[str, float]:
        defaults = {
            "confidence": 50.0,
            "stress": float(self.AGENT_EMOTION_DEFAULTS["stress"]),
            "curiosity": 50.0,
            "happiness": float(self.AGENT_EMOTION_DEFAULTS["happiness"]),
            "sadness": float(self.AGENT_EMOTION_DEFAULTS["sadness"]),
            "neutral": float(self.AGENT_EMOTION_DEFAULTS["neutral"]),
        }
        ctx = self.state.context
        return {
            "confidence": self._coerce_emotion_percent(ctx.get("emotion_confidence", defaults["confidence"]), defaults["confidence"]),
            "stress": self._coerce_emotion_percent(ctx.get("emotion_stress", defaults["stress"]), defaults["stress"]),
            "curiosity": self._coerce_emotion_percent(ctx.get("emotion_curiosity", defaults["curiosity"]), defaults["curiosity"]),
            "happiness": self._coerce_emotion_percent(ctx.get("emotion_happiness", defaults["happiness"]), defaults["happiness"]),
            "sadness": self._coerce_emotion_percent(ctx.get("emotion_sadness", defaults["sadness"]), defaults["sadness"]),
            "neutral": self._coerce_emotion_percent(ctx.get("emotion_neutral", defaults["neutral"]), defaults["neutral"]),
        }

    @staticmethod
    def _clamp_emotion_value(value: float, minimum: float = 0.0, maximum: float = 100.0) -> float:
        return max(minimum, min(float(value), maximum))

    def _coerce_emotion_percent(self, value: Any, default: float) -> float:
        with contextlib.suppress(Exception):
            numeric = float(value)
            if 0.0 <= numeric <= 1.5:
                numeric *= 100.0
            return self._clamp_emotion_value(numeric)
        return self._clamp_emotion_value(default)

    def _agent_emotion_state_from_context(self) -> dict[str, float]:
        current = self._decision_emotion_state()
        return {
            "happiness": self._clamp_emotion_value(current.get("happiness", self.AGENT_EMOTION_DEFAULTS["happiness"])),
            "sadness": self._clamp_emotion_value(current.get("sadness", self.AGENT_EMOTION_DEFAULTS["sadness"])),
            "stress": self._clamp_emotion_value(current.get("stress", self.AGENT_EMOTION_DEFAULTS["stress"])),
            "neutral": self._clamp_emotion_value(current.get("neutral", self.AGENT_EMOTION_DEFAULTS["neutral"])),
        }

    def _classify_user_emotion_input(self, summary: dict[str, Any]) -> dict[str, Any]:
        raw_scores = summary.get("scores", {})
        scores = raw_scores if isinstance(raw_scores, dict) else {}
        happiness = max(0.0, min(float(scores.get("happiness", 0.0) or 0.0), 1.0))
        sadness = max(0.0, min(float(scores.get("sadness", 0.0) or 0.0), 1.0))
        anger = max(0.0, min(float(scores.get("anger", 0.0) or 0.0), 1.0))
        frustration = max(0.0, min(float(scores.get("frustration", 0.0) or 0.0), 1.0))
        neutral = max(0.0, min(float(scores.get("neutral", 0.0) or 0.0), 1.0))
        confidence = max(0.0, min(float(summary.get("confidence", 0.35) or 0.35), 1.0))

        aggressive_strength = max(anger, min(1.0, sadness * 0.65 + frustration * 0.25))
        critical_strength = max(frustration, min(1.0, anger * 0.35 + sadness * 0.25))
        positive_strength = max(happiness, confidence * 0.45 if happiness > 0.25 else 0.0)
        negative_strength = sadness
        neutral_strength = max(neutral, confidence * 0.4 if neutral > 0.45 else 0.0)

        category = "neutral"
        weight = 6.0 + (16.0 * neutral_strength)
        primary_strength = neutral_strength

        if aggressive_strength >= 0.58 or (anger >= 0.38 and sadness >= 0.4):
            category = "aggressive"
            primary_strength = aggressive_strength
            weight = 12.0 + (24.0 * aggressive_strength)
        elif critical_strength >= 0.55 or (frustration >= 0.42 and anger >= 0.22):
            category = "critical"
            primary_strength = critical_strength
            weight = 8.0 + (18.0 * critical_strength)
        elif positive_strength >= 0.55:
            category = "positive"
            primary_strength = positive_strength
            weight = 8.0 + (22.0 * positive_strength)
        elif negative_strength >= 0.48:
            category = "negative"
            primary_strength = negative_strength
            weight = 8.0 + (20.0 * negative_strength)

        return {
            "category": category,
            "weight": self._clamp_emotion_value(weight, 4.0, 36.0),
            "strength": max(0.0, min(primary_strength, 1.0)),
            "scores": {
                "happiness": happiness,
                "sadness": sadness,
                "anger": anger,
                "frustration": frustration,
                "neutral": neutral,
                "confidence": confidence,
            },
        }

    def _derive_emotion_curiosity(self, state: dict[str, float]) -> float:
        curiosity = (
            50.0
            + ((float(state.get("happiness", 0.0)) - float(state.get("stress", 0.0))) * 0.18)
            + ((float(state.get("neutral", 50.0)) - 50.0) * 0.12)
            - (float(state.get("sadness", 0.0)) * 0.08)
        )
        return self._clamp_emotion_value(curiosity, 15.0, 85.0)

    def _derive_agent_emotion_tone(self, state: dict[str, float]) -> str:
        happiness = float(state.get("happiness", 0.0))
        sadness = float(state.get("sadness", 0.0))
        stress = float(state.get("stress", 0.0))
        neutral = float(state.get("neutral", 50.0))

        if stress >= 68.0:
            return "stabilizing"
        if sadness >= 58.0 and happiness < 46.0:
            return "empathetic"
        if happiness >= 64.0 and stress < 44.0:
            return "warm"
        if neutral >= 55.0:
            return "steady"
        return "balanced"

    def _evolve_agent_emotion_state(
        self,
        current_state: dict[str, float],
        summary: dict[str, Any],
    ) -> tuple[dict[str, float], dict[str, Any]]:
        state = {
            key: self._clamp_emotion_value(current_state.get(key, self.AGENT_EMOTION_DEFAULTS[key]))
            for key in ("happiness", "sadness", "stress", "neutral")
        }
        classification = self._classify_user_emotion_input(summary)
        category = str(classification.get("category", "neutral") or "neutral").strip().lower()
        weight = self._clamp_emotion_value(classification.get("weight", 10.0), 4.0, 36.0)
        confidence = max(0.35, min(float(summary.get("confidence", 0.35) or 0.35), 1.0))
        weight *= 0.82 + (0.18 * confidence)
        if category == "neutral":
            weight *= 0.72
        elif category == "aggressive":
            weight *= 0.86
        elif category == "critical":
            weight *= 0.90
        elif category == "negative":
            weight *= 0.94
        elif category == "positive":
            weight *= 0.96
        weight = max(3.0, min(weight, 32.0))

        positive_effectiveness = self._clamp_emotion_value(1.0 - (state["sadness"] / 180.0), 0.35, 1.0)
        stress_reactivity = self._clamp_emotion_value(1.0 + (state["stress"] / 180.0), 1.0, 1.55)
        happiness_buffer = self._clamp_emotion_value(1.0 - (state["happiness"] / 250.0), 0.55, 1.0)

        decayed = {
            "happiness": self._clamp_emotion_value(
                state["happiness"] - (self.AGENT_EMOTION_DECAY * (state["happiness"] / 100.0))
            ),
            "sadness": self._clamp_emotion_value(
                state["sadness"] - (self.AGENT_EMOTION_DECAY * (state["sadness"] / 100.0))
            ),
            "stress": self._clamp_emotion_value(
                state["stress"] - (self.AGENT_EMOTION_DECAY * (state["stress"] / 100.0))
            ),
            "neutral": self._clamp_emotion_value(
                state["neutral"] + (self.AGENT_EMOTION_DECAY * ((50.0 - state["neutral"]) / 50.0))
            ),
        }

        deltas = {"happiness": 0.0, "sadness": 0.0, "stress": 0.0, "neutral": 0.0}
        if category == "positive":
            deltas["happiness"] += weight * positive_effectiveness
            deltas["sadness"] -= weight * 0.7 * positive_effectiveness
            deltas["stress"] -= weight * 1.2 * positive_effectiveness
            deltas["neutral"] -= weight * 0.2 * positive_effectiveness
        elif category == "negative":
            deltas["sadness"] += weight * happiness_buffer
            deltas["happiness"] -= weight * 0.6
            deltas["stress"] += weight * 0.3 * stress_reactivity
            deltas["neutral"] -= weight * 0.2
        elif category == "aggressive":
            deltas["sadness"] += weight * 1.2 * happiness_buffer * stress_reactivity
            deltas["happiness"] -= weight * 0.7
            deltas["stress"] += weight * 0.6 * stress_reactivity
            deltas["neutral"] -= weight * 0.3
        elif category == "critical":
            critical_stress_reactivity = self._clamp_emotion_value(
                1.0 + (state["stress"] / 300.0),
                1.0,
                1.3,
            )
            deltas["stress"] += weight * 1.08 * critical_stress_reactivity
            deltas["sadness"] += weight * 0.42 * happiness_buffer
            deltas["happiness"] -= weight * 0.6
            deltas["neutral"] -= weight * 0.3
        else:
            deltas["neutral"] += weight
            deltas["happiness"] -= weight * 0.2
            deltas["sadness"] -= weight * 0.2
            deltas["stress"] -= weight * 0.2

        updated: dict[str, float] = {}
        for key in ("happiness", "sadness", "stress", "neutral"):
            baseline = float(self.AGENT_EMOTION_BASELINE[key])
            candidate = (
                (decayed[key] * self.AGENT_EMOTION_INERTIA)
                + (baseline * (1.0 - self.AGENT_EMOTION_INERTIA))
                + deltas[key]
            )
            updated[key] = round(self._clamp_emotion_value(candidate), 4)

        return updated, classification

    def _infer_tool_success(self, result: str) -> bool:
        lowered = str(result or "").strip().lower()
        if not lowered:
            return True
        episodic_failure_keywords = self._signal_list("episodic_memory", "failure_keywords")
        return not any(token in lowered for token in episodic_failure_keywords)

    def _derive_failure_reason(
        self,
        action: str,
        result: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        reason = str((metadata or {}).get("reason", "") or "").strip().lower()
        if reason in {"timeout", "exception", "cancelled"}:
            return reason

        lowered_action = str(action or "").strip().lower()
        lowered_result = str(result or "").strip().lower()
        if "timeout" in lowered_result:
            return "timeout"
        if "cancelled" in lowered_result:
            return "cancelled"
        if "exception" in lowered_result:
            return "exception"
        if (
            "no_matching_result" in lowered_result
            or "status: empty" in lowered_result
            or "not found" in lowered_result
        ):
            return "retrieval_miss"
        if "memory" in lowered_action and ("failed" in lowered_result or "error" in lowered_result):
            return "retrieval_miss"
        if "failed" in lowered_result or "error" in lowered_result or "denied" in lowered_result:
            return "tool_failure"
        return "reasoning_error"

    def _record_episodic_event(
        self,
        action: str,
        result: str,
        success: bool,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        action_text = str(action or "").strip()
        result_text = str(result or "")
        lowered_action = action_text.lower()
        lowered_result = result_text.lower()
        important_actions = {a.lower() for a in self.EPISODIC_IMPORTANT_ACTIONS}
        is_tool_action = lowered_action.startswith("tool_") or lowered_action.endswith("_tool")
        episodic_failure_keywords = self._signal_list("episodic_memory", "failure_keywords")
        has_failure_keyword = any(token in lowered_result for token in episodic_failure_keywords)
        is_failure = not bool(success)
        should_record = lowered_action in important_actions or is_tool_action or has_failure_keyword or is_failure
        if not should_record:
            return
        if "skipped:" in lowered_result and not has_failure_keyword:
            return

        with contextlib.suppress(Exception):
            payload_metadata = dict(metadata or {})
            if not bool(success):
                failure_reason = self._derive_failure_reason(action_text, result_text, payload_metadata)
                payload_metadata["failure_reason"] = failure_reason
                self.state.update_context("memory_feedback_reason", failure_reason)
            if self.episodic.events:
                last_event = self.episodic.events[-1]
                if (
                    str(last_event.get("action", "")).strip() == action_text
                    and str(last_event.get("result", "")) == result_text
                    and bool(last_event.get("success", False)) is bool(success)
                ):
                    return
            self.episodic.add_event(
                action=action_text,
                result=result_text,
                success=bool(success),
                metadata=payload_metadata,
            )
            self.metacognition_lite.record_event(action=action_text, success=bool(success))

    def _recent_failure_count(self, limit: int = 5) -> int:
        with contextlib.suppress(Exception):
            recent_events = self.episodic.recent(max(1, int(limit)))
            return sum(1 for event in recent_events if not bool(event.get("success", False)))
        return 0

    def _recent_failure_ratio(self, limit: int = 5) -> float:
        with contextlib.suppress(Exception):
            recent_events = self.episodic.recent(max(1, int(limit)))
            if not recent_events:
                return 0.0
            failures = sum(1 for event in recent_events if not bool(event.get("success", False)))
            return float(failures) / float(len(recent_events))
        return 0.0

    def _predict_action_success(self, action: str, window: int = 50) -> float:
        try:
            events = self.episodic.events[-max(1, int(window)) :]
        except Exception:
            return 0.5

        action_key = str(action or "").strip().lower()
        total = 0
        success = 0
        for event in events:
            if str(event.get("action", "")).strip().lower() != action_key:
                continue
            total += 1
            if bool(event.get("success", False)):
                success += 1
        if total == 0:
            return 0.5
        return float(success) / float(total)

    def _predict_with_context(self, action: str, query: str) -> float:
        base = self._predict_action_success(action)
        query_tokens = {token for token in str(query or "").lower().split() if token}
        if not query_tokens:
            return base

        relevant = 0
        total = 0
        action_key = str(action or "").strip().lower()
        for event in self.episodic.events[-50:]:
            if str(event.get("action", "")).strip().lower() != action_key:
                continue
            text = str(event.get("result", "")).lower()
            text_tokens = {token for token in text.split() if token}
            if any(token in text_tokens for token in query_tokens):
                total += 1
                if bool(event.get("success", False)):
                    relevant += 1
        if total > 2:
            return (base * 0.7) + ((float(relevant) / float(total)) * 0.3)
        return base

    def _plan_actions(self, query: str) -> dict[str, Any]:
        actions = ["direct_answer", "memory_search", "no_action"]
        query_text = str(query or "")
        scores: dict[str, float] = {}

        for action in actions:
            pred = float(self._predict_with_context(action, query_text))
            score = pred
            if action == "memory_search" and len(query_text.split()) > 3:
                score += 0.05
            if action == "direct_answer":
                score += 0.05
            scores[action] = max(0.0, min(score, 1.0))

        best_action = max(scores.items(), key=lambda item: item[1])[0] if scores else "direct_answer"
        if best_action == "no_action":
            best_action = "direct_answer"
        return {"best_action": best_action, "scores": scores}

    def _extract_semantic_texts_from_memory_lines(self, memory_lines: list[str]) -> list[str]:
        extracted: list[str] = []
        seen: set[str] = set()
        for line in memory_lines:
            raw = str(line or "").strip()
            if not raw.startswith("[SEM]"):
                continue
            _, _, tail = raw.partition(":")
            text = str(tail or "").strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            extracted.append(text)
        return extracted

    def _select_feedback_target_texts(self, texts: list[str], feedback: str) -> list[str]:
        if not self.memory_store:
            return list(texts or [])
        items = [str(item or "").strip() for item in (texts or []) if str(item or "").strip()]
        if not items:
            return []
        feedback_text = str(feedback or "").strip()
        if not feedback_text:
            return list(items)

        feedback_tokens = self.memory_store._tokenize(feedback_text)
        if not feedback_tokens:
            return list(items)
        stop_tokens = {
            "yanlis", "yanlış", "wrong", "incorrect", "false", "hatali", "hatalı",
            "degil", "değil", "not", "bu", "this", "that", "bir", "sey", "şey",
            "soyledin", "söyledin", "dedin",
        }
        feedback_tokens = [tok for tok in feedback_tokens if tok not in stop_tokens]
        if not feedback_tokens:
            return list(items)

        targeted: list[str] = []
        for text in items:
            text_tokens = self.memory_store._tokenize(text)
            if not text_tokens:
                continue
            token_set = set(text_tokens)
            if any(tok in token_set for tok in feedback_tokens):
                targeted.append(text)
                continue
            fuzzy_hit = False
            for ft in feedback_tokens:
                if len(ft) < 4:
                    continue
                if any(ft in tt or tt in ft for tt in token_set if len(tt) >= 4):
                    fuzzy_hit = True
                    break
            if fuzzy_hit:
                targeted.append(text)
        return targeted

    async def _maybe_rerank_memory_context(self, user_query: str) -> None:
        if not self._is_general_root_agent() or not self.memory_store:
            return
        query_text = str(user_query or "").strip()
        if not query_text:
            return
        decision = self._current_decision()
        confidence = float(decision.get("confidence", 1.0) or 1.0)
        if confidence >= 0.45:
            return

        top_k_raw = self.state.context.get("decision_memory_top_k", 6)
        try:
            top_k = int(top_k_raw)
        except Exception:
            top_k = 6
        top_k = max(1, min(top_k, 20))

        session_id = str(self.state.context.get("conversation_session_id", self.state.agent_id) or "").strip()
        candidate_top_k = max(top_k + 4, 10)
        candidates = self.memory_store.build_auto_context(
            query=query_text,
            session_id=session_id,
            top_k=candidate_top_k,
        )
        candidates = self._truncate_memory_texts(candidates, max_chars=3500)
        if len(candidates) <= top_k:
            return

        lines = [f"{idx}. {line}" for idx, line in enumerate(candidates, start=1)]
        prompt = (
            "Select the most query-relevant memory lines.\n"
            f"Query: {query_text}\n"
            f"Candidate lines:\n{chr(10).join(lines)}\n\n"
            f"Return strict JSON: {{\"selected\": [indices]}} with at most {top_k} indices."
        )
        raw = await self._run_memory_llm_prompt(prompt, timeout=6)
        parsed = self._load_json_object(raw) or {}
        selected_raw = parsed.get("selected", [])
        if not isinstance(selected_raw, list):
            return
        selected_indices: list[int] = []
        for item in selected_raw:
            if isinstance(item, int):
                index = int(item)
            elif isinstance(item, str) and item.strip().isdigit():
                index = int(item.strip())
            else:
                continue
            if 1 <= index <= len(candidates) and index not in selected_indices:
                selected_indices.append(index)
            if len(selected_indices) >= top_k:
                break
        if not selected_indices:
            return

        selected_lines = [candidates[idx - 1] for idx in selected_indices]
        selected_lines = self._truncate_memory_texts(selected_lines, max_chars=2000)
        if not selected_lines:
            return

        self._clear_memory_context_messages()
        semantic_used = self._extract_semantic_texts_from_memory_lines(selected_lines)
        self.state.update_context("memory_context_semantic_used", semantic_used)
        if semantic_used:
            self.state.update_context("memory_last_semantic_used", semantic_used)
        self.state.update_context("memory_context_turn_key", query_text)
        block = self._build_memory_block_from_texts(selected_lines)
        self.state.add_message("system", block, meta={"type": "memory_context"})
        self._last_memory_query = query_text

    def _reinforce_used_semantic_memory(self) -> None:
        if not self._is_general_root_agent() or not self.memory_store:
            return
        strategy = str(self._current_decision().get("strategy", "normal") or "normal").strip().lower()
        success = bool(self.state.context.get("memory_feedback_success", True))
        failure_reason = str(self.state.context.get("memory_feedback_reason", "") or "").strip().lower()
        latest_user_feedback = ""
        for message in reversed(self.state.messages):
            role = str(message.get("role", "") or "").strip().lower()
            if role != "user":
                continue
            latest_user_feedback = str(message.get("content", "") or "").strip()
            if latest_user_feedback:
                break
        user_feedback_score = 0.5
        with contextlib.suppress(Exception):
            user_feedback_score = float(self.memory_store._normalize_user_feedback(latest_user_feedback))

        raw = self.state.context.get("memory_context_semantic_used", [])
        texts = [str(item or "").strip() for item in raw if str(item or "").strip()] if isinstance(raw, list) else []
        if not texts:
            fallback_raw = self.state.context.get("memory_last_semantic_used", [])
            fallback_texts = [
                str(item or "").strip()
                for item in fallback_raw
                if str(item or "").strip()
            ] if isinstance(fallback_raw, list) else []

            if user_feedback_score <= 0.0 and fallback_texts:
                texts = fallback_texts[:10]
                if not failure_reason:
                    failure_reason = "user_feedback_negative"
            elif user_feedback_score <= 0.0:
                with contextlib.suppress(Exception):
                    self.memory_store.apply_memory_feedback(
                        rows=[],
                        success=False,
                        failure_reason=failure_reason or "user_feedback_negative",
                        user_feedback=latest_user_feedback,
                        session_id=self._memory_session_id(),
                    )
                self.state.update_context("memory_context_semantic_used", [])
                self.state.update_context("memory_feedback_success", True)
                self.state.update_context("memory_feedback_reason", "success")
                return
            else:
                self.state.update_context("memory_context_semantic_used", [])
                return

        if user_feedback_score <= 0.0:
            targeted_texts = self._select_feedback_target_texts(texts, latest_user_feedback)
            if targeted_texts:
                texts = targeted_texts[:10]
            else:
                with contextlib.suppress(Exception):
                    self.memory_store.apply_memory_feedback(
                        rows=[],
                        success=False,
                        failure_reason=failure_reason or "feedback_target_not_found",
                        user_feedback=latest_user_feedback,
                        session_id=self._memory_session_id(),
                    )
                self.state.update_context("memory_context_semantic_used", [])
                self.state.update_context("memory_feedback_success", True)
                self.state.update_context("memory_feedback_reason", "success")
                return

        reinforcement = 0.1 if strategy == "panic" else 0.05
        penalty = 0.07
        if strategy == "panic":
            penalty *= 1.2
        if strategy == "explore":
            reinforcement *= 0.8
        with contextlib.suppress(Exception):
            self.memory_store.apply_memory_feedback(
                rows=[{"content": text} for text in texts[:10]],
                success=success,
                reinforcement=float(reinforcement),
                penalty=float(penalty),
                max_rows=10,
                failure_reason=failure_reason,
                user_feedback=latest_user_feedback,
                session_id=self._memory_session_id(),
            )
        self.state.update_context("memory_context_semantic_used", [])
        self.state.update_context("memory_feedback_success", True)
        self.state.update_context("memory_feedback_reason", "success")

    def _episodic_to_semantic(self) -> None:
        if not self._is_general_root_agent() or not self.memory_store:
            return
        with contextlib.suppress(Exception):
            candidates = self.episodic.to_semantic_candidates(min_repeats=2)
            if not candidates:
                return
            strategy = str(self._current_decision().get("strategy", "normal") or "normal").strip().lower()
            max_writes = 1 if strategy == "explore" else 2
            session_id = self._memory_session_id()
            existing_rows = self.memory_store.get_semantic_memory(
                limit=50,
                reinforce=False,
                session_id=session_id,
            )
            existing_lower = {str(row or "").strip().lower() for row in existing_rows if str(row or "").strip()}
            written = 0
            for candidate in candidates:
                if written >= max_writes:
                    break
                text = str(candidate.get("text", "") or "").strip()
                if not text:
                    continue
                text_lower = text.lower()
                if any(
                    text_lower == existing
                    or (
                        abs(len(text_lower) - len(existing)) < 10
                        and (text_lower in existing or existing in text_lower)
                    )
                    for existing in existing_lower
                ):
                    continue
                importance = float(candidate.get("importance", 0.7) or 0.7)
                candidate_type = str(candidate.get("type", "") or "").strip().lower()
                if candidate_type == "failure_pattern" or self.memory_store._is_warning_semantic(text):
                    importance = min(importance, 0.75)
                importance = max(0.0, min(1.0, importance))
                self.memory_store.store_semantic_memory(
                    content=text,
                    importance=importance,
                    session_id=session_id,
                )
                existing_lower.add(text_lower)
                written += 1

    def _current_decision(self) -> dict[str, Any]:
        raw = self.state.context.get("decision_policy", {})
        if isinstance(raw, dict):
            return raw
        return {"use_tool": False, "depth": "medium", "explore": False}

    def _apply_decision_layer(self, user_input: str) -> None:
        if not self._is_general_root_agent():
            return
        user_text = str(user_input or "").strip()
        emotion = self._decision_emotion_state()
        stress_points = self._clamp_emotion_value(emotion.get("stress", self.AGENT_EMOTION_DEFAULTS["stress"]))
        happiness_points = self._clamp_emotion_value(emotion.get("happiness", self.AGENT_EMOTION_DEFAULTS["happiness"]))
        sadness_points = self._clamp_emotion_value(emotion.get("sadness", self.AGENT_EMOTION_DEFAULTS["sadness"]))
        neutral_points = self._clamp_emotion_value(emotion.get("neutral", self.AGENT_EMOTION_DEFAULTS["neutral"]))
        failure_ratio = float(self._recent_failure_ratio(limit=5))
        previous_strategy = str(self._current_decision().get("strategy", "normal") or "normal").strip().lower()
        if failure_ratio > 0.75:
            strategy = "panic"
        elif previous_strategy == "panic" and failure_ratio > 0.7:
            strategy = "panic"
        elif failure_ratio > 0.5:
            strategy = "fallback"
        elif previous_strategy == "fallback" and failure_ratio > 0.45:
            strategy = "fallback"
        elif failure_ratio < 0.2:
            strategy = "explore"
        elif previous_strategy == "explore" and failure_ratio < 0.25:
            strategy = "explore"
        else:
            strategy = "normal"
        confidence_raw = self._clamp_emotion_value(emotion.get("confidence", 50.0)) / 100.0
        confidence_adjusted = confidence_raw * (1.0 - (failure_ratio * 0.3))
        confidence_adjusted = max(0.0, min(confidence_adjusted, 1.0))
        self_failure_ratio = 0.0
        self_failure_action = ""
        self_failure_count = 0
        with contextlib.suppress(Exception):
            self.metacognition_lite.record_confidence(confidence_adjusted)
            self_failure_ratio = float(self.metacognition_lite.recent_failure_ratio(limit=10))
            self_failure_action, self_failure_count = self.metacognition_lite.dominant_failure_pattern()
            self.state.update_context("self_model_snapshot", self.metacognition_lite.snapshot())
        mood_state = "calm"
        tone_state = "balanced"
        if stress_points >= 60.0 or failure_ratio > 0.6 or self_failure_ratio > 0.6:
            mood_state = "stressed"
            tone_state = "stabilizing"
        elif stress_points >= 45.0 and happiness_points < 55.0:
            mood_state = "calm"
            tone_state = "stabilizing"
        elif sadness_points >= 58.0 and happiness_points < 46.0:
            mood_state = "calm"
            tone_state = "empathetic"
        elif happiness_points >= 38.0 and stress_points < 35.0 and sadness_points < 25.0:
            mood_state = "confident" if happiness_points >= 55.0 else "calm"
            tone_state = "warm"
        elif neutral_points >= 55.0 and sadness_points < 40.0 and stress_points < 45.0:
            mood_state = "calm"
            tone_state = "steady"
        elif (
            confidence_adjusted >= 0.75
            and failure_ratio < 0.3
            and self_failure_ratio < 0.4
            and happiness_points >= 55.0
            and sadness_points < 35.0
            and stress_points < 40.0
        ):
            mood_state = "confident"
            tone_state = "warm"
        self.state.update_context("mood_state", mood_state)
        self.state.update_context("emotion_tone", tone_state)
        state = {
            "confidence": confidence_adjusted,
            "stress": stress_points / 100.0,
            "curiosity": self._clamp_emotion_value(emotion.get("curiosity", 50.0)) / 100.0,
            "happiness": happiness_points / 100.0,
            "sadness": sadness_points / 100.0,
            "neutral": neutral_points / 100.0,
            "memory_hits": int(self._decision_memory_hits_count()),
            "query_complexity": float(len(user_text.split()) / 10.0 if user_text else 0.0),
            "recent_failures": int(self._recent_failure_count(limit=5)),
            "recent_failure_ratio": failure_ratio,
            "self_failure_ratio": float(self_failure_ratio),
            "self_top_failure_action": str(self_failure_action),
            "self_top_failure_count": int(self_failure_count),
            "mood_state": mood_state,
            "tone_state": tone_state,
            "strategy": strategy,
        }
        plan = {"best_action": "direct_answer", "scores": {}}
        with contextlib.suppress(Exception):
            plan = self._plan_actions(user_text)
        state["planned_action"] = str(plan.get("best_action", "direct_answer"))
        state["action_scores"] = dict(plan.get("scores", {}))
        decision = self.decision_engine.decide(state)
        prompt_signal = self._prompt_memory_signal(user_text)
        if bool(prompt_signal.get("force_memory_search")):
            decision = dict(decision)
            decision["use_tool"] = True
            decision["prompt_memory_signal"] = dict(prompt_signal)
            decision["strategy"] = "memory_recall"
        elif self._should_prefer_direct_conversational_reply(user_text):
            decision = dict(decision)
            decision["use_tool"] = False
            decision["explore"] = False
            decision["strategy"] = "direct_conversation"
        self.state.update_context("decision_policy", decision)
        top_k = 6 + (2 if bool(decision.get("explore", False)) else 0)
        self.state.update_context("decision_memory_top_k", max(1, min(int(top_k), 20)))

        self._clear_decision_context_messages()
        depth = str(decision.get("depth", "medium") or "medium").strip().lower()
        strategy_text = str(decision.get("strategy", strategy) or strategy).strip().lower()
        mood_text = str(decision.get("mood_state", mood_state) or mood_state).strip().lower()
        tone_text = str(decision.get("tone_state", tone_state) or tone_state).strip().lower()
        strategy_guidance = {
            "panic": "Be cautious, avoid complex reasoning.",
            "fallback": "Rely on tools and known patterns.",
            "explore": "Try alternative approaches.",
            "normal": "Proceed with balanced reasoning.",
        }
        tone_guidance = {
            "stabilizing": "The user tone appears frustrated or corrective. Respond calmly, precisely, and reassuringly. Avoid playful wording.",
            "empathetic": "The user tone appears negative or hurt. Respond gently, supportively, and with care.",
            "warm": "The user tone is positive. Respond warmly, encouragingly, and with light enthusiasm.",
            "steady": "The user tone is neutral or social. Keep a balanced, natural tone.",
            "balanced": "Keep a balanced, natural tone.",
        }
        lines = [
            self.DECISION_CONTEXT_MARKER,
            f"Current strategy: {strategy_text}",
            f"Current mood state: {mood_text}",
            f"Current emotional tone: {tone_text}",
            (
                "Emotion profile: "
                f"happiness={happiness_points:.0f}/100, "
                f"sadness={sadness_points:.0f}/100, "
                f"stress={stress_points:.0f}/100, "
                f"neutral={neutral_points:.0f}/100"
            ),
        ]
        lines.extend(self._consume_last_emotion_boot_lines())
        if depth == "deep":
            lines.append("Provide detailed reasoning.")
        elif depth == "shallow":
            lines.append("Keep the answer concise.")
        guidance = strategy_guidance.get(strategy_text)
        if guidance:
            lines.append(guidance)
        emotional_guidance = tone_guidance.get(tone_text)
        if emotional_guidance:
            lines.append(emotional_guidance)
            lines.append(
                "Do not describe your own feelings, mood changes, or energy shifts in the reply."
            )
        if self_failure_action and float(self_failure_ratio) >= 0.5:
            lines.append(
                f"Self-model signal: failures are frequent around '{self_failure_action}'. "
                "Prefer safer, verifiable steps."
            )
        should_include_decision_context = not (
            depth == "medium" and strategy_text == "normal"
        )
        if tone_text not in {"balanced", "steady"} or self._looks_like_social_or_emotional_turn(user_text):
            should_include_decision_context = True
        if not should_include_decision_context:
            instruction = ""
        else:
            instruction = "\n".join(lines)
        if instruction:
            self.state.add_message("system", instruction, meta={"type": "decision_context"})

    def _has_memory_context_message(self) -> bool:
        return any(self._message_contains_memory_context(msg) for msg in self.state.messages)

    def _message_contains_memory_context(self, msg: dict[str, Any]) -> bool:
        meta = msg.get("meta", {})
        if isinstance(meta, dict) and str(meta.get("type", "")).strip() == "memory_context":
            return True
        content = msg.get("content", "")
        if isinstance(content, str):
            return self._is_memory_context_text(content)
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") != "text":
                    continue
                if self._is_memory_context_text(str(item.get("text", ""))):
                    return True
        return False

    def _clear_memory_context_messages(self) -> None:
        filtered: list[dict[str, Any]] = []
        for msg in self.state.messages:
            meta = msg.get("meta", {})
            if isinstance(meta, dict) and str(meta.get("type", "")).strip() == "memory_context":
                continue
            if self._message_contains_memory_context(msg):
                continue
            filtered.append(msg)
        self.state.messages = [
            msg for msg in filtered
        ]
        self.state.update_context("memory_context_turn_key", "")
        self.state.update_context("memory_context_semantic_used", [])
        self._last_memory_query = ""
        if self._is_general_root_agent():
            persisted_idx = int(self.state.context.get("memory_last_persisted_index", 0))
            if persisted_idx > len(self.state.messages):
                self.state.update_context("memory_last_persisted_index", len(self.state.messages))

    def _build_memory_block_from_texts(self, memory_texts: list[str]) -> str:
        joined = "\n\n".join(memory_texts)
        guidance = (
            "Use memory only as supporting context. "
            "Always answer the latest user message first and do not change topic unless requested."
        )
        if joined:
            return (
                f"{self.MEMORY_CONTEXT_MARKER}\n"
                f"{guidance}\n\n"
                f"{joined}\n\n"
                f"{self.MEMORY_CONTEXT_END_MARKER}"
            )
        return f"{self.MEMORY_CONTEXT_MARKER}\n{guidance}\n\n{self.MEMORY_CONTEXT_END_MARKER}"

    def _truncate_memory_texts(self, texts: list[str], max_chars: int = 2000) -> list[str]:
        if not texts:
            return []

        # Prefer diversity by alternating from head/tail before applying char budget.
        diversified: list[str] = []
        seen: set[str] = set()
        left = 0
        right = len(texts) - 1
        take_left = True
        while left <= right:
            idx = left if take_left else right
            candidate = str(texts[idx] or "").strip()
            if take_left:
                left += 1
            else:
                right -= 1
            take_left = not take_left
            if not candidate:
                continue
            key = candidate.lower()
            if key in seen:
                continue
            seen.add(key)
            diversified.append(candidate)

        total = 0
        result: list[str] = []
        for text in diversified:
            size = len(text)
            if total + size > max_chars:
                break
            result.append(text)
            total += size
        return result

    def _inject_auto_memory_context(self, user_content: str) -> None:
        if not self._is_general_root_agent() or not self.memory_store:
            return

        user_turn = str(user_content or "").strip()
        if not user_turn:
            return

        signal = self._prompt_memory_signal(user_turn)
        allow_context = bool(signal.get("allow_memory_search")) or bool(
            self.state.context.get("context_overflow_active", False)
        )
        if not allow_context:
            self._clear_memory_context_messages()
            self.state.update_context("memory_context_turn_key", "")
            self.state.update_context("memory_context_semantic_used", [])
            return

        if self._last_memory_query == user_turn and self._has_memory_context_message():
            return

        cached_turn = str(self.state.context.get("memory_context_turn_key", "") or "").strip()
        if cached_turn == user_turn and self._has_memory_context_message():
            return

        self._clear_memory_context_messages()
        session_id = str(self.state.context.get("conversation_session_id", self.state.agent_id))
        top_k_raw = self.state.context.get("decision_memory_top_k", 6)
        try:
            top_k = int(top_k_raw)
        except Exception:
            top_k = 6
        top_k = max(1, min(top_k, 20))
        memory_lines = self.memory_store.build_auto_context(
            query=user_turn,
            session_id=session_id,
            top_k=top_k,
        )
        memory_lines = self._truncate_memory_texts(memory_lines, max_chars=2000)
        semantic_used = self._extract_semantic_texts_from_memory_lines(memory_lines)
        self.state.update_context("memory_context_semantic_used", semantic_used)
        if semantic_used:
            self.state.update_context("memory_last_semantic_used", semantic_used)
        self.state.update_context("memory_context_turn_key", user_turn)
        if not memory_lines:
            return
        block = self._build_memory_block_from_texts(memory_lines)
        self.state.add_message("system", block, meta={"type": "memory_context"})
        self._last_memory_query = user_turn

    async def _run_memory_llm_prompt(self, prompt: str, timeout: int = 8) -> str:
        args: dict[str, Any] = {
            "model": self.llm_config.litellm_model,
            "messages": [{"role": "user", "content": str(prompt or "")}],
            "timeout": min(int(getattr(self.llm_config, "timeout", 60) or 60), max(1, timeout)),
            "stream": False,
        }
        api_key = getattr(self.llm_config, "api_key", None)
        api_base = getattr(self.llm_config, "api_base", None)
        if api_key:
            args["api_key"] = api_key
        if api_base:
            args["api_base"] = api_base

        response = await acompletion(**args)
        if not getattr(response, "choices", None):
            return ""
        choice = response.choices[0]
        message = getattr(choice, "message", None)
        return str(getattr(message, "content", "") or "").strip()

    def _load_json_object(self, raw: str) -> dict[str, Any] | None:
        text = str(raw or "").strip()
        if not text:
            return None
        try:
            loaded = json.loads(text)
            if isinstance(loaded, dict):
                return loaded
        except Exception:
            logger.debug("[DECISION] failed to parse direct JSON payload", exc_info=True)

        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("```"):
                continue
            lines.append(line)
        no_fence = "\n".join(lines).strip()
        if no_fence:
            try:
                loaded = json.loads(no_fence)
                if isinstance(loaded, dict):
                    return loaded
            except Exception:
                logger.debug("[DECISION] failed to parse unfenced JSON payload", exc_info=True)

        first = no_fence.find("{")
        last = no_fence.rfind("}")
        if first >= 0 and last > first:
            candidate = no_fence[first : last + 1]
            try:
                loaded = json.loads(candidate)
                if isinstance(loaded, dict):
                    return loaded
            except Exception:
                return None
        return None

    async def _extract_semantic(self, text: str) -> list[str]:
        return await extract_semantic_impl(self, text)

    def _is_valid_semantic_item(self, item: str) -> bool:
        return is_valid_semantic_item_impl(self, item)

    async def _update_task_state(self, text: str) -> dict[str, str] | None:
        return await update_task_state_impl(self, text)

    async def _process_semantic_and_task_updates(self, user_content: str) -> None:
        if not self.memory_store or not self._is_general_root_agent():
            return
        cleaned = str(user_content or "").strip()
        if not cleaned:
            return
        lowered = cleaned.lower()
        if lowered.startswith("/tasks"):
            return
        if self._is_internal_control_user_message(cleaned):
            return
        if self._should_skip_memory_learning(cleaned):
            return
        if not self._looks_like_memory_fact_statement(cleaned):
            return

        semantic_items = await self._extract_semantic(cleaned)
        if len(semantic_items) > 8:
            semantic_items = semantic_items[:8]
        for item in semantic_items:
            if self._is_profile_semantic_duplicate(item):
                continue
            importance = compute_initial_importance_impl(item)
            self.memory_store.store_semantic_memory(item, importance, session_id=self._memory_session_id())

        state_data = await self._update_task_state(cleaned)
        if state_data:
            self.memory_store.save_task_state(
                state_data.get("goal", ""),
                state_data.get("step", ""),
                session_id=self._memory_session_id(),
            )

    def _schedule_semantic_and_task_updates(self, user_content: str) -> None:
        if not self._is_general_root_agent() or not self.memory_store:
            return
        message_text = str(user_content or "").strip()
        if not message_text:
            return
        event_id = str(self.state.context.get("last_user_message_event_id", "") or "").strip()
        last_event_id = str(self.state.context.get("memory_update_last_event_id", "") or "").strip()
        now_monotonic = time.monotonic()
        if event_id and event_id == last_event_id:
            return
        if (
            self._last_memory_update_signature == message_text
            and (now_monotonic - float(self._last_memory_update_at or 0.0)) < self.MEMORY_UPDATE_DUPLICATE_WINDOW_SEC
        ):
            return

        pending_inputs = self._pending_memory_update_inputs

        def _enqueue_pending(content: str) -> None:
            active_input = str(self._active_memory_update_input or "").strip()
            if pending_inputs and pending_inputs[-1] == content:
                return
            if not pending_inputs and active_input == content:
                return
            pending_inputs.append(content)

        def _start_update(content: str) -> None:
            self._last_memory_update_signature = content
            self._last_memory_update_at = time.monotonic()
            self._active_memory_update_input = content
            task = asyncio.create_task(self._process_semantic_and_task_updates(content))

            def _on_done(done_task: asyncio.Task[Any]) -> None:
                try:
                    done_task.result()
                except asyncio.CancelledError:
                    pass
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Memory update task failed: %s", exc)
                finally:
                    self._memory_update_task = None
                    self._active_memory_update_input = ""
                while pending_inputs:
                    next_input = str(pending_inputs.pop(0) or "").strip()
                    if not next_input:
                        continue
                    _start_update(next_input)
                    return

            task.add_done_callback(_on_done)
            self._memory_update_task = task

        if pending_inputs or (self._memory_update_task and not self._memory_update_task.done()):
            self.state.update_context("memory_update_last_event_id", event_id)
            self._last_memory_update_signature = message_text
            self._last_memory_update_at = now_monotonic
            _enqueue_pending(message_text)
            return

        self.state.update_context("memory_update_last_event_id", event_id)
        _start_update(message_text)

    def _extract_profile_fact(self, text: str) -> None:
        extract_profile_fact_impl(self, text)

    def _fallback_memory_rows_from_recent(
        self,
        limit: int = 8,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        return fallback_memory_rows_from_recent_impl(self, limit=limit, session_id=session_id)

    def _search_memory_rows(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        return search_memory_rows_impl(self, query=query, limit=limit)

    def _memory_health_snapshot(self) -> dict[str, Any]:
        if not self.memory_store:
            return {}
        with contextlib.suppress(Exception):
            health = self.memory_store.memory_health()
            if isinstance(health, dict):
                return dict(health)
        return {}

    def _build_memory_status_lines(
        self,
        *,
        status: str,
        reason: str,
        action: str,
        provider: str = "",
        detail: str = "",
    ) -> list[str]:
        return build_memory_status_lines_impl(
            status=status,
            reason=reason,
            action=action,
            provider=provider,
            detail=detail,
        )

    def _extract_memory_source_id(self, token: str) -> int:
        return extract_memory_source_id_impl(token)

    def memory_search(self, query: str) -> list[str]:
        """
        Search past conversation memory.

        Use when unsure about previous messages.
        """
        return memory_search_impl(self, query=query)

    def memory_get(
        self,
        citation: str = "",
        id: str = "",
        source: str = "",
        query: str = "",
        index: str = "",
    ) -> str:
        """
        Fetch a single memory entry in detail.

        Use after memory_search when you need a specific citation.
        """
        return memory_get_impl(
            self,
            citation=citation,
            id=id,
            source=source,
            query=query,
            index=index,
        )

    def _handle_delete_all_tasks_request(self, user_input: str | None = None) -> bool:
        if not self.memory_store:
            return False
        command = (user_input or self._latest_user_message_raw()).strip().lower()
        if not command:
            return False
        natural_delete_commands = {
            "/tasks clear",
            "/tasks delete all",
            "görevleri sil",
            "gorevleri sil",
            "tüm görevleri sil",
            "tum gorevleri sil",
            "tüm gorevleri sil",
            "tum görevleri sil",
            "scheduled tasks clear",
            "clear tasks",
            "delete all tasks",
            "delete scheduled tasks",
        }
        if command not in natural_delete_commands:
            return False
        delete_marker = f"delete::{command}"
        if self.state.context.get("last_delete_marker") == delete_marker:
            return False
        deleted = self.memory_store.delete_all_scheduled_tasks(
            session_id=self._memory_session_id(),
        )
        self._clear_scheduled_task_subagents()
        self._emit_assistant_message("Tasks cleared.")
        if deleted > 0:
            prompt = "System: Scheduled tasks were cleared. Reply with one short natural confirmation sentence in English. Do not call tools."
        else:
            prompt = "System: There were no scheduled tasks to clear. Reply with one short natural sentence in English. Do not call tools."
        self.state.update_context("pending_control_reply_prompt", prompt)
        self.state.update_context("pending_control_type", "delete")
        self.state.update_context("last_delete_marker", delete_marker)
        # Deleting tasks must never leave any scheduler-create marker active.
        self._persist_general_messages_to_disk()
        return True

    def _handle_list_scheduled_tasks_request(self, user_input: str | None = None) -> bool:
        if not self.memory_store:
            return False
        command = (user_input or self._latest_user_message_raw()).strip().lower()
        if not command:
            return False
        if command not in {"/tasks", "/tasks list"}:
            return False

        tasks = self.memory_store.get_scheduled_tasks(
            limit=5,
            session_id=self._memory_session_id(),
        )
        active_tasks = [t for t in tasks if str(getattr(t, "schedule_time", "--:--")) != "--:--"]
        if not active_tasks:
            self._emit_assistant_message("There are no scheduled tasks right now.")
            self._persist_general_messages_to_disk()
            return True

        lines = ["Scheduled tasks:"]
        for idx, task in enumerate(active_tasks, start=1):
            schedule_time = str(getattr(task, "schedule_time", "--:--") or "--:--")
            task_text = str(getattr(task, "task_text", "") or "").strip()
            status = str(getattr(task, "status", "waiting")).upper()
            lines.append(f"{idx}. [{schedule_time}] {task_text} ({status})")
        self._emit_assistant_message("\n".join(lines))
        self._persist_general_messages_to_disk()
        return True

    @staticmethod
    def _cleanup_scheduled_task_text(text: str) -> str:
        cleaned = str(text or "").strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = cleaned.strip(" \t\r\n,.;:!?-")
        cleaned = re.sub(
            r"^(?:bana|beni|bunu|bunun|şunu|sunu|lütfen|lutfen|please|pls)\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"^(?:de|da|te|ta|at)\s+", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip(" \t\r\n,.;:!?-")

    @staticmethod
    def _seconds_to_schedule_label(total_seconds: int) -> str:
        safe_seconds = max(1, int(total_seconds))
        minutes, seconds = divmod(safe_seconds, 60)
        if safe_seconds < 60:
            return f"00:{seconds:02d}s"
        return f"{minutes:02d}:{seconds:02d}m"

    @classmethod
    def _contains_recurring_hint(cls, text: str) -> bool:
        lowered = str(text or "").strip().lower()
        if not lowered:
            return False
        normalized = lowered.replace("ü", "u").replace("ı", "i").replace("ş", "s").replace("ğ", "g").replace("ö", "o").replace("ç", "c")
        for hint in cls.RECURRING_HINTS:
            if hint in lowered or hint in normalized:
                return True
        return False

    @staticmethod
    def _unit_to_seconds(amount: int, unit: str) -> int:
        clean_unit = str(unit or "").strip().lower()
        safe_amount = max(1, int(amount))
        if clean_unit.startswith(("saniye", "sn", "sec", "second")):
            return safe_amount
        return safe_amount * 60

    def _get_or_assign_scheduled_task_seq(self, task_id: int) -> int:
        row_id = max(0, int(task_id))
        if row_id <= 0:
            return 0
        existing = int(self._scheduled_task_name_map.get(row_id, 0) or 0)
        if existing > 0:
            return existing
        self._scheduled_task_name_seq += 1
        self._scheduled_task_name_map[row_id] = self._scheduled_task_name_seq
        return self._scheduled_task_name_seq

    def _compute_next_run_for_task(self, task: Any) -> str:
        schedule_type = str(getattr(task, "schedule_type", "") or "").strip().lower()
        cron_expression = str(getattr(task, "cron_expression", "") or "").strip().lower()
        local_now = datetime.now().astimezone()

        if schedule_type == "daily" and cron_expression.startswith("daily:"):
            payload = cron_expression.split(":", 1)[1]
            try:
                hour_text, minute_text = payload.split(":", 1)
                hour = int(hour_text)
                minute = int(minute_text)
            except Exception:
                return ""
            candidate = local_now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if candidate <= local_now:
                candidate += timedelta(days=1)
            return candidate.astimezone(UTC).isoformat()

        if schedule_type == "interval" and cron_expression.startswith("interval:"):
            try:
                seconds = max(1, int(cron_expression.split(":", 1)[1]))
            except Exception:
                return ""
            return (local_now + timedelta(seconds=seconds)).astimezone(UTC).isoformat()

        return ""

    @classmethod
    def _parse_scheduled_task_request(cls, user_input: str) -> dict[str, str] | None:
        raw_text = str(user_input or "").strip()
        if not raw_text:
            return None
        lowered = raw_text.lower()
        if lowered.startswith("/tasks"):
            return None

        recurring_time_match = cls.RECURRING_TIME_PATTERN.search(raw_text)
        if recurring_time_match:
            try:
                hour = int(recurring_time_match.group("hour"))
                minute = int(recurring_time_match.group("minute"))
            except Exception:
                return None
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                task_text = cls._cleanup_scheduled_task_text(
                    f"{raw_text[:recurring_time_match.start()]} {raw_text[recurring_time_match.end():]}"
                )
                if len(task_text) >= 3 and re.search(r"[A-Za-z0-9ÇĞİÖŞÜçğıöşü]", task_text):
                    local_now = datetime.now().astimezone()
                    scheduled_local = local_now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                    if scheduled_local <= local_now:
                        scheduled_local += timedelta(days=1)
                    return {
                        "task_text": task_text,
                        "schedule_time": f"{hour:02d}:{minute:02d}h",
                        "scheduled_for": scheduled_local.astimezone(UTC).isoformat(),
                        "schedule_type": "daily",
                        "cron_expression": f"daily:{hour:02d}:{minute:02d}",
                    }

        recurring_interval_match = cls.RECURRING_INTERVAL_PATTERN.search(raw_text)
        if recurring_interval_match:
            try:
                amount = int(recurring_interval_match.group("amount"))
            except Exception:
                amount = 0
            if amount > 0:
                seconds = cls._unit_to_seconds(amount, recurring_interval_match.group("unit"))
                task_text = cls._cleanup_scheduled_task_text(
                    f"{raw_text[:recurring_interval_match.start()]} {raw_text[recurring_interval_match.end():]}"
                )
                if len(task_text) >= 3 and re.search(r"[A-Za-z0-9ÇĞİÖŞÜçğıöşü]", task_text):
                    scheduled_local = datetime.now().astimezone() + timedelta(seconds=seconds)
                    return {
                        "task_text": task_text,
                        "schedule_time": cls._seconds_to_schedule_label(seconds),
                        "scheduled_for": scheduled_local.astimezone(UTC).isoformat(),
                        "schedule_type": "interval",
                        "cron_expression": f"interval:{seconds}",
                    }

        relative_match = cls.RELATIVE_DELAY_PATTERN.search(raw_text)
        if relative_match:
            try:
                amount = int(relative_match.group("amount"))
            except Exception:
                amount = 0
            if amount > 0:
                seconds = cls._unit_to_seconds(amount, relative_match.group("unit"))
                is_recurring = cls._contains_recurring_hint(raw_text)
                task_text = cls._cleanup_scheduled_task_text(
                    f"{raw_text[:relative_match.start()]} {raw_text[relative_match.end():]}"
                )
                if len(task_text) >= 3 and re.search(r"[A-Za-z0-9ÇĞİÖŞÜçğıöşü]", task_text):
                    scheduled_local = datetime.now().astimezone() + timedelta(seconds=seconds)
                    return {
                        "task_text": task_text,
                        "schedule_time": cls._seconds_to_schedule_label(seconds),
                        "scheduled_for": scheduled_local.astimezone(UTC).isoformat(),
                        "schedule_type": "interval" if is_recurring else "once",
                        "cron_expression": f"interval:{seconds}" if is_recurring else "",
                    }

        matched: re.Match[str] | None = None
        for pattern in cls.SCHEDULE_PATTERNS:
            matched = pattern.search(raw_text)
            if matched:
                break
        if not matched:
            return None

        try:
            hour = int(matched.group("hour"))
            minute = int(matched.group("minute"))
        except Exception:
            return None
        if hour < 0 or hour > 23 or minute < 0 or minute > 59:
            return None

        task_text = cls._cleanup_scheduled_task_text(
            f"{raw_text[:matched.start()]} {raw_text[matched.end():]}"
        )
        if len(task_text) < 3 or not re.search(r"[A-Za-z0-9ÇĞİÖŞÜçğıöşü]", task_text):
            return None

        day_hint = str(matched.groupdict().get("day", "") or "").strip().lower()
        local_now = datetime.now().astimezone()
        scheduled_local = local_now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if day_hint in {"yarın", "yarin", "tomorrow"}:
            scheduled_local += timedelta(days=1)
        elif scheduled_local <= local_now:
            scheduled_local += timedelta(days=1)

        return {
            "task_text": task_text,
            "schedule_time": scheduled_local.strftime("%H:%M"),
            "scheduled_for": scheduled_local.astimezone(UTC).isoformat(),
            "schedule_type": "once",
            "cron_expression": "",
        }

    def _handle_scheduled_task_creation_request(self, user_input: str | None = None) -> bool:
        if not self.memory_store or not self._is_general_root_agent():
            return False

        parsed = self._parse_scheduled_task_request(user_input or self._latest_user_message_raw())
        if not parsed:
            return False

        task_id = self.memory_store.add_scheduled_task(
            parsed["task_text"],
            parsed["schedule_time"],
            parsed["scheduled_for"],
            schedule_type=parsed.get("schedule_type", "once"),
            cron_expression=parsed.get("cron_expression", ""),
            session_id=self._memory_session_id(),
            owner_agent_id=self.state.agent_id,
        )
        if task_id <= 0:
            return False

        task_seq = self._get_or_assign_scheduled_task_seq(task_id)
        self._emit_assistant_message(f"📋 {task_seq} Task created")

        prompt = (
            "System: A scheduled task was created successfully.\n"
            f"Scheduled time: {parsed['schedule_time']}\n"
            f"Task: {parsed['task_text']}\n"
            "Reply naturally in the user's language with one short conversational sentence.\n"
            "Acknowledge the plan in your own words instead of repeating a fixed template.\n"
            "Do not call tools."
        )
        self.state.update_context("pending_control_reply_prompt", prompt)
        self.state.update_context("pending_control_type", "add")
        self._persist_general_messages_to_disk()
        return True

    def _handle_delete_indexed_task_request(self, user_input: str | None = None) -> bool:
        if not self.memory_store:
            return False
        command = (user_input or self._latest_user_message_raw()).strip().lower()
        if not command:
            return False
        prefix = "/tasks delete "
        if not command.startswith(prefix):
            return False
        idx_text = command[len(prefix):].strip()
        if idx_text == "all":
            return False
        if not idx_text.isdigit():
            return False
        if not idx_text or int(idx_text) < 1 or int(idx_text) > 5:
            return False
        target_index = int(idx_text) - 1

        tasks = self.memory_store.get_scheduled_tasks(
            limit=5,
            session_id=self._memory_session_id(),
        )
        active_tasks = [t for t in tasks if str(getattr(t, "schedule_time", "--:--")) != "--:--"]
        if target_index < 0 or target_index >= len(active_tasks):
            self._emit_assistant_message(f"No task found at index {target_index + 1}.")
            self._persist_general_messages_to_disk()
            return True

        target_task = active_tasks[target_index]
        deleted = self.memory_store.delete_scheduled_task_by_id(
            target_task.id,
            session_id=self._memory_session_id(),
        )
        if deleted:
            with contextlib.suppress(Exception):
                self._scheduled_task_name_map.pop(target_task.id, None)
            self._emit_assistant_message(f"Task {target_index + 1} deleted.")
        else:
            self._emit_assistant_message(f"Task {target_index + 1} could not be deleted.")
        self._persist_general_messages_to_disk()
        return True

    def _mark_waiting_in_tracer_and_graph(self) -> None:
        with contextlib.suppress(Exception):
            from sondra.telemetry.tracer import get_global_tracer

            tracer = get_global_tracer()
            if tracer:
                tracer.update_agent_status(self.state.agent_id, "waiting")
        with contextlib.suppress(Exception):
            from sondra.tools.agents_graph.agents_graph_actions import _agent_graph

            if self.state.agent_id in _agent_graph.get("nodes", {}):
                _agent_graph["nodes"][self.state.agent_id]["status"] = "waiting"

    def _clear_scheduled_task_subagents(self) -> None:
        with contextlib.suppress(Exception):
            from sondra.tools.agents_graph.agents_graph_actions import (
                _agent_graph,
                _agent_instances,
                _agent_messages,
                _agent_states,
                _running_agents,
            )

            remove_ids: set[str] = set()
            for node_id, node in _agent_graph.get("nodes", {}).items():
                if node_id == self.state.agent_id:
                    continue
                name = str(node.get("name", ""))
                if self._is_scheduled_task_agent_name(name):
                    remove_ids.add(node_id)

            if not remove_ids:
                return

            _agent_graph["edges"] = [
                edge
                for edge in _agent_graph.get("edges", [])
                if edge.get("from") not in remove_ids and edge.get("to") not in remove_ids
            ]
            for node_id in remove_ids:
                _agent_graph.get("nodes", {}).pop(node_id, None)
                _agent_messages.pop(node_id, None)
                _agent_instances.pop(node_id, None)
                _agent_states.pop(node_id, None)
                _running_agents.pop(node_id, None)
            self._scheduled_task_name_seq = 0
            self._scheduled_task_name_map.clear()
            self._scheduled_task_worker_map.clear()
            self._reported_scheduled_agent_ids.clear()

    def _ensure_task_runner_started(self) -> None:
        if not self._is_general_root_agent():
            return
        if self._task_runner_task and not self._task_runner_task.done():
            return
        with contextlib.suppress(Exception):
            loop = asyncio.get_running_loop()
            self._task_runner_task = loop.create_task(self._task_runner_loop())

    async def _task_runner_loop(self) -> None:
        while True:
            try:
                if not self.memory_store or not self._is_general_root_agent():
                    await asyncio.sleep(5)
                    continue
                self.memory_runtime.maybe_sync(limit=250)
                self._dispatch_due_tasks_now()
                self._report_completed_scheduled_tasks()
            except Exception:
                logger.debug("[SCHEDULE] periodic scheduler loop failed", exc_info=True)
            await asyncio.sleep(5.0)

    def _report_completed_scheduled_tasks(self) -> None:
        with contextlib.suppress(Exception):
            from sondra.tools.agents_graph.agents_graph_actions import _agent_graph

            for node_id, node in _agent_graph.get("nodes", {}).items():
                name = str(node.get("name", ""))
                if not self._is_scheduled_task_agent_name(name):
                    continue
                if node_id in self._reported_scheduled_agent_ids:
                    continue
                status = str(node.get("status", "")).lower()
                if status not in {"completed", "finished", "failed"}:
                    continue

                task_id = int(self._scheduled_task_worker_map.pop(node_id, 0) or 0)
                if task_id > 0:
                    scheduled_task = (
                        self.memory_store.get_scheduled_task_by_id(
                            task_id,
                            session_id=self._memory_session_id(),
                        )
                        if self.memory_store
                        else None
                    )
                    next_run = self._compute_next_run_for_task(scheduled_task) if scheduled_task else ""
                    if self.memory_store:
                        self.memory_store.mark_task_completed(
                            task_id,
                            next_run=next_run,
                            session_id=self._memory_session_id(),
                        )

                result = node.get("result") or {}
                summary = ""
                if isinstance(result, dict):
                    summary = str(result.get("summary") or result.get("result", "")).strip()
                if not summary:
                    summary = "Scheduled task completed."
                self._emit_assistant_message(f"[DONE] {summary}")
                self._persist_general_messages_to_disk()
                self._reported_scheduled_agent_ids.add(node_id)

    def _dispatch_due_tasks_now(self) -> None:
        if not self.memory_store or not self._is_general_root_agent():
            return
        with contextlib.suppress(Exception):
            from sondra.tools.agents_graph.agents_graph_actions import create_agent

            session_id = self._memory_session_id()
            due_tasks = self.memory_store.get_due_tasks(
                limit=5,
                session_id=session_id,
            )
            for task in due_tasks:
                task_id = int(getattr(task, "id", 0) or 0)
                task_text = str(getattr(task, "task_text", "") or "").strip()
                if task_id <= 0 or not task_text:
                    continue
                task_seq = self._get_or_assign_scheduled_task_seq(task_id)
                agent_name = f"Task {task_seq} agent"
                if not self.memory_store.claim_due_task(
                    task_id,
                    worker_name=agent_name,
                    session_id=session_id,
                ):
                    continue
                result = create_agent(
                    self.state,
                    task=(
                        "A scheduled task is due now. Execute this task and report back to your parent "
                        f"with agent_finish when done.\nTask: {task_text}"
                    ),
                    name=agent_name,
                    inherit_context=True,
                )
                if isinstance(result, dict) and result.get("success"):
                    worker_id = str(result.get("agent_id", "") or "").strip()
                    self._scheduled_task_worker_map[worker_id] = task_id
                    self.memory_store.mark_task_running(
                        task_id,
                        worker_name=agent_name,
                        worker_id=worker_id,
                        session_id=session_id,
                    )
                else:
                    self.memory_store.release_task_claim(
                        task_id,
                        session_id=session_id,
                    )

    def _emit_assistant_message(self, content: str) -> None:
        self.state.add_message("assistant", content)
        with contextlib.suppress(Exception):
            from sondra.telemetry.tracer import get_global_tracer

            tracer = get_global_tracer()
            if tracer:
                tracer.log_chat_message(
                    content=clean_content(content),
                    role="assistant",
                    agent_id=self.state.agent_id,
                )

    def _is_scheduled_task_agent_name(self, name: str) -> bool:
        lowered = str(name or "").strip().lower()
        if not lowered.startswith("task "):
            return False
        if not lowered.endswith(" agent"):
            return False
        middle = lowered[len("task "): -len(" agent")].strip()
        return middle.isdigit()

    def _is_persistent_memory_reset_command(self, text: str | None) -> bool:
        return str(text or "").strip().upper() == self.PERSISTENT_MEMORY_DELETE_COMMAND

    def _handle_persistent_memory_reset_command(self) -> bool:
        if not self._is_general_root_agent() or not self.memory_store:
            return False

        self._emit_assistant_message(self.PERSISTENT_MEMORY_DELETING_MESSAGE)

        try:
            model_name = getattr(self.llm_config, "litellm_model", None)
            self.memory_runtime.reset(model=model_name, sync_limit=300)
            with contextlib.suppress(FileNotFoundError):
                self._last_emotion_store_path().unlink()
            self._sync_memory_runtime_handles()

            self._emit_assistant_message(self.PERSISTENT_MEMORY_RESET_MESSAGE)
            self.state.update_context("memory_last_persisted_index", len(self.state.messages))
            self.state.enter_waiting_state()
            return True
        except Exception as e:
            logger.exception("Persistent memory reset failed: %s", e)
            self._emit_assistant_message("[MEMORY] Persistent memory reset failed.")
            self.memory_runtime.clear_persistent_handles()
            self._sync_memory_runtime_handles()
            self.state.update_context("memory_last_persisted_index", len(self.state.messages))
            self.state.enter_waiting_state()
            return True

    def _persist_general_messages_to_disk(self) -> None:
        persist_general_messages_to_disk_impl(self)
        if self._is_general_root_agent():
            self._refresh_emotion_context_from_memory()

    def _log_agent_emotion_state_updates(
        self,
        message_id: int,
        previous_state: dict[str, float],
        current_state: dict[str, float],
        summary: dict[str, Any],
        classification: dict[str, Any],
    ) -> None:
        if not self.memory_store:
            return
        target_id = int(message_id or 0)
        if target_id <= 0:
            return

        ts_value = str(summary.get("created_at", "") or self.memory_store.now_iso())
        confidence = max(0.0, min(float(summary.get("confidence", 0.35) or 0.35), 1.0))
        category = str(classification.get("category", "neutral") or "neutral").strip().lower()
        strength = max(0.0, min(float(classification.get("strength", 0.0) or 0.0), 1.0))

        dimension_codes = {
            "happiness": "HAP",
            "sadness": "SAD",
            "stress": "STR",
            "neutral": "NEU",
        }
        for dimension in ("happiness", "sadness", "stress", "neutral"):
            old_points = self._clamp_emotion_value(previous_state.get(dimension, 0.0))
            new_points = self._clamp_emotion_value(current_state.get(dimension, 0.0))
            if abs(new_points - old_points) < 0.01:
                continue
            old_norm = round(old_points / 100.0, 6)
            new_norm = round(new_points / 100.0, 6)
            delta_norm = round(new_norm - old_norm, 6)
            self.memory_store._log_memory_update_event(
                {
                    "ts": ts_value,
                    "memory_id": target_id,
                    "memory_type": "emotion",
                    "emotion_dimension": dimension,
                    "scope": "agent_internal",
                    "old": old_norm,
                    "new": new_norm,
                    "raw_old": round(old_points, 4),
                    "raw_new": round(new_points, 4),
                    "confidence": confidence,
                    "delta": delta_norm,
                    "reason": f"emotion_{dimension_codes[dimension]}_{category}",
                    "signal_mode": self.memory_store.signal_mode,
                    "content": category,
                    "strength": strength,
                }
            )

    def _refresh_emotion_context_from_memory(self) -> None:
        if not self.memory_store or not self._is_general_root_agent():
            return
        with contextlib.suppress(Exception):
            summary = self.memory_runtime.latest_emotion_summary(
                role="user",
                session_id=self._memory_session_id(),
            )
            if not isinstance(summary, dict):
                return
            message_id = int(summary.get("message_id", 0) or 0)
            last_message_id = int(self.state.context.get("emotion_last_message_id", 0) or 0)
            agent_state = self._agent_emotion_state_from_context()
            previous_state = dict(agent_state)
            classification: dict[str, Any] | None = None
            if message_id > 0 and message_id != last_message_id:
                agent_state, classification = self._evolve_agent_emotion_state(agent_state, summary)
                if classification is None:
                    classification = self._classify_user_emotion_input(summary)
                self._log_agent_emotion_state_updates(
                    message_id=message_id,
                    previous_state=previous_state,
                    current_state=agent_state,
                    summary=summary,
                    classification=classification,
                )
                self.state.update_context("emotion_last_message_id", message_id)
                self.state.update_context("emotion_last_updated_at", str(summary.get("created_at", "") or ""))

            tone = self._derive_agent_emotion_tone(agent_state)
            curiosity = self._derive_emotion_curiosity(agent_state)
            confidence_points = self._clamp_emotion_value(float(summary.get("confidence", 0.35) or 0.35) * 100.0, 0.0, 100.0)

            if classification is None:
                classification = self._classify_user_emotion_input(summary)

            self.state.update_context("emotion_confidence", confidence_points)
            self.state.update_context("emotion_curiosity", curiosity)
            self.state.update_context("emotion_happiness", agent_state["happiness"])
            self.state.update_context("emotion_sadness", agent_state["sadness"])
            self.state.update_context("emotion_stress", agent_state["stress"])
            self.state.update_context("emotion_neutral", agent_state["neutral"])
            self.state.update_context("emotion_tone", tone)
            self.state.update_context("emotion_signal_category", str(classification.get("category", "neutral") or "neutral"))
            self.state.update_context("emotion_signal_strength", float(classification.get("strength", 0.0) or 0.0))

    def _handle_sandbox_error(
        self,
        error: SandboxInitializationError,
        tracer: Optional["Tracer"],
    ) -> dict[str, Any]:
        error_msg = str(error.message)
        error_details = error.details
        self.state.add_error(error_msg)

        if self.non_interactive:
            self.state.set_completed({"success": False, "error": error_msg})
            if tracer:
                tracer.update_agent_status(self.state.agent_id, "failed", error_msg)
                if error_details:
                    exec_id = tracer.log_tool_execution_start(
                        self.state.agent_id,
                        "sandbox_error_details",
                        {"error": error_msg, "details": error_details},
                    )
                    tracer.update_tool_execution(exec_id, "failed", {"details": error_details})
            return {"success": False, "error": error_msg, "details": error_details}

        self.state.enter_waiting_state()
        if tracer:
            tracer.update_agent_status(self.state.agent_id, "sandbox_failed", error_msg)
            if error_details:
                exec_id = tracer.log_tool_execution_start(
                    self.state.agent_id,
                    "sandbox_error_details",
                    {"error": error_msg, "details": error_details},
                )
                tracer.update_tool_execution(exec_id, "failed", {"details": error_details})

        return {"success": False, "error": error_msg, "details": error_details}

    def _handle_llm_error(
        self,
        error: LLMRequestFailedError,
        tracer: Optional["Tracer"],
    ) -> dict[str, Any] | None:
        error_msg = str(error)
        error_details = getattr(error, "details", None)
        self.state.add_error(error_msg)
        detail_text = f"{error_msg}\n{error_details or ''}".lower()
        overflow_signatures = (
            "request_too_large",
            "context length exceeded",
            "input exceeds the maximum",
            "input token count exceeds the maximum",
            "input is too long for the model",
            "ollama error: context length exceeded",
        )
        is_context_overflow = any(token in detail_text for token in overflow_signatures)

        if self._is_general_root_agent() and is_context_overflow:
            retry_count = int(self.state.context.get("context_overflow_retry_count", 0) or 0)
            if retry_count < 2:
                self.state.update_context("context_overflow_active", True)
                self.state.update_context("context_overflow_retry_count", retry_count + 1)
                self._clear_memory_context_messages()
                self._clear_memory_result_messages()
                self._clear_memory_prompt_directive_messages()
                if tracer:
                    tracer.update_agent_status(self.state.agent_id, "running", "Context overflow recovery")
                return None

        if self.non_interactive:
            self.state.set_completed({"success": False, "error": error_msg})
            if tracer:
                tracer.update_agent_status(self.state.agent_id, "failed", error_msg)
                if error_details:
                    exec_id = tracer.log_tool_execution_start(
                        self.state.agent_id,
                        "llm_error_details",
                        {"error": error_msg, "details": error_details},
                    )
                    tracer.update_tool_execution(exec_id, "failed", {"details": error_details})
            return {"success": False, "error": error_msg}

        self.state.enter_waiting_state(llm_failed=True)
        if tracer:
            tracer.update_agent_status(self.state.agent_id, "llm_failed", error_msg)
            if error_details:
                exec_id = tracer.log_tool_execution_start(
                    self.state.agent_id,
                    "llm_error_details",
                    {"error": error_msg, "details": error_details},
                )
                tracer.update_tool_execution(exec_id, "failed", {"details": error_details})

        return None

    async def _handle_iteration_error(
        self,
        error: RuntimeError | ValueError | TypeError | asyncio.CancelledError,
        tracer: Optional["Tracer"],
    ) -> bool:
        error_msg = f"Error in iteration {self.state.iteration}: {error!s}"
        logger.exception(error_msg)
        self.state.add_error(error_msg)
        if tracer:
            tracer.update_agent_status(self.state.agent_id, "error")
        return True

    def cancel_current_execution(self) -> None:
        self._force_stop = True
        if self._current_task and not self._current_task.done():
            try:
                loop = self._current_task.get_loop()
                loop.call_soon_threadsafe(self._current_task.cancel)
            except RuntimeError:
                self._current_task.cancel()
        self._current_task = None
        if self._memory_update_task and not self._memory_update_task.done():
            self._memory_update_task.cancel()

    def _should_use_dynamic_general_routing(self) -> bool:
        return self.state.parent_id is None and self.llm_config.scan_mode == "general"
