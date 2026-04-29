from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import platform
import random
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

import tomllib
from rich.text import Text


def _load_tui_colors() -> tuple[str, str, str, str]:
    default_text = "#FFFFFF"
    default_alert = "#ef4444"
    default_na = "#ff1744"
    default_ok = "#00FCC1"
    tcss_path = Path(__file__).resolve().parents[1] / "interface" / "assets" / "tui_styles.tcss"

    with contextlib.suppress(Exception):
        content = tcss_path.read_text(encoding="utf-8")
        token_matches = dict(
            (k.upper(), v)
            for k, v in re.findall(
                r"BOOT_COLOR_(TEXT|ALERT|NA|OK)\s*:\s*(#[0-9A-Fa-f]{6})",
                content,
                flags=re.IGNORECASE,
            )
        )
        default_text = token_matches.get("TEXT", default_text)
        default_alert = token_matches.get("ALERT", default_alert)
        default_na = token_matches.get("NA", default_na)
        default_ok = token_matches.get("OK", default_ok)

    return default_text, default_alert, default_na, default_ok


WHITE, ALERT_RED, NA_RED, OK_GREEN = _load_tui_colors()
NEON_RED = ALERT_RED

EmitLine = Callable[[Any, str], Awaitable[None]]
UpdateLine = Callable[[Any, str, bool], Awaitable[None]]
WaitForEnter = Callable[[], Awaitable[None]]


@dataclass
class BootCheck:
    label: str
    status: str
    color: str = WHITE
    critical: bool = False


@dataclass
class BootSnapshot:
    mode: str
    processor: str
    firmware: str
    inference_bus: str
    memory_kb: int
    checks: list[BootCheck] = field(default_factory=list)
    ready: bool = True
    missing_components: list[str] = field(default_factory=list)


@dataclass
class BootRunResult:
    snapshot: BootSnapshot
    continued: bool


def _load_boot_banner_lines() -> list[str]:
    banner_path = Path(__file__).resolve().with_name("banner.txt")
    with contextlib.suppress(Exception):
        lines = banner_path.read_text(encoding="utf-8").splitlines()
        if any(str(line).strip() for line in lines):
            return lines
    return [
        "             _______  ",
        "            /       /",
        "   ___     /   ____/   ",
        r"  \   \  /   /\            ___       ______   ______   _   __   ___ ",
        r"   \   \/___/  \          /   |     / ____/  / ____/  / | / /  /   | ",
        r"    \       \   \        / /| |    / / __   / __/    /  |/ /  / /| | ",
        r"     \_______\   \      / ___ |   / /_/ /  / /___   / /|  /  / ___ | ",
        r"             /   /     /_/  |_|   \____/  /_____/  /_/ |_/  /_/  |_|   ",
        "             /   /        ",
        "             \\  /      Award Modular SONDRA BIOS v1.00SG, Agena Systems Build",
        "              \\/       Copyright (C) 2026, Agena Memory Systems ",
    ]


def _format_banner_line(line: str, logo_width: int = 18) -> Text:
    text = Text()
    raw = str(line or "")
    if not raw:
        return text

    split = min(max(0, int(logo_width)), len(raw))
    left = raw[:split]
    right = raw[split:]

    if left:
        text.append(left, style=NEON_RED)
    if right:
        text.append(right, style=WHITE)
    return text


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_memory_info() -> tuple[str, str, str]:
    memory_name = "UNKNOWN"
    memory_type = "UNKNOWN"
    memory_version = "UNKNOWN"

    candidates = [
        _project_root() / "sondra" / "memory" / "info.txt",
        _project_root() / "memory" / "info.txt",
    ]

    content = ""
    for info_path in candidates:
        with contextlib.suppress(Exception):
            if info_path.exists():
                content = info_path.read_text(encoding="utf-8")
                break

    if not content:
        return memory_name, memory_type, memory_version

    for raw_line in content.splitlines():
        line = str(raw_line or "").strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        k = key.strip().upper()
        v = value.strip()
        if not v:
            continue
        if k == "MEMORY NAME":
            memory_name = v
        elif k == "MEMORY TYPE":
            memory_type = v
        elif k == "VERSION":
            memory_version = v

    return memory_name, memory_type, memory_version


def _read_firmware_version() -> str:
    pyproject = _project_root() / "pyproject.toml"
    with contextlib.suppress(Exception):
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        version = str(data.get("tool", {}).get("poetry", {}).get("version", "")).strip()
        if version:
            return f"SONDRA v{version}"
    return "SONDRA v2.6.14"


def _detect_processor() -> str:
    with contextlib.suppress(Exception):
        if os.name == "nt":
            import subprocess

            result = subprocess.run(
                ["wmic", "cpu", "get", "name"],
                check=False,
                capture_output=True,
                text=True,
            )
            lines = [line.strip() for line in str(result.stdout or "").splitlines() if line.strip()]
            if len(lines) >= 2:
                return lines[1]

    with contextlib.suppress(Exception):
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore")
        for line in cpuinfo.splitlines():
            if line.lower().startswith("model name"):
                _, value = line.split(":", 1)
                model = value.strip()
                if model:
                    return model

    cpu = platform.processor().strip() or platform.machine().strip()
    return cpu or "UNKNOWN"


def _detect_inference_bus() -> str:
    from sondra.config import Config

    llm = str(Config.get("sondra_llm") or "").strip().lower()
    api_base = (
        str(os.getenv("LLM_API_BASE", "")).strip().lower()
        or str(os.getenv("OPENAI_API_BASE", "")).strip().lower()
        or str(os.getenv("LITELLM_BASE_URL", "")).strip().lower()
        or str(os.getenv("OLLAMA_API_BASE", "")).strip().lower()
    )

    if llm.startswith("ollama/") or "11434" in api_base or "ollama" in api_base:
        return "OLLAMA"
    if "localhost" in api_base or "127.0.0.1" in api_base:
        return "LOCAL"
    return "LOCAL"


def _detect_memory_total_kb() -> int:
    with contextlib.suppress(Exception):
        import psutil  # type: ignore

        return max(65536, int(psutil.virtual_memory().total // 1024))

    if os.name == "posix":
        with contextlib.suppress(Exception):
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            page_count = int(os.sysconf("SC_PHYS_PAGES"))
            return max(65536, (page_size * page_count) // 1024)

    return 384_938


def _table_exists(db_path: Path, table: str) -> bool:
    with contextlib.suppress(Exception):
        conn = sqlite3.connect(db_path)
        try:
            row = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
                (table,),
            ).fetchone()
            return bool(row)
        finally:
            conn.close()
    return False


def detect_boot_components(mode: str = "general") -> BootSnapshot:
    mode_name = str(mode or "general").strip().lower() or "general"
    is_general = mode_name == "general"

    snapshot = BootSnapshot(
        mode=mode_name,
        processor=_detect_processor(),
        firmware=_read_firmware_version(),
        inference_bus=_detect_inference_bus(),
        memory_kb=_detect_memory_total_kb(),
    )

    store = None
    index_probe: dict[str, int | float | str | bool] = {}
    embed_backend = "nomic-embed-text"
    memory_name, memory_type, memory_version = _read_memory_info()

    def add_check(
        label: str,
        ok: bool,
        ok_status: str = "OK",
        fail_status: str = "FAIL",
        *,
        critical: bool = False,
        general_only: bool = False,
    ) -> None:
        if general_only and not is_general:
            snapshot.checks.append(BootCheck(label=label, status="N/A", color=NA_RED, critical=False))
            return

        status = ok_status if ok else fail_status
        color = WHITE if ok else NEON_RED
        snapshot.checks.append(BootCheck(label=label, status=status, color=color, critical=critical))
        if critical and is_general and not ok:
            snapshot.ready = False
            snapshot.missing_components.append(label)

    with contextlib.suppress(Exception):
        from sondra.memory.persistent_memory import PersistentMemoryStore

        store = PersistentMemoryStore()

    ag_memory_ok = bool(store and Path(store.db_path).exists())

    semantic_ok = bool(store and _table_exists(Path(store.db_path), "semantic_memory"))
    profile_ok = bool(store and _table_exists(Path(store.db_path), "profile_facts"))
    task_ok = bool(store and _table_exists(Path(store.db_path), "task_state"))

    add_check(f"Detecting {memory_name} Memory", ag_memory_ok, critical=True)
    add_check("Detecting Semantic Layer", semantic_ok, critical=True)
    add_check("Detecting Profile Store", profile_ok, critical=True)
    add_check("Detecting Task Queue", task_ok, critical=True)

    snapshot.checks.append(BootCheck(label="Memory Core", status=memory_name, color=WHITE))
    snapshot.checks.append(BootCheck(label="Memory Type", status=memory_type, color=WHITE))
    snapshot.checks.append(BootCheck(label="Memory Version", status=memory_version, color=WHITE))
    add_check(
        f"{memory_name} Memory Status",
        ag_memory_ok,
        ok_status="CONNECTED",
        fail_status="DISCONNECTED",
        critical=True,
    )
    add_check("Persistent Memory", ag_memory_ok, critical=True)

    if store:
        with contextlib.suppress(Exception):
            from sondra.memory.index_manager import MemoryIndexManager

            index_probe = MemoryIndexManager(store).probe()

    index_ok = bool(index_probe)
    add_check("Memory Index", index_ok, critical=True)

    found_snapshot = False
    if store:
        with contextlib.suppress(Exception):
            profile_facts = store.get_profile_facts(top_k=1)
            semantic = store.get_semantic_memory(limit=1, reinforce=False)
            task_state = store.get_task_state() or {}
            found_snapshot = bool(profile_facts or semantic or task_state.get("goal") or task_state.get("current_step"))

    snapshot_status = "FOUND" if found_snapshot else "EMPTY"
    snapshot.checks.append(BootCheck(label="Session Snapshot", status=snapshot_status, color=WHITE))

    context_ok = False
    if store:
        with contextlib.suppress(Exception):
            restored = store.build_auto_context(query="boot probe", session_id="", top_k=2)
            context_ok = isinstance(restored, list)
    add_check("Context Restore", context_ok)

    if store:
        with contextlib.suppress(Exception):
            embed_backend = str(getattr(store, "embed_model", "") or "nomic-embed-text")
    snapshot.checks.append(BootCheck(label="Embedding Backend", status=embed_backend, color=WHITE))

    vector_ok = bool(index_probe.get("vector_available", True)) if index_probe else False
    vector_status = "ACTIVE" if vector_ok else "INACTIVE"
    vector_color = WHITE if vector_ok else NEON_RED
    snapshot.checks.append(BootCheck(label="Vector Store", status=vector_status, color=vector_color))

    decision_ok = bool(store and hasattr(store, "amygdala"))
    add_check("Decision Engine", decision_ok, critical=True)

    runtime_ok = bool(store and hasattr(store, "memory_health"))
    add_check("Memory Runtime", runtime_ok, critical=True)

    prompt_router_ok = False
    with contextlib.suppress(Exception):
        from sondra.agents.base_agent import BaseAgent

        prompt_router_ok = hasattr(BaseAgent, "_schedule_semantic_and_task_updates") and hasattr(
            BaseAgent,
            "_process_semantic_and_task_updates",
        )
    add_check("Prompt Router", prompt_router_ok, critical=True)

    tool_registry_ok = False
    with contextlib.suppress(Exception):
        from sondra.interface.tool_components.registry import ToolTUIRegistry

        _ = ToolTUIRegistry.list_tools()
        tool_registry_ok = True
    add_check("Tool Registry", tool_registry_ok)

    recall_ok = False
    if store:
        with contextlib.suppress(Exception):
            _ = store.get_profile_facts(top_k=1)
            _ = store.get_semantic_memory(limit=1, reinforce=False)
            _ = store.get_task_state()
            recall_ok = True
    add_check("Recall Test", recall_ok, ok_status="PASS", fail_status="FAIL", general_only=True)

    routing_ok = prompt_router_ok
    add_check("Routing Test", routing_ok, ok_status="PASS", fail_status="FAIL", general_only=True)

    return snapshot


def _format_status_line(label: str, status: str) -> str:
    dots = "." * max(2, 32 - len(label))
    return f"{label} {dots} {status}"


def _format_boot_status_text(label: str, status: str, base_color: str) -> Text:
    text = Text()
    dots = "." * max(2, 32 - len(label))
    text.append(f"{label} {dots} ", style=base_color)
    text.append(status, style=base_color)
    return text


def _format_memory_testing_text(current_kb: int) -> Text:
    text = Text()
    text.append(f"Memory Testing : {current_kb} KB OK", style=WHITE)
    return text


def _format_press_enter_text() -> Text:
    text = Text()
    text.append("Press ", style=WHITE)
    text.append("ENTER", style=OK_GREEN)
    text.append(" to continue...", style=WHITE)
    return text


async def print_slow_line(emit_line: EmitLine, text: Any, color: str = WHITE, delay: float = 0.08) -> None:
    await emit_line(text, color)
    await asyncio.sleep(max(0.0, delay))


async def run_memory_test(
    update_line: UpdateLine,
    final_kb: int,
    *,
    duration: float = 4.0,
    ticks: int = 40,
) -> int:
    from sondra.sounds.play_sound import play_boot_sound

    target = max(65536, int(final_kb))
    current = 0
    steps = max(8, int(ticks))
    tick_delay = max(0.01, float(duration) / steps)

    play_boot_sound()

    for step in range(steps):
        if step == steps - 1:
            current = target
        else:
            progress = (step + 1) / steps
            jitter = random.uniform(0.94, 1.03)
            current = min(target, int(target * progress * jitter))

        await update_line(_format_memory_testing_text(current), WHITE, False)
        await asyncio.sleep(tick_delay)

    await update_line(_format_memory_testing_text(target), WHITE, True)
    return target

async def wait_for_enter(wait_fn: WaitForEnter) -> None:
    await wait_fn()


async def run_boot_sequence(
    mode: str,
    emit_line: EmitLine,
    update_line: UpdateLine,
    wait_for_enter_fn: WaitForEnter,
) -> BootRunResult:
    snapshot = detect_boot_components(mode=mode)

    for banner_line in _load_boot_banner_lines():
        if str(banner_line or "").strip():
            await print_slow_line(emit_line, _format_banner_line(banner_line), WHITE, delay=0.06)
        else:
            await print_slow_line(emit_line, "", WHITE, delay=0.03)
    await print_slow_line(emit_line, "", WHITE, delay=0.05)

    await print_slow_line(emit_line, f"Main Processor : {snapshot.processor}", WHITE)
    await print_slow_line(emit_line, f"Boot Firmware  : {snapshot.firmware}", WHITE)
    await print_slow_line(emit_line, f"Inference Bus  : {snapshot.inference_bus}", WHITE)
    await print_slow_line(emit_line, "", WHITE, delay=0.05)

    await run_memory_test(
    update_line,
    final_kb=snapshot.memory_kb,
    duration=4.0,
    ticks=40,
)
    await print_slow_line(emit_line, "", WHITE, delay=0.04)
    await print_slow_line(emit_line, "", WHITE, delay=0.04)

    check_map = {check.label: check for check in snapshot.checks}
    memory_name, _, _ = _read_memory_info()
    groups: list[list[str]] = [
        [
            f"Detecting {memory_name} Memory",
            "Detecting Semantic Layer",
            "Detecting Profile Store",
            "Detecting Task Queue",
        ],
        [
            "Memory Core",
            "Memory Type",
            "Memory Version",
            f"{memory_name} Memory Status",
        ],
        [
            "Persistent Memory",
            "Memory Index",
            "Session Snapshot",
            "Context Restore",
        ],
        [
            "Embedding Backend",
            "Vector Store",
        ],
        [
            "Decision Engine",
            "Memory Runtime",
            "Prompt Router",
            "Tool Registry",
        ],
        [
            "Recall Test",
            "Routing Test",
        ],
    ]

    for gi, labels in enumerate(groups):
        for label in labels:
            check = check_map.get(label)
            if not check:
                continue
            await print_slow_line(
                emit_line,
                _format_boot_status_text(check.label, check.status, check.color),
                check.color,
                delay=0.07,
            )
        if gi < len(groups) - 1:
            await print_slow_line(emit_line, "", WHITE, delay=0.04)

    await print_slow_line(emit_line, "", WHITE, delay=0.04)
    await print_slow_line(emit_line, "", WHITE, delay=0.04)

    if snapshot.ready:
        await print_slow_line(emit_line, "SONDRA READY.", WHITE, delay=0.09)
        await print_slow_line(emit_line, "", WHITE, delay=0.04)
        await print_slow_line(emit_line, "Press DEL to enter SETUP", WHITE, delay=0.06)
        await print_slow_line(emit_line, _format_press_enter_text(), WHITE, delay=0.06)
        await wait_for_enter(wait_for_enter_fn)
        return BootRunResult(snapshot=snapshot, continued=True)

    await print_slow_line(emit_line, "SONDRA NOT READY.", NEON_RED, delay=0.06)
    await print_slow_line(emit_line, "Missing required components:", NEON_RED, delay=0.02)
    for missing in snapshot.missing_components:
        await print_slow_line(emit_line, f"- {missing}", NEON_RED, delay=0.02)
    await print_slow_line(emit_line, "Startup blocked in GENERAL mode.", NEON_RED, delay=0.03)
    return BootRunResult(snapshot=snapshot, continued=False)


async def _console_emit_line(text: Any, color: str = WHITE) -> None:
    _ = color
    print(text.plain if isinstance(text, Text) else str(text), flush=True)


async def _console_update_line(text: Any, color: str = WHITE, final: bool = False) -> None:
    _ = color
    raw = text.plain if isinstance(text, Text) else str(text)
    sys.stdout.write(f"\r{raw}")
    if final:
        sys.stdout.write("\n")
    sys.stdout.flush()


async def _console_wait_for_enter() -> None:
    await asyncio.to_thread(input)


def main() -> int:
    parser = argparse.ArgumentParser(description="SONDRA BIOS boot screen")
    parser.add_argument("--mode", default="general", help="Sondra mode (general/osint/pentest/...)" )
    args = parser.parse_args()

    result = asyncio.run(
        run_boot_sequence(
            args.mode,
            _console_emit_line,
            _console_update_line,
            _console_wait_for_enter,
        )
    )
    return 0 if result.continued else 1


if __name__ == "__main__":
    raise SystemExit(main())
