#!/usr/bin/env python3
"""
Sondra Agent Interface
"""

import argparse
import asyncio
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import litellm
import tomllib
from docker.errors import DockerException
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from sondra.config import Config, apply_saved_config, save_current_config
from sondra.config.config import resolve_llm_config
from sondra.llm.utils import resolve_sondra_model


apply_saved_config()

from sondra.interface.auto_intent import (  # noqa: E402
    build_auto_execute_command,
    resolve_auto_intent,
)
from sondra.interface.utils import (  # noqa: E402
    assign_workspace_subdirs,
    check_docker_connection,
    clone_repository,
    collect_local_sources,
    generate_run_name,
    image_exists,
    infer_target_type,
    process_pull_line,
    rewrite_localhost_targets,
    validate_config_file,
    validate_llm_response,
)
from sondra.runtime.docker_runtime import HOST_GATEWAY_HOSTNAME  # noqa: E402
from sondra.telemetry import posthog  # noqa: E402
from sondra.telemetry.tracer import get_global_tracer  # noqa: E402


logging.getLogger().setLevel(logging.ERROR)


def _build_sondra_error_panel(error_text: Text) -> Panel:
    return Panel(
        error_text,
        title="[bold #00FCC1]𝙎 𝙊 𝙉 𝘿 𝙍 𝘼",
        title_align="left",
        border_style="#00FCC1",
        padding=(1, 2),
    )


def _confirm_continue_after_llm_error(console: Console) -> bool:
    while True:
        answer = console.input("[bold white]⚠️ Do you want to continue? \\[y/n\\]: [/]").strip().lower()
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False


def _should_retry_llm_error(error: Exception) -> bool:
    code = getattr(error, "status_code", None) or getattr(
        getattr(error, "response", None), "status_code", None
    )
    return code is None or litellm._should_retry(code)


def validate_environment() -> None:  # noqa: PLR0912, PLR0915
    console = Console()
    missing_required_vars = []
    missing_optional_vars = []

    sondra_llm = Config.get("sondra_llm")
    uses_sondra_router = sondra_llm and sondra_llm.startswith("sondra/")

    if not sondra_llm:
        missing_required_vars.append("SONDRA_LLM")

    has_base_url = uses_sondra_router or any(
        [
            Config.get("llm_api_base"),
            Config.get("openai_api_base"),
            Config.get("litellm_base_url"),
            Config.get("ollama_api_base"),
        ]
    )

    if not Config.get("llm_api_key"):
        missing_optional_vars.append("LLM_API_KEY")

    if not has_base_url:
        missing_optional_vars.append("LLM_API_BASE")

    if not Config.get("perplexity_api_key"):
        missing_optional_vars.append("PERPLEXITY_API_KEY")

    if not Config.get("sondra_reasoning_effort"):
        missing_optional_vars.append("SONDRA_REASONING_EFFORT")

    if missing_required_vars:
        error_text = Text()
        error_text.append("MISSING REQUIRED ENVIRONMENT VARIABLES", style="bold red")
        error_text.append("\n\n", style="white")

        for var in missing_required_vars:
            error_text.append(f"• {var}", style="bold yellow")
            error_text.append(" is not set\n", style="white")

        if missing_optional_vars:
            error_text.append("\nOptional environment variables:\n", style="dim white")
            for var in missing_optional_vars:
                error_text.append(f"• {var}", style="dim yellow")
                error_text.append(" is not set\n", style="dim white")

        error_text.append("\nRequired environment variables:\n", style="white")
        for var in missing_required_vars:
            if var == "SONDRA_LLM":
                error_text.append("• ", style="white")
                error_text.append("SONDRA_LLM", style="bold cyan")
                error_text.append(
                    " - Model name to use with litellm (e.g., 'openai/gpt-5')\n",
                    style="white",
                )

        if missing_optional_vars:
            error_text.append("\nOptional environment variables:\n", style="white")
            for var in missing_optional_vars:
                if var == "LLM_API_KEY":
                    error_text.append("• ", style="white")
                    error_text.append("LLM_API_KEY", style="bold cyan")
                    error_text.append(
                        " - API key for the LLM provider "
                        "(not needed for local models, Vertex AI, AWS, etc.)\n",
                        style="white",
                    )
                elif var == "LLM_API_BASE":
                    error_text.append("• ", style="white")
                    error_text.append("LLM_API_BASE", style="bold cyan")
                    error_text.append(
                        " - Custom API base URL if using local models (e.g., Ollama, LMStudio)\n",
                        style="white",
                    )
                elif var == "PERPLEXITY_API_KEY":
                    error_text.append("• ", style="white")
                    error_text.append("PERPLEXITY_API_KEY", style="bold cyan")
                    error_text.append(
                        " - API key for Perplexity AI web search (enables real-time research)\n",
                        style="white",
                    )
                elif var == "SONDRA_REASONING_EFFORT":
                    error_text.append("• ", style="white")
                    error_text.append("SONDRA_REASONING_EFFORT", style="bold cyan")
                    error_text.append(
                        " - Reasoning effort level: none, minimal, low, medium, high, xhigh "
                        "(default: high)\n",
                        style="white",
                    )

        error_text.append("\nExample setup:\n", style="white")
        if uses_sondra_router:
            error_text.append("export SONDRA_LLM='sondra/gpt-5'\n", style="dim white")
        else:
            error_text.append("export SONDRA_LLM='openai/gpt-5'\n", style="dim white")

        if missing_optional_vars:
            for var in missing_optional_vars:
                if var == "LLM_API_KEY":
                    error_text.append(
                        "export LLM_API_KEY='your-api-key-here'  "
                        "# not needed for local models, Vertex AI, AWS, etc.\n",
                        style="dim white",
                    )
                elif var == "LLM_API_BASE":
                    error_text.append(
                        "export LLM_API_BASE='http://localhost:11434'  "
                        "# needed for local models only\n",
                        style="dim white",
                    )
                elif var == "PERPLEXITY_API_KEY":
                    error_text.append(
                        "export PERPLEXITY_API_KEY='your-perplexity-key-here'\n", style="dim white"
                    )
                elif var == "SONDRA_REASONING_EFFORT":
                    error_text.append(
                        "export SONDRA_REASONING_EFFORT='high'\n",
                        style="dim white",
                    )

        panel = _build_sondra_error_panel(error_text)

        console.print("\n")
        console.print(panel)
        console.print()
        sys.exit(1)


def check_docker_installed() -> None:
    if shutil.which("docker") is None:
        console = Console()
        error_text = Text()
        error_text.append("DOCKER NOT INSTALLED", style="bold red")
        error_text.append("\n\n", style="white")
        error_text.append("The 'docker' CLI was not found in your PATH.\n", style="white")
        error_text.append(
            "Please install Docker and ensure the 'docker' command is available.\n\n", style="white"
        )

        panel = _build_sondra_error_panel(error_text)
        console.print("\n", panel, "\n")
        sys.exit(1)


async def warm_up_llm(retry_attempts: int | None = None) -> None:
    console = Console()
    max_retries = max(0, int(retry_attempts or 0))

    try:
        model_name, api_key, api_base = resolve_llm_config()
        litellm_model, _ = resolve_sondra_model(model_name)
        litellm_model = litellm_model or model_name

        test_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Reply with just 'OK'."},
        ]

        llm_timeout = int(Config.get("llm_timeout") or "300")

        completion_kwargs: dict[str, Any] = {
            "model": litellm_model,
            "messages": test_messages,
            "timeout": llm_timeout,
        }
        if api_key:
            completion_kwargs["api_key"] = api_key
        if api_base:
            completion_kwargs["api_base"] = api_base

        for attempt in range(max_retries + 1):
            try:
                response = litellm.completion(**completion_kwargs)
                break
            except Exception as e:  # noqa: BLE001
                if attempt >= max_retries or not _should_retry_llm_error(e):
                    raise
                console.print(f"🔄 Reconnecting {attempt + 1}/{max_retries} ...")
                await asyncio.sleep(min(10, 2 * (2**attempt)))

        validate_llm_response(response)

    except Exception as e:  # noqa: BLE001
        error_text = Text()
        error_text.append("LLM CONNECTION FAILED", style="bold red")
        error_text.append("\n\n", style="bold white")
        error_text.append("Could not establish connection to the language model.\n", style="bold white")
        error_text.append("Please check your configuration and try again.\n", style="bold white")
        error_text.append("\nError: ", style="bold red")
        error_text.append(str(e), style="red")

        panel = _build_sondra_error_panel(error_text)

        console.print("\n")
        console.print(panel)
        console.print()
        if not _confirm_continue_after_llm_error(console):
            sys.exit(1)


def get_version() -> str:
    try:
        project_root = Path(__file__).resolve().parents[2]
        pyproject = project_root / "pyproject.toml"
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        version = str(data.get("tool", {}).get("poetry", {}).get("version", "") or "").strip()
        if version:
            return version
    except Exception:  # noqa: BLE001
        pass

    try:
        from importlib.metadata import version

        return version("sondra-agent")
    except Exception:  # noqa: BLE001
        return "unknown"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sondra",
        description="Sondra Multi-Agent Autonomous Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # OSINT mode (+ level required)
  sondra -m osint -l standard --instruction "Investigate this email target"

  # General autonomous mode (chat/tasks without required target, level optional)
  sondra -m general

  # ADB control mode (instruction or instruction file required, level required)
  sondra -m adb -l standard --instruction-file ./adb_steps.txt

  # Pentest mode (targets required, level required)
  sondra -m pentest -l deep --target https://example.com

  # Auto mode (mode, level, target/instruction inferred from the request)
  sondra -a "Please visit https://example.com and analyze the site"
        """,
    )

    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"sondra {get_version()}",
    )

    parser.add_argument(
        "-t",
        "--target",
        type=str,
        required=False,
        action="append",
        help="Target to test (URL, repository, local directory path, domain name, or IP address). "
        "Can be specified multiple times for multi-target scans.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        help="Custom task instructions for OSINT/general/ADB workflows.",
    )

    parser.add_argument(
        "--instruction-file",
        type=str,
        help="Path to a file containing detailed custom task instructions.",
    )

    parser.add_argument(
        "-a",
        "--auto",
        type=str,
        help=(
            "Infer scan mode, level, target, and instruction from a natural-language request."
        ),
    )

    parser.add_argument(
        "-n",
        "--non-interactive",
        action="store_true",
        help=(
            "Run in non-interactive mode (no TUI, exits on completion). "
            "Default is interactive mode with TUI."
        ),
    )

    parser.add_argument(
        "-m",
        "--scan-mode",
        type=str,
        choices=["osint", "general", "adb", "pentest"],
        help=(
            "Scan mode: "
            "'osint' for open-source intelligence tasks, "
            "'general' for general autonomous user-defined tasks, "
            "'adb' for Android ADB control workflows driven by instructions, "
            "'pentest' for original Sondra-compatible penetration testing behavior."
        ),
    )

    parser.add_argument(
        "-l",
        "--scan-level",
        type=str,
        choices=["quick", "standard", "deep"],
        default=None,
        help=(
            "Execution level for non-general modes: "
            "'quick' for fast CI/CD checks, "
            "'standard' for routine testing, "
            "'deep' for thorough security reviews. "
            "Required unless --scan-mode is general."
        ),
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to a custom config file (JSON) to use instead of ~/.sondra/cli-config.json",
    )

    parser.add_argument(
        "--subagents",
        type=int,
        default=None,
        help=(
            "Force orchestrator to run with exactly this many sub-agents. "
            "If omitted, agent count is decided automatically."
        ),
    )

    parser.add_argument(
        "--retry",
        type=int,
        default=None,
        help=(
            "Retry failed LLM requests this many times before surfacing the error. "
            "If omitted, failed LLM requests are not retried."
        ),
    )

    parser.add_argument(
        "--voice-speech",
        action="store_true",
        help=(
            "Enable local Piper voice output for assistant chat replies only. "
            "Tool and command outputs are not spoken."
        ),
    )

    args = parser.parse_args()

    if args.subagents is not None and args.subagents < 1:
        parser.error("--subagents must be a positive integer")

    if args.retry is not None and args.retry < 0:
        parser.error("--retry must be a non-negative integer")

    if args.auto is not None:
        if not args.auto.strip():
            parser.error("--auto requires a non-empty request")
        conflicting_options = []
        if args.scan_mode:
            conflicting_options.append("--scan-mode")
        if args.scan_level:
            conflicting_options.append("--scan-level")
        if args.target:
            conflicting_options.append("--target")
        if args.instruction:
            conflicting_options.append("--instruction")
        if args.instruction_file:
            conflicting_options.append("--instruction-file")
        if conflicting_options:
            parser.error(
                "--auto chooses task parameters automatically; do not combine it with "
                + ", ".join(conflicting_options)
            )

        auto_result = resolve_auto_intent(args.auto)
        args.scan_mode = auto_result.scan_mode
        args.scan_level = auto_result.scan_level
        args.instruction = auto_result.instruction
        args.target = auto_result.targets
        args.instruction_file = None
        args.auto_execute_command = build_auto_execute_command(args)

    if not args.scan_mode:
        parser.error("--scan-mode (-m) is required unless --auto is used")

    if args.scan_mode != "general" and not args.scan_level:
        parser.error("--scan-level (-l) is required unless --scan-mode is general")
    if args.scan_mode == "general" and not args.scan_level:
        args.scan_level = "standard"

    if args.instruction and args.instruction_file:
        parser.error(
            "Cannot specify both --instruction and --instruction-file. Use one or the other."
        )

    if args.scan_mode == "adb" and not (args.instruction_file or args.instruction):
        parser.error("ADB mode requires --instruction or --instruction-file")
    if args.instruction_file:
        instruction_path = Path(args.instruction_file)
        try:
            with instruction_path.open(encoding="utf-8") as f:
                args.instruction = f.read().strip()
                if not args.instruction:
                    parser.error(f"Instruction file '{instruction_path}' is empty")
        except Exception as e:  # noqa: BLE001
            parser.error(f"Failed to read instruction file '{instruction_path}': {e}")

    if args.scan_mode == "general":
        args.target = args.target or []
    elif args.scan_mode == "pentest":
        if not args.target:
            parser.error("pentest mode requires at least one --target")
    elif args.scan_mode == "adb":
        args.target = []
    elif args.scan_mode == "osint":
        if not args.instruction:
            parser.error("OSINT mode requires --instruction or --instruction-file")
        args.target = []

    args.targets_info = []
    for target in args.target:
        try:
            target_type, target_dict = infer_target_type(target)

            if target_type == "local_code":
                display_target = target_dict.get("target_path", target)
            else:
                display_target = target

            args.targets_info.append(
                {"type": target_type, "details": target_dict, "original": display_target}
            )
        except ValueError:
            parser.error(f"Invalid target '{target}'")

    assign_workspace_subdirs(args.targets_info)
    rewrite_localhost_targets(args.targets_info, HOST_GATEWAY_HOSTNAME)

    return args


def display_completion_message(args: argparse.Namespace, results_path: Path) -> None:
    console = Console()
    tracer = get_global_tracer()

    scan_completed = False
    if tracer and tracer.scan_results:
        scan_completed = tracer.scan_results.get("scan_completed", False)

    has_vulnerabilities = tracer and len(tracer.vulnerability_reports) > 0

    def _resolve_exposure_count() -> int:
        """Resolve OSINT exposure count from scan results with safe fallbacks."""
        if not tracer:
            return 0

        scan_results = tracer.scan_results or {}
        for key in ("exposures", "exposure_count", "findings_count", "links_count"):
            value = scan_results.get(key)
            if isinstance(value, int):
                return max(value, 0)
            if isinstance(value, list):
                return len(value)
            if isinstance(value, dict):
                return len(value)

        for key in ("findings", "links"):
            value = scan_results.get(key)
            if isinstance(value, list):
                return len(value)
            if isinstance(value, dict):
                return len(value)

        return len(getattr(tracer, "vulnerability_reports", []) or [])

    def _build_mode_stats_text() -> Text:
        """Build mode-aware summary lines for final panel."""
        stats_text = Text()

        if args.scan_mode == "osint":
            exposure_count = _resolve_exposure_count()
            stats_text.append("Exposures", style="#00FCC1")
            stats_text.append("  ")
            stats_text.append(str(exposure_count), style="bold white")
            stats_text.append("\n")

        agent_count = len(tracer.agents) if tracer else 0
        stats_text.append("Agents", style="#00FCC1")
        stats_text.append("  ")
        stats_text.append(str(agent_count), style="bold white")
        stats_text.append("\n")

        total_cost = 0.0
        if tracer:
            llm_stats = tracer.get_total_llm_stats()
            total_cost = float(llm_stats.get("total", {}).get("cost", 0.0) or 0.0)

        stats_text.append("Cost", style="#00FCC1")
        stats_text.append(" ")
        stats_text.append(f"${total_cost:.4f}", style="bold white")
        return stats_text

    completion_text = Text()
    if scan_completed:
        completion_text.append("Session completed", style="bold #00FCC1")
    else:
        completion_text.append("SESSION ENDED", style="bold #00FCC1")

    panel_parts = [completion_text]

    stats_text = _build_mode_stats_text()
    if stats_text.plain:
        panel_parts.extend(["\n", stats_text])

    if scan_completed or has_vulnerabilities:
        results_text = Text()
        results_text.append("\n")
        results_text.append("Output", style="dim")
        results_text.append("  ")
        results_text.append(str(results_path), style="#60a5fa")
        panel_parts.extend(["\n", results_text])

    panel_content = Text.assemble(*panel_parts)

    panel = Panel(
        panel_content,
        title="[bold #00FCC1]𝙎 𝙊 𝙉 𝘿 𝙍 𝘼",
        title_align="left",
        border_style="#00FCC1",
        padding=(1, 2),
    )

    console.print("\n")
    console.print(panel)
    console.print()
    console.print("[#60a5fa]models.sondra.ai[/]  [dim]·[/]  [#60a5fa]discord.gg/sondra-ai[/]")
    console.print()


def pull_docker_image() -> None:
    console = Console()
    client = check_docker_connection()

    if image_exists(client, Config.get("sondra_image")):  # type: ignore[arg-type]
        return

    console.print()
    console.print(f"[dim]Pulling image[/] {Config.get('sondra_image')}")
    console.print("[dim yellow]This only happens on first run and may take a few minutes...[/]")
    console.print()

    with console.status("[bold cyan]Downloading image layers...", spinner="dots") as status:
        try:
            layers_info: dict[str, str] = {}
            last_update = ""

            for line in client.api.pull(Config.get("sondra_image"), stream=True, decode=True):
                last_update = process_pull_line(line, layers_info, status, last_update)

        except DockerException as e:
            console.print()
            error_text = Text()
            error_text.append("FAILED TO PULL IMAGE", style="bold red")
            error_text.append("\n\n", style="white")
            error_text.append(f"Could not download: {Config.get('sondra_image')}\n", style="white")
            error_text.append(str(e), style="dim red")

            panel = _build_sondra_error_panel(error_text)
            console.print(panel, "\n")
            sys.exit(1)

    success_text = Text()
    success_text.append("Docker image ready", style="#22c55e")
    console.print(success_text)
    console.print()


def apply_config_override(config_path: str) -> None:
    Config._config_file_override = validate_config_file(config_path)
    apply_saved_config(force=True)


def persist_config() -> None:
    if Config._config_file_override is None:
        save_current_config()


def main() -> None:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    args = parse_arguments()
    if getattr(args, "auto", None):
        console = Console()
        console.print("\n🔎 Request is being reviewed …\n")
        console.print(f'▶ Execute command: "{args.auto_execute_command}"')
        console.print()

    if args.scan_mode == "pentest":
        from sondra.interface.pentest_runtime import run_pentest_cli as run_cli  # noqa: E402
        from sondra.interface.pentest_runtime import run_pentest_tui as run_tui  # noqa: E402
    else:
        from sondra.interface.cli import run_cli  # noqa: E402
        from sondra.interface.tui import run_tui  # noqa: E402

    if args.config:
        apply_config_override(args.config)

    check_docker_installed()
    pull_docker_image()

    validate_environment()
    asyncio.run(warm_up_llm(args.retry))

    persist_config()

    args.run_name = generate_run_name(args.targets_info)

    for target_info in args.targets_info:
        if target_info["type"] == "repository":
            repo_url = target_info["details"]["target_repo"]
            dest_name = target_info["details"].get("workspace_subdir")
            cloned_path = clone_repository(repo_url, args.run_name, dest_name)
            target_info["details"]["cloned_repo_path"] = cloned_path

    args.local_sources = collect_local_sources(args.targets_info)

    is_whitebox = bool(args.local_sources)

    posthog.start(
        model=Config.get("sondra_llm"),
        scan_mode=args.scan_mode,
        is_whitebox=is_whitebox,
        interactive=not args.non_interactive,
        has_instructions=bool(args.instruction),
    )

    exit_reason = "user_exit"
    try:
        if args.non_interactive:
            asyncio.run(run_cli(args))
        else:
            asyncio.run(run_tui(args))
    except KeyboardInterrupt:
        exit_reason = "interrupted"
    except Exception as e:
        exit_reason = "error"
        posthog.error("unhandled_exception", str(e))
        raise
    finally:
        tracer = get_global_tracer()
        if tracer:
            posthog.end(tracer, exit_reason=exit_reason)

    results_path = Path("sondra_runs") / args.run_name
    display_completion_message(args, results_path)

    if args.non_interactive:
        tracer = get_global_tracer()
        if tracer and tracer.vulnerability_reports:
            sys.exit(2)


if __name__ == "__main__":
    main()
