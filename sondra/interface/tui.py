import argparse
import asyncio
import atexit
import contextlib
import importlib
import importlib.util
import logging
import os
import signal
import sys
import threading
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import tomllib

if os.name == "posix":
    import termios


if TYPE_CHECKING:
    from textual.timer import Timer

from rich.align import Align
from rich.console import Group
from rich.panel import Panel
from rich.style import Style
from rich.text import Text
from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static, TextArea, Tree
from textual.widgets.tree import TreeNode

from sondra.agents.SondraAgent import SondraAgent
from sondra.interface.streaming_parser import parse_streaming_content
from sondra.interface.tool_components.agent_message_renderer import AgentMessageRenderer
from sondra.interface.tool_components.registry import get_tool_renderer
from sondra.interface.tool_components.user_message_renderer import UserMessageRenderer
from sondra.interface.utils import build_tui_stats_text
from sondra.llm.config import LLMConfig
from sondra.sounds.play_sound import play_choose_sound
from sondra.speech.voice_tts import VoiceSpeechEngine
from sondra.telemetry.tracer import Tracer, set_global_tracer
from sondra.interface.widgets.network_graph import NetworkGraph

logger = logging.getLogger(__name__)


def get_package_version() -> str:
    # Prefer local project metadata so UI reflects repository version immediately.
    with contextlib.suppress(Exception):
        project_root = Path(__file__).resolve().parents[2]
        pyproject = project_root / "pyproject.toml"
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        return str(data.get("tool", {}).get("poetry", {}).get("version", "") or "").strip() or "dev"

    try:
        return pkg_version("sondra-agent")
    except PackageNotFoundError:
        return "dev"

def get_display_version_label() -> str:
    version = get_package_version()
    if version.endswith("b0"):
        base = version[:-2]
        return f"S-{base} Beta"
    return f"v{version}"

class ChatTextArea(TextArea):  # type: ignore[misc]
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._app_reference: SondraTUIApp | None = None

    def set_app_reference(self, app: "SondraTUIApp") -> None:
        self._app_reference = app

    def on_mount(self) -> None:
        self._update_height()

    def _on_key(self, event: events.Key) -> None:
        if event.key == "shift+enter":
            self.insert("\n")
            event.prevent_default()
            return

        if event.key == "enter" and self._app_reference:
            text_content = str(self.text)  # type: ignore[has-type]
            message = text_content.strip()
            if message:
                self.text = ""

                self._app_reference._send_user_message(message)

                event.prevent_default()
                return

        super()._on_key(event)

    @on(TextArea.Changed)  # type: ignore[misc]
    def _update_height(self, _event: TextArea.Changed | None = None) -> None:
        if not self.parent:
            return

        line_count = self.document.line_count
        target_lines = min(max(1, line_count), 8)

        new_height = target_lines + 2

        if self.parent.styles.height != new_height:
            self.parent.styles.height = new_height
            self.scroll_cursor_visible()


class SplashScreen(Static):  # type: ignore[misc]
    ALLOW_SELECT = False
    PRIMARY_GREEN = "#00FCC1"
    PANEL_WIDTH = 78
    BANNER = r"""
    
           _______  _____  __   _ ______   ______ _______
           |______ |     | | \  | |     \ |_____/ |_____|
            ______||_____| |  \_| |_____/ |    \_ |     |                                           

                        ᴀɢᴇɴᴀ ᴍᴇᴍᴏʀʏ sʏsᴛᴇᴍs
                        · · ──── ·⟡· ──── · ·
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._animation_step = 0
        self._animation_timer: Timer | None = None
        self._panel_static: Static | None = None
        self._version = "dev"

    def compose(self) -> ComposeResult:
        self._version = get_package_version()
        self._animation_step = 0
        start_line = self._build_start_line_text(self._animation_step)
        panel = self._build_panel(start_line)

        panel_static = Static(panel, id="splash_content")
        self._panel_static = panel_static
        yield panel_static

    def on_mount(self) -> None:
        self._animation_timer = self.set_interval(0.05, self._animate_start_line)

    def on_unmount(self) -> None:
        if self._animation_timer is not None:
            self._animation_timer.stop()
            self._animation_timer = None

    def _animate_start_line(self) -> None:
        if not self._panel_static:
            return

        self._animation_step += 1
        start_line = self._build_start_line_text(self._animation_step)
        panel = self._build_panel(start_line)
        self._panel_static.update(panel)

    def _build_panel(self, start_line: Text) -> Panel:
        banner = Text.from_ansi(self.BANNER.strip("\n"))
        banner.stylize(self.PRIMARY_GREEN)

        content = Group(
            Align.center(banner),
            Align.center(Text(" ")),
            Align.center(self._build_welcome_text()),
            Align.center(self._build_version_text()),
            Align.center(self._build_tagline_text()),
            Align.center(Text(" ")),
            Align.center(start_line.copy()),
            Align.center(Text(" ")),
            Align.center(self._build_url_text()),
        )

        return Panel(
            content,
            border_style=self.PRIMARY_GREEN,
            padding=(1, 3),
            width=self.PANEL_WIDTH,
        )

    def _build_url_text(self) -> Text:
        return Text("sondra.ai", style=Style(color=self.PRIMARY_GREEN, bold=True))

    def _build_welcome_text(self) -> Text:
        text = Text("Welcome to ", style=Style(color="white", bold=True))
        text.append("Sondra", style=Style(color=self.PRIMARY_GREEN, bold=True))
        text.append("!", style=Style(color="white", bold=True))
        return text

    def _build_version_text(self) -> Text:
        return Text(get_display_version_label(), style=Style(color="white", dim=True))

    def _build_tagline_text(self) -> Text:
        return Text("An open-source AI agent for all-purpose tasks", style=Style(color="white", dim=True))

    def _build_start_line_text(self, phase: int) -> Text:
        full_text = "Starting Sondra Agent"
        text_len = len(full_text)

        shine_pos = phase % (text_len + 8)

        text = Text()
        for i, char in enumerate(full_text):
            dist = abs(i - shine_pos)

            if dist <= 1:
                style = Style(color="bright_white", bold=True)
            elif dist <= 3:
                style = Style(color="white", bold=True)
            elif dist <= 5:
                style = Style(color="#a3a3a3")
            else:
                style = Style(color="#525252")

            text.append(char, style=style)

        return text


class HelpScreen(ModalScreen):  # type: ignore[misc]
    def compose(self) -> ComposeResult:
        yield Grid(
            Label("Sondra Help", id="help_title"),
            Label(
                "F1        Help\nCtrl+Q/C  Quit\nESC       Stop Agent\n"
                "Enter     Send message to agent\nTab       Switch panels\n↑/↓       Navigate tree",
                id="help_content",
            ),
            id="dialog",
        )

    def on_key(self, event: events.Key) -> None:
        self.app.pop_screen()


class StopAgentScreen(ModalScreen):  # type: ignore[misc]
    def __init__(self, agent_name: str, agent_id: str):
        super().__init__()
        self.agent_name = agent_name
        self.agent_id = agent_id

    def compose(self) -> ComposeResult:
        yield Grid(
            Label(f"🛑 Stop '{self.agent_name}'?", id="stop_agent_title"),
            Grid(
                Button("Yes", variant="error", id="stop_agent"),
                Button("No", variant="default", id="cancel_stop"),
                id="stop_agent_buttons",
            ),
            id="stop_agent_dialog",
        )

    def on_mount(self) -> None:
        cancel_button = self.query_one("#cancel_stop", Button)
        cancel_button.focus()

    def on_key(self, event: events.Key) -> None:
        if event.key in ("left", "right", "up", "down"):
            focused = self.focused

            if focused and focused.id == "stop_agent":
                cancel_button = self.query_one("#cancel_stop", Button)
                cancel_button.focus()
            else:
                stop_button = self.query_one("#stop_agent", Button)
                stop_button.focus()

            event.prevent_default()
        elif event.key == "enter":
            focused = self.focused
            if focused and isinstance(focused, Button):
                focused.press()
            event.prevent_default()
        elif event.key == "escape":
            self.app.pop_screen()
            event.prevent_default()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "stop_agent":
            self.app.action_confirm_stop_agent(self.agent_id)
        else:
            self.app.pop_screen()


class VulnerabilityDetailScreen(ModalScreen):  # type: ignore[misc]
    """Modal screen to display vulnerability details."""

    SEVERITY_COLORS: ClassVar[dict[str, str]] = {
        "critical": "#dc2626",  # Red
        "high": "#ea580c",  # Orange
        "medium": "#d97706",  # Amber
        "low": "#22c55e",  # Green
        "info": "#3b82f6",  # Blue
    }

    FIELD_STYLE: ClassVar[str] = "bold #4ade80"

    def __init__(self, vulnerability: dict[str, Any]) -> None:
        super().__init__()
        self.vulnerability = vulnerability

    def compose(self) -> ComposeResult:
        content = self._render_vulnerability()
        yield Grid(
            VerticalScroll(Static(content, id="vuln_detail_content"), id="vuln_detail_scroll"),
            Horizontal(
                Button("Copy", variant="default", id="copy_vuln_detail"),
                Button("Done", variant="default", id="close_vuln_detail"),
                id="vuln_detail_buttons",
            ),
            id="vuln_detail_dialog",
        )

    def on_mount(self) -> None:
        close_button = self.query_one("#close_vuln_detail", Button)
        close_button.focus()

    def _get_cvss_color(self, cvss_score: float) -> str:
        if cvss_score >= 9.0:
            return "#dc2626"
        if cvss_score >= 7.0:
            return "#ea580c"
        if cvss_score >= 4.0:
            return "#d97706"
        if cvss_score >= 0.1:
            return "#65a30d"
        return "#6b7280"

    def _highlight_python(self, code: str) -> Text:
        try:
            from pygments.lexers import PythonLexer
            from pygments.styles import get_style_by_name

            lexer = PythonLexer()
            style = get_style_by_name("native")
            colors = {
                token: f"#{style_def['color']}" for token, style_def in style if style_def["color"]
            }

            text = Text()
            for token_type, token_value in lexer.get_tokens(code):
                if not token_value:
                    continue
                color = None
                tt = token_type
                while tt:
                    if tt in colors:
                        color = colors[tt]
                        break
                    tt = tt.parent
                text.append(token_value, style=color)
        except (ImportError, KeyError, AttributeError):
            return Text(code)
        else:
            return text

    def _render_vulnerability(self) -> Text:  # noqa: PLR0912, PLR0915
        vuln = self.vulnerability
        text = Text()

        text.append("🐞 ")
        text.append("Vulnerability Report", style="bold #ea580c")

        agent_name = vuln.get("agent_name", "")
        if agent_name:
            text.append("\n\n")
            text.append("Agent: ", style=self.FIELD_STYLE)
            text.append(agent_name)

        title = vuln.get("title", "")
        if title:
            text.append("\n\n")
            text.append("Title: ", style=self.FIELD_STYLE)
            text.append(title)

        severity = vuln.get("severity", "")
        if severity:
            text.append("\n\n")
            text.append("Severity: ", style=self.FIELD_STYLE)
            severity_color = self.SEVERITY_COLORS.get(severity.lower(), "#6b7280")
            text.append(severity.upper(), style=f"bold {severity_color}")

        cvss_score = vuln.get("cvss")
        if cvss_score is not None:
            text.append("\n\n")
            text.append("CVSS Score: ", style=self.FIELD_STYLE)
            cvss_color = self._get_cvss_color(float(cvss_score))
            text.append(str(cvss_score), style=f"bold {cvss_color}")

        target = vuln.get("target", "")
        if target:
            text.append("\n\n")
            text.append("Target: ", style=self.FIELD_STYLE)
            text.append(target)

        endpoint = vuln.get("endpoint", "")
        if endpoint:
            text.append("\n\n")
            text.append("Endpoint: ", style=self.FIELD_STYLE)
            text.append(endpoint)

        method = vuln.get("method", "")
        if method:
            text.append("\n\n")
            text.append("Method: ", style=self.FIELD_STYLE)
            text.append(method)

        cve = vuln.get("cve", "")
        if cve:
            text.append("\n\n")
            text.append("CVE: ", style=self.FIELD_STYLE)
            text.append(cve)

        # CVSS breakdown
        cvss_breakdown = vuln.get("cvss_breakdown", {})
        if cvss_breakdown:
            cvss_parts = []
            if cvss_breakdown.get("attack_vector"):
                cvss_parts.append(f"AV:{cvss_breakdown['attack_vector']}")
            if cvss_breakdown.get("attack_complexity"):
                cvss_parts.append(f"AC:{cvss_breakdown['attack_complexity']}")
            if cvss_breakdown.get("privileges_required"):
                cvss_parts.append(f"PR:{cvss_breakdown['privileges_required']}")
            if cvss_breakdown.get("user_interaction"):
                cvss_parts.append(f"UI:{cvss_breakdown['user_interaction']}")
            if cvss_breakdown.get("scope"):
                cvss_parts.append(f"S:{cvss_breakdown['scope']}")
            if cvss_breakdown.get("confidentiality"):
                cvss_parts.append(f"C:{cvss_breakdown['confidentiality']}")
            if cvss_breakdown.get("integrity"):
                cvss_parts.append(f"I:{cvss_breakdown['integrity']}")
            if cvss_breakdown.get("availability"):
                cvss_parts.append(f"A:{cvss_breakdown['availability']}")
            if cvss_parts:
                text.append("\n\n")
                text.append("CVSS Vector: ", style=self.FIELD_STYLE)
                text.append("/".join(cvss_parts), style="dim")

        description = vuln.get("description", "")
        if description:
            text.append("\n\n")
            text.append("Description", style=self.FIELD_STYLE)
            text.append("\n")
            text.append(description)

        impact = vuln.get("impact", "")
        if impact:
            text.append("\n\n")
            text.append("Impact", style=self.FIELD_STYLE)
            text.append("\n")
            text.append(impact)

        technical_analysis = vuln.get("technical_analysis", "")
        if technical_analysis:
            text.append("\n\n")
            text.append("Technical Analysis", style=self.FIELD_STYLE)
            text.append("\n")
            text.append(technical_analysis)

        poc_description = vuln.get("poc_description", "")
        if poc_description:
            text.append("\n\n")
            text.append("PoC Description", style=self.FIELD_STYLE)
            text.append("\n")
            text.append(poc_description)

        poc_script_code = vuln.get("poc_script_code", "")
        if poc_script_code:
            text.append("\n\n")
            text.append("PoC Code", style=self.FIELD_STYLE)
            text.append("\n")
            text.append_text(self._highlight_python(poc_script_code))

        remediation_steps = vuln.get("remediation_steps", "")
        if remediation_steps:
            text.append("\n\n")
            text.append("Remediation", style=self.FIELD_STYLE)
            text.append("\n")
            text.append(remediation_steps)

        return text

    def _get_markdown_report(self) -> str:  # noqa: PLR0912, PLR0915
        """Get Markdown version of vulnerability report for clipboard."""
        vuln = self.vulnerability
        lines: list[str] = []

        # Title
        title = vuln.get("title", "Untitled Vulnerability")
        lines.append(f"# {title}")
        lines.append("")

        # Metadata
        if vuln.get("id"):
            lines.append(f"**ID:** {vuln['id']}")
        if vuln.get("severity"):
            lines.append(f"**Severity:** {vuln['severity'].upper()}")
        if vuln.get("timestamp"):
            lines.append(f"**Found:** {vuln['timestamp']}")
        if vuln.get("agent_name"):
            lines.append(f"**Agent:** {vuln['agent_name']}")
        if vuln.get("target"):
            lines.append(f"**Target:** {vuln['target']}")
        if vuln.get("endpoint"):
            lines.append(f"**Endpoint:** {vuln['endpoint']}")
        if vuln.get("method"):
            lines.append(f"**Method:** {vuln['method']}")
        if vuln.get("cve"):
            lines.append(f"**CVE:** {vuln['cve']}")
        if vuln.get("cvss") is not None:
            lines.append(f"**CVSS:** {vuln['cvss']}")

        # CVSS Vector
        cvss_breakdown = vuln.get("cvss_breakdown", {})
        if cvss_breakdown:
            abbrevs = {
                "attack_vector": "AV",
                "attack_complexity": "AC",
                "privileges_required": "PR",
                "user_interaction": "UI",
                "scope": "S",
                "confidentiality": "C",
                "integrity": "I",
                "availability": "A",
            }
            parts = [
                f"{abbrevs.get(k, k)}:{v}" for k, v in cvss_breakdown.items() if v and k in abbrevs
            ]
            if parts:
                lines.append(f"**CVSS Vector:** {'/'.join(parts)}")

        # Description
        lines.append("")
        lines.append("## Description")
        lines.append("")
        lines.append(vuln.get("description") or "No description provided.")

        # Impact
        if vuln.get("impact"):
            lines.extend(["", "## Impact", "", vuln["impact"]])

        # Technical Analysis
        if vuln.get("technical_analysis"):
            lines.extend(["", "## Technical Analysis", "", vuln["technical_analysis"]])

        # Proof of Concept
        if vuln.get("poc_description") or vuln.get("poc_script_code"):
            lines.extend(["", "## Proof of Concept", ""])
            if vuln.get("poc_description"):
                lines.append(vuln["poc_description"])
                lines.append("")
            if vuln.get("poc_script_code"):
                lines.append("```python")
                lines.append(vuln["poc_script_code"])
                lines.append("```")

        # Code Analysis
        if vuln.get("code_locations"):
            lines.extend(["", "## Code Analysis", ""])
            for i, loc in enumerate(vuln["code_locations"]):
                file_ref = loc.get("file", "unknown")
                line_ref = ""
                if loc.get("start_line") is not None:
                    if loc.get("end_line") and loc["end_line"] != loc["start_line"]:
                        line_ref = f" (lines {loc['start_line']}-{loc['end_line']})"
                    else:
                        line_ref = f" (line {loc['start_line']})"
                lines.append(f"**Location {i + 1}:** `{file_ref}`{line_ref}")
                if loc.get("label"):
                    lines.append(f"  {loc['label']}")
                if loc.get("snippet"):
                    lines.append(f"```\n{loc['snippet']}\n```")
                if loc.get("fix_before") or loc.get("fix_after"):
                    lines.append("**Suggested Fix:**")
                    lines.append("```diff")
                    if loc.get("fix_before"):
                        lines.extend(f"- {line}" for line in loc["fix_before"].splitlines())
                    if loc.get("fix_after"):
                        lines.extend(f"+ {line}" for line in loc["fix_after"].splitlines())
                    lines.append("```")
                lines.append("")

        # Remediation
        if vuln.get("remediation_steps"):
            lines.extend(["", "## Remediation", "", vuln["remediation_steps"]])

        lines.append("")
        return "\n".join(lines)

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            self.app.pop_screen()
            event.prevent_default()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "copy_vuln_detail":
            markdown_text = self._get_markdown_report()
            self.app.copy_to_clipboard(markdown_text)

            copy_button = self.query_one("#copy_vuln_detail", Button)
            copy_button.label = "Copied!"
            self.set_timer(1.5, lambda: setattr(copy_button, "label", "Copy"))
        elif event.button.id == "close_vuln_detail":
            self.app.pop_screen()


class VulnerabilityItem(Static):  # type: ignore[misc]
    """A clickable vulnerability item."""

    def __init__(self, label: Text, vuln_data: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(label, **kwargs)
        self.vuln_data = vuln_data

    def on_click(self, _event: events.Click) -> None:
        """Handle click to open vulnerability detail."""
        self.app.push_screen(VulnerabilityDetailScreen(self.vuln_data))


class VulnerabilitiesPanel(VerticalScroll):  # type: ignore[misc]
    """A scrollable panel showing found vulnerabilities with severity-colored dots."""

    SEVERITY_COLORS: ClassVar[dict[str, str]] = {
        "critical": "#dc2626",  # Red
        "high": "#ea580c",  # Orange
        "medium": "#d97706",  # Amber
        "low": "#22c55e",  # Green
        "info": "#3b82f6",  # Blue
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._vulnerabilities: list[dict[str, Any]] = []

    def compose(self) -> ComposeResult:
        return []

    def update_vulnerabilities(self, vulnerabilities: list[dict[str, Any]]) -> None:
        """Update the list of vulnerabilities and re-render."""
        if self._vulnerabilities == vulnerabilities:
            return
        self._vulnerabilities = list(vulnerabilities)
        self._render_panel()

    def _render_panel(self) -> None:
        """Render the vulnerabilities panel content."""
        for child in list(self.children):
            if isinstance(child, VulnerabilityItem):
                child.remove()

        if not self._vulnerabilities:
            return

        for vuln in self._vulnerabilities:
            severity = vuln.get("severity", "info").lower()
            title = vuln.get("title", "Unknown Vulnerability")
            color = self.SEVERITY_COLORS.get(severity, "#3b82f6")

            label = Text()
            label.append("● ", style=Style(color=color))
            label.append(title, style=Style(color="#d4d4d4"))

            item = VulnerabilityItem(label, vuln, classes="vuln-item")
            self.mount(item)


class QuitScreen(ModalScreen):  # type: ignore[misc]
    def compose(self) -> ComposeResult:
        yield Grid(
            Label("Quit Sondra?", id="quit_title"),
            Grid(
                Button("Yes", variant="error", id="quit"),
                Button("No", variant="default", id="cancel"),
                id="quit_buttons",
            ),
            id="quit_dialog",
        )

    def on_mount(self) -> None:
        cancel_button = self.query_one("#cancel", Button)
        cancel_button.focus()

    def on_key(self, event: events.Key) -> None:
        if event.key in ("left", "right", "up", "down"):
            focused = self.focused

            if focused and focused.id == "quit":
                cancel_button = self.query_one("#cancel", Button)
                cancel_button.focus()
            else:
                quit_button = self.query_one("#quit", Button)
                quit_button.focus()

            event.prevent_default()
        elif event.key == "enter":
            focused = self.focused
            if focused and isinstance(focused, Button):
                focused.press()
            event.prevent_default()
        elif event.key == "escape":
            self.app.pop_screen()
            event.prevent_default()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "quit":
            self.app.action_custom_quit()
        else:
            self.app.pop_screen()


class SondraTUIApp(App):  # type: ignore[misc]
    CSS_PATH = "assets/tui_styles.tcss"
    ALLOW_SELECT = True

    SIDEBAR_MIN_WIDTH = 120

    selected_agent_id: reactive[str | None] = reactive(default=None)
    show_splash: reactive[bool] = reactive(default=True)

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("f1", "toggle_help", "Help", priority=True),
        Binding("ctrl+q", "request_quit", "Quit", priority=True),
        Binding("ctrl+c", "request_quit", "Quit", priority=True),
        Binding("escape", "stop_selected_agent", "Stop Agent", priority=True),
    ]

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        self.scan_config = self._build_scan_config(args)
        self.agent_config = self._build_agent_config(args)

        self.tracer = Tracer(self.scan_config["run_name"])
        self.tracer.set_scan_config(self.scan_config)
        set_global_tracer(self.tracer)

        self.agent_nodes: dict[str, TreeNode] = {}

        self._displayed_agents: set[str] = set()
        self._displayed_events: list[str] = []

        self._streaming_render_cache: dict[str, tuple[int, Any]] = {}
        self._last_streaming_len: dict[str, int] = {}

        self._scan_thread: threading.Thread | None = None
        self._scan_stop_event = threading.Event()
        self._scan_completed = threading.Event()

        self._boot_lines: list[tuple[str, str]] = []
        self._boot_in_progress = False
        self._boot_blocked = False
        self._boot_waiting_for_enter = False
        self._boot_enter_event: asyncio.Event | None = None
        self._boot_task: asyncio.Task[Any] | None = None

        self._spinner_frame_index: int = 0  # Current animation frame index
        self._sweep_num_squares: int = 6  # Number of squares in sweep animation
        self._sweep_colors: list[str] = [
            "#000000",  # Dimmest (shows dot)
            "#031a09",
            "#052e16",
            "#0d4a2a",
            "#15803d",
            "#22c55e",
            "#4ade80",
            "#86efac",  # Brightest
        ]
        self._dot_animation_timer: Any | None = None

        self.voice_speech_enabled = bool(getattr(args, "voice_speech", False))
        self._voice_engine: VoiceSpeechEngine | None = None
        self._spoken_message_ids: set[int] = set()
        if self.voice_speech_enabled:
            self._voice_engine = VoiceSpeechEngine()
            if not self._voice_engine.is_available():
                logger.warning(
                    "Voice speech unavailable: %s",
                    self._voice_engine.init_error or "initialization failed",
                )

        self._setup_cleanup_handlers()

    def _build_scan_config(self, args: argparse.Namespace) -> dict[str, Any]:
        from sondra.config import Config

        return {
            "scan_id": args.run_name,
            "targets": args.targets_info,
            "user_instructions": args.instruction or "",
            "run_name": args.run_name,
            "scan_mode": getattr(args, "scan_mode", "general"),
            "scan_level": getattr(args, "scan_level", "standard"),
            "model_name": str(Config.get("sondra_llm") or ""),
        }

    def _build_agent_config(self, args: argparse.Namespace) -> dict[str, Any]:
        scan_mode = getattr(args, "scan_mode", "general")
        scan_level = getattr(args, "scan_level", "standard")
        llm_config = LLMConfig(
            scan_mode=scan_mode,
            scan_level=scan_level,
            interactive=False,
            fixed_agents=getattr(args, "subagents", None),
            retry_attempts=getattr(args, "retry", None),
        )

        config = {
            "llm_config": llm_config,
            "max_iterations": 300,
        }

        if getattr(args, "local_sources", None):
            config["local_sources"] = args.local_sources

        return config

    def _setup_cleanup_handlers(self) -> None:
        def cleanup_on_exit() -> None:
            from sondra.runtime import cleanup_runtime

            self._persist_root_agent_last_emotion()
            self.tracer.cleanup()
            cleanup_runtime()

        def signal_handler(_signum: int, _frame: Any) -> None:
            self._persist_root_agent_last_emotion()
            self.tracer.cleanup()
            sys.exit(0)

        atexit.register(cleanup_on_exit)
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, signal_handler)

    def _persist_root_agent_last_emotion(self) -> None:
        with contextlib.suppress(Exception):
            graph_actions = self._get_active_graph_actions_module()
            _agent_instances = getattr(graph_actions, "_agent_instances", {})
            _root_agent_id = getattr(graph_actions, "_root_agent_id", None)

            if not _root_agent_id:
                return
            agent = _agent_instances.get(_root_agent_id)
            if not agent:
                return
            persist_fn = getattr(agent, "persist_last_emotion_snapshot", None)
            if callable(persist_fn):
                persist_fn()

    def _get_active_graph_actions_module(self) -> Any:
        scan_mode = str(self.scan_config.get("scan_mode", "general") or "general").strip().lower()
        module_name = (
            "sondra.tools.agents_graph.agents_graph_actions"
            if scan_mode == "general"
            else "sondra.pentest_tools.agents_graph.agents_graph_actions"
        )
        return importlib.import_module(module_name)

    def compose(self) -> ComposeResult:
        if self.show_splash:
            yield SplashScreen(id="splash_screen")

    @staticmethod
    def _build_command_hint_text() -> Text:
        hint_text = Text()
        hint_text.append("  ↑↓  ", style="bold #FFFFFF on #0b7a64")
        hint_text.append("  Scroll terminal  ", style="bold #FFFFFF")
        hint_text.append("  ◄ ►  ", style="bold #FFFFFF on #0b7a64")
        hint_text.append("  Change indicator page  ", style="bold #FFFFFF")
        hint_text.append(" ✖  ", style="bold #FFFFFF on #0b7a64")
        hint_text.append("  CTRL + Q Exit  ", style="bold #FFFFFF")
        hint_text.append(" ⊞  ", style="bold #FFFFFF on #0b7a64")
        hint_text.append("  Set window: CTRL +/-  ", style="bold #FFFFFF")
        return hint_text

    def _build_voice_volume_text(self) -> Text:
        volume_text = Text()
        if not self.voice_speech_enabled or not self._voice_engine or not self._voice_engine.is_available():
            return volume_text

        percent = self._voice_engine.get_volume_percent()
        segments = 12
        filled = max(0, min(segments, int(round((percent / 100) * segments))))
        bar = ("▰" * filled) + ("▱" * (segments - filled))
        volume_text.append("🔊  ", style="bold #FFFFFF")
        volume_text.append(bar, style="bold #FFFFFF")
        volume_text.append(f"  {percent}%", style="bold #FFFFFF")
        return volume_text

    def _update_voice_volume_indicator(self) -> None:
        try:
            volume_widget = self.query_one("#voice_volume_text", Static)
        except (ValueError, Exception):
            return

        if not self._is_widget_safe(volume_widget):
            return

        self._safe_widget_operation(volume_widget.update, self._build_voice_volume_text())

    @staticmethod
    def _build_stopped_hint_text() -> Text:
        stopped_text = Text()
        stopped_text.append("  ▶  ", style="bold #FFFFFF on #0b7a64")
        stopped_text.append("  Agent stopped  ", style="bold #FFFFFF")
        return stopped_text

    def watch_show_splash(self, show_splash: bool) -> None:
        if not show_splash and self.is_mounted:
            try:
                splash = self.query_one("#splash_screen")
                splash.remove()
            except ValueError:
                pass

            main_container = Vertical(id="main_container")

            self.mount(main_container)

            content_container = Horizontal(id="content_container")
            main_container.mount(content_container)

            chat_area_container = Vertical(id="chat_area_container")

            chat_display = Static("", id="chat_display")
            command_hint_text = Static(
                self._build_command_hint_text(),
                id="command_hint_text",
            )
            command_hint_text.ALLOW_SELECT = False
            voice_volume_text = Static(self._build_voice_volume_text(), id="voice_volume_text")
            voice_volume_text.ALLOW_SELECT = False
            command_hint_bar = Horizontal(
                command_hint_text,
                voice_volume_text,
                id="command_hint_bar",
            )
            command_hint_bar.ALLOW_SELECT = False
            chat_history = VerticalScroll(chat_display, command_hint_bar, id="chat_history")
            chat_history.border_title = Text(
               " TERMINAL ",
               style="bold #00FCC1"
            )

            chat_history.styles.border_title_align = "center"
            chat_history.can_focus = True

            status_text = Static("", id="status_text")
            status_text.ALLOW_SELECT = False
            keymap_indicator = Static("", id="keymap_indicator")
            keymap_indicator.ALLOW_SELECT = False

            agent_status_display = Horizontal(
                status_text, keymap_indicator, id="agent_status_display", classes="hidden"
            )
            chat_prompt = Static(">_ ", id="chat_prompt")
            chat_prompt.ALLOW_SELECT = False
            chat_input = ChatTextArea(
                "",
                id="chat_input",
                show_line_numbers=False,
            )
            chat_input.set_app_reference(self)
            chat_input_container = Horizontal(chat_prompt, chat_input, id="chat_input_container")

            agents_tree = Tree("Agents", id="agents_tree")
            agents_tree.border_title = Text(
              " AGENT MAP ",
              style="bold #00FCC1"
            )

            agents_tree.styles.border_title_align = "center"
            agents_tree.root.expand()
            agents_tree.show_root = False

            agents_tree.show_guide = True
            agents_tree.guide_depth = 3
            agents_tree.guide_style = "dashed"

            stats_display = Static("", id="stats_display")
            stats_scroll = VerticalScroll(stats_display, id="stats_scroll")

            vulnerabilities_panel = VulnerabilitiesPanel(id="vulnerabilities_panel")

            network_graph = NetworkGraph(id="network_graph")
            network_graph.scan_mode = str(self.scan_config.get("scan_mode", "general") or "general")
            network_graph.task_session_id = ""
            network_graph.border_title = Text(
               " SUBSYSTEM MENU ",
               style="bold #00FCC1"
            )

            network_graph.styles.border_title_align = "center"

            sidebar = Vertical(
              agents_tree,
              vulnerabilities_panel,
              network_graph,
              stats_scroll,
              id="sidebar"
            )

            content_container.mount(chat_area_container)
            content_container.mount(sidebar)

            chat_area_container.mount(chat_history)
            chat_area_container.mount(agent_status_display)
            chat_area_container.mount(chat_input_container)

            self.call_after_refresh(self._set_chat_input_visible, False)
            self.call_after_refresh(self._update_voice_volume_indicator)
            self.call_after_refresh(self._start_boot_sequence)

    def on_key(self, event: events.Key) -> None:
        if self._boot_waiting_for_enter and event.key == "enter":
            if self._boot_enter_event and not self._boot_enter_event.is_set():
                self._boot_enter_event.set()
            event.prevent_default()
            return

        is_plus = event.character == "+" or event.key == "plus"
        is_minus = event.character == "-" or event.key == "minus"
        if is_plus or is_minus:
            if self.voice_speech_enabled and self._voice_engine and self._voice_engine.is_available():
                if is_plus:
                    self._voice_engine.increase_volume()
                else:
                    self._voice_engine.decrease_volume()
                self._update_voice_volume_indicator()
                event.prevent_default()
                return

        if event.key not in ("up", "down"):
            return
        try:
            chat_history = self.query_one("#chat_history", VerticalScroll)
            chat_input = self.query_one("#chat_input", ChatTextArea)
        except (ValueError, Exception):
            return
        if self.focused not in (chat_history, chat_input):
            return
        if event.key == "up":
            chat_history.scroll_up(animate=False)
        else:
            chat_history.scroll_down(animate=False)
        event.prevent_default()

    def _focus_chat_input(self) -> None:
        if len(self.screen_stack) > 1 or self.show_splash:
            return

        if not self.is_mounted:
            return

        try:
            chat_input = self.query_one("#chat_input", ChatTextArea)
            chat_input.show_vertical_scrollbar = False
            chat_input.show_horizontal_scrollbar = False
            chat_input.focus()
        except (ValueError, Exception):
            self.call_after_refresh(self._focus_chat_input)

    def _focus_agents_tree(self) -> None:
        if len(self.screen_stack) > 1 or self.show_splash:
            return

        if not self.is_mounted:
            return

        try:
            agents_tree = self.query_one("#agents_tree", Tree)
            agents_tree.focus()

            if agents_tree.root.children:
                first_node = agents_tree.root.children[0]
                agents_tree.select_node(first_node)
        except (ValueError, Exception):
            self.call_after_refresh(self._focus_agents_tree)

    def _enable_ctrl_q_capture_on_posix(self) -> None:
        if os.name != "posix":
            return

        if not sys.stdin.isatty():
            return

        with contextlib.suppress(Exception):
            fd = sys.stdin.fileno()
            attrs = termios.tcgetattr(fd)

            attrs[0] &= ~getattr(termios, "IXON", 0)
            attrs[0] &= ~getattr(termios, "IXOFF", 0)

            termios.tcsetattr(fd, termios.TCSANOW, attrs)

    def on_mount(self) -> None:
        self._enable_ctrl_q_capture_on_posix()
        self.title = "sondra"

        self.set_timer(4.5, self._hide_splash_screen)

    def _hide_splash_screen(self) -> None:
        self.show_splash = False

    def _set_chat_input_visible(self, visible: bool) -> None:
        with contextlib.suppress(Exception):
            chat_input_container = self.query_one("#chat_input_container", Horizontal)
            chat_input_container.styles.display = "block" if visible else "none"

    def _focus_chat_terminal(self) -> None:
        with contextlib.suppress(Exception):
            chat_history = self.query_one("#chat_history", VerticalScroll)
            chat_history.focus()

    def _load_boot_module(self) -> Any | None:
        with contextlib.suppress(Exception):
            from sondra.boot import boot as boot_module

            return boot_module

        boot_path = Path(__file__).resolve().parents[1] / "boot" / "boot.py"
        if not boot_path.exists():
            return None

        try:
            spec = importlib.util.spec_from_file_location("sondra_boot", boot_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
        except Exception:
            logger.exception("Failed to load boot module from %s", boot_path)
        return None

    def _start_boot_sequence(self) -> None:
        if self._boot_task and not self._boot_task.done():
            return

        self._boot_lines.clear()
        self._boot_in_progress = True
        self._boot_blocked = False
        self._set_chat_input_visible(False)
        self.call_after_refresh(self._focus_chat_terminal)
        self._boot_task = asyncio.create_task(self._run_boot_sequence())

    def _render_boot_lines(self) -> None:
        rendered = Text()
        for text_line, text_color in self._boot_lines:
            if isinstance(text_line, Text):
                rendered.append_text(text_line)
            else:
                rendered.append(str(text_line or ""), style=text_color)
            rendered.append("\n")

        with contextlib.suppress(Exception):
            chat_display = self.query_one("#chat_display", Static)
            if self._is_widget_safe(chat_display):
                self._safe_widget_operation(chat_display.update, rendered)

        with contextlib.suppress(Exception):
            chat_history = self.query_one("#chat_history", VerticalScroll)
            if self._is_widget_safe(chat_history):
                self._safe_widget_operation(chat_history.scroll_end, animate=False)

    async def _boot_emit_line(self, line: Any, color: str = "white") -> None:
        rendered_line = line if isinstance(line, Text) else str(line or "")
        self._boot_lines.append((rendered_line, str(color or "white")))
        self._render_boot_lines()

    async def _boot_update_line(self, line: Any, color: str = "white", final: bool = False) -> None:
        _ = final
        line_text = line if isinstance(line, Text) else str(line or "")
        line_plain = line_text.plain if isinstance(line_text, Text) else str(line_text)
        line_color = str(color or "white")
        if self._boot_lines:
            prev = self._boot_lines[-1][0]
            prev_plain = prev.plain if isinstance(prev, Text) else str(prev)
        else:
            prev_plain = ""

        if prev_plain.startswith("Memory Testing :"):
            self._boot_lines[-1] = (line_text, line_color)
        else:
            self._boot_lines.append((line_text, line_color))
        self._render_boot_lines()

    async def _boot_wait_for_enter(self) -> None:
        self._boot_enter_event = asyncio.Event()
        self._boot_waiting_for_enter = True
        self.call_after_refresh(self._focus_chat_terminal)
        await self._boot_enter_event.wait()
        self._boot_waiting_for_enter = False

    async def _run_boot_sequence(self) -> None:
        mode = str(self.scan_config.get("scan_mode", "general") or "general").strip().lower()
        boot_module = self._load_boot_module()

        if boot_module is None:
            await self._boot_emit_line("SONDRA BOOT MODULE NOT FOUND. CONTINUING...", "#ff1744")
            self._boot_in_progress = False
            self._set_chat_input_visible(True)
            self._start_scan_thread()
            self.set_interval(0.35, self._update_ui_from_tracer)
            self.call_after_refresh(self._focus_chat_input)
            return

        try:
            result = await boot_module.run_boot_sequence(
                mode=mode,
                emit_line=self._boot_emit_line,
                update_line=self._boot_update_line,
                wait_for_enter_fn=self._boot_wait_for_enter,
            )
        except Exception:
            logger.exception("Boot sequence failed unexpectedly")
            await self._boot_emit_line("SONDRA BOOT ERROR. CONTINUING SAFE STARTUP...", "#ff1744")
            self._boot_in_progress = False
            self._set_chat_input_visible(True)
            self._start_scan_thread()
            self.set_interval(0.35, self._update_ui_from_tracer)
            self.call_after_refresh(self._focus_chat_input)
            return

        self._boot_in_progress = False
        continued = bool(getattr(result, "continued", True))

        if not continued:
            self._boot_blocked = True
            self._set_chat_input_visible(False)
            return

        self._set_chat_input_visible(True)
        self._start_scan_thread()
        self.set_interval(0.35, self._update_ui_from_tracer)
        self.call_after_refresh(self._focus_chat_input)

    def _update_ui_from_tracer(self) -> None:
        if self.show_splash:
            return

        if len(self.screen_stack) > 1:
            return

        if not self.is_mounted:
            return

        try:
            chat_history = self.query_one("#chat_history", VerticalScroll)
            agents_tree = self.query_one("#agents_tree", Tree)

            if not self._is_widget_safe(chat_history) or not self._is_widget_safe(agents_tree):
                return
        except (ValueError, Exception):
            return

        agent_updates = False
        for agent_id, agent_data in list(self.tracer.agents.items()):
            if agent_id not in self._displayed_agents:
                self._add_agent_node(agent_data)
                self._displayed_agents.add(agent_id)
                agent_updates = True
            elif self._update_agent_node(agent_id, agent_data):
                agent_updates = True

        if agent_updates:
            self._expand_new_agent_nodes()

        self._update_chat_view()
        self._speak_new_assistant_messages()

        self._update_agent_status_display()
        self._update_indicator_execution_button()

        self._update_stats_display()

        self._update_vulnerabilities_panel()

    def _update_agent_node(self, agent_id: str, agent_data: dict[str, Any]) -> bool:
        if agent_id not in self.agent_nodes:
            return False

        try:
            agent_node = self.agent_nodes[agent_id]
            agent_name_raw = agent_data.get("name", "Agent")
            status = agent_data.get("status", "running")

            status_indicators = {
                "running": "⚪",
                "waiting": "🛑",
                "completed": "🟢",
                "failed": "⚠️ ",
                "stopped": "◼",
                "stopping": "⟳",
                "llm_failed": "⚠️ ",
            }

            status_icon = status_indicators.get(status, "○")
            vuln_count = self._agent_vulnerability_count(agent_id)
            vuln_indicator = f" ({vuln_count})" if vuln_count > 0 else ""
            agent_name = f"{status_icon} {agent_name_raw}{vuln_indicator}"

            if agent_node.label != agent_name:
                agent_node.set_label(agent_name)
                return True

        except (KeyError, AttributeError, ValueError) as e:
            import logging

            logging.warning(f"Failed to update agent node label: {e}")

        return False

    def _get_chat_content(
        self,
    ) -> tuple[Any, str | None]:
        if not self.selected_agent_id:
            return self._get_chat_placeholder_content(
                "Select an agent from the tree to see its activity.", "placeholder-no-agent"
            )

        events = self._gather_agent_events(self.selected_agent_id)
        streaming = self.tracer.get_streaming_content(self.selected_agent_id)

        if not events and not streaming:
            return self._get_chat_placeholder_content(
                "Starting agent...", "placeholder-no-activity"
            )

        current_event_ids = [e["id"] for e in events]
        current_streaming_len = len(streaming) if streaming else 0
        last_streaming_len = self._last_streaming_len.get(self.selected_agent_id, 0)

        if (
            current_event_ids == self._displayed_events
            and current_streaming_len == last_streaming_len
        ):
            return None, None

        self._displayed_events = current_event_ids
        self._last_streaming_len[self.selected_agent_id] = current_streaming_len
        return self._get_rendered_events_content(events), "chat-content"

    def _update_chat_view(self) -> None:
        if len(self.screen_stack) > 1 or self.show_splash or not self.is_mounted:
            return

        try:
            chat_history = self.query_one("#chat_history", VerticalScroll)
        except (ValueError, Exception):
            return

        if not self._is_widget_safe(chat_history):
            return

        try:
            is_at_bottom = chat_history.scroll_y >= chat_history.max_scroll_y
        except (AttributeError, ValueError):
            is_at_bottom = True

        content, css_class = self._get_chat_content()
        if content is None:
            return

        chat_display = self.query_one("#chat_display", Static)
        self._safe_widget_operation(chat_display.update, content)
        chat_display.set_classes(css_class)

        if is_at_bottom:
            self.call_later(chat_history.scroll_end, animate=False)

    def _get_chat_placeholder_content(
        self, message: str, placeholder_class: str
    ) -> tuple[Text, str]:
        self._displayed_events = [placeholder_class]
        text = Text()
        text.append(message)
        return text, f"chat-placeholder {placeholder_class}"

    @staticmethod
    def _merge_renderables(renderables: list[Any]) -> Text:
        """Merge renderables into a single Text for mouse text selection support."""
        combined = Text()
        for i, item in enumerate(renderables):
            if i > 0:
                combined.append("\n")
            SondraTUIApp._append_renderable(combined, item)
        return combined

    @staticmethod
    def _append_renderable(combined: Text, item: Any) -> None:
        """Recursively append a renderable's text content to a combined Text."""
        if isinstance(item, Text):
            combined.append_text(item)
        elif isinstance(item, Group):
            for j, sub in enumerate(item.renderables):
                if j > 0:
                    combined.append("\n")
                SondraTUIApp._append_renderable(combined, sub)
        else:
            inner = getattr(item, "renderable", None)
            if inner is not None:
                SondraTUIApp._append_renderable(combined, inner)
            else:
                combined.append(str(item))

    def _get_rendered_events_content(self, events: list[dict[str, Any]]) -> Any:
        renderables: list[Any] = []

        if not events:
            return Text()

        for event in events:
            content: Any = None

            if event["type"] == "chat":
                content = self._render_chat_content(event["data"])
            elif event["type"] == "tool":
                content = self._render_tool_content_simple(event["data"])

            if content:
                if renderables:
                    renderables.append(Text(""))
                renderables.append(content)

        if self.selected_agent_id:
            streaming = self.tracer.get_streaming_content(self.selected_agent_id)
            if streaming:
                streaming_text = self._render_streaming_content(streaming)
                if streaming_text:
                    if renderables:
                        renderables.append(Text(""))
                    renderables.append(streaming_text)

        if not renderables:
            return Text()

        if len(renderables) == 1 and isinstance(renderables[0], Text):
            return renderables[0]

        return self._merge_renderables(renderables)

    def _render_streaming_content(self, content: str, agent_id: str | None = None) -> Any:
        cache_key = agent_id or self.selected_agent_id or ""
        content_len = len(content)

        if cache_key in self._streaming_render_cache:
            cached_len, cached_output = self._streaming_render_cache[cache_key]
            if cached_len == content_len:
                return cached_output

        renderables: list[Any] = []
        thinking_renderable = self._render_streaming_thinking(content)
        if thinking_renderable is not None:
            renderables.append(thinking_renderable)
            content = self._strip_streaming_thinking(content)
        segments = parse_streaming_content(content)

        for segment in segments:
            if segment.type == "text":
                text_content = AgentMessageRenderer.render_simple(segment.content)
                if renderables:
                    renderables.append(Text(""))
                renderables.append(text_content)

            elif segment.type == "tool":
                tool_renderable = self._render_streaming_tool(
                    segment.tool_name or "unknown",
                    segment.args or {},
                    segment.is_complete,
                )
                if renderables:
                    renderables.append(Text(""))
                renderables.append(tool_renderable)

        if not renderables:
            result = Text()
        elif len(renderables) == 1 and isinstance(renderables[0], Text):
            result = renderables[0]
        else:
            result = self._merge_renderables(renderables)

        self._streaming_render_cache[cache_key] = (content_len, result)
        return result

    def _render_streaming_thinking(self, content: str) -> Text | None:
        raw = str(content or "")
        if not raw.startswith("[THINKING]"):
            return None
        thinking_body = raw[len("[THINKING]") :].strip()
        if "[END THINKING]" in thinking_body:
            thinking_body = thinking_body.split("[END THINKING]", 1)[0].strip()
        text = Text()
        text.append("🧠 Thinking ...", style="#8a8a8a")
        if thinking_body and thinking_body != "🧠 Thinking ...":
            text.append("\n")
            text.append(thinking_body, style="#6b7280")
        return text

    def _strip_streaming_thinking(self, content: str) -> str:
        raw = str(content or "")
        if not raw.startswith("[THINKING]"):
            return raw
        thinking_body = raw[len("[THINKING]") :]
        if "[END THINKING]" in thinking_body:
            return thinking_body.split("[END THINKING]", 1)[1].strip()
        return ""

    def _render_streaming_tool(
        self, tool_name: str, args: dict[str, str], is_complete: bool
    ) -> Any:
        tool_data = {
            "tool_name": tool_name,
            "args": args,
            "status": "completed" if is_complete else "running",
            "result": None,
        }

        renderer = get_tool_renderer(tool_name)
        if renderer:
            widget = renderer.render(tool_data)
            return widget.renderable

        return self._render_default_streaming_tool(tool_name, args, is_complete)

    def _render_default_streaming_tool(
        self, tool_name: str, args: dict[str, str], is_complete: bool
    ) -> Text:
        normalized_tool = str(tool_name or "").strip().lower()
        if normalized_tool in {"memory_search", "memory_get"}:
            return Text()
        text = Text()

        if is_complete:
            text.append("✓ ", style="green")
        else:
            text.append("● ", style="yellow")

        text.append("Using tool ", style="dim")
        text.append(tool_name, style="bold blue")

        if args:
            for key, value in list(args.items())[:3]:
                text.append("\n  ")
                text.append(key, style="dim")
                text.append(": ")
                display_value = value if len(value) <= 100 else value[:97] + "..."
                text.append(display_value, style="italic" if not is_complete else None)

        return text

    def _get_status_display_content(
        self, agent_id: str, agent_data: dict[str, Any]
    ) -> tuple[Text | None, Text, bool]:
        status = agent_data.get("status", "running")

        def keymap_styled(keys: list[tuple[str, str]]) -> Text:
            t = Text()
            for i, (key, action) in enumerate(keys):
                if i > 0:
                    t.append(" · ", style="dim")
                t.append(key, style="white")
                t.append(" ", style="dim")
                t.append(action, style="dim")
            return t

        simple_statuses: dict[str, tuple[str, str]] = {
            "stopping": ("Agent stopping...", ""),
            "stopped": ("Agent stopped", ""),
            "completed": ("Agent completed", ""),
        }

        if status in simple_statuses:
            msg, _ = simple_statuses[status]
            text = Text()
            text.append(msg)
            return (text, Text(), False)

        if status == "llm_failed":
            error_msg = agent_data.get("error_message", "")
            text = Text()
            if error_msg:
                text.append(error_msg, style="red")
            else:
                text.append("LLM request failed", style="red")
            self._stop_dot_animation()
            keymap = Text()
            keymap.append("Send message to retry", style="dim")
            return (text, keymap, False)

        if status == "waiting":
            waiting_text = Text()
            waiting_text.append("• ", style="#00FCC1")
            waiting_text.append("Done", style="dim")
            keymap = Text()
            keymap.append("Send message to resume", style="dim")
            return (waiting_text, keymap, False)

        if status == "running":
            if self._agent_has_real_activity(agent_id):
                animated_text = Text()
                animated_text.append_text(self._get_sweep_animation(self._sweep_colors))
                animated_text.append("esc", style="white")
                animated_text.append(" ", style="dim")
                animated_text.append("stop", style="dim")
                return (animated_text, Text(), True)
            animated_text = self._get_animated_verb_text(agent_id, "Initializing")
            return (animated_text, Text(), True)

        return (None, Text(), False)

    def _update_agent_status_display(self) -> None:
        try:
            status_display = self.query_one("#agent_status_display", Horizontal)
            status_text = self.query_one("#status_text", Static)
            keymap_indicator = self.query_one("#keymap_indicator", Static)
            command_hint_text = self.query_one("#command_hint_text", Static)
        except (ValueError, Exception):
            return

        widgets = [status_display, status_text, keymap_indicator, command_hint_text]
        if not all(self._is_widget_safe(w) for w in widgets):
            return

        self._update_voice_volume_indicator()

        if not self.selected_agent_id:
            self._safe_widget_operation(status_display.add_class, "hidden")
            self._safe_widget_operation(command_hint_text.update, self._build_command_hint_text())
            return

        try:
            agent_data = self.tracer.agents[self.selected_agent_id]
            status = str(agent_data.get("status", "running"))
            if status == "stopped":
                self._safe_widget_operation(command_hint_text.update, self._build_stopped_hint_text())
                self._safe_widget_operation(status_display.add_class, "hidden")
                return
            self._safe_widget_operation(command_hint_text.update, self._build_command_hint_text())
            content, keymap, should_animate = self._get_status_display_content(
                self.selected_agent_id, agent_data
            )

            if not content:
                self._safe_widget_operation(status_display.add_class, "hidden")
                return

            self._safe_widget_operation(status_text.update, content)
            self._safe_widget_operation(keymap_indicator.update, keymap)
            self._safe_widget_operation(status_display.remove_class, "hidden")

            if should_animate:
                self._start_dot_animation()

        except (KeyError, Exception):
            self._safe_widget_operation(status_display.add_class, "hidden")

    def _update_stats_display(self) -> None:
        try:
            stats_display = self.query_one("#stats_display", Static)
        except (ValueError, Exception):
            return

        if not self._is_widget_safe(stats_display):
            return

        if self.screen.selections:
            return

        stats_content = Text()

        stats_text = build_tui_stats_text(self.tracer, self.agent_config)
        if stats_text:
            stats_content.append(stats_text)

        stats_content.append(f"\n{get_display_version_label()}", style="white")

        self._safe_widget_operation(stats_display.update, stats_content)

    def _update_vulnerabilities_panel(self) -> None:
        """Update the vulnerabilities panel with current vulnerability data."""
        try:
            vuln_panel = self.query_one("#vulnerabilities_panel", VulnerabilitiesPanel)
        except (ValueError, Exception):
            return

        if not self._is_widget_safe(vuln_panel):
            return

        vulnerabilities = self.tracer.vulnerability_reports

        if not vulnerabilities:
            self._safe_widget_operation(vuln_panel.add_class, "hidden")
            return

        enriched_vulns = []
        for vuln in vulnerabilities:
            enriched = dict(vuln)
            report_id = vuln.get("id", "")
            agent_name = self._get_agent_name_for_vulnerability(report_id)
            if agent_name:
                enriched["agent_name"] = agent_name
            enriched_vulns.append(enriched)

        self._safe_widget_operation(vuln_panel.remove_class, "hidden")
        vuln_panel.update_vulnerabilities(enriched_vulns)

    def _get_agent_name_for_vulnerability(self, report_id: str) -> str | None:
        """Find the agent name that created a vulnerability report."""
        for _exec_id, tool_data in list(self.tracer.tool_executions.items()):
            if tool_data.get("tool_name") == "create_vulnerability_report":
                result = tool_data.get("result", {})
                if isinstance(result, dict) and result.get("report_id") == report_id:
                    agent_id = tool_data.get("agent_id")
                    if agent_id and agent_id in self.tracer.agents:
                        name: str = self.tracer.agents[agent_id].get("name", "Unknown Agent")
                        return name
        return None

    def _get_sweep_animation(self, color_palette: list[str]) -> Text:
        text = Text()
        num_squares = self._sweep_num_squares
        num_colors = len(color_palette)

        offset = num_colors - 1
        max_pos = (num_squares - 1) + offset
        total_range = max_pos + offset
        cycle_length = total_range * 2
        frame_in_cycle = self._spinner_frame_index % cycle_length

        wave_pos = total_range - abs(total_range - frame_in_cycle)
        sweep_pos = wave_pos - offset

        dark_color = (0x00, 0x2A, 0x20)
        bright_color = (0x00, 0xFC, 0xC1)

        for i in range(num_squares):
            dist = abs(i - sweep_pos)
            color_idx = max(0, num_colors - 1 - dist)
            t = color_idx / max(1, num_colors - 1)
            r = int(dark_color[0] + (bright_color[0] - dark_color[0]) * t)
            g = int(dark_color[1] + (bright_color[1] - dark_color[1]) * t)
            b = int(dark_color[2] + (bright_color[2] - dark_color[2]) * t)
            shade = f"#{r:02X}{g:02X}{b:02X}"
            text.append("▪", style=Style(color=shade))

        text.append(" ")
        return text

    def _get_animated_verb_text(self, agent_id: str, verb: str) -> Text:  # noqa: ARG002
        text = Text()
        sweep = self._get_sweep_animation(self._sweep_colors)
        text.append_text(sweep)
        parts = verb.split(" ", 1)
        text.append(parts[0], style="white")
        if len(parts) > 1:
            text.append(" ", style="dim")
            text.append(parts[1], style="dim")
        return text

    def _start_dot_animation(self) -> None:
        if self._dot_animation_timer is None:
            self._dot_animation_timer = self.set_interval(0.06, self._animate_dots)

    def _stop_dot_animation(self) -> None:
        if self._dot_animation_timer is not None:
            self._dot_animation_timer.stop()
            self._dot_animation_timer = None

    def _animate_dots(self) -> None:
        has_active_agents = False

        if self.selected_agent_id and self.selected_agent_id in self.tracer.agents:
            agent_data = self.tracer.agents[self.selected_agent_id]
            status = agent_data.get("status", "running")
            if status in ["running", "waiting"]:
                has_active_agents = True
                num_colors = len(self._sweep_colors)
                offset = num_colors - 1
                max_pos = (self._sweep_num_squares - 1) + offset
                total_range = max_pos + offset
                cycle_length = total_range * 2
                self._spinner_frame_index = (self._spinner_frame_index + 1) % cycle_length
                self._update_agent_status_display()

        if not has_active_agents:
            has_active_agents = any(
                agent_data.get("status", "running") in ["running", "waiting"]
                for agent_data in self.tracer.agents.values()
            )

        if not has_active_agents:
            self._stop_dot_animation()
            self._spinner_frame_index = 0

    def _agent_has_real_activity(self, agent_id: str) -> bool:
        initial_tools = {"scan_start_info", "subagent_start_info"}

        for _exec_id, tool_data in list(self.tracer.tool_executions.items()):
            if tool_data.get("agent_id") == agent_id:
                tool_name = tool_data.get("tool_name", "")
                if tool_name not in initial_tools:
                    return True

        streaming = self.tracer.get_streaming_content(agent_id)
        return bool(streaming and streaming.strip())

    def _agent_vulnerability_count(self, agent_id: str) -> int:
        count = 0
        for _exec_id, tool_data in list(self.tracer.tool_executions.items()):
            if tool_data.get("agent_id") == agent_id:
                tool_name = tool_data.get("tool_name", "")
                if tool_name == "create_vulnerability_report":
                    status = tool_data.get("status", "")
                    if status == "completed":
                        result = tool_data.get("result", {})
                        if isinstance(result, dict) and result.get("success"):
                            count += 1
        return count

    def _gather_agent_events(self, agent_id: str) -> list[dict[str, Any]]:
        chat_events = [
            {
                "type": "chat",
                "timestamp": msg["timestamp"],
                "id": f"chat_{msg['message_id']}",
                "data": msg,
            }
            for msg in self.tracer.chat_messages
            if msg.get("agent_id") == agent_id
        ]

        tool_events = [
            {
                "type": "tool",
                "timestamp": tool_data["timestamp"],
                "id": f"tool_{exec_id}",
                "data": tool_data,
            }
            for exec_id, tool_data in list(self.tracer.tool_executions.items())
            if tool_data.get("agent_id") == agent_id
        ]

        events = chat_events + tool_events
        events.sort(key=lambda e: (e["timestamp"], e["id"]))
        return events

    def watch_selected_agent_id(self, _agent_id: str | None) -> None:
        if len(self.screen_stack) > 1 or self.show_splash:
            return

        if not self.is_mounted:
            return

        self._displayed_events.clear()
        self._streaming_render_cache.clear()
        self._last_streaming_len.clear()

        self.call_later(self._update_chat_view)
        self._update_agent_status_display()
        self._update_indicator_execution_button()

    def _set_task_panel_session_id(self, session_id: str) -> None:
        if not self.is_mounted:
            return
        with contextlib.suppress(Exception):
            network_graph = self.query_one("#network_graph", NetworkGraph)
            network_graph.task_session_id = str(session_id or "").strip()
            network_graph.update_ui()

    def _start_scan_thread(self) -> None:
        def scan_target() -> None:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                try:
                    agent = SondraAgent(self.agent_config)
                    if str(self.scan_config.get("scan_mode", "general") or "general").strip().lower() == "general":
                        session_id = ""
                        memory_session_id = getattr(agent, "_memory_session_id", None)
                        if callable(memory_session_id):
                            session_id = str(memory_session_id() or "").strip()
                        if session_id:
                            self.call_from_thread(self._set_task_panel_session_id, session_id)

                    if not self._scan_stop_event.is_set():
                        loop.run_until_complete(agent.execute_scan(self.scan_config))

                except (KeyboardInterrupt, asyncio.CancelledError):
                    logging.info("Scan interrupted by user")
                except (ConnectionError, TimeoutError):
                    logging.exception("Network error during scan")
                except RuntimeError:
                    logging.exception("Runtime error during scan")
                except Exception:
                    logging.exception("Unexpected error during scan")
                finally:
                    loop.close()
                    self._scan_completed.set()

            except Exception:
                logging.exception("Error setting up scan thread")
                self._scan_completed.set()

        self._scan_thread = threading.Thread(target=scan_target, daemon=True)
        self._scan_thread.start()

    def _add_agent_node(self, agent_data: dict[str, Any]) -> None:
        if len(self.screen_stack) > 1 or self.show_splash:
            return

        if not self.is_mounted:
            return

        agent_id = agent_data["id"]
        parent_id = agent_data.get("parent_id")
        status = agent_data.get("status", "running")

        try:
            agents_tree = self.query_one("#agents_tree", Tree)
        except (ValueError, Exception):
            return

        agent_name_raw = agent_data.get("name", "Agent")

        status_indicators = {
            "running": "⚪",
            "waiting": "🛑",
            "completed": "🟢",
            "failed": "⚠️ ",
            "stopped": "◼",
            "stopping": "⟳",
            "llm_failed": "⚠️ ",
        }

        status_icon = status_indicators.get(status, "○")
        vuln_count = self._agent_vulnerability_count(agent_id)
        vuln_indicator = f" ({vuln_count})" if vuln_count > 0 else ""
        agent_name = f"{status_icon} {agent_name_raw}{vuln_indicator}"

        try:
            if parent_id and parent_id in self.agent_nodes:
                parent_node = self.agent_nodes[parent_id]
                agent_node = parent_node.add(
                    agent_name,
                    data={"agent_id": agent_id},
                )
                parent_node.allow_expand = True
            else:
                agent_node = agents_tree.root.add(
                    agent_name,
                    data={"agent_id": agent_id},
                )

            agent_node.allow_expand = False
            agent_node.expand()
            self.agent_nodes[agent_id] = agent_node

            if len(self.agent_nodes) == 1:
                agents_tree.select_node(agent_node)
                self.selected_agent_id = agent_id

            self._reorganize_orphaned_agents(agent_id)
        except (AttributeError, ValueError, RuntimeError) as e:
            import logging

            logging.warning(f"Failed to add agent node {agent_id}: {e}")

    def _expand_new_agent_nodes(self) -> None:
        if len(self.screen_stack) > 1 or self.show_splash:
            return

        if not self.is_mounted:
            return

    def _expand_all_agent_nodes(self) -> None:
        if len(self.screen_stack) > 1 or self.show_splash:
            return

        if not self.is_mounted:
            return

        try:
            agents_tree = self.query_one("#agents_tree", Tree)
            self._expand_node_recursively(agents_tree.root)
        except (ValueError, Exception):
            logging.debug("Tree not ready for expanding nodes")

    def _expand_node_recursively(self, node: TreeNode) -> None:
        if not node.is_expanded:
            node.expand()
        for child in node.children:
            self._expand_node_recursively(child)

    def _copy_node_under(self, node_to_copy: TreeNode, new_parent: TreeNode) -> None:
        agent_id = node_to_copy.data["agent_id"]
        agent_data = self.tracer.agents.get(agent_id, {})
        agent_name_raw = agent_data.get("name", "Agent")
        status = agent_data.get("status", "running")

        status_indicators = {
            "running": "⚪",
            "waiting": "🛑",
            "completed": "🟢",
            "failed": "⚠️ ",
            "stopped": "◼",
            "stopping": "⟳",
            "llm_failed": "⚠️ ",
        }

        status_icon = status_indicators.get(status, "○")
        vuln_count = self._agent_vulnerability_count(agent_id)
        vuln_indicator = f" ({vuln_count})" if vuln_count > 0 else ""
        agent_name = f"{status_icon} {agent_name_raw}{vuln_indicator}"

        new_node = new_parent.add(
            agent_name,
            data=node_to_copy.data,
        )
        new_node.allow_expand = node_to_copy.allow_expand

        self.agent_nodes[agent_id] = new_node

        for child in node_to_copy.children:
            self._copy_node_under(child, new_node)

        if node_to_copy.is_expanded:
            new_node.expand()

    def _reorganize_orphaned_agents(self, new_parent_id: str) -> None:
        agents_to_move = []

        for agent_id, agent_data in list(self.tracer.agents.items()):
            if (
                agent_data.get("parent_id") == new_parent_id
                and agent_id in self.agent_nodes
                and agent_id != new_parent_id
            ):
                agents_to_move.append(agent_id)

        if not agents_to_move:
            return

        parent_node = self.agent_nodes[new_parent_id]

        for child_agent_id in agents_to_move:
            if child_agent_id in self.agent_nodes:
                old_node = self.agent_nodes[child_agent_id]

                if old_node.parent is parent_node:
                    continue

                self._copy_node_under(old_node, parent_node)

                old_node.remove()

        parent_node.allow_expand = True
        parent_node.expand()

    def _render_chat_content(self, msg_data: dict[str, Any]) -> Any:
        role = msg_data.get("role")
        content = msg_data.get("content", "")
        metadata = msg_data.get("metadata", {})

        if not content:
            return None

        if role == "user":
            return UserMessageRenderer.render_simple(content)

        if metadata.get("interrupted"):
            streaming_result = self._render_streaming_content(content)
            interrupted_text = Text()
            interrupted_text.append("\n")
            interrupted_text.append("⚠ ", style="yellow")
            interrupted_text.append("Interrupted by user", style="yellow dim")
            return self._merge_renderables([streaming_result, interrupted_text])

        return AgentMessageRenderer.render_simple(content)

    def _render_tool_content_simple(self, tool_data: dict[str, Any]) -> Any:
        tool_name = tool_data.get("tool_name", "Unknown Tool")
        normalized_tool = str(tool_name or "").strip().lower()
        if normalized_tool in {"memory_search", "memory_get"}:
            return Text()
        args = tool_data.get("args", {})
        status = tool_data.get("status", "unknown")
        result = tool_data.get("result")

        renderer = get_tool_renderer(tool_name)

        if renderer:
            widget = renderer.render(tool_data)
            return widget.renderable

        text = Text()

        if tool_name in ("llm_error_details", "sandbox_error_details"):
            return self._render_error_details(text, tool_name, args)

        text.append("→ Using tool ")
        text.append(tool_name, style="bold blue")

        status_styles = {
            "running": ("▶︎", "yellow"),
            "completed": ("⚡︎", "green"),
            "failed": ("⚠︎︎", "red"),
            "error": ("⚠︎︎", "red"),
        }
        icon, style = status_styles.get(status, ("○", "dim"))
        text.append(" ")
        text.append(icon, style=style)

        if args:
            for k, v in list(args.items())[:5]:
                str_v = str(v)
                if len(str_v) > 500:
                    str_v = str_v[:497] + "..."
                text.append("\n  ")
                text.append(k, style="dim")
                text.append(": ")
                text.append(str_v)

        if status in ["completed", "failed", "error"] and result:
            result_str = str(result)
            if len(result_str) > 1000:
                result_str = result_str[:997] + "..."
            text.append("\n")
            text.append("Result: ", style="bold")
            text.append(result_str)

        return text

    def _render_error_details(self, text: Any, tool_name: str, args: dict[str, Any]) -> Any:
        if tool_name == "llm_error_details":
            text.append("⚠︎︎ LLM Request Failed", style="red")
        else:
            text.append("⚠︎︎ Sandbox Initialization Failed", style="red")
            if args.get("error"):
                text.append(f"\n{args['error']}", style="bold red")
        if args.get("details"):
            details = str(args["details"])
            if len(details) > 1000:
                details = details[:997] + "..."
            text.append("\nDetails: ", style="dim")
            text.append(details)
        return text

    @on(Tree.NodeHighlighted)  # type: ignore[misc]
    def handle_tree_highlight(self, event: Tree.NodeHighlighted) -> None:
        if len(self.screen_stack) > 1 or self.show_splash:
            return

        if not self.is_mounted:
            return

        node = event.node

        try:
            agents_tree = self.query_one("#agents_tree", Tree)
        except (ValueError, Exception):
            return

        if self.focused == agents_tree and node.data:
            agent_id = node.data.get("agent_id")
            if agent_id:
                self.selected_agent_id = agent_id

    @on(Tree.NodeSelected)  # type: ignore[misc]
    def handle_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        if len(self.screen_stack) > 1 or self.show_splash:
            return

        if not self.is_mounted:
            return

        node = event.node

        if node.allow_expand:
            if node.is_expanded:
                node.collapse()
            else:
                node.expand()

    def _send_user_message(self, message: str) -> None:
        if self._boot_in_progress or self._boot_waiting_for_enter or self._boot_blocked:
            return

        if not self.selected_agent_id:
            return
        if self._voice_engine and self._voice_engine.is_available():
            self._voice_engine.update_language_from_text(message)

        if self.tracer:
            streaming_content = self.tracer.get_streaming_content(self.selected_agent_id)
            if streaming_content and streaming_content.strip():
                self.tracer.clear_streaming_content(self.selected_agent_id)
                self.tracer.interrupted_content[self.selected_agent_id] = streaming_content
                self.tracer.log_chat_message(
                    content=streaming_content,
                    role="assistant",
                    agent_id=self.selected_agent_id,
                    metadata={"interrupted": True},
                )

        try:
            _agent_instances = getattr(self._get_active_graph_actions_module(), "_agent_instances", {})

            if self.selected_agent_id in _agent_instances:
                agent_instance = _agent_instances[self.selected_agent_id]
                if hasattr(agent_instance, "cancel_current_execution"):
                    agent_instance.cancel_current_execution()
        except (ImportError, AttributeError, KeyError):
            pass

        if self.tracer:
            self.tracer.log_chat_message(
                content=message,
                role="user",
                agent_id=self.selected_agent_id,
            )

        try:
            send_user_message_to_agent = getattr(
                self._get_active_graph_actions_module(),
                "send_user_message_to_agent",
            )

            send_user_message_to_agent(self.selected_agent_id, message)

        except (ImportError, AttributeError) as e:
            import logging

            logging.warning(f"Failed to send message to agent {self.selected_agent_id}: {e}")

        self._displayed_events.clear()
        self._update_chat_view()

        self.call_after_refresh(self._focus_chat_input)

    def _get_agent_name(self, agent_id: str) -> str:
        try:
            if self.tracer and agent_id in self.tracer.agents:
                agent_name = self.tracer.agents[agent_id].get("name")
                if isinstance(agent_name, str):
                    return agent_name
        except (KeyError, AttributeError) as e:
            logging.warning(f"Could not retrieve agent name for {agent_id}: {e}")
        return "Unknown Agent"

    def _update_indicator_execution_button(self) -> None:
        try:
            network_graph = self.query_one("#network_graph", NetworkGraph)
        except (ValueError, Exception):
            return

        if not self.selected_agent_id:
            network_graph.set_execution_button_state(None)
            return

        agent_data = self.tracer.agents.get(self.selected_agent_id, {})
        status = str(agent_data.get("status", "running"))
        network_graph.set_execution_button_state(status)

    def toggle_selected_agent_from_indicator(self) -> None:
        if not self.selected_agent_id:
            self.notify("Select an agent first", timeout=2)
            return

        agent_data = self.tracer.agents.get(self.selected_agent_id, {})
        status = str(agent_data.get("status", "running"))

        if status in {"running", "stopping"}:
            self._stop_agent_immediately(self.selected_agent_id)
            return

        self._resume_agent_immediately(self.selected_agent_id)

    def _stop_agent_immediately(self, agent_id: str) -> None:
        try:
            stop_agent = getattr(self._get_active_graph_actions_module(), "stop_agent")

            result = stop_agent(agent_id)
            if result.get("success"):
                self.notify("Agent stopped", timeout=2)
            else:
                self.notify("Failed to stop agent", timeout=2)
        except Exception:
            logging.exception(f"Failed to stop agent {agent_id}")
            self.notify("Failed to stop agent", timeout=2)

    def _resume_agent_immediately(self, agent_id: str) -> None:
        try:
            send_user_message_to_agent = getattr(
                self._get_active_graph_actions_module(),
                "send_user_message_to_agent",
            )

            result = send_user_message_to_agent(
                agent_id,
                "Resume execution from where you paused and continue the current task.",
            )
            if result.get("success"):
                self.notify("Agent resumed", timeout=2)
            else:
                self.notify("Failed to resume agent", timeout=2)
        except Exception:
            logging.exception(f"Failed to resume agent {agent_id}")
            self.notify("Failed to resume agent", timeout=2)

    def action_toggle_help(self) -> None:
        if self.show_splash or not self.is_mounted:
            return

        try:
            self.query_one("#main_container")
        except (ValueError, Exception):
            return

        if isinstance(self.screen, HelpScreen):
            self.pop_screen()
            return

        if len(self.screen_stack) > 1:
            return

        self.push_screen(HelpScreen())

    def action_request_quit(self) -> None:
        play_choose_sound()

        if self.show_splash or not self.is_mounted:
            self.action_custom_quit()
            return

        if len(self.screen_stack) > 1:
            return

        try:
            self.query_one("#main_container")
        except (ValueError, Exception):
            self.action_custom_quit()
            return

        self.push_screen(QuitScreen())

    def action_stop_selected_agent(self) -> None:
        play_choose_sound()

        if self.show_splash or not self.is_mounted:
            return

        if len(self.screen_stack) > 1:
            self.pop_screen()
            return

        if not self.selected_agent_id:
            return

        agent_name, should_stop = self._validate_agent_for_stopping()
        if not should_stop:
            return

        try:
            self.query_one("#main_container")
        except (ValueError, Exception):
            return

        self.push_screen(StopAgentScreen(agent_name, self.selected_agent_id))

    def _validate_agent_for_stopping(self) -> tuple[str, bool]:
        agent_name = "Unknown Agent"

        try:
            if self.tracer and self.selected_agent_id in self.tracer.agents:
                agent_data = self.tracer.agents[self.selected_agent_id]
                agent_name = agent_data.get("name", "Unknown Agent")

                agent_status = agent_data.get("status", "running")
                if agent_status not in ["running"]:
                    return agent_name, False

                agent_events = self._gather_agent_events(self.selected_agent_id)
                if not agent_events:
                    return agent_name, False

                return agent_name, True

        except (KeyError, AttributeError, ValueError) as e:
            import logging

            logging.warning(f"Failed to gather agent events: {e}")

        return agent_name, False

    def action_confirm_stop_agent(self, agent_id: str) -> None:
        self.pop_screen()

        try:
            stop_agent = getattr(self._get_active_graph_actions_module(), "stop_agent")

            result = stop_agent(agent_id)

            import logging

            if result.get("success"):
                logging.info(f"Stop request sent to agent: {result.get('message', 'Unknown')}")
            else:
                logging.warning(f"Failed to stop agent: {result.get('error', 'Unknown error')}")

        except Exception:
            import logging

            logging.exception(f"Failed to stop agent {agent_id}")

    def action_custom_quit(self) -> None:
        if self._boot_task and not self._boot_task.done():
            self._boot_task.cancel()

        if self._scan_thread and self._scan_thread.is_alive():
            self._scan_stop_event.set()

            self._scan_thread.join(timeout=1.0)

        if self._voice_engine:
            self._voice_engine.close()

        self._persist_root_agent_last_emotion()
        self.tracer.cleanup()

        self.exit()

    def _speak_new_assistant_messages(self) -> None:
        if not self._voice_engine or not self._voice_engine.is_available():
            return

        for msg in self.tracer.chat_messages:
            message_id = msg.get("message_id")
            if not isinstance(message_id, int):
                continue

            if message_id in self._spoken_message_ids:
                continue
            self._spoken_message_ids.add(message_id)

            if msg.get("role") != "assistant":
                continue

            metadata = msg.get("metadata") or {}
            if isinstance(metadata, dict) and metadata.get("interrupted"):
                continue

            content = str(msg.get("content", "") or "").strip()
            if not content:
                continue

            self._voice_engine.enqueue(content)

    def _is_widget_safe(self, widget: Any) -> bool:
        try:
            _ = widget.screen
        except (AttributeError, ValueError, Exception):
            return False
        else:
            return bool(widget.is_mounted)

    def _safe_widget_operation(
        self, operation: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> bool:
        try:
            operation(*args, **kwargs)
        except (AttributeError, ValueError, Exception):
            return False
        else:
            return True

    def on_resize(self, event: events.Resize) -> None:
        if self.show_splash or not self.is_mounted:
            return

        try:
            sidebar = self.query_one("#sidebar", Vertical)
            chat_area = self.query_one("#chat_area_container", Vertical)
        except (ValueError, Exception):
            return

        if event.size.width < self.SIDEBAR_MIN_WIDTH:
            sidebar.add_class("-hidden")
            chat_area.add_class("-full-width")
        else:
            sidebar.remove_class("-hidden")
            chat_area.remove_class("-full-width")

    def on_mouse_up(self, _event: events.MouseUp) -> None:
        self.set_timer(0.05, self._auto_copy_selection)

    _ICON_PREFIXES: ClassVar[tuple[str, ...]] = (
        "🐞 ",
        "🌐 ",
        "📋 ",
        "🧠 ",
        "◆ ",
        "◇ ",
        "◈ ",
        "→ ",
        "○ ",
        "● ",
        "✓ ",
        "⚠︎︎ ",
        "⚠ ",
        "▍ ",
        "▍",
        "┃ ",
        "• ",
        ">_ ",
        "</> ",
        "<~> ",
        "[ ] ",
        "[~] ",
        "[•] ",
    )

    _DECORATIVE_LINES: ClassVar[frozenset[str]] = frozenset(
        {
            "● In progress...",
            "✓ Done",
            "⚠︎︎ Failed",
            "⚠︎︎ Error",
            "○ Unknown",
        }
    )

    @staticmethod
    def _clean_copied_text(text: str) -> str:
        lines = text.split("\n")
        cleaned: list[str] = []
        for line in lines:
            stripped = line.lstrip()
            if stripped in SondraTUIApp._DECORATIVE_LINES:
                continue
            if stripped and all(c == "─" for c in stripped):
                continue
            out = line
            for prefix in SondraTUIApp._ICON_PREFIXES:
                if stripped.startswith(prefix):
                    leading = line[: len(line) - len(line.lstrip())]
                    out = leading + stripped[len(prefix) :]
                    break
            cleaned.append(out)
        return "\n".join(cleaned)

    def _auto_copy_selection(self) -> None:
        copied = False

        try:
            if self.screen.selections:
                selected = self.screen.get_selected_text()
                self.screen.clear_selection()
                if selected and selected.strip():
                    cleaned = self._clean_copied_text(selected)
                    self.copy_to_clipboard(cleaned if cleaned.strip() else selected)
                    copied = True
        except Exception:  # noqa: BLE001
            logger.debug("Failed to copy screen selection", exc_info=True)

        if not copied:
            try:
                chat_input = self.query_one("#chat_input", ChatTextArea)
                selected = chat_input.selected_text
                if selected and selected.strip():
                    self.copy_to_clipboard(selected)
                    chat_input.move_cursor(chat_input.cursor_location)
                    copied = True
            except Exception:  # noqa: BLE001
                logger.debug("Failed to copy chat input selection", exc_info=True)

        if copied:
            self.notify("Copied to clipboard", timeout=2)


async def run_tui(args: argparse.Namespace) -> None:
    """Run sondra in interactive TUI mode with textual."""
    app = SondraTUIApp(args)
    await app.run_async()
