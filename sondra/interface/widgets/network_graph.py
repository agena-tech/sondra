from collections import deque
import time
import contextlib
import re
from datetime import datetime
from pathlib import Path
import json
import psutil

from rich.console import Group
from rich.ansi import AnsiDecoder
from rich.text import Text
from rich.panel import Panel
from rich.align import Align

from textual.widget import Widget
from textual.reactive import reactive
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.widgets import Button, Static

import plotext as plt

from sondra.telemetry.tracer import get_global_tracer
from sondra.memory import PersistentMemoryStore


class NetworkGraph(Widget):

    req = reactive(deque(maxlen=500))
    res = reactive(deque(maxlen=500))

    decoder = AnsiDecoder()

    last_req = 0
    last_res = 0
    last_time = time.time()

    last_net_sent = psutil.net_io_counters().bytes_sent
    last_net_recv = psutil.net_io_counters().bytes_recv

    offset = reactive(0)
    window = reactive(60)

    page = reactive(1)

    sent_bytes = reactive(0.0)
    recv_bytes = reactive(0.0)
    start_timestamp = reactive(time.time())
    req_samples = deque(maxlen=120)
    res_samples = deque(maxlen=120)
    rate_window_seconds = 6.0
    task_store = None
    task_session_id = ""
    scan_mode = "general"
    _last_emotion_panel_scores: dict[str, float] | None = None

    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("", id="graph_canvas"),
            Static("", id="graph_stats"),
            Static("", id="graph_spacer"),
            Horizontal(
                Button("◀", id="btn_back", compact=True),
                Button("■", id="btn_enter", compact=True),
                Button("▶", id="btn_next", compact=True),
                id="graph_buttons",
            ),
            id="graph_root",
        )

    def on_mount(self) -> None:
        self.start_timestamp = time.time()
        self._last_emotion_panel_scores = {}
        try:
            self.task_store = PersistentMemoryStore()
        except Exception:  # noqa: BLE001
            self.task_store = None
        self.set_interval(0.5, self.update_graph)
        self.update_ui()

    def on_button_pressed(self, event: Button.Pressed) -> None:

        if event.button.id == "btn_next":
            self.page = 1 if self.page >= 6 else self.page + 1

        elif event.button.id == "btn_back":
            self.page = 6 if self.page <= 1 else self.page - 1

        elif event.button.id == "btn_enter":
            app = self.app
            toggle_fn = getattr(app, "toggle_selected_agent_from_indicator", None)
            if callable(toggle_fn):
                toggle_fn()

        self.update_ui()

    def set_execution_button_state(self, status: str | None) -> None:
        try:
            button = self.query_one("#btn_enter", Button)
        except Exception:  # noqa: BLE001
            return

        button.remove_class("-stop")
        button.remove_class("-resume")
        button.remove_class("-idle")

        if status in {"running", "stopping"}:
            button.label = "■"
            button.add_class("-stop")
            return

        if status in {"waiting", "stopped", "llm_failed", "error", "failed"}:
            button.label = "▶"
            button.add_class("-resume")
            return

        button.label = "■"
        button.add_class("-idle")

    # --------------------------------------------------

    def update_graph(self) -> None:

        tracer = get_global_tracer()

        if not tracer:
            return

        now = time.time()
        elapsed = now - self.last_time

        if elapsed <= 0:
            return

        current_req = len(tracer.tool_executions)

        current_res = sum(
            1 for m in tracer.chat_messages
            if m.get("role") == "assistant"
        )

        req_rate = (current_req - self.last_req) / elapsed
        res_rate = (current_res - self.last_res) / elapsed

        self.last_req = current_req
        self.last_res = current_res

        net = psutil.net_io_counters()

        delta_sent = net.bytes_sent - self.last_net_sent
        delta_recv = net.bytes_recv - self.last_net_recv

        self.last_net_sent = net.bytes_sent
        self.last_net_recv = net.bytes_recv

        self.sent_bytes = delta_sent / elapsed
        self.recv_bytes = delta_recv / elapsed

        self.last_time = now
        self.req_samples.append((now, current_req))
        self.res_samples.append((now, current_res))

        smooth_req_rate = self._compute_window_rate(self.req_samples, now)
        smooth_res_rate = self._compute_window_rate(self.res_samples, now)

        self.req.append(smooth_req_rate if smooth_req_rate >= 0 else 0.0)
        self.res.append(smooth_res_rate if smooth_res_rate >= 0 else 0.0)

        max_offset = max(0, len(self.req) - self.window)

        if self.offset >= max_offset - 5:
            self.offset = max_offset

        self.update_ui()

    def _compute_window_rate(self, samples: deque, now: float) -> float:
        if len(samples) < 2:
            return 0.0

        oldest_idx = 0
        for i, (ts, _count) in enumerate(samples):
            if now - ts <= self.rate_window_seconds:
                oldest_idx = i
                break
        old_ts, old_count = samples[oldest_idx]
        new_ts, new_count = samples[-1]

        delta_t = new_ts - old_ts
        if delta_t <= 0:
            return 0.0

        delta_c = new_count - old_count
        if delta_c < 0:
            return 0.0

        return delta_c / delta_t

    # --------------------------------------------------
    # GRAPH PANEL
    # --------------------------------------------------

    def build_graph_panel(self) -> Panel:

        width = max(10, self.size.width - 6)
        height = 10

        plt.clf()

        start = self.offset
        end = start + self.window

        req = list(self.req)[start:end]
        res = list(self.res)[start:end]

        if not req:
            req = [0]
            res = [0]
            x = [0]
        else:
            x = list(range(len(req)))

        plt.theme("dark")
        plt.plotsize(width, height)

        plt.canvas_color("black")
        plt.axes_color("black")

        plt.grid(False, False)
        plt.xaxes(False)
        plt.frame(False)
        plt.yaxes(False)
        plt.xticks([])
        plt.yticks([])

        plt.plot(x, req, color=84, marker="braille")
        plt.plot(x, res, color=85, marker="braille")
        canvas = plt.build()

        graph = Group(*self.decoder.decode(canvas))

        return Panel(
            graph,
            border_style="#00FCC1",
            padding=(0, 1),
            style="black",
            height=9
        )

    # --------------------------------------------------
    # SYSTEM PANEL (PAGE 3)
    # --------------------------------------------------

    def build_tokens_panel(self) -> Panel:

        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        disk = psutil.disk_usage("/").percent
        swap = psutil.swap_memory().percent

        free = 100 - ram

        body = Text()

        def line(prefix, label, value=None, warn=False):

            body.append(prefix, style="#00FCC1")            # ascii semboller
            body.append(label, style="bold #FFFFFF")        # yazi

            if value is not None:
                body.append(" : ", style="#00FCC1")         # :
                body.append(f"%{int(value)}", style="bold #FFFFFF")  # sayi

            if warn:
                body.append("  -- ⚠", style="bold red")

            body.append("\n")

        line("┌─► ", "CPU  ", cpu, cpu >= 85)
        line("├─► ", "RAM  ", ram)
        line("├─► ", "DISK ", disk, disk >= 90)
        line("├─► ", "SWAP ", swap, swap >= 80)
        line("└─► ", "FREE ", free)

        return Panel(
            body,
            border_style="#00FCC1",
            padding=(1, 1),
            style="black",
            height=9
        )

    # --------------------------------------------------
    # AGENT INFO PANEL (PAGE 2)
    # --------------------------------------------------

    def _format_model_name(self, model_name: str) -> str:
        short = model_name.split("/")[-1] if "/" in model_name else model_name
        # Prevent overflow in the compact subsystem panel:
        # if model includes variant suffixes (e.g. gemini-3-flash),
        # keep only the family prefix (GEMINI).
        if "-" in short:
            short = short.split("-", 1)[0]
        if short.lower() == "gpt-4o":
            return "GPT-4o"
        return short.upper()

    def _format_mode_name(self, scan_mode: str) -> str:
        mode_map = {
            "osint": "OSINT",
            "general": "GENERAL",
            "adb": "ADB",
        }
        return mode_map.get(scan_mode.lower(), scan_mode.upper())

    def _get_live_agent_count(self, tracer: object | None) -> int:
        if not tracer or not hasattr(tracer, "agents"):
            return 0
        agents = getattr(tracer, "agents", {})
        if not isinstance(agents, dict):
            return 0
        return len(agents)

    def _get_live_status(self, tracer: object | None) -> str:
        if not tracer or not hasattr(tracer, "agents"):
            return "STOPPED"
        agents = getattr(tracer, "agents", {})
        if not isinstance(agents, dict) or not agents:
            return "STOPPED"

        for agent_data in agents.values():
            if not isinstance(agent_data, dict):
                continue
            status = str(agent_data.get("status", "")).lower()
            if status in {"running", "waiting", "stopping"}:
                return "RUNNING"
        return "STOPPED"

    def _get_elapsed_hms(self) -> str:
        elapsed = max(0, int(time.time() - self.start_timestamp))
        hours = elapsed // 3600
        minutes = (elapsed % 3600) // 60
        return f"{hours:02}:{minutes:02}"

    def build_agent_info_panel(self) -> Panel:
        app = self.app
        llm_cfg = getattr(app, "agent_config", {}).get("llm_config")
        model = "UNKNOWN"
        mode = "UNKNOWN"

        if llm_cfg is not None:
            model = self._format_model_name(str(getattr(llm_cfg, "litellm_model", "UNKNOWN")))
            mode = self._format_mode_name(str(getattr(llm_cfg, "scan_mode", "unknown")))

        tracer = get_global_tracer()
        agents_count = self._get_live_agent_count(tracer)
        status = self._get_live_status(tracer)
        elapsed_hms = self._get_elapsed_hms()

        body = Text()

        def line(prefix: str, label: str, value: str, *, endline: bool = True) -> None:
            body.append(prefix, style="#00FCC1")
            body.append(label, style="bold #FFFFFF")
            body.append(" : ", style="#00FCC1")
            body.append(value or "00:00:00", style="bold #FFFFFF")
            if endline:
                body.append("\n")

        line("┌─► ", "LLM  ", model)
        line("├─► ", "MODE ", mode)
        line("├─► ", "AGNTS", str(agents_count))
        line("├─► ", "PROGR", status)
        line("└─► ", "TIME ", elapsed_hms, endline=False)

        return Panel(
            body,
            border_style="#00FCC1",
            padding=(1, 1),
            style="black",
            height=9,
        )


    # --------------------------------------------------
    # STATS PAGE 1
    # --------------------------------------------------

    def build_stats_page1(self) -> Text:

        stats_text = Text()

        stats_text.append("")
        stats_text.append("\n---------- :::: ----------", style="#00FCC1")

        req_val = self.req[-1] if self.req else 0
        res_val = self.res[-1] if self.res else 0

        stats_text.append("\n\nRequest rate ", style="bold #FFFFFF")
        stats_text.append(":", style="#00FCC1")
        stats_text.append(f" {req_val:.2f} req/s\n", style="bold #FFFFFF")

        stats_text.append("Respons rate ", style="bold #FFFFFF")
        stats_text.append(":", style="#00FCC1")
        stats_text.append(f" {res_val:.2f} res/s\n", style="bold #FFFFFF")

        stats_text.append("Request sent ", style="bold #FFFFFF")
        stats_text.append(":", style="#00FCC1")
        stats_text.append(
            f" {self.sent_bytes/1024:.2f} KB/s\n", style="bold #FFFFFF"
        )

        stats_text.append("Respons recv ", style="bold #FFFFFF")
        stats_text.append(":", style="#00FCC1")
        stats_text.append(
            f" {self.recv_bytes/1024:.2f} KB/s\n", style="bold #FFFFFF"
        )

        stats_text.append("\n---------- :::: ----------", style="#00FCC1")

        return stats_text


    # --------------------------------------------------
    # STATS PAGE 2
    # --------------------------------------------------

    def build_stats_page2(self) -> Text:

        stats_text = Text()
        stats_text.append("")
        stats_text.append("\n---------- :::: ----------", style="#00FCC1")

        req_val = self.req[-1] if self.req else 0
        res_val = self.res[-1] if self.res else 0

        stats_text.append("\n\nRequest rate ", style="bold #FFFFFF")
        stats_text.append(":", style="#00FCC1")
        stats_text.append(f" {req_val:.2f} req/s\n", style="bold #FFFFFF")

        stats_text.append("Respons rate ", style="bold #FFFFFF")
        stats_text.append(":", style="#00FCC1")
        stats_text.append(f" {res_val:.2f} res/s\n", style="bold #FFFFFF")

        stats_text.append("Request sent ", style="bold #FFFFFF")
        stats_text.append(":", style="#00FCC1")
        stats_text.append(
            f" {self.sent_bytes/1024:.2f} KB/s\n", style="bold #FFFFFF"
        )

        stats_text.append("Respons recv ", style="bold #FFFFFF")
        stats_text.append(":", style="#00FCC1")
        stats_text.append(
            f" {self.recv_bytes/1024:.2f} KB/s\n", style="bold #FFFFFF"
        )

        stats_text.append("\n---------- :::: ----------", style="#00FCC1")

        return stats_text
    def build_tasks_panel(self) -> Panel:
        if not self._tasks_supported_in_current_mode():
            return Panel(
                Align.center(
                    Text("GENERAL MODE SUPPORTED.", style="bold #FFFFFF"),
                    vertical="middle",
                    height=5,
                ),
                border_style="#00FCC1",
                padding=(1, 1),
                style="black",
                height=9,
            )

        body = Text()

        tasks = []
        if self.task_store is not None:
            try:
                session_id = str(self.task_session_id or "").strip()
                all_tasks = (
                    self.task_store.get_scheduled_tasks(limit=5, session_id=session_id)
                    if session_id
                    else []
                )
                tasks = [t for t in all_tasks if t.schedule_time and t.schedule_time != "--:--"][:5]
            except Exception:  # noqa: BLE001
                tasks = []

        now_dt = time.time()
        for idx in range(5):
            task = tasks[idx] if idx < len(tasks) else None
            if idx == 0:
                prefix = "┌─► "
            elif idx < 4:
                prefix = "├─► "
            else:
                prefix = "└─► "
            due = False
            if task is not None and task.next_run:
                try:
                    due = datetime.fromisoformat(task.next_run).timestamp() <= now_dt
                except Exception:  # noqa: BLE001
                    due = False
            completed_recent = False
            running_now = False
            status_lower = ""
            if task is not None:
                status_lower = str(task.status).lower()
                running_now = status_lower == "running"
                if getattr(task, "last_run", None):
                    try:
                        last_run_ts = datetime.fromisoformat(str(task.last_run)).timestamp()
                        completed_recent = (now_dt - last_run_ts) <= 60.0
                    except Exception:  # noqa: BLE001
                        completed_recent = False
            bullet_style = (
                "bold green"
                if (task is not None and (due or running_now or completed_recent))
                else "bold red"
            )
            task_time = "--:--"
            base_task_time = "--:--"
            is_looping = False
            try:
                if task is not None:
                    stype = str(getattr(task, "schedule_type", "")).lower()
                    cron_expr = str(getattr(task, "cron_expression", "") or "").strip()
                    is_looping = bool(cron_expr) or stype in {"daily", "weekly", "recurring", "interval"}
                if (
                    task is not None
                    and isinstance(task.schedule_time, str)
                    and re.match(r"^\d{2}:\d{2}[hms]$", task.schedule_time.strip(), flags=re.IGNORECASE)
                ):
                    base_task_time = task.schedule_time.strip()
                elif (
                    task is not None
                    and isinstance(task.schedule_time, str)
                    and len(task.schedule_time) >= 5
                    and task.schedule_time[2] == ":"
                    and task.schedule_time[:5].replace(":", "").isdigit()
                ):
                    base_task_time = task.schedule_time[:5]
                elif (
                    task is not None
                    and isinstance(task.schedule_time, str)
                    and re.match(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}", task.schedule_time.strip())
                ):
                    base_task_time = task.schedule_time.strip()[-5:]
                elif (
                    task is not None
                    and isinstance(task.schedule_time, str)
                    and re.match(r"^\d+(S|MIN|H)$", task.schedule_time.upper())
                ):
                    base_task_time = task.schedule_time.upper()
                elif task is not None and task.next_run:
                    base_task_time = datetime.fromisoformat(task.next_run).astimezone().strftime("%H:%M")
            except Exception:  # noqa: BLE001
                base_task_time = "--:--"

            if base_task_time != "--:--" and is_looping:
                base_task_time = f"{base_task_time}+"

            if running_now:
                task_time = "EXEC"
            elif completed_recent:
                task_time = "CMPLT"
            elif task is not None and status_lower in {"done", "completed", "success"} and not is_looping:
                task_time = "--:--"
            else:
                task_time = base_task_time

            task_display = f"{task_time:<7}"
            body.append(prefix, style="#00FCC1")
            body.append("TASK ", style="bold #FFFFFF")
            body.append(" : ", style="#00FCC1")
            body.append(task_display, style="bold #FFFFFF")
            body.append("•", style=bullet_style)
            body.append("\n")

        return Panel(
            body,
            border_style="#00FCC1",
            padding=(1, 1),
            style="black",
            height=9,
        )

    # --------------------------------------------------
    # MEMORY LAYERS PANEL (PAGE 5)
    # --------------------------------------------------

    def _resolve_memory_events_path(self) -> Path:
        if self.task_store is not None:
            path_obj = getattr(self.task_store, "memory_events_jsonl_path", None)
            if path_obj:
                return Path(path_obj)
        return Path.cwd() / "logs" / "memory_events.jsonl"

    def _load_recent_memory_events(self, limit: int = 60) -> list[dict]:
        path = self._resolve_memory_events_path()
        if not path.exists():
            return []
        try:
            with path.open("rb") as fh:
                fh.seek(0, 2)
                size = fh.tell()
                if size <= 0:
                    return []
                read_size = min(size, 131072)
                fh.seek(max(0, size - read_size))
                payload = fh.read(read_size)
        except Exception:  # noqa: BLE001
            return []

        events: list[dict] = []
        lines = payload.decode("utf-8", errors="replace").splitlines()
        for line in lines[-max(1, int(limit) * 3):]:
            raw = str(line or "").strip()
            if not raw:
                continue
            try:
                loaded = json.loads(raw)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(loaded, dict):
                events.append(loaded)
                if len(events) >= max(1, int(limit)):
                    continue
        return events[-max(1, int(limit)) :]

    @staticmethod
    def _trend_arrow_and_style(value: float) -> tuple[str, str]:
        if value > 0:
            return "↑", "bold #22c55e"
        if value < 0:
            return "↓", "bold #ef4444"
        return "→", "bold #FFFFFF"

    @staticmethod
    def _memory_signal_code(reason: str, signal_mode: str = "prompt") -> str:
        normalized = str(reason or "").strip().lower()
        if not normalized:
            fallback = str(signal_mode or "prompt").strip().upper()
            return (fallback[:3] or "SIG").ljust(3, "G")
        if "no_update" in normalized:
            return "NUP"
        code_map = {
            "save_task_state": "STS",
            "store_semantic_memory": "SEM",
            "store_profile_fact": "PRF",
            "success": "SCS",
            "failure": "FLR",
            "feedback_positive": "FBP",
            "feedback_negative": "FBN",
            "prompt_signal": "PRM",
        }
        for key, value in code_map.items():
            if key in normalized:
                return value
        compact = "".join(ch for ch in normalized.upper() if ch.isalpha())
        return (compact[:3] or "SIG").ljust(3, "G")

    @staticmethod
    def _memory_entry_code(memory_type: str, memory_id: int) -> str:
        prefix_map = {
            "semantic": "SEM",
            "profile": "PRF",
            "task": "TSK",
            "episodic": "EPI",
        }
        prefix = prefix_map.get(str(memory_type or "").strip().lower(), "MEM")
        return f"{prefix}{max(0, int(memory_id or 0))}"

    @staticmethod
    def _emotion_code(emotion_name: str) -> str:
        normalized = str(emotion_name or "").strip().lower()
        emotion_map = {
            "neutral": "NTRL",
            "anger": "ANGR",
            "angry": "ANGR",
            "frustration": "FRST",
            "frustrated": "FRST",
            "sadness": "SADN",
            "sad": "SADN",
            "happiness": "HAPP",
            "happy": "HAPP",
            "joy": "JOY",
            "fear": "FEAR",
            "surprise": "SURP",
            "curiosity": "CURI",
        }
        if normalized in emotion_map:
            return emotion_map[normalized]
        compact = "".join(ch for ch in normalized.upper() if ch.isalpha())
        return (compact[:4] or "NTRL").ljust(4, "L")

    def _load_recent_emotion_signals(self, limit: int = 2) -> list[dict]:
        if self.task_store is None:
            return []
        safe_limit = max(1, int(limit))
        try:
            conn = self.task_store._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT message_id, anger, frustration, happiness, sadness, neutral, confidence, created_at
                    FROM message_emotions
                    WHERE role = ?
                    ORDER BY message_id DESC
                    LIMIT ?
                    """,
                    ("user", safe_limit),
                ).fetchall()
            finally:
                conn.close()
        except Exception:  # noqa: BLE001
            return []

        signals: list[dict] = []
        for row in rows:
            anger = float(row[1] or 0.0)
            frustration = float(row[2] or 0.0)
            happiness = float(row[3] or 0.0)
            sadness = float(row[4] or 0.0)
            neutral = float(row[5] or 0.0)
            stress = max(
                frustration,
                anger,
                min(1.0, (frustration * 0.72) + (anger * 0.58) + (sadness * 0.18)),
            )
            signals.append(
                {
                    "message_id": int(row[0] or 0),
                    "created_at": str(row[7] or ""),
                    "confidence": float(row[6] or 0.0),
                    "scores": {
                        "happy": happiness,
                        "sadness": sadness,
                        "stress": stress,
                        "neutral": neutral,
                    },
                }
            )
        return signals

    @staticmethod
    def _normalize_emotion_weight(value: float) -> float:
        numeric = float(value or 0.0)
        if 0.0 <= numeric <= 1.5:
            return max(0.0, min(numeric, 1.0))
        return max(0.0, min(numeric / 100.0, 1.0))

    def _load_live_agent_emotion_scores(self) -> dict[str, float]:
        try:
            from sondra.tools.agents_graph.agents_graph_actions import _agent_instances
        except Exception:  # noqa: BLE001
            return {}

        selected_agent_id = str(getattr(self.app, "selected_agent_id", "") or "").strip()
        agent_instance = None
        if selected_agent_id and selected_agent_id in _agent_instances:
            agent_instance = _agent_instances.get(selected_agent_id)
        elif _agent_instances:
            with contextlib.suppress(Exception):
                agent_instance = next(iter(_agent_instances.values()))
        if agent_instance is None:
            return {}

        state = getattr(agent_instance, "state", None)
        context = getattr(state, "context", None)
        if not isinstance(context, dict):
            return {}

        return {
            "happy": self._normalize_emotion_weight(context.get("emotion_happiness", 0.0)),
            "sadness": self._normalize_emotion_weight(context.get("emotion_sadness", 0.0)),
            "stress": self._normalize_emotion_weight(context.get("emotion_stress", 0.0)),
            "neutral": self._normalize_emotion_weight(context.get("emotion_neutral", 0.0)),
        }

    @staticmethod
    def _active_memory_layers_count(health: dict, latest_event: dict, latest_semantic: dict) -> int:
        codes: list[str] = []
        if int(health.get("profile_facts", 0) or 0) > 0:
            codes.append("PRF")
        if int(latest_semantic.get("id", 0) or 0) > 0:
            codes.append("SEM")
        if int(health.get("total_messages", 0) or 0) > 0:
            codes.append("EPI")
        latest_type = str(latest_event.get("memory_type", "") or "").strip().lower()
        latest_reason = str(latest_event.get("reason", "") or "").strip().lower()
        if latest_type == "task" or "task" in latest_reason:
            codes.append("TSK")
        unique_codes: list[str] = []
        for code in codes:
            if code not in unique_codes:
                unique_codes.append(code)
        return len(unique_codes)

    def build_memory_layers_panel(self) -> Panel:
        body = Text()

        snapshot: dict = {}
        if self.task_store is not None:
            with contextlib.suppress(Exception):
                snapshot = dict(getattr(self.task_store, "memory_signal_snapshot")() or {})

        events = [
            event
            for event in self._load_recent_memory_events(limit=80)
            if str(event.get("memory_type", "") or "").strip().lower() != "emotion"
        ]
        latest_event = dict(snapshot.get("latest_event", {}) or (events[-1] if events else {}))
        if str(latest_event.get("memory_type", "") or "").strip().lower() == "emotion":
            latest_event = dict(events[-1] if events else {})
        previous = events[-2] if len(events) >= 2 else {}
        latest_semantic = dict(snapshot.get("latest_semantic", {}) or {})
        latest_profile = dict(snapshot.get("latest_profile", {}) or {})
        latest_task = dict(snapshot.get("latest_task", {}) or {})
        health = dict(snapshot.get("health", {}) or {})

        memory_type = str(latest_event.get("memory_type", "") or "").strip().lower()
        semantic_id = int(latest_semantic.get("id", 0) or 0)
        event_id = int(latest_event.get("memory_id", 0) or 0)
        if memory_type == "semantic":
            entry_id = semantic_id or event_id
        elif memory_type == "profile":
            entry_id = int(latest_profile.get("id", event_id) or 0)
        elif memory_type == "task":
            entry_id = int(latest_task.get("id", event_id) or 0)
        else:
            entry_id = event_id or semantic_id
        if not memory_type and entry_id > 0:
            memory_type = "semantic"
        weight_value = float(latest_semantic.get("importance", latest_event.get("new", 0.0)) or 0.0)
        old_val = float(latest_event.get("old", 0.0) or 0.0)
        delta_value = float(latest_event.get("delta", weight_value - old_val) or 0.0)
        imp_arrow, imp_style = self._trend_arrow_and_style(delta_value)

        confidence = float(latest_event.get("confidence", 0.5) or 0.5)
        prev_conf = float(previous.get("confidence", confidence) or confidence)
        conf_arrow, conf_style = self._trend_arrow_and_style(confidence - prev_conf)

        signal_reason = str(
            latest_event.get("reason")
            or latest_event.get("signal_mode")
            or snapshot.get("signal_mode", "prompt")
        ).strip()
        signal_text = self._memory_signal_code(signal_reason, str(snapshot.get("signal_mode", "prompt") or "prompt"))
        signal_style = "bold #FFFFFF"
        layers_count = self._active_memory_layers_count(health, latest_event, latest_semantic)

        def line(prefix: str, label: str, value: str, *, value_style: str = "bold #FFFFFF") -> None:
            body.append(prefix, style="#00FCC1")
            body.append(label, style="bold #FFFFFF")
            body.append(" : ", style="#00FCC1")
            body.append(value, style=value_style)
            body.append("\n")

        def line_with_arrow(prefix: str, label: str, number_text: str, arrow_text: str, *, arrow_style: str) -> None:
            body.append(prefix, style="#00FCC1")
            body.append(label, style="bold #FFFFFF")
            body.append(" : ", style="#00FCC1")
            body.append(number_text, style="bold #FFFFFF")
            body.append(" ", style="bold #FFFFFF")
            body.append(arrow_text, style=arrow_style)
            body.append("\n")

        line("┌─► ", "ENTRY     ", self._memory_entry_code(memory_type, entry_id))
        line_with_arrow("├─► ", "WEIGHT    ", f"{weight_value:.1f}", imp_arrow, arrow_style=imp_style)
        line_with_arrow("├─► ", "CONFIDENCE", f"{confidence:.1f}", conf_arrow, arrow_style=conf_style)
        line("├─► ", "SIGNAL    ", signal_text, value_style=signal_style)
        line("└─► ", "LAYERS    ", str(layers_count))

        return Panel(
            body,
            border_style="#00FCC1",
            padding=(1, 1),
            style="black",
            height=9,
        )

    def _tasks_supported_in_current_mode(self) -> bool:
        return str(getattr(self, "scan_mode", "general") or "general").strip().lower() == "general"

    def build_emotion_weights_panel(self) -> Panel:
        body = Text()

        live_scores = self._load_live_agent_emotion_scores()
        if live_scores:
            current_scores = dict(live_scores)
            previous_scores = dict(getattr(self, "_last_emotion_panel_scores", {}) or {})
            self._last_emotion_panel_scores = dict(current_scores)
        else:
            signals = self._load_recent_emotion_signals(limit=2)
            latest = signals[0] if signals else {"scores": {}}
            previous = signals[1] if len(signals) >= 2 else latest
            current_scores = {
                "happy": self._normalize_emotion_weight(dict(latest.get("scores", {}) or {}).get("happy", 0.0)),
                "sadness": self._normalize_emotion_weight(dict(latest.get("scores", {}) or {}).get("sadness", 0.0)),
                "stress": self._normalize_emotion_weight(dict(latest.get("scores", {}) or {}).get("stress", 0.0)),
                "neutral": self._normalize_emotion_weight(dict(latest.get("scores", {}) or {}).get("neutral", 1.0)),
            }
            previous_scores = {
                "happy": self._normalize_emotion_weight(dict(previous.get("scores", {}) or {}).get("happy", 0.0)),
                "sadness": self._normalize_emotion_weight(dict(previous.get("scores", {}) or {}).get("sadness", 0.0)),
                "stress": self._normalize_emotion_weight(dict(previous.get("scores", {}) or {}).get("stress", 0.0)),
                "neutral": self._normalize_emotion_weight(dict(previous.get("scores", {}) or {}).get("neutral", 1.0)),
            }

        def line_with_arrow(prefix: str, label: str, value: float, prev: float) -> None:
            arrow, arrow_style = self._trend_arrow_and_style(value - prev)
            body.append(prefix, style="#00FCC1")
            body.append(label, style="bold #FFFFFF")
            body.append(" : ", style="#00FCC1")
            body.append(f"{value:.1f}", style="bold #FFFFFF")
            body.append(" ", style="bold #FFFFFF")
            body.append(arrow, style=arrow_style)
            body.append("\n")

        line_with_arrow("┌─► ", "HAPPY   ", float(current_scores.get("happy", 0.0) or 0.0), float(previous_scores.get("happy", 0.0) or 0.0))
        line_with_arrow("├─► ", "SADNESS ", float(current_scores.get("sadness", 0.0) or 0.0), float(previous_scores.get("sadness", 0.0) or 0.0))
        line_with_arrow("├─► ", "STRESS  ", float(current_scores.get("stress", 0.0) or 0.0), float(previous_scores.get("stress", 0.0) or 0.0))
        line_with_arrow("└─► ", "NEUTRAL ", float(current_scores.get("neutral", 1.0) or 1.0), float(previous_scores.get("neutral", 1.0) or 1.0))

        return Panel(
            body,
            border_style="#00FCC1",
            padding=(1, 1),
            style="black",
            height=9,
        )

    # --------------------------------------------------
    # UI UPDATE
    # --------------------------------------------------

    def update_ui(self) -> None:

        canvas = self.query_one("#graph_canvas", Static)
        stats = self.query_one("#graph_stats", Static)
        if self.page == 1:
            canvas.update(self.build_graph_panel())
            stats.update(self.build_stats_page1())
        elif self.page == 2:
            canvas.update(self.build_agent_info_panel())
            stats.update(self.build_stats_page2())
        elif self.page == 3:
            canvas.update(self.build_tokens_panel())
            stats.update(self.build_stats_page2())
        elif self.page == 4:
            canvas.update(self.build_tasks_panel())
            stats.update(self.build_stats_page2())
        elif self.page == 5:
            canvas.update(self.build_memory_layers_panel())
            stats.update(self.build_stats_page2())
        else:
            canvas.update(self.build_emotion_weights_panel())
            stats.update(self.build_stats_page2())



