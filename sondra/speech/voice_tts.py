from __future__ import annotations

import logging
import os
import platform
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

try:
    from langdetect import LangDetectException, detect
except Exception:  # noqa: BLE001
    LangDetectException = Exception  # type: ignore[assignment]
    detect = None  # type: ignore[assignment]


DEBUG = os.getenv("SONDRA_DEBUG", "False") == "True" or "--test" in sys.argv

logger = logging.getLogger(__name__)


class VoiceSpeechEngine:
    _TASK_BLOCK_RE = re.compile(r"<task_add>.*?</task_add>", flags=re.IGNORECASE | re.DOTALL)
    _TOOL_BLOCK_RE = re.compile(r"<tool_result>.*?</tool_result>", flags=re.IGNORECASE | re.DOTALL)
    _TAG_RE = re.compile(r"<[^>]+>")

    def __init__(
        self,
        *,
        model_path: Path | None = None,
        length_scale: float = 0.9,
        noise_scale: float = 0.7,
        noise_w_scale: float = 0.8,
    ) -> None:
        if DEBUG:
            print("[DEBUG] Initializing VoiceSpeechEngine")

        self._queue: queue.Queue[str | None] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._voice_lock = threading.RLock()
        self._current_lang = "tr"
        self._volume_percent = 30

        self._model_path = model_path or self._resolve_default_model_path()
        self._length_scale = length_scale
        self._noise_scale = noise_scale
        self._noise_w_scale = noise_w_scale

        self._available = False
        self._init_error: str | None = None
        self._voice = None
        self._syn_config = None
        self._player_cmd = self._resolve_player_command()
        self._ffmpeg_cmd = self._resolve_ffmpeg_binary()

        self._initialize_engine()

    def is_available(self) -> bool:
        return self._available

    @property
    def init_error(self) -> str | None:
        return self._init_error

    @staticmethod
    def _is_wsl() -> bool:
        system = platform.system()
        release = platform.release().lower()
        return system == "Linux" and ("microsoft" in release or "wsl" in release)

    @staticmethod
    def _resolve_project_root() -> Path:
        return Path(__file__).resolve().parents[2]

    @classmethod
    def _resolve_model_from_dir(cls, lang_dir_name: str) -> Path | None:
        model_dir = cls._resolve_project_root() / "models" / lang_dir_name
        if not model_dir.exists():
            return None
        models = sorted(model_dir.glob("*.onnx"))
        return models[0] if models else None

    @classmethod
    def _resolve_model_for_lang(cls, lang_code: str) -> Path | None:
        if lang_code == "en":
            return cls._resolve_model_from_dir("en")
        return cls._resolve_model_from_dir("tr")

    @classmethod
    def _resolve_default_model_path(cls) -> Path:
        explicit = cls._resolve_model_for_lang("tr")
        if explicit:
            return explicit

        project_root = cls._resolve_project_root()
        models_dir = project_root / "models"

        fallback_candidates = [
            models_dir / "tr_TR-fettah-medium.onnx",
            models_dir / "tr" / "tr_TR-fettah-medium.onnx",
        ]

        for candidate in fallback_candidates:
            if candidate.exists():
                return candidate

        if models_dir.exists():
            turkish_models = sorted(models_dir.glob("**/*tr*.onnx"))
            if turkish_models:
                return turkish_models[0]

            any_models = sorted(models_dir.glob("**/*.onnx"))
            if any_models:
                return any_models[0]

        return models_dir / "tr_TR-fettah-medium.onnx"

    @staticmethod
    def _is_windows_executable(command: str | None) -> bool:
        return bool(command and str(command).lower().endswith(".exe"))

    @classmethod
    def _resolve_tool_command(cls, tool_name: str) -> str | None:
        is_wsl = cls._is_wsl()
        system = platform.system()

        if system != "Windows" and not is_wsl:
            return tool_name

        command = os.getenv("FFPLAY_COMMAND_DIR")
        if not command:
            return None

        command_path = Path(command)
        windows_tool = f"{tool_name}.exe"

        if command_path.is_dir():
            return str(command_path / windows_tool)

        command_name = command_path.name.lower()
        if command_name in {"ffplay.exe", "ffmpeg.exe"}:
            return str(command_path.with_name(windows_tool))

        if command_path.suffix.lower() == ".exe":
            return str(command_path.with_name(windows_tool))

        return command

    @classmethod
    def _resolve_player_command(cls) -> list[str] | None:
        ffplay = cls._resolve_tool_command("ffplay")
        if not ffplay:
            return None
        return [ffplay, "-nodisp", "-autoexit", "-loglevel", "quiet"]

    @classmethod
    def _resolve_ffmpeg_binary(cls) -> str | None:
        return cls._resolve_tool_command("ffmpeg")

    @classmethod
    def _resolve_command_argument_path(cls, path: Path, command: str | None) -> str:
        resolved_path = path.resolve()

        if cls._is_wsl() and cls._is_windows_executable(command):
            try:
                return subprocess.check_output(
                    ["wslpath", "-w", str(resolved_path)],
                    text=True,
                ).strip()
            except Exception:  # noqa: BLE001
                return str(resolved_path)

        return str(resolved_path)

    def _detect_input_language(self, text: str) -> str:
        sample = (text or "").strip()
        if not sample:
            return "tr"

        if re.search(r"[çğıöşüÇĞİÖŞÜ]", sample):
            return "tr"

        if detect is None:
            return "tr"

        try:
            lang = detect(sample).lower()
        except LangDetectException:
            return "tr"
        except Exception:  # noqa: BLE001
            return "tr"

        return "en" if lang.startswith("en") else "tr"

    def update_language_from_text(self, text: str) -> None:
        if not self._available:
            return

        target_lang = self._detect_input_language(text)
        if target_lang == self._current_lang:
            return

        target_model = self._resolve_model_for_lang(target_lang)
        if not target_model or not target_model.exists():
            logger.warning("Voice model for '%s' not found. Keeping '%s'.", target_lang, self._current_lang)
            return

        try:
            from piper import PiperVoice
        except Exception as exc:
            logger.warning("Failed to import PiperVoice while switching model: %s", exc)
            return

        try:
            with self._voice_lock:
                self._voice = PiperVoice.load(str(target_model))
                self._model_path = target_model
                self._current_lang = target_lang
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to switch voice model to '%s': %s", target_model, exc)

    def enqueue(self, text: str) -> None:
        if not self._available:
            return

        cleaned = self._sanitize_text(text)
        if not cleaned:
            return

        self._queue.put(cleaned)

    def get_volume_percent(self) -> int:
        with self._voice_lock:
            return int(self._volume_percent)

    def increase_volume(self) -> int:
        with self._voice_lock:
            self._volume_percent = min(100, self._volume_percent + 10)
            return int(self._volume_percent)

    def decrease_volume(self) -> int:
        with self._voice_lock:
            self._volume_percent = max(0, self._volume_percent - 5)
            return int(self._volume_percent)

    def _get_ffmpeg_volume_value(self) -> float:
        with self._voice_lock:
            return round(self._volume_percent / 100.0, 2)

    def close(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._queue.put(None)
            self._thread.join(timeout=3.0)

    def _initialize_engine(self) -> None:
        try:
            from piper import PiperVoice, SynthesisConfig
        except Exception as exc:
            self._init_error = str(exc)
            return

        if not self._model_path.exists():
            self._init_error = "model yok"
            return

        if not self._player_cmd:
            self._init_error = "player yok"
            return

        self._voice = PiperVoice.load(str(self._model_path))
        self._current_lang = "en" if "/en/" in str(self._model_path).replace("\\", "/") else "tr"
        self._syn_config = SynthesisConfig(
            length_scale=self._length_scale,
            noise_scale=self._noise_scale,
            noise_w_scale=self._noise_w_scale,
        )

        self._available = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                text = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if text is None:
                continue

            self._speak_text(text)

    def _speak_text(self, text: str) -> None:
        with self._voice_lock:
            voice = self._voice
            syn_config = self._syn_config

        if not voice:
            return

        base = Path("/mnt/c/Temp/sondra")
        base.mkdir(parents=True, exist_ok=True)

        tmp = base / "live"
        tmp.mkdir(exist_ok=True)

        out_wav = tmp / "out.wav"
        processed_wav = tmp / "processed.wav"

        with wave.open(str(out_wav), "wb") as f:
            voice.synthesize_wav(text, f, syn_config=syn_config)

        ffmpeg = self._ffmpeg_cmd
        if ffmpeg:
            input_path = self._resolve_command_argument_path(out_wav, ffmpeg)
            output_path = self._resolve_command_argument_path(processed_wav, ffmpeg)
            volume_value = self._get_ffmpeg_volume_value()
            af_filter = (
                f"asetrate=22050*1.06,aresample=22050,volume={volume_value},"
                "aecho=0.4:0.5:350:0.15"
            )

            subprocess.run(
                [
                    ffmpeg,
                    "-loglevel",
                    "quiet",
                    "-y",
                    "-i",
                    input_path,
                    "-af",
                    af_filter,
                    output_path,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            if processed_wav.exists() and processed_wav.stat().st_size > 1000:
                shutil.copyfile(processed_wav, out_wav)

        play_path = self._resolve_command_argument_path(out_wav, self._player_cmd[0])

        subprocess.run(
            [*self._player_cmd, play_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    @classmethod
    def _sanitize_text(cls, text: str) -> str:
        cleaned = text.strip()
        cleaned = cls._TASK_BLOCK_RE.sub("", cleaned)
        cleaned = cls._TOOL_BLOCK_RE.sub("", cleaned)
        cleaned = cls._TAG_RE.sub("", cleaned)
        cleaned = cleaned.replace("💾 Reading from disk ...", "")
        cleaned = cleaned.replace("⏱️ 1 Görev eklendi", "")
        cleaned = cleaned.replace("⏱️ Tasks cleared", "")

        lines = []
        for raw_line in cleaned.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            low = line.lower()
            if low.startswith(
                (
                    "launching ",
                    "clicking",
                    "executing javascript",
                    "closing browser",
                    "waiting for the scheduled time.",
                    "o waiting",
                    ">_",
                    "$ ",
                )
            ):
                continue

            lines.append(line)

        return " ".join(lines).strip()


if __name__ == "__main__":
    engine = VoiceSpeechEngine()

    if engine.is_available():
        engine.enqueue("Merhaba, bu bir test konuşmasıdır.")
        time.sleep(3)

    engine.close()
