from __future__ import annotations

import os
import platform
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path


EXIT_AND_STOP_SOUND = "choose.mp3"
BOOT_SOUND = "boot.mp3"
OUT_LOG_PATH = Path(__file__).resolve().parents[2] / "out.log"


def _resolve_ffplay_command() -> str | None:
    system = platform.system()
    release = platform.release().lower()
    is_wsl = "microsoft" in release or "wsl" in release

    if system != "Windows" and not is_wsl:
        return "ffplay"

    command = os.getenv("FFPLAY_COMMAND_DIR")
    if not command:
        return None

    command_path = Path(command)
    if command_path.is_dir():
        return str(command_path / "ffplay.exe")
    return command


def _resolve_sound_argument_path(sound_path: Path) -> str:
    system = platform.system()
    release = platform.release().lower()
    is_wsl = "microsoft" in release or "wsl" in release

    if is_wsl:
        try:
            return subprocess.check_output(
                ["wslpath", "-w", str(sound_path)],
                text=True,
            ).strip()
        except Exception:
            return str(sound_path)

    return str(sound_path)


def _append_out_log(message: str) -> None:
    OUT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(message)
        if not message.endswith("\n"):
            log_file.write("\n")


def _play_sound_file(filename: str, log_label: str) -> None:
    ffplay = _resolve_ffplay_command()
    if not ffplay:
        _append_out_log(
            f"[{datetime.now().isoformat(timespec='seconds')}] {log_label} skipped: FFPLAY command not resolved"
        )
        return

    def _play() -> None:
        sound_path = Path(__file__).resolve().with_name(filename)
        sound_arg = _resolve_sound_argument_path(sound_path)
        command = [ffplay, "-nodisp", "-autoexit", "-loglevel", "quiet", sound_arg]

        _append_out_log(
            "\n".join(
                [
                    f"[{datetime.now().isoformat(timespec='seconds')}] {log_label} start",
                    f"cwd={Path.cwd()}",
                    f"sound_path={sound_path}",
                    f"sound_arg={sound_arg}",
                    f"command={command!r}",
                ]
            )
        )

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
            )
            _append_out_log(
                "\n".join(
                    [
                        f"[{datetime.now().isoformat(timespec='seconds')}] {log_label} end",
                        f"returncode={result.returncode}",
                        "--- ffplay stdout ---",
                        result.stdout or "<empty>",
                        "--- ffplay stderr ---",
                        result.stderr or "<empty>",
                    ]
                )
            )
        except OSError as exc:
            _append_out_log(
                "\n".join(
                    [
                        f"[{datetime.now().isoformat(timespec='seconds')}] {log_label} error",
                        f"command={command!r}",
                        f"error={exc}",
                    ]
                )
            )

    threading.Thread(target=_play, daemon=True).start()


def exit_and_stop_sound() -> None:
    _play_sound_file(EXIT_AND_STOP_SOUND, "choose_sound")


def play_boot_sound() -> None:
    _play_sound_file(BOOT_SOUND, "boot_sound")


def play_choose_sound() -> None:
    exit_and_stop_sound()


def prime_choose_sound() -> None:
    return


exit_and_stop_agent = exit_and_stop_sound


# --- MAIN ---
def main() -> None:
    print("Test: ses çalınıyor...")
    play_choose_sound()
    time.sleep(3)


if __name__ == "__main__":
    main()