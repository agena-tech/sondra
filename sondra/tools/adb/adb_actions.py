import json
import os
from pathlib import Path

import httpx

from sondra.tools.registry import register_tool


@register_tool(sandbox_execution=False)
def adb_exec(cmd: str) -> str:
    command = (cmd or "").strip()
    if not command:
        return "Error: cmd is required."

    bridge_url = (os.getenv("SONDRA_ADB_URL") or "").strip()
    if not bridge_url:
        return "Error: SONDRA_ADB_URL is not set."

    timeout = float(os.getenv("SONDRA_ADB_TIMEOUT", "120"))

    try:
        response = httpx.post(
            bridge_url,
            json={"cmd": command},
            timeout=timeout,
        )
        response.raise_for_status()
    except Exception as exc:
        return f"Error executing adb command via bridge: {exc}"

    content_type = str(response.headers.get("content-type", "")).lower()

    # Binary screenshots from `exec-out screencap -p` are returned as raw bytes
    # by the bridge (application/octet-stream). Save them deterministically so
    # downstream steps (e.g. analyze_image) can consume a concrete file path.
    is_screencap = command.startswith("exec-out") and "screencap" in command and "-p" in command
    if is_screencap and "application/json" not in content_type:
        default_path = "/workspace/adb_screencap.png"
        target_path = str(os.getenv("SONDRA_ADB_SCREENSHOT_PATH") or default_path).strip() or default_path
        try:
            path_obj = Path(target_path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            path_obj.write_bytes(response.content)
        except Exception as exc:
            return f"Error: screenshot binary received but failed to save ({exc})"
        return f"Saved screenshot: {path_obj}"

    if "application/json" not in content_type:
        return response.text

    try:
        payload = response.json()
    except Exception:
        return response.text

    if isinstance(payload, dict):
        for key in ("output", "stdout", "result"):
            if key in payload:
                return str(payload.get(key, ""))
        return json.dumps(payload, ensure_ascii=False)

    if isinstance(payload, str):
        return payload

    return json.dumps(payload, ensure_ascii=False)
