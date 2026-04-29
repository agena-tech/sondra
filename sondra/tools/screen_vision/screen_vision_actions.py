import base64
import contextlib
import os
import tarfile
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from openai import OpenAI

from sondra.config import Config
from sondra.tools.registry import register_tool

# DEBUG FLAG
DEBUG = os.getenv("SONDRA_DEBUG", "False").lower() == "true"


def _debug_print(*args):
    if DEBUG:
        print("[DEBUG]", *args)


def _log_error(message: str) -> None:
    try:
        log_path = Path.cwd() / "analyze_image_errors.log"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().isoformat()}] {message}\n")
    except Exception:
        pass


def _is_general_mode(agent_state: Any | None) -> bool:
    if DEBUG:
        return True
    if agent_state is None:
        return False
    context = getattr(agent_state, "context", {}) or {}
    mode = str(context.get("scan_mode", "")).lower()
    return mode in {"general", "adb"}


def _download_image(image_url: str) -> tuple[bytes | None, str | None]:
    try:
        response = requests.get(image_url, timeout=20)
    except Exception as e:  # noqa: BLE001
        _log_error(f"Image download failed: {e}")
        return None, f"Failed to download image_url: {e}"

    if response.status_code != 200:
        _log_error(f"Image download status error: {response.status_code}")
        return None, f"image_url returned status_code={response.status_code}"

    return response.content, None


def _resolve_existing_image_path(image_path: str) -> Path | None:
    raw = str(image_path or "").strip()
    if not raw:
        return None

    p = Path(raw).expanduser()
    if p.exists():
        return p

    return None


def _read_image_bytes_from_sandbox(
    agent_state: Any | None, image_path: str
) -> tuple[bytes | None, str | None]:
    sandbox_id = getattr(agent_state, "sandbox_id", None) if agent_state is not None else None
    if not sandbox_id:
        return None, "sandbox_id not available"

    try:
        import docker
    except Exception as e:  # noqa: BLE001
        return None, f"docker import failed: {e}"

    raw = str(image_path or "").strip()
    if not raw:
        return None, "empty image_path"

    candidate_paths = [raw]
    if not raw.startswith("/"):
        candidate_paths.append(f"/workspace/{raw}")

    try:
        client = docker.from_env(timeout=30)
        container = client.containers.get(sandbox_id)
    except Exception as e:  # noqa: BLE001
        return None, f"sandbox container access failed: {e}"

    last_error = "image not found in sandbox"
    for candidate in candidate_paths:
        try:
            stream, _ = container.get_archive(candidate)
            tar_bytes = b"".join(stream)
            with tarfile.open(fileobj=BytesIO(tar_bytes), mode="r:*") as tar:
                for member in tar.getmembers():
                    if member.isfile():
                        extracted = tar.extractfile(member)
                        if extracted is None:
                            continue
                        return extracted.read(), None
            last_error = f"no regular file extracted from sandbox path: {candidate}"
        except Exception as e:  # noqa: BLE001
            last_error = f"sandbox image read failed for {candidate}: {e}"

    return None, last_error


def _read_image_bytes(
    image_path: str, agent_state: Any | None = None
) -> tuple[bytes | None, str | None]:
    path = _resolve_existing_image_path(image_path)
    if path is not None:
        try:
            return path.read_bytes(), None
        except Exception as e:  # noqa: BLE001
            return None, str(e)

    sandbox_bytes, sandbox_err = _read_image_bytes_from_sandbox(agent_state, image_path)
    if sandbox_bytes is not None:
        return sandbox_bytes, None

    return None, f"image_path does not exist locally and sandbox read failed: {sandbox_err}"


# MODEL PREFIX NORMALIZER
def _normalize_model_name(model: str) -> str:
    if "/" in model:
        cleaned = model.split("/", 1)[1]
        _debug_print(f"MODEL NORMALIZED: {model} -> {cleaned}")
        return cleaned
    return model


# STRICT ENV
def _resolve_vision_config() -> tuple[str, str, str]:
    model = Config.get("sondra_llm")
    base_url = os.getenv("LLM_API_BASE")
    key = os.getenv("LLM_API_KEY")

    if not model:
        raise RuntimeError("SONDRA_LLM is not set")
    if not base_url:
        raise RuntimeError("LLM_API_BASE is not set")
    if not key:
        raise RuntimeError("LLM_API_KEY is not set")

    model = _normalize_model_name(model.strip())
    base_url = base_url.strip()
    key = key.strip()

    _debug_print("MODEL  :", model)
    _debug_print("BASE_URL:", base_url)
    _debug_print("API_KEY :", key[:6] + "...")

    return model, base_url, key


def _extract_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join([i.get("text", "") for i in content if isinstance(i, dict)])
    return str(content)


def _run_openai_vision(image_bytes: bytes, analysis_prompt: str):
    model, base_url, api_key = _resolve_vision_config()
    image_b64 = base64.b64encode(image_bytes).decode()
    vision_timeout = float(os.getenv("SONDRA_VISION_TIMEOUT", "90"))

    _debug_print("REQUEST -> model:", model)
    _debug_print("REQUEST -> prompt:", analysis_prompt)

    try:
        client = OpenAI(base_url=base_url, api_key=api_key)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": analysis_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}"
                            },
                        },
                    ],
                }
            ],
            timeout=vision_timeout,
        )

    except Exception as e:  # noqa: BLE001
        _log_error(f"Vision request failed (timeout={vision_timeout}s): {e}")
        _debug_print("REQUEST FAILED")
        _debug_print("MODEL:", model)
        _debug_print("BASE:", base_url)
        _debug_print("ERROR:", e)

        return "", f"Vision request failed (timeout={vision_timeout}s): {e}", model, base_url

    text = ""
    with contextlib.suppress(Exception):
        text = _extract_message_text(response.choices[0].message.content)

    return text, None, model, base_url


def _analyze_image_impl(agent_state, image_url=None, image_path=None, analysis_prompt="Analyze"):
    if not _is_general_mode(agent_state):
        return {"success": False, "error": "Not allowed in this mode"}

    if image_url:
        image_bytes, err = _download_image(image_url)
    elif image_path:
        image_bytes, err = _read_image_bytes(image_path, agent_state=agent_state)
    else:
        return {"success": False, "error": "No image"}

    if err:
        return {"success": False, "error": err}

    text, err, _, _ = _run_openai_vision(image_bytes, analysis_prompt)

    if err:
        return {"success": False, "error": err}

    return {"success": True, "analysis_text": text}


@register_tool(sandbox_execution=False)
def analyze_image(agent_state, image_url=None, image_path=None, analysis_prompt="Analyze image"):
    return _analyze_image_impl(agent_state, image_url, image_path, analysis_prompt)


if DEBUG and __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--url")
    parser.add_argument("--path")
    parser.add_argument("--prompt", default="Analyze image")
    args = parser.parse_args()

    result = _analyze_image_impl(
        None,
        image_url=args.url,
        image_path=args.path,
        analysis_prompt=args.prompt,
    )

    print("\n==== RESULT ====\n")
    print(result)
