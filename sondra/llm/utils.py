import html
import json
import re
from typing import Any


_INVOKE_OPEN = re.compile(r'<invoke\s+name=["\']([^"\']+)["\']>')
_PARAM_NAME_ATTR = re.compile(r'<parameter\s+name=["\']([^"\']+)["\']>')
_FUNCTION_CALLS_TAG = re.compile(r"</?function_calls>")
_STRIP_TAG_QUOTES = re.compile(r"<(function|parameter)\s*=\s*([^>]*?)>")
_LEGACY_TOOL_TAGS = (
    "browser_action",
    "web_search",
    "memory_search",
    "memory_get",
    "python_action",
    "terminal_execute",
    "search_files",
    "list_files",
)


def normalize_tool_format(content: str) -> str:
    """Convert alternative tool-call XML formats to the expected one.

    Handles:
      <function_calls>...</function_calls>  → stripped
      <invoke name="X">                     → <function=X>
      <parameter name="X">                  → <parameter=X>
      </invoke>                             → </function>
      <function="X">                        → <function=X>
      <parameter="X">                       → <parameter=X>
      <browser_action><action>...</action>  → <function=browser_action>...
    """
    if "<invoke" in content or "<function_calls" in content:
        content = _FUNCTION_CALLS_TAG.sub("", content)
        content = _INVOKE_OPEN.sub(r"<function=\1>", content)
        content = _PARAM_NAME_ATTR.sub(r"<parameter=\1>", content)
        content = content.replace("</invoke>", "</function>")

    content = _normalize_legacy_tool_tags(content)

    return _STRIP_TAG_QUOTES.sub(
        lambda m: f"<{m.group(1)}={m.group(2).strip().strip(chr(34) + chr(39))}>", content
    )


def _normalize_legacy_tool_tags(content: str) -> str:
    """Normalize model-emitted ``<tool><arg>...</arg></tool>`` calls.

    Smaller local models often copy schema names as XML tags instead of using
    Sondra's canonical ``<function=tool>`` format. Restrict conversion to known
    tool names so ordinary XML-like text stays untouched.
    """
    normalized = str(content or "")
    for tool_name in _LEGACY_TOOL_TAGS:
        pattern = re.compile(
            rf"<{re.escape(tool_name)}>\s*(.*?)\s*</{re.escape(tool_name)}>",
            re.DOTALL | re.IGNORECASE,
        )

        def _replace(match: re.Match[str]) -> str:
            body = match.group(1)
            params: list[str] = []
            for param_match in re.finditer(
                r"<([A-Za-z_][\w-]*)>(.*?)</\1>",
                body,
                re.DOTALL,
            ):
                param_name = param_match.group(1).strip()
                param_value = param_match.group(2).strip()
                if not param_name:
                    continue
                params.append(f"<parameter={param_name}>{param_value}</parameter>")
            if not params:
                return match.group(0)
            return f"<function={tool_name}>\n" + "\n".join(params) + "\n</function>"

        normalized = pattern.sub(_replace, normalized)
    return normalized


SONDRA_MODEL_MAP: dict[str, str] = {
    "claude-sonnet-4.6": "anthropic/claude-sonnet-4-6",
    "claude-opus-4.6": "anthropic/claude-opus-4-6",
    "gpt-5.2": "openai/gpt-5.2",
    "gpt-5.1": "openai/gpt-5.1",
    "gpt-5": "openai/gpt-5",
    "gemini-3-pro-preview": "gemini/gemini-3-pro-preview",
    "gemini-3-flash-preview": "gemini/gemini-3-flash-preview",
    "glm-5": "openrouter/z-ai/glm-5",
    "glm-4.7": "openrouter/z-ai/glm-4.7",
}


def resolve_sondra_model(model_name: str | None) -> tuple[str | None, str | None]:
    """Resolve a sondra/ model into names for API calls and capability lookups.

    Returns (api_model, canonical_model):
    - api_model: openai/<base> for API calls (Sondra API is OpenAI-compatible)
    - canonical_model: actual provider model name for litellm capability lookups
    Non-sondra models return the same name for both.
    """
    if not model_name or not model_name.startswith("sondra/"):
        return model_name, model_name

    base_model = model_name[len("sondra/"):]
    api_model = f"openai/{base_model}"
    canonical_model = SONDRA_MODEL_MAP.get(base_model, api_model)
    return api_model, canonical_model


def _truncate_to_first_function(content: str) -> str:
    if not content:
        return content

    function_starts = [
        match.start() for match in re.finditer(r"<function=|<invoke\s+name=", content)
    ]

    if len(function_starts) >= 2:
        second_function_start = function_starts[1]

        return content[:second_function_start].rstrip()

    return content


def parse_tool_invocations(content: str) -> list[dict[str, Any]] | None:
    content = normalize_tool_format(content)
    content = fix_incomplete_tool_call(content)

    tool_invocations: list[dict[str, Any]] = []

    fn_regex_pattern = r"<function=([^>]+)>\n?(.*?)</function>"
    fn_param_regex_pattern = r"<parameter=([^>]+)>(.*?)</parameter>"
    fn_param_inline_pattern = r"<parameter=([^>]+)>\s*"

    fn_matches = re.finditer(fn_regex_pattern, content, re.DOTALL)

    for fn_match in fn_matches:
        fn_name = fn_match.group(1).strip()
        fn_body = fn_match.group(2)

        param_matches = re.finditer(fn_param_regex_pattern, fn_body, re.DOTALL)

        args = {}
        for param_match in param_matches:
            param_name = param_match.group(1).strip()
            param_value = param_match.group(2).strip()

            param_name_unescaped = html.unescape(param_name)
            param_value_unescaped = html.unescape(param_value)

            if not param_value_unescaped and _merge_json_parameter_spec(args, param_name_unescaped):
                continue

            args[param_name_unescaped] = param_value_unescaped

        remaining_body = re.sub(fn_param_regex_pattern, "", fn_body, flags=re.DOTALL)
        inline_matches = re.finditer(fn_param_inline_pattern, remaining_body, re.DOTALL)
        for inline_match in inline_matches:
            inline_spec = html.unescape(inline_match.group(1).strip())
            _merge_json_parameter_spec(args, inline_spec)

        tool_invocations.append({"toolName": fn_name, "args": args})

    if tool_invocations:
        return tool_invocations

    return _parse_json_tool_invocations(content)


def _merge_json_parameter_spec(args: dict[str, Any], spec: str) -> bool:
    text = str(spec or "").strip()
    if not text.startswith("{") or not text.endswith("}"):
        return False
    try:
        loaded = json.loads(text)
    except Exception:
        return False
    if not isinstance(loaded, dict):
        return False

    for key, value in loaded.items():
        param_name = str(key).strip()
        if not param_name:
            continue
        args[param_name] = value
    return True


def _parse_json_tool_invocations(content: str) -> list[dict[str, Any]] | None:
    payload_text = _strip_markdown_code_fence(content)
    if not payload_text:
        return None

    try:
        payload = json.loads(payload_text)
    except Exception:
        return None

    return _json_payload_to_tool_invocations(payload)


def _strip_markdown_code_fence(content: str) -> str:
    raw = str(content or "").strip()
    if not raw.startswith("```") or not raw.endswith("```"):
        return raw

    lines = raw.splitlines()
    if len(lines) < 3:
        return raw

    if not lines[-1].strip().startswith("```"):
        return raw

    body = "\n".join(lines[1:-1]).strip()
    return body


def _json_payload_to_tool_invocations(payload: Any) -> list[dict[str, Any]] | None:
    if isinstance(payload, dict):
        sequence = _extract_json_tool_sequence(payload)
        if sequence is not None:
            parsed = _collect_json_tool_items(sequence)
            return parsed or None

        item = _normalize_json_tool_item(payload)
        return [item] if item else None

    if isinstance(payload, list):
        parsed = _collect_json_tool_items(payload)
        return parsed or None

    return None


def _extract_json_tool_sequence(payload: dict[str, Any]) -> list[Any] | None:
    for key in ("tool_calls", "tools", "calls", "invocations"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return None


def _collect_json_tool_items(items: list[Any]) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for item in items:
        normalized = _normalize_json_tool_item(item)
        if normalized:
            parsed.append(normalized)
    return parsed


def _normalize_json_tool_item(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None

    alias_map = {
        "memorysearch": "memory_search",
        "memoryget": "memory_get",
        "websearch": "web_search",
        "search": "web_search",
    }

    function_payload = item.get("function")
    if isinstance(function_payload, dict):
        tool_name = str(function_payload.get("name", "") or "").strip()
        args_payload = function_payload.get("arguments", {})
    elif isinstance(function_payload, str):
        tool_name = str(function_payload or "").strip()
        args_payload = item.get(
            "args",
            item.get(
                "arguments",
                item.get("parameters", item.get("params", item.get("tool_input", item.get("input", {})))),
            ),
        )
    else:
        tool_name = str(
            item.get("toolName")
            or item.get("tool_name")
            or item.get("toolname")
            or item.get("action")
            or item.get("tool")
            or item.get("name")
            or ""
        ).strip()
        args_payload = item.get(
            "args",
            item.get(
                "arguments",
                item.get("parameters", item.get("params", item.get("tool_input", item.get("input", {})))),
            ),
        )

    if not tool_name:
        return None

    normalized_tool = tool_name.strip().lower()
    tool_name = alias_map.get(normalized_tool, tool_name)
    normalized_tool = tool_name.strip().lower()
    if normalized_tool == "think":
        if not args_payload:
            thought_value = (
                item.get("thought")
                or item.get("message")
                or item.get("content")
                or item.get("analysis")
                or item.get("reasoning")
                or ""
            )
            if str(thought_value or "").strip():
                args_payload = {"thought": str(thought_value)}
        elif isinstance(args_payload, dict) and not str(args_payload.get("thought", "") or "").strip():
            thought_value = (
                item.get("thought")
                or item.get("message")
                or item.get("content")
                or item.get("analysis")
                or item.get("reasoning")
                or ""
            )
            if str(thought_value or "").strip():
                args_payload = {**args_payload, "thought": str(thought_value)}

    return {"toolName": tool_name, "args": _coerce_tool_args(args_payload)}


def _coerce_tool_args(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except Exception:
            return {}
        if isinstance(parsed, dict):
            return parsed
        return {}

    return {}


def fix_incomplete_tool_call(content: str) -> str:
    """Fix incomplete tool calls by adding missing closing tag.

    Handles both ``<function=…>`` and ``<invoke name="…">`` formats.
    """
    has_open = "<function=" in content or "<invoke " in content
    count_open = content.count("<function=") + content.count("<invoke ")
    has_close = "</function>" in content or "</invoke>" in content
    if has_open and count_open == 1 and not has_close:
        content = content.rstrip()
        content = content + "function>" if content.endswith("</") else content + "\n</function>"
    return content


def format_tool_call(tool_name: str, args: dict[str, Any]) -> str:
    xml_parts = [f"<function={tool_name}>"]

    for key, value in args.items():
        xml_parts.append(f"<parameter={key}>{value}</parameter>")

    xml_parts.append("</function>")

    return "\n".join(xml_parts)


def clean_content(content: str) -> str:
    if not content:
        return ""

    content = normalize_tool_format(content)
    content = fix_incomplete_tool_call(content)

    tool_pattern = r"<function=[^>]+>.*?</function>"
    cleaned = re.sub(tool_pattern, "", content, flags=re.DOTALL)

    incomplete_tool_pattern = r"<function=[^>]+>.*$"
    cleaned = re.sub(incomplete_tool_pattern, "", cleaned, flags=re.DOTALL)

    partial_tag_pattern = r"<f(?:u(?:n(?:c(?:t(?:i(?:o(?:n(?:=(?:[^>]*)?)?)?)?)?)?)?)?)?$"
    cleaned = re.sub(partial_tag_pattern, "", cleaned)

    hidden_xml_patterns = [
        r"<inter_agent_message>.*?</inter_agent_message>",
        r"<agent_completion_report>.*?</agent_completion_report>",
    ]
    for pattern in hidden_xml_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL | re.IGNORECASE)

    cleaned = re.sub(r"\n\s*\n", "\n\n", cleaned)

    return cleaned.strip()
