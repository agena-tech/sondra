from types import SimpleNamespace

from sondra.agents.base_agent import BaseAgent
from sondra.memory.signal_catalog import get_memory_signal_catalog


def _build_bare_agent() -> BaseAgent:
    agent = BaseAgent.__new__(BaseAgent)
    agent.signal_catalog = get_memory_signal_catalog()
    agent.state = SimpleNamespace(messages=[], context={})
    return agent


def test_low_quality_direct_reply_detects_echoed_how_are_you() -> None:
    agent = _build_bare_agent()
    agent.state.messages = [{"role": "user", "content": "Nasilsin?"}]

    assert agent._is_how_are_you_turn("Nasilsin?")
    assert agent._looks_like_low_quality_direct_reply("Nasilsin?", "Nasilsin?")


def test_social_tone_rewrite_ignores_emoji_only_reply() -> None:
    agent = _build_bare_agent()
    agent.state.context = {
        "emotion_tone": "stabilizing",
        "emotion_signal_category": "critical",
    }
    response_text = "Buradayim " + chr(0x1F60A) + " nasil yardimci olayim?"

    assert not agent._response_needs_social_tone_rewrite(
        "Selam",
        response_text,
    )


def test_how_are_you_prefers_direct_conversation() -> None:
    agent = _build_bare_agent()
    agent.memory_store = None

    assert agent._is_how_are_you_turn("nasılsın?")
    assert agent._should_prefer_direct_conversational_reply("nasılsın?")


def test_social_reply_detects_tool_or_command_output() -> None:
    agent = _build_bare_agent()

    assert agent._looks_like_tool_or_command_output_reply("> $ pwd")
    assert agent._looks_like_tool_or_command_output_reply("Executing terminal command: pwd")
    assert agent._looks_like_tool_or_command_output_reply("<function=memory_search>")
    assert not agent._looks_like_tool_or_command_output_reply("Buradayım, yardımcı olabilirim.")


def test_python_code_writing_request_does_not_require_execution_tool() -> None:
    agent = _build_bare_agent()

    assert agent._looks_like_code_writing_request("basit bir python kodu yazarmısın")
    assert not agent._looks_like_terminal_request("basit bir python kodu yazarmısın")
    assert not agent._looks_like_python_request("basit bir python kodu yazarmısın")
    assert agent._looks_like_python_request("python kodunu çalıştır")


def test_successful_browser_launch_pauses_without_rechecking_latest_user_route() -> None:
    agent = _build_bare_agent()
    agent._is_general_root_agent = lambda: True
    agent._is_ollama_tool_guard_enabled = lambda: True

    actions = [
        {
            "toolName": "browser_action",
            "args": {"action": "launch", "url": "https://example.com"},
        }
    ]

    assert agent._should_pause_after_browser_open(actions, operation_success=True)


def test_routed_tool_match_uses_cached_user_turn_after_internal_guidance() -> None:
    agent = _build_bare_agent()
    agent.state.context = {"last_user_turn_raw": "terminalde pwd çalıştır"}
    agent.state.messages = [
        {"role": "user", "content": "terminalde pwd çalıştır"},
        {"role": "user", "content": "External action is required. Use terminal_execute."},
    ]
    agent._is_general_root_agent = lambda: True
    agent._is_ollama_tool_guard_enabled = lambda: True
    agent._route_obvious_tool_request = (
        lambda text: "terminal_execute" if text == "terminalde pwd çalıştır" else ""
    )

    actions = [{"toolName": "terminal_execute", "args": {"command": "pwd"}}]

    assert agent._matched_routed_tool_name(actions) == "terminal_execute"


def test_original_user_turn_raw_prefers_cached_value() -> None:
    agent = _build_bare_agent()
    agent.state.context = {"last_user_turn_raw": "http://scaa.us sitesine gir"}
    agent.state.messages = [{"role": "user", "content": "Use browser_action now."}]

    assert agent._original_user_turn_raw() == "http://scaa.us sitesine gir"


def test_ollama_tool_guard_requires_sondra_llm_ollama(monkeypatch) -> None:
    agent = _build_bare_agent()
    agent.llm_config = SimpleNamespace(
        litellm_model="deepseek-r1:8b",
        api_base="http://127.0.0.1:11434",
    )

    monkeypatch.setenv("SONDRA_LLM", "openai/gpt-5-mini")
    assert not agent._is_ollama_tool_guard_enabled()

    monkeypatch.setenv("SONDRA_LLM", "ollama/gemma3")
    assert agent._is_ollama_tool_guard_enabled()


def test_forced_tool_fallback_builds_terminal_action() -> None:
    agent = _build_bare_agent()

    action = agent._build_forced_tool_fallback_action(
        "terminal_execute",
        "terminalde pwd komutunu çalıştır",
    )

    assert action == {
        "toolName": "terminal_execute",
        "args": {"command": "pwd"},
    }


def test_forced_tool_fallback_builds_list_files_action() -> None:
    agent = _build_bare_agent()

    action = agent._build_forced_tool_fallback_action(
        "list_files",
        "/workspace dizinini listele",
    )

    assert action == {
        "toolName": "list_files",
        "args": {"path": "/workspace"},
    }


def test_forced_tool_fallback_builds_python_action_from_code() -> None:
    agent = _build_bare_agent()

    action = agent._build_forced_tool_fallback_action(
        "python_action",
        "python çalıştır `print(123)`",
    )

    assert action == {
        "toolName": "python_action",
        "args": {"action": "execute", "code": "print(123)"},
    }


def test_extract_python_code_supports_code_before_python_calistir() -> None:
    agent = _build_bare_agent()

    assert agent._extract_python_code('print("hello") python çalıştır') == 'print("hello")'


def test_forced_tool_system_directive_requires_real_tool_call() -> None:
    agent = _build_bare_agent()

    directive = agent._build_forced_tool_system_directive("browser_action")

    assert "browser_action" in directive
    assert "Output only the real tool call." in directive


def test_route_obvious_tool_request_uses_normalized_browser_marker() -> None:
    agent = _build_bare_agent()
    agent.TOOLS = {"browser_action": object()}
    agent.state.context = {"last_user_turn_raw": "http://scaa.us siteyi aç"}
    agent.memory_store = None

    assert agent._route_obvious_tool_request("http://scaa.us siteyi aç") == "browser_action"
