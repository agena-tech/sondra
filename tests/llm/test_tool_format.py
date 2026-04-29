import importlib.util
from pathlib import Path


def _load_llm_utils():
    utils_path = Path(__file__).resolve().parents[2] / "sondra" / "llm" / "utils.py"
    spec = importlib.util.spec_from_file_location("llm_utils_under_test", utils_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_legacy_browser_action_xml() -> None:
    utils = _load_llm_utils()

    parsed = utils.parse_tool_invocations(
        "<browser_action>\n"
        "<action>launch</action>\n"
        "<url>http://scaa.us</url>\n"
        "</browser_action>"
    )

    assert parsed == [
        {
            "toolName": "browser_action",
            "args": {"action": "launch", "url": "http://scaa.us"},
        }
    ]


def test_parse_canonical_browser_action_xml_still_works() -> None:
    utils = _load_llm_utils()

    parsed = utils.parse_tool_invocations(
        "<function=browser_action>\n"
        "<parameter=action>launch</parameter>\n"
        "<parameter=url>http://scaa.us</parameter>\n"
        "</function>"
    )

    assert parsed == [
        {
            "toolName": "browser_action",
            "args": {"action": "launch", "url": "http://scaa.us"},
        }
    ]
