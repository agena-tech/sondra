from typing import Any, ClassVar

from rich.text import Text
from textual.widgets import Static

from .base_renderer import BaseToolRenderer
from .registry import register_tool_renderer


@register_tool_renderer
class AdbExecRenderer(BaseToolRenderer):
    tool_name: ClassVar[str] = "adb_exec"
    css_classes: ClassVar[list[str]] = ["tool-call", "adb-tool"]

    @classmethod
    def render(cls, tool_data: dict[str, Any]) -> Static:
        text = Text()
        text.append("📱 Adb connection", style="#8a8a8a")

        css_classes = cls.get_css_classes(tool_data.get("status", "unknown"))
        return Static(text, classes=css_classes)
