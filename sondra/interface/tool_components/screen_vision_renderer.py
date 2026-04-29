from typing import Any, ClassVar

from rich.text import Text
from textual.widgets import Static

from .base_renderer import BaseToolRenderer
from .registry import register_tool_renderer


@register_tool_renderer
class ScreenVisionRenderer(BaseToolRenderer):
    tool_name: ClassVar[str] = "analyze_image"
    css_classes: ClassVar[list[str]] = ["tool-call", "screen-vision-tool"]

    @classmethod
    def render(cls, tool_data: dict[str, Any]) -> Static:
        args = tool_data.get("args", {}) or {}
        image_path = str(args.get("image_path", "") or "").strip()

        text = Text()
        text.append("🔍 Analyzing image ...", style="#8a8a8a")

        css_classes = cls.get_css_classes(tool_data.get("status", "unknown"))
        return Static(text, classes=css_classes)

