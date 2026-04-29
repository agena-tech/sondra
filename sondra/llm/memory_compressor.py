from typing import Any


class MemoryCompressor:
    """
    Keeps prompt history within deterministic limits.
    It does not perform retrieval.
    """

    def __init__(
        self,
        max_images: int = 3,
        model_name: str | None = None,
        timeout: int | None = None,
        keep_recent_messages: int = 14,
        keep_system_messages: int = 4,
    ):
        self.max_images = max_images
        self.keep_recent_messages = keep_recent_messages
        self.keep_system_messages = keep_system_messages

    @staticmethod
    def _message_size(msg: dict[str, Any]) -> int:
        content = msg.get("content", "")
        if isinstance(content, str):
            return len(content)
        if isinstance(content, list):
            total = 0
            for item in content:
                if isinstance(item, dict):
                    total += len(str(item.get("text", "")))
                else:
                    total += len(str(item))
            return total
        return len(str(content))

    @staticmethod
    def _trim_message_images(
        msg: dict[str, Any],
        available_images: int,
    ) -> tuple[dict[str, Any] | None, int]:
        content = msg.get("content", "")
        if not isinstance(content, list):
            return dict(msg), 0

        used_images = 0
        trimmed: list[Any] = []
        for item in content:
            if not isinstance(item, dict):
                trimmed.append(item)
                continue
            if item.get("type") == "text":
                trimmed.append(item)
                continue
            if used_images >= max(0, int(available_images)):
                continue
            trimmed.append(item)
            used_images += 1

        if not trimmed:
            return None, 0

        result = dict(msg)
        result["content"] = trimmed
        return result, used_images

    def compress_history(
        self,
        messages: list[dict[str, Any]],
        max_chars: int = 12000,
        max_messages: int = 40,
    ) -> list[dict[str, Any]]:
        if not messages:
            return []

        system_msgs: list[dict[str, Any]] = []
        normal_msgs: list[dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") == "system":
                system_msgs.append(msg)
            else:
                normal_msgs.append(msg)

        system_msgs = system_msgs[-self.keep_system_messages :]
        recent_msgs = normal_msgs[-self.keep_recent_messages :]
        selected = (system_msgs + recent_msgs)[-max_messages:]

        total = 0
        result_reversed: list[dict[str, Any]] = []
        remaining_images = max(0, int(self.max_images))
        for msg in reversed(selected):
            prepared_msg, used_images = self._trim_message_images(msg, remaining_images)
            if not prepared_msg:
                continue
            size = self._message_size(prepared_msg)
            if result_reversed and (total + size) > max_chars:
                continue
            result_reversed.append(prepared_msg)
            total += size
            remaining_images = max(0, remaining_images - used_images)

        return list(reversed(result_reversed))
