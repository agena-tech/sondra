import json
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any


_SIGNAL_TEXT_REPLACEMENTS = (
    ("\u00c3\u00a7", "c"),
    ("\u00c4\u0178", "g"),
    ("\u00c4\u00b1", "i"),
    ("\u00c3\u00b6", "o"),
    ("\u00c5\u0178", "s"),
    ("\u00c3\u00bc", "u"),
    ("\u00c3\u00a2", "a"),
    ("\u00c3\u00ae", "i"),
    ("\u00c3\u00bb", "u"),
    ("\u00e2\u20ac\u2122", "'"),
    ("\u2019", "'"),
    ("`", "'"),
)

_SIGNAL_TEXT_TRANSLATION = str.maketrans(
    {
        "\u00e7": "c",
        "\u011f": "g",
        "\u0131": "i",
        "\u00f6": "o",
        "\u015f": "s",
        "\u00fc": "u",
        "\u00e2": "a",
        "\u00ee": "i",
        "\u00fb": "u",
    }
)


def normalize_signal_text(text: str) -> str:
    value = str(text or "").strip().lower()
    if not value:
        return ""
    for source, target in _SIGNAL_TEXT_REPLACEMENTS:
        value = value.replace(source, target)
    value = value.translate(_SIGNAL_TEXT_TRANSLATION)
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii")
    value = value.replace("'", " ")
    return " ".join(value.split())


def _normalize_signal_data(value: Any) -> Any:
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = normalize_signal_text(key) if isinstance(key, str) else key
            normalized[normalized_key] = _normalize_signal_data(item)
        return normalized
    if isinstance(value, list):
        return [_normalize_signal_data(item) for item in value]
    if isinstance(value, str):
        return normalize_signal_text(value)
    return value


class MemorySignalCatalog:
    def __init__(self, base_dir: Path | None = None):
        root = base_dir or (Path(__file__).resolve().parent / "memory_signals")
        self.base_dir = Path(root)
        self._data: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        loaded: dict[str, Any] = {}
        for path in sorted(self.base_dir.glob("*.json")):
            loaded[path.stem] = _normalize_signal_data(json.loads(path.read_text(encoding="utf-8")))
        self._data = loaded

    def get(self, file_name: str, *path: str, default: Any = None) -> Any:
        current = self._data.get(str(file_name or "").strip(), default)
        if current is default:
            return default
        for key in path:
            if not isinstance(current, dict):
                return default
            current = current.get(key, default)
            if current is default:
                return default
        return current

    def get_list(self, file_name: str, *path: str) -> list[Any]:
        value = self.get(file_name, *path, default=[])
        return list(value) if isinstance(value, list) else []

    def get_mapping(self, file_name: str, *path: str) -> dict[str, Any]:
        value = self.get(file_name, *path, default={})
        return dict(value) if isinstance(value, dict) else {}

    def get_tuple(self, file_name: str, *path: str) -> tuple[Any, ...]:
        return tuple(self.get_list(file_name, *path))

    def get_value(self, file_name: str, *path: str, default: Any = None) -> Any:
        return self.get(file_name, *path, default=default)


@lru_cache(maxsize=1)
def get_memory_signal_catalog() -> MemorySignalCatalog:
    return MemorySignalCatalog()
