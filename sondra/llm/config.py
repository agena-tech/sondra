from sondra.config import Config
from sondra.config.config import resolve_llm_config
from sondra.llm.utils import resolve_sondra_model


class LLMConfig:
    def __init__(
        self,
        model_name: str | None = None,
        enable_prompt_caching: bool = True,
        skills: list[str] | None = None,
        timeout: int | None = None,
        scan_mode: str = "general",
        scan_level: str = "standard",
        interactive: bool = False,
        fixed_agents: int | None = None,
        retry_attempts: int | None = None,
    ):
        resolved_model, self.api_key, self.api_base = resolve_llm_config()
        self.model_name = model_name or resolved_model

        if not self.model_name:
            raise ValueError("SONDRA_LLM environment variable must be set and not empty")

        api_model, canonical = resolve_sondra_model(self.model_name)
        self.litellm_model: str = api_model or self.model_name
        self.canonical_model: str = canonical or self.model_name

        self.enable_prompt_caching = enable_prompt_caching
        self.skills = skills or []

        self.timeout = timeout or int(Config.get("llm_timeout") or "300")

        normalized_mode = (scan_mode or "general").lower()
        normalized_level = (scan_level or "standard").lower()

        if normalized_mode in {"quick", "standard", "deep"}:
            normalized_level = normalized_mode
            normalized_mode = "general"

        if normalized_mode == "special":
            normalized_mode = "general"

        if normalized_mode not in {"osint", "general", "adb"}:
            normalized_mode = "general"

        if normalized_level not in {"quick", "standard", "deep"}:
            normalized_level = "standard"

        self.scan_mode = normalized_mode
        self.scan_level = normalized_level
        self.effective_scan_mode = self.scan_mode
        self.interactive = interactive
        self.fixed_agents = fixed_agents
        self.retry_attempts = max(0, int(retry_attempts)) if retry_attempts is not None else None
