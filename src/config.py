"""Configuration management for the LLM Agent Platform.

Supports multi-provider role-based LLM routing with fallback chains so the
pipeline can survive free-tier rate limits by rotating to a backup provider.
"""

import os
import threading as _threading
import time as _time
from typing import Any, ClassVar, Dict, List, Optional
from typing import Dict as _Dict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


# ─── Provider registry ──────────────────────────────────────────────────
#
# Each profile declares:
#   - api_type:     "groq" (native ChatGroq), or "openai_compat" (generic)
#   - base_url:     endpoint for OpenAI-compatible providers
#   - env_key:      env variable name that holds the API key
#   - default_model: model id to use when caller doesn't specify one
#
# Add a provider here and it becomes usable everywhere (coding role, judge
# role, fallback chains, etc.) with no other code changes.
# ────────────────────────────────────────────────────────────────────────

PROVIDER_PROFILES: Dict[str, Dict[str, Any]] = {
    "groq": {
        "api_type": "groq",
        "base_url": None,
        "env_key": "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile",
    },
    "gemini": {
        "api_type": "openai_compat",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "env_key": "GOOGLE_API_KEY",
        "default_model": "gemini-2.5-flash",
    },
    "cerebras": {
        "api_type": "openai_compat",
        "base_url": "https://api.cerebras.ai/v1",
        "env_key": "CEREBRAS_API_KEY",
        "default_model": "qwen-3-coder-480b",
    },
    "mistral": {
        "api_type": "openai_compat",
        "base_url": "https://api.mistral.ai/v1",
        "env_key": "MISTRAL_API_KEY",
        "default_model": "codestral-latest",
    },
    "github": {
        "api_type": "openai_compat",
        "base_url": "https://models.inference.ai.azure.com",
        "env_key": "GITHUB_TOKEN",
        "default_model": "gpt-4o-mini",
    },
    "sambanova": {
        "api_type": "openai_compat",
        "base_url": "https://api.sambanova.ai/v1",
        "env_key": "SAMBANOVA_API_KEY",
        "default_model": "Meta-Llama-3.3-70B-Instruct",
    },
    "openrouter": {
        "api_type": "openai_compat",
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
        "default_model": "qwen/qwen-2.5-coder-32b-instruct:free",
    },
    # Ollama Cloud (Pro / Max plans) -- https://ollama.com/cloud
    # Pro ($20/mo): 3 concurrent cloud models, 50x free usage, 5h session
    # + 7d weekly resets. Quota is measured in GPU-time, not tokens, so
    # long-context coding prompts are quite cheap relative to free tiers.
    "ollama": {
        "api_type": "openai_compat",
        "base_url": "https://ollama.com/v1",
        "env_key": "OLLAMA_API_KEY",
        "default_model": "glm-5.1:cloud",
    },
    # OpenCode Zen -- curated cloud inference gateway
    # OpenAI-compatible endpoint for Qwen, GLM, Kimi, MiniMax, Nemotron models.
    # GPT models use /responses, Claude uses /messages -- both unsupported.
    # API key from https://opencode.ai/auth
    "opencode": {
        "api_type": "openai_compat",
        "base_url": None,  # set via OPENCODE_BASE_URL env var
        "env_key": "OPENCODE_API_KEY",
        "default_model": None,  # caller must set model
    },
    "openai_compat": {
        "api_type": "openai_compat",
        "base_url": None,  # caller MUST set OPENAI_COMPAT_BASE_URL
        "env_key": "OPENAI_COMPAT_API_KEY",
        "default_model": None,  # caller MUST set model
    },
}


def _resolve_api_key(provider: str) -> Optional[str]:
    profile = PROVIDER_PROFILES.get(provider)
    if not profile:
        return None
    return os.getenv(profile["env_key"])


def _resolve_default_model(provider: str) -> Optional[str]:
    profile = PROVIDER_PROFILES.get(provider)
    if not profile:
        return None
    return profile.get("default_model")


def _resolve_base_url(provider: str) -> Optional[str]:
    profile = PROVIDER_PROFILES.get(provider)
    if not profile:
        return None
    base = profile.get("base_url")
    if provider == "openai_compat":
        return os.getenv("OPENAI_COMPAT_BASE_URL", base)
    if provider == "opencode":
        return os.getenv("OPENCODE_BASE_URL", base or "https://opencode.ai/zen/go/v1")
    return base


# ─── Config models ──────────────────────────────────────────────────────


class LLMConfig(BaseModel):
    """Configuration for a single LLM endpoint (one provider + one model)."""

    provider: str = Field(default="groq")
    model: str = Field(default="llama-3.3-70b-versatile")
    api_key: Optional[str] = Field(default=None)
    base_url: Optional[str] = Field(default=None)
    temperature: float = Field(default=0.1)
    max_tokens: int = Field(default=8192)

    @classmethod
    def from_env(cls, provider: Optional[str] = None) -> "LLMConfig":
        provider = provider or os.getenv("LLM_PROVIDER", "groq")

        if provider not in PROVIDER_PROFILES:
            raise ValueError(
                f"Unsupported provider: {provider}. "
                f"Known providers: {sorted(PROVIDER_PROFILES)}"
            )

        api_key = _resolve_api_key(provider)
        model = os.getenv("DEFAULT_MODEL") or _resolve_default_model(provider)
        base_url = _resolve_base_url(provider)

        return cls(
            provider=provider,
            model=model or "unknown",
            api_key=api_key,
            base_url=base_url,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "8192")),
        )

    @classmethod
    def from_spec(cls, spec: str, temperature: float = 0.1, max_tokens: int = 8192) -> "LLMConfig":
        """Build config from a 'provider:model' spec string.

        Examples:
            'gemini:gemini-2.5-flash'
            'groq:llama-3.3-70b-versatile'
            'cerebras' (uses default model for provider)
        """
        if ":" in spec:
            provider, model = spec.split(":", 1)
        else:
            provider, model = spec, None

        provider = provider.strip()
        if provider not in PROVIDER_PROFILES:
            raise ValueError(
                f"Unsupported provider '{provider}' in spec '{spec}'. "
                f"Known providers: {sorted(PROVIDER_PROFILES)}"
            )

        model = (model or "").strip() or _resolve_default_model(provider)
        if not model:
            raise ValueError(f"No model specified for provider '{provider}'")

        return cls(
            provider=provider,
            model=model,
            api_key=_resolve_api_key(provider),
            base_url=_resolve_base_url(provider),
            temperature=temperature,
            max_tokens=max_tokens,
        )


def _parse_chain(env_value: Optional[str]) -> List[str]:
    if not env_value:
        return []
    return [s.strip() for s in env_value.split(",") if s.strip()]


class RoleConfig(BaseModel):
    """Primary LLM + ordered fallbacks for a given role (coding or judge)."""

    primary: LLMConfig
    fallbacks: List[LLMConfig] = Field(default_factory=list)

    @property
    def all_endpoints(self) -> List[LLMConfig]:
        return [self.primary] + self.fallbacks

    def provenance(self) -> Dict[str, Any]:
        """Return a JSON-serializable description for audit logs/papers."""
        return {
            "primary": {"provider": self.primary.provider, "model": self.primary.model},
            "fallbacks": [{"provider": f.provider, "model": f.model} for f in self.fallbacks],
        }


class SandboxConfig(BaseModel):
    image_name: str = Field(default="llm-agent-sandbox")
    timeout: int = Field(default=60)
    memory_limit: str = Field(default="512m")
    cpu_limit: float = Field(default=1.0)
    network_disabled: bool = Field(default=True)

    @classmethod
    def from_env(cls) -> "SandboxConfig":
        return cls(
            image_name=os.getenv("SANDBOX_IMAGE", "llm-agent-sandbox"),
            timeout=int(os.getenv("SANDBOX_TIMEOUT", "60")),
            memory_limit=os.getenv("SANDBOX_MEMORY_LIMIT", "512m"),
            cpu_limit=float(os.getenv("SANDBOX_CPU_LIMIT", "1.0")),
            network_disabled=os.getenv("SANDBOX_NETWORK_DISABLED", "true").lower() == "true",
        )


class SastConfig(BaseModel):
    enabled: bool = Field(default=True)
    semgrep_rules: str = Field(default="auto")
    bandit_enabled: bool = Field(default=True)
    timeout: int = Field(default=60)

    @classmethod
    def from_env(cls) -> "SastConfig":
        return cls(
            enabled=os.getenv("SAST_ENABLED", "true").lower() == "true",
            semgrep_rules=os.getenv("SEMGREP_RULES", "auto"),
            bandit_enabled=os.getenv("BANDIT_ENABLED", "true").lower() == "true",
            timeout=int(os.getenv("SAST_TIMEOUT", "60")),
        )


class DependencyConfig(BaseModel):
    enabled: bool = Field(default=True)
    pypi_timeout: int = Field(default=10)

    @classmethod
    def from_env(cls) -> "DependencyConfig":
        return cls(
            enabled=os.getenv("DEPENDENCY_CHECK_ENABLED", "true").lower() == "true",
            pypi_timeout=int(os.getenv("PYPI_TIMEOUT", "10")),
        )


class JudgeConfig(BaseModel):
    enabled: bool = Field(default=True)
    provider: Optional[str] = Field(default=None)
    model: Optional[str] = Field(default=None)

    @classmethod
    def from_env(cls) -> "JudgeConfig":
        return cls(
            enabled=os.getenv("JUDGE_ENABLED", "true").lower() == "true",
            provider=os.getenv("JUDGE_PROVIDER"),
            model=os.getenv("JUDGE_MODEL"),
        )


class RelevanceConfig(BaseModel):
    """Anti-gaming gate: ensures generated tests actually exercise the
    function under test. Disabled by default to preserve baseline
    behaviour; enable for evaluations where benchmark gaming is a
    concern (e.g. ULT). See ``src/verification/relevance.py``."""

    enabled: bool = Field(default=False)
    source_module: str = Field(default="source_module")
    min_signals: int = Field(default=1)

    @classmethod
    def from_env(cls) -> "RelevanceConfig":
        return cls(
            enabled=os.getenv("RELEVANCE_GATE_ENABLED", "false").lower() == "true",
            source_module=os.getenv("RELEVANCE_SOURCE_MODULE", "source_module"),
            min_signals=int(os.getenv("RELEVANCE_MIN_SIGNALS", "1")),
        )


class UITestConfig(BaseModel):
    enabled: bool = Field(default=True)
    playwright_image: str = Field(default="llm-agent-playwright")
    timeout: int = Field(default=120)
    retry_budget: int = Field(default=5)
    headless: bool = Field(default=True)
    network_enabled: bool = Field(default=True)
    memory_limit: str = Field(default="1g")

    @classmethod
    def from_env(cls) -> "UITestConfig":
        return cls(
            enabled=os.getenv("UI_TEST_ENABLED", "true").lower() == "true",
            playwright_image=os.getenv("PLAYWRIGHT_IMAGE", "llm-agent-playwright"),
            timeout=int(os.getenv("UI_TEST_TIMEOUT", "120")),
            retry_budget=int(os.getenv("UI_TEST_RETRY_BUDGET", "5")),
            headless=os.getenv("UI_TEST_HEADLESS", "true").lower() == "true",
            network_enabled=os.getenv("UI_TEST_NETWORK", "true").lower() == "true",
            memory_limit=os.getenv("UI_TEST_MEMORY_LIMIT", "1g"),
        )


class JsSandboxConfig(BaseModel):
    enabled: bool = Field(default=True)
    image_name: str = Field(default="llm-agent-node")
    timeout: int = Field(default=60)
    memory_limit: str = Field(default="512m")
    network_disabled: bool = Field(default=True)

    @classmethod
    def from_env(cls) -> "JsSandboxConfig":
        return cls(
            enabled=os.getenv("JS_SANDBOX_ENABLED", "true").lower() == "true",
            image_name=os.getenv("JS_SANDBOX_IMAGE", "llm-agent-node"),
            timeout=int(os.getenv("JS_SANDBOX_TIMEOUT", "60")),
            memory_limit=os.getenv("JS_SANDBOX_MEMORY_LIMIT", "512m"),
            network_disabled=os.getenv("JS_SANDBOX_NETWORK_DISABLED", "true").lower() == "true",
        )


class ExplanationConfig(BaseModel):
    enabled: bool = Field(default=True)
    max_retries: int = Field(default=2)
    judge_enabled: bool = Field(default=True)
    complexity_check_enabled: bool = Field(default=True)

    @classmethod
    def from_env(cls) -> "ExplanationConfig":
        return cls(
            enabled=os.getenv("EXPLANATION_ENABLED", "true").lower() == "true",
            max_retries=int(os.getenv("EXPLANATION_MAX_RETRIES", "2")),
            judge_enabled=os.getenv("EXPLANATION_JUDGE_ENABLED", "true").lower() == "true",
            complexity_check_enabled=os.getenv("EXPLANATION_COMPLEXITY_CHECK", "true").lower()
            == "true",
        )


class EvalConfig(BaseModel):
    data_dir: str = Field(default="data/benchmarks")
    results_dir: str = Field(default="eval_results")
    max_cases: Optional[int] = None
    parallel: int = Field(default=1)

    quality_mode: str = Field(default="fast")
    mutation_max_mutants: int = Field(default=25)
    reliability_repeats: int = Field(default=0)

    execution_context: str = Field(default="auto")
    repo_setup: str = Field(default="auto")
    gate_policy: str = Field(default="strict")
    repo_pytest_timeout: int = Field(default=120)

    # TestGenEval official bridge settings
    testgeneval_repo_dir: str = Field(default="")
    testgeneval_namespace: str = Field(default="kdjain")
    testgeneval_timeout: int = Field(default=900)
    testgeneval_num_processes: int = Field(default=1)
    testgeneval_skip_mutation: bool = Field(default=False)

    _EXECUTION_CONTEXTS: ClassVar[tuple[str, ...]] = ("auto", "single-file", "repo")
    _REPO_SETUPS: ClassVar[tuple[str, ...]] = ("auto", "prepare", "reuse", "off")
    _GATE_POLICIES: ClassVar[tuple[str, ...]] = ("strict", "balanced")

    _QUALITY_MODES: ClassVar[tuple[str, ...]] = ("off", "fast", "full")

    @classmethod
    def _normalize_quality_mode(cls, v: str) -> str:
        value = str(v or "fast").lower()
        if value not in cls._QUALITY_MODES:
            return "fast"
        return value

    def model_post_init(self, __context: Any) -> None:
        self.quality_mode = self._normalize_quality_mode(self.quality_mode)
        ec = str(self.execution_context or "auto").lower()
        self.execution_context = ec if ec in self._EXECUTION_CONTEXTS else "auto"
        rs = str(self.repo_setup or "auto").lower()
        self.repo_setup = rs if rs in self._REPO_SETUPS else "auto"
        gp = str(self.gate_policy or "strict").lower()
        self.gate_policy = gp if gp in self._GATE_POLICIES else "strict"
        if self.mutation_max_mutants < 0:
            self.mutation_max_mutants = 0
        if self.reliability_repeats < 0:
            self.reliability_repeats = 0
        if self.repo_pytest_timeout <= 0:
            self.repo_pytest_timeout = 120
        if self.testgeneval_timeout <= 0:
            self.testgeneval_timeout = 900
        if self.testgeneval_num_processes <= 0:
            self.testgeneval_num_processes = 1

    @classmethod
    def from_env(cls) -> "EvalConfig":
        max_cases_raw = os.getenv("EVAL_MAX_CASES")
        return cls(
            data_dir=os.getenv("EVAL_DATA_DIR", "data/benchmarks"),
            results_dir=os.getenv("EVAL_RESULTS_DIR", "eval_results"),
            max_cases=int(max_cases_raw) if max_cases_raw else None,
            parallel=int(os.getenv("EVAL_PARALLEL", "1")),
            quality_mode=os.getenv("EVAL_QUALITY_MODE", "fast"),
            mutation_max_mutants=int(os.getenv("EVAL_MUTATION_MAX_MUTANTS", "25")),
            reliability_repeats=int(os.getenv("EVAL_RELIABILITY_REPEATS", "0")),
            execution_context=os.getenv("EVAL_EXECUTION_CONTEXT", "auto"),
            repo_setup=os.getenv("EVAL_REPO_SETUP", "auto"),
            gate_policy=os.getenv("EVAL_GATE_POLICY", "strict"),
            repo_pytest_timeout=int(os.getenv("EVAL_REPO_PYTEST_TIMEOUT", "120")),
            testgeneval_repo_dir=os.getenv("TESTGENEVAL_REPO_DIR", ""),
            testgeneval_namespace=os.getenv("TESTGENEVAL_NAMESPACE", "kdjain"),
            testgeneval_timeout=int(os.getenv("TESTGENEVAL_TIMEOUT", "900")),
            testgeneval_num_processes=int(os.getenv("TESTGENEVAL_NUM_PROCESSES", "1")),
            testgeneval_skip_mutation=(
                os.getenv("TESTGENEVAL_SKIP_MUTATION", "false").lower() == "true"
            ),
        )


class PipelineConfig(BaseModel):
    max_retries: int = Field(default=3)
    verbose: bool = Field(default=False)
    audit_log_dir: str = Field(default="audit_logs")

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        return cls(
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            verbose=os.getenv("VERBOSE", "false").lower() == "true",
            audit_log_dir=os.getenv("AUDIT_LOG_DIR", "audit_logs"),
        )


class Config(BaseModel):
    """Main configuration container."""

    llm: LLMConfig = Field(default_factory=LLMConfig.from_env)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig.from_env)
    sast: SastConfig = Field(default_factory=SastConfig.from_env)
    dependency: DependencyConfig = Field(default_factory=DependencyConfig.from_env)
    judge: JudgeConfig = Field(default_factory=JudgeConfig.from_env)
    relevance: RelevanceConfig = Field(default_factory=RelevanceConfig.from_env)
    ui_test: UITestConfig = Field(default_factory=UITestConfig.from_env)
    js_sandbox: JsSandboxConfig = Field(default_factory=JsSandboxConfig.from_env)
    explanation: ExplanationConfig = Field(default_factory=ExplanationConfig.from_env)
    evaluation: EvalConfig = Field(default_factory=EvalConfig.from_env)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig.from_env)

    # Role-based routing (populated by load()); these may be None if only the
    # legacy single-provider 'llm' field is configured.
    coding_role: Optional[RoleConfig] = None
    judge_role: Optional[RoleConfig] = None

    @classmethod
    def load(cls, provider: Optional[str] = None) -> "Config":
        coding_role, judge_role, llm_cfg = _build_role_configs(provider)

        return cls(
            llm=llm_cfg,
            sandbox=SandboxConfig.from_env(),
            sast=SastConfig.from_env(),
            dependency=DependencyConfig.from_env(),
            judge=JudgeConfig.from_env(),
            relevance=RelevanceConfig.from_env(),
            ui_test=UITestConfig.from_env(),
            js_sandbox=JsSandboxConfig.from_env(),
            explanation=ExplanationConfig.from_env(),
            evaluation=EvalConfig.from_env(),
            pipeline=PipelineConfig.from_env(),
            coding_role=coding_role,
            judge_role=judge_role,
        )


def _build_role_configs(
    provider_override: Optional[str] = None,
) -> tuple[Optional[RoleConfig], Optional[RoleConfig], LLMConfig]:
    """Resolve coding/judge role configs from env, with sensible fallbacks.

    Priority order for the coding role's primary:
        1. --provider CLI override
        2. CODING_PROVIDER / CODING_MODEL
        3. LLM_PROVIDER / DEFAULT_MODEL (legacy single-provider path)
    """
    # Legacy single-provider compat. For an explicit CLI provider override,
    # use the provider profile's default model instead of DEFAULT_MODEL from
    # .env, because DEFAULT_MODEL may belong to a different provider.
    if provider_override:
        legacy_llm = LLMConfig.from_spec(provider_override)
    else:
        legacy_provider = os.getenv("LLM_PROVIDER", "groq")
        legacy_llm = LLMConfig.from_env(legacy_provider)

    def _role_from_env(
        primary_prov_key: str, primary_model_key: str, fallback_key: str
    ) -> Optional[RoleConfig]:
        provider = os.getenv(primary_prov_key)
        if not provider:
            return None  # role not configured via new-style env
        model = os.getenv(primary_model_key) or _resolve_default_model(provider)
        if not model:
            return None
        primary = LLMConfig.from_spec(f"{provider}:{model}")

        fallback_specs = _parse_chain(os.getenv(fallback_key))
        fallbacks: List[LLMConfig] = []
        for spec in fallback_specs:
            try:
                cfg = LLMConfig.from_spec(spec)
                # Skip fallbacks without an API key so they don't crash at runtime
                if cfg.api_key:
                    fallbacks.append(cfg)
            except Exception:
                continue

        return RoleConfig(primary=primary, fallbacks=fallbacks)

    if provider_override:
        fallbacks: List[LLMConfig] = []
        for spec in _parse_chain(os.getenv("CODING_FALLBACKS")):
            try:
                cfg = LLMConfig.from_spec(spec)
                same_endpoint = (
                    cfg.provider == legacy_llm.provider and cfg.model == legacy_llm.model
                )
                if cfg.api_key and not same_endpoint:
                    fallbacks.append(cfg)
            except Exception:
                continue
        coding_role = RoleConfig(primary=legacy_llm, fallbacks=fallbacks)
    else:
        coding_role = _role_from_env("CODING_PROVIDER", "CODING_MODEL", "CODING_FALLBACKS")
    judge_role = _role_from_env("JUDGE_PROVIDER", "JUDGE_MODEL", "JUDGE_FALLBACKS")

    # If new-style env isn't set, synthesize a one-endpoint RoleConfig from
    # the legacy llm config so downstream code can always use role-based API.
    if coding_role is None and legacy_llm.api_key:
        coding_role = RoleConfig(primary=legacy_llm, fallbacks=[])

    return coding_role, judge_role, legacy_llm


# ─── LLM factory & fallback chain ───────────────────────────────────────


def get_llm(config: Optional[LLMConfig] = None):
    """Get an LLM instance for a single endpoint."""
    if config is None:
        config = LLMConfig.from_env()

    if not config.api_key:
        profile = PROVIDER_PROFILES.get(config.provider, {})
        env_key = profile.get("env_key", "API_KEY")
        raise ValueError(
            f"API key not found for provider '{config.provider}'. "
            f"Please set {env_key} in your environment."
        )

    profile = PROVIDER_PROFILES.get(config.provider)
    if not profile:
        raise ValueError(f"Unsupported provider: {config.provider}")

    api_type = profile["api_type"]

    if api_type == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=config.model,
            api_key=config.api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    if api_type == "openai_compat":
        base_url = config.base_url or profile.get("base_url")
        if not base_url:
            raise ValueError(
                f"Provider '{config.provider}' requires base_url " f"(set OPENAI_COMPAT_BASE_URL)"
            )
        return _build_openai_compat_chat(
            api_key=config.api_key,
            base_url=base_url,
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            provider_tag=config.provider,
        )

    raise ValueError(f"Unknown api_type '{api_type}' for provider '{config.provider}'")


def _build_openai_compat_chat(
    api_key: str, base_url: str, model: str, temperature: float, max_tokens: int, provider_tag: str
):
    """Generic OpenAI-compatible chat model (works for Gemini, Cerebras,
    Mistral, SambaNova, OpenRouter, GitHub Models, custom self-hosted, ...)."""
    import httpx
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    class OpenAICompatChat(BaseChatModel):
        api_key: str
        base_url: str
        model: str
        temperature: float = 0.1
        max_tokens: int = 8192
        provider_tag: str = "openai_compat"

        @property
        def _llm_type(self) -> str:
            return f"openai_compat:{self.provider_tag}"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            formatted: List[Dict[str, str]] = []
            for m in messages:
                role = "user"
                if m.type == "system":
                    role = "system"
                elif m.type == "ai":
                    role = "assistant"
                elif m.type == "human":
                    role = "user"
                formatted.append({"role": role, "content": m.content})

            url = self.base_url.rstrip("/") + "/chat/completions"
            # Split connect vs read timeout. Ollama Cloud and other large
            # cloud models can take 2-3 min to stream a full response for
            # long coding prompts (up to ``max_tokens=8192``), so the read
            # timeout has to be generous. Connect stays short so unreachable
            # endpoints fail fast and rotate to the next fallback quickly.
            read_timeout = float(os.getenv("LLM_READ_TIMEOUT", "300"))
            connect_timeout = float(os.getenv("LLM_CONNECT_TIMEOUT", "15"))
            response = httpx.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": formatted,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                },
                timeout=httpx.Timeout(read_timeout, connect=connect_timeout),
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]

            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    return OpenAICompatChat(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        provider_tag=provider_tag,
    )


# ─── Per-provider cooldown (module-global, shared across cases) ────────
#
# When a provider returns 429/5xx we stop hitting it for a short window so
# other benchmark cases don't burn CPU spamming an exhausted endpoint. Keys
# are "provider:model"; values are the earliest wall-clock time (epoch
# seconds) when that endpoint may be tried again.
# ────────────────────────────────────────────────────────────────────────

_PROVIDER_COOLDOWN: _Dict[str, float] = {}
_COOLDOWN_LOCK = _threading.Lock()

# Per-error-class default cooldowns (seconds)
_COOLDOWN_RATE_LIMIT = 45.0  # 429 with no Retry-After -> wait ~1 min
_COOLDOWN_SERVER = 20.0  # 5xx / timeout
_COOLDOWN_AUTH = 3600.0  # 401/403 -> don't retry this session
_COOLDOWN_NOT_FOUND = 3600.0  # 404 (wrong model id) -> don't retry


def _cooldown_key(provider: str, model: str) -> str:
    return f"{provider}:{model}"


def _get_cooldown(key: str) -> float:
    """Return remaining cooldown seconds (<=0 means ready)."""
    with _COOLDOWN_LOCK:
        deadline = _PROVIDER_COOLDOWN.get(key, 0.0)
    return max(0.0, deadline - _time.time())


def _set_cooldown(key: str, seconds: float) -> None:
    with _COOLDOWN_LOCK:
        now = _time.time()
        existing = _PROVIDER_COOLDOWN.get(key, 0.0)
        _PROVIDER_COOLDOWN[key] = max(existing, now + seconds)


def _extract_retry_after(exc: BaseException) -> Optional[float]:
    """Pull Retry-After (seconds) from an httpx.HTTPStatusError if present."""
    import httpx

    if not isinstance(exc, httpx.HTTPStatusError):
        return None
    header = exc.response.headers.get("Retry-After") or exc.response.headers.get("retry-after")
    if not header:
        return None
    try:
        return max(0.0, float(header))
    except ValueError:
        return None


def _classify_and_cooldown(cfg: "LLMConfig", exc: BaseException) -> str:
    """Record the right cooldown for this error class and return a tag.

    Returns one of: 'rate_limit', 'server', 'auth', 'not_found', 'other'.
    """
    import httpx

    key = _cooldown_key(cfg.provider, cfg.model)

    status = None
    if isinstance(exc, httpx.HTTPStatusError):
        status = getattr(exc.response, "status_code", None)

    msg = str(exc).lower()

    # 413 / context-window overflows must be detected BEFORE the rate-limit
    # heuristic because some providers (e.g. Groq) return
    # `"code": "rate_limit_exceeded"` inside a 413 body when a single request
    # exceeds per-model TPM caps. Those failures will recur for the SAME
    # prompt no matter how long we wait, so we park the endpoint for the
    # session and let the fallback chain route to a bigger-context model.
    if (
        status == 413
        or "request too large" in msg
        or "too large for model" in msg
        or "context_length_exceeded" in msg
        or "context length" in msg
        or "maximum context" in msg
        or "string too long" in msg
    ):
        _set_cooldown(key, _COOLDOWN_AUTH)  # effectively session-permanent
        return "too_large"
    if status == 429 or "rate" in msg or "quota" in msg or "429" in msg:
        retry_after = _extract_retry_after(exc)
        _set_cooldown(key, retry_after if retry_after else _COOLDOWN_RATE_LIMIT)
        return "rate_limit"
    is_timeout_exc = isinstance(
        exc,
        (
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.NetworkError,
            httpx.RemoteProtocolError,
        ),
    )
    if (
        status in (500, 502, 503, 504)
        or "service_unavailable" in msg
        or "timeout" in msg
        or "timed out" in msg
        or is_timeout_exc
    ):
        _set_cooldown(key, _COOLDOWN_SERVER)
        return "server"
    if status in (401, 403) or "unauthorized" in msg:
        _set_cooldown(key, _COOLDOWN_AUTH)
        return "auth"
    if (
        status == 404
        or "model_not_found" in msg
        or "does not exist" in msg
        or "no endpoints found" in msg
    ):
        _set_cooldown(key, _COOLDOWN_NOT_FOUND)
        return "not_found"
    return "other"


def _should_rotate(exc: BaseException) -> bool:
    """Return True if the fallback chain should rotate to the next endpoint.

    Rotates on:
      - 429 / quota / rate-limit
      - 401 / 403 (bad/expired key for this provider)
      - 404 (model id not available on this account/tier)
      - 5xx (provider down)
      - httpx timeouts (read/connect/pool) -- treated like a transient
        server problem because cloud models can cold-start unpredictably.
    Does NOT rotate on genuine client-side bugs like 400 with "invalid request".
    """
    import httpx

    # Transient network/connection problems: rotate to next fallback so a
    # single stalled provider doesn't fail the whole case.
    if isinstance(
        exc,
        (
            httpx.TimeoutException,
            httpx.ReadTimeout,
            httpx.ConnectTimeout,
            httpx.WriteTimeout,
            httpx.PoolTimeout,
            httpx.ConnectError,
            httpx.NetworkError,
            httpx.RemoteProtocolError,
        ),
    ):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = getattr(exc.response, "status_code", None)
        if status in (401, 403, 404, 408, 413, 429, 500, 502, 503, 504):
            return True

    text = str(exc).lower()
    return any(
        kw in text
        for kw in (
            "rate limit",
            "ratelimit",
            "rate_limit",
            "quota",
            "too many requests",
            "429",
            "resource_exhausted",
            "service_unavailable",
            "model_not_found",
            "model not found",
            "does not exist or you do not have access",
            "no endpoints found",
            "unauthorized",
            "request too large",
            "too large for model",
            "context_length_exceeded",
            "context length",
            "maximum context",
            "string too long",
            # Timeout phrasing seen in the wild across httpx/requests/stdlib
            "read operation timed out",
            "read timed out",
            "connection timed out",
            "timeout",
            # DNS / name-resolution and TLS handshake glitches -- always
            # rotate-eligible (paired with intra-endpoint transient retry).
            "getaddrinfo",
            "name or service not known",
            "temporary failure in name resolution",
            "ssl: wrong_version_number",
            "wrong_version_number",
            "ssl handshake",
        )
    )


# Backwards-compatible alias used by tests
_is_rate_limit_error = _should_rotate


def _is_transient_retryable(exc: BaseException) -> bool:
    """Return True if ``exc`` is the kind of transient error worth retrying
    on the *same* endpoint after a short backoff (instead of rotating to
    a different provider).

    Transient = expected to clear within a few seconds:
      - httpx network/timeout exceptions (ReadTimeout, ConnectError, ...)
      - HTTP 5xx (502/503/504 + 500 once)
      - "server disconnected without sending a response"

    Explicitly NOT transient (rotate immediately):
      - 401/403 (auth)         -- retrying won't fix a bad key
      - 404 (model_not_found)  -- model doesn't exist on this account
      - 413 (too large)        -- prompt won't shrink on retry
      - 429 (rate limit)       -- handled by the cooldown mechanism
    """
    import httpx

    if isinstance(
        exc,
        (
            httpx.ReadTimeout,
            httpx.ConnectTimeout,
            httpx.WriteTimeout,
            httpx.PoolTimeout,
            httpx.ConnectError,
            httpx.RemoteProtocolError,
            httpx.NetworkError,
        ),
    ):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = getattr(exc.response, "status_code", None)
        if status in (500, 502, 503, 504):
            return True
        return False  # other HTTP errors are NOT transient

    text = str(exc).lower()
    return any(
        kw in text
        for kw in (
            "server disconnected",
            "connection reset",
            "remote protocol error",
            "service unavailable",
            "read operation timed out",
            "read timed out",
            "connection timed out",
            # DNS / name-resolution failures (Windows: "getaddrinfo failed";
            # Linux: "name or service not known"). httpx wraps these into
            # ConnectError, but bare ``socket.gaierror`` can leak through
            # custom HTTP plumbing in some providers.
            "getaddrinfo",
            "name or service not known",
            "temporary failure in name resolution",
            # TLS handshake glitches -- usually transient on retry
            "ssl: wrong_version_number",
            "wrong_version_number",
            "ssl handshake",
        )
    )


def get_llm_with_fallback(role: RoleConfig):
    """Return an LLM instance that automatically rotates through fallbacks
    when the primary hits a rate limit / quota error.

    If ``role`` has no fallbacks, returns the primary LLM directly.
    """
    endpoints = role.all_endpoints
    if not endpoints:
        raise ValueError("RoleConfig has no endpoints")
    if len(endpoints) == 1:
        return get_llm(endpoints[0])

    from langchain_core.language_models.chat_models import BaseChatModel

    built = [(cfg, get_llm(cfg)) for cfg in endpoints if cfg.api_key]
    if not built:
        raise ValueError(
            "No endpoints in the RoleConfig have API keys set. "
            "Configure at least one of: "
            + ", ".join(PROVIDER_PROFILES[c.provider]["env_key"] for c in endpoints)
        )

    class FallbackChatModel(BaseChatModel):
        model_config = {"arbitrary_types_allowed": True}

        endpoints_: list = []

        @property
        def _llm_type(self) -> str:
            return "fallback_chain"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            import logging

            log = logging.getLogger(__name__)

            last_exc: Optional[BaseException] = None
            max_wait_cycles = 3  # at most 3 full sleeps before giving up

            # Same-endpoint retry config. Defaults: retry transient errors
            # twice with 3s -> 6s exponential backoff before rotating to
            # a fallback. This keeps cross-model benchmark data
            # uncontaminated when the primary has a brief 503/timeout
            # blip but is otherwise healthy.
            transient_retries = max(0, int(os.getenv("LLM_TRANSIENT_RETRIES", "2")))
            backoff_base = max(0.0, float(os.getenv("LLM_TRANSIENT_BACKOFF", "3.0")))

            for cycle in range(max_wait_cycles):
                # First pass: try any endpoint that isn't cooling down.
                for cfg, llm in self.endpoints_:
                    key = _cooldown_key(cfg.provider, cfg.model)
                    remaining = _get_cooldown(key)
                    if remaining > 0:
                        continue  # skip cooling endpoint

                    # ─── Same-endpoint retry loop (transient errors only) ───
                    final_exc: Optional[BaseException] = None
                    for attempt in range(transient_retries + 1):
                        try:
                            return llm._generate(
                                messages, stop=stop, run_manager=run_manager, **kwargs
                            )
                        except Exception as exc:  # noqa: BLE001
                            final_exc = exc
                            if attempt < transient_retries and _is_transient_retryable(exc):
                                wait_s = backoff_base * (2**attempt)
                                msg = str(exc)
                                if len(msg) > 100:
                                    msg = msg[:97] + "..."
                                log.warning(
                                    "Transient error on %s:%s "
                                    "(attempt %d/%d), retrying in %.1fs: %s",
                                    cfg.provider,
                                    cfg.model,
                                    attempt + 1,
                                    transient_retries + 1,
                                    wait_s,
                                    msg,
                                )
                                _time.sleep(wait_s)
                                continue
                            break  # not transient, or retries exhausted

                    # All same-endpoint attempts done. Decide rotate-or-raise.
                    last_exc = final_exc
                    if final_exc is not None and _should_rotate(final_exc):
                        tag = _classify_and_cooldown(cfg, final_exc)
                        wait = _get_cooldown(key)
                        msg = str(final_exc)
                        if len(msg) > 120:
                            msg = msg[:117] + "..."
                        log.warning(
                            "Rotating past %s:%s [%s, cooldown=%.0fs] %s",
                            cfg.provider,
                            cfg.model,
                            tag,
                            wait,
                            msg,
                        )
                        continue
                    if final_exc is not None:
                        raise final_exc

                # All endpoints are cooling (or we just 429'd them all this
                # cycle). Sleep until the earliest one recovers, then retry.
                deadlines = []
                for cfg, _ in self.endpoints_:
                    key = _cooldown_key(cfg.provider, cfg.model)
                    r = _get_cooldown(key)
                    if 0 < r < _COOLDOWN_AUTH:  # ignore auth/404 "permanent"
                        deadlines.append(r)
                if not deadlines:
                    break  # nothing recoverable; bail
                sleep_for = min(deadlines) + 0.5
                log.warning(
                    "All coding/judge endpoints cooling; sleeping %.0fs then retrying",
                    sleep_for,
                )
                _time.sleep(min(sleep_for, 90.0))  # cap at 90s per cycle

            assert last_exc is not None
            raise last_exc

    instance = FallbackChatModel()
    instance.endpoints_ = built
    return instance


def get_role_llm(config: Config, role: str):
    """Convenience: get the LLM for 'coding' or 'judge' from a Config.

    Falls back to config.llm if the role-based config is not set.
    """
    if role == "coding":
        rc = config.coding_role
    elif role == "judge":
        # When judge provider isn't explicitly configured, reuse the coding
        # role so both use the same primary. This preserves the legacy
        # behavior (judge uses same LLM as generation) while still allowing
        # a dedicated judge chain when JUDGE_PROVIDER is set.
        rc = config.judge_role or config.coding_role
    else:
        raise ValueError(f"Unknown role '{role}'")

    if rc is None:
        return get_llm(config.llm)
    return get_llm_with_fallback(rc)
