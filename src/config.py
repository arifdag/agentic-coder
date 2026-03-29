"""Configuration management for the LLM Agent Platform."""

import os
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class LLMConfig(BaseModel):
    """Configuration for LLM providers."""
    
    provider: str = Field(default="groq", description="LLM provider: groq or openrouter")
    model: str = Field(default="llama-3.3-70b-versatile", description="Model name")
    api_key: Optional[str] = Field(default=None, description="API key for the provider")
    temperature: float = Field(default=0.1, description="Generation temperature")
    max_tokens: int = Field(default=4096, description="Maximum tokens in response")
    
    @classmethod
    def from_env(cls, provider: Optional[str] = None) -> "LLMConfig":
        """Create configuration from environment variables."""
        provider = provider or os.getenv("LLM_PROVIDER", "groq")
        
        if provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            model = os.getenv("DEFAULT_MODEL", "llama-3.3-70b-versatile")
        elif provider == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY")
            model = os.getenv("DEFAULT_MODEL", "meta-llama/llama-3.1-70b-instruct:free")
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        return cls(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
        )


class SandboxConfig(BaseModel):
    """Configuration for the Docker sandbox."""
    
    image_name: str = Field(default="llm-agent-sandbox", description="Docker image name")
    timeout: int = Field(default=60, description="Execution timeout in seconds")
    memory_limit: str = Field(default="512m", description="Memory limit")
    cpu_limit: float = Field(default=1.0, description="CPU limit")
    network_disabled: bool = Field(default=True, description="Disable network access")
    
    @classmethod
    def from_env(cls) -> "SandboxConfig":
        """Create configuration from environment variables."""
        return cls(
            image_name=os.getenv("SANDBOX_IMAGE", "llm-agent-sandbox"),
            timeout=int(os.getenv("SANDBOX_TIMEOUT", "60")),
            memory_limit=os.getenv("SANDBOX_MEMORY_LIMIT", "512m"),
            cpu_limit=float(os.getenv("SANDBOX_CPU_LIMIT", "1.0")),
            network_disabled=os.getenv("SANDBOX_NETWORK_DISABLED", "true").lower() == "true",
        )


class SastConfig(BaseModel):
    """Configuration for the SAST gate."""

    enabled: bool = Field(default=True, description="Enable SAST scanning")
    semgrep_rules: str = Field(default="auto", description="Semgrep rule config")
    bandit_enabled: bool = Field(default=True, description="Enable Bandit scanning")
    timeout: int = Field(default=60, description="Tool timeout in seconds")

    @classmethod
    def from_env(cls) -> "SastConfig":
        return cls(
            enabled=os.getenv("SAST_ENABLED", "true").lower() == "true",
            semgrep_rules=os.getenv("SEMGREP_RULES", "auto"),
            bandit_enabled=os.getenv("BANDIT_ENABLED", "true").lower() == "true",
            timeout=int(os.getenv("SAST_TIMEOUT", "60")),
        )


class DependencyConfig(BaseModel):
    """Configuration for the dependency validation gate."""

    enabled: bool = Field(default=True, description="Enable dependency validation")
    pypi_timeout: int = Field(default=10, description="PyPI request timeout in seconds")

    @classmethod
    def from_env(cls) -> "DependencyConfig":
        return cls(
            enabled=os.getenv("DEPENDENCY_CHECK_ENABLED", "true").lower() == "true",
            pypi_timeout=int(os.getenv("PYPI_TIMEOUT", "10")),
        )


class JudgeConfig(BaseModel):
    """Configuration for the LLM-as-judge false-positive filter."""

    enabled: bool = Field(default=True, description="Enable LLM judge for SAST findings")
    provider: Optional[str] = Field(default=None, description="Override LLM provider for judge")
    model: Optional[str] = Field(default=None, description="Override model for judge")

    @classmethod
    def from_env(cls) -> "JudgeConfig":
        return cls(
            enabled=os.getenv("JUDGE_ENABLED", "true").lower() == "true",
            provider=os.getenv("JUDGE_PROVIDER"),
            model=os.getenv("JUDGE_MODEL"),
        )


class UITestConfig(BaseModel):
    """Configuration for the Playwright UI test sandbox."""

    enabled: bool = Field(default=True, description="Enable UI test generation")
    playwright_image: str = Field(
        default="llm-agent-playwright", description="Docker image for Playwright sandbox"
    )
    timeout: int = Field(default=120, description="Execution timeout in seconds")
    retry_budget: int = Field(default=5, description="Max repair iterations for UI tests")
    headless: bool = Field(default=True, description="Run browser in headless mode")
    network_enabled: bool = Field(default=True, description="Allow network access (needed for URL targets)")
    memory_limit: str = Field(default="1g", description="Container memory limit")

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


class PipelineConfig(BaseModel):
    """Configuration for the GDR pipeline."""
    
    max_retries: int = Field(default=3, description="Maximum repair iterations (k)")
    verbose: bool = Field(default=False, description="Enable verbose output")
    audit_log_dir: str = Field(default="audit_logs", description="Directory for audit logs")
    
    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Create configuration from environment variables."""
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
    ui_test: UITestConfig = Field(default_factory=UITestConfig.from_env)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig.from_env)
    
    @classmethod
    def load(cls, provider: Optional[str] = None) -> "Config":
        """Load configuration from environment."""
        return cls(
            llm=LLMConfig.from_env(provider),
            sandbox=SandboxConfig.from_env(),
            sast=SastConfig.from_env(),
            dependency=DependencyConfig.from_env(),
            judge=JudgeConfig.from_env(),
            ui_test=UITestConfig.from_env(),
            pipeline=PipelineConfig.from_env(),
        )


def get_llm(config: Optional[LLMConfig] = None):
    """Get an LLM instance based on configuration."""
    if config is None:
        config = LLMConfig.from_env()
    
    if not config.api_key:
        raise ValueError(
            f"API key not found for provider '{config.provider}'. "
            f"Please set the appropriate environment variable "
            f"(GROQ_API_KEY or OPENROUTER_API_KEY)."
        )
    
    if config.provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=config.model,
            api_key=config.api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
    elif config.provider == "openrouter":
        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_core.messages import BaseMessage, AIMessage
        import httpx
        
        class OpenRouterChat(BaseChatModel):
            """Simple OpenRouter chat model wrapper."""
            
            api_key: str
            model: str
            temperature: float = 0.1
            max_tokens: int = 4096
            
            @property
            def _llm_type(self) -> str:
                return "openrouter"
            
            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                from langchain_core.outputs import ChatGeneration, ChatResult
                
                formatted_messages = [
                    {"role": "user" if m.type == "human" else "assistant", "content": m.content}
                    for m in messages
                ]
                
                response = httpx.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": formatted_messages,
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens,
                    },
                    timeout=120.0,
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                
                return ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content=content))]
                )
        
        return OpenRouterChat(
            api_key=config.api_key,
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
    else:
        raise ValueError(f"Unsupported provider: {config.provider}")
