"""Tests for multi-provider routing, JSON hardening, and fallback chains."""

import os
from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.config import (
    _PROVIDER_COOLDOWN,
    PROVIDER_PROFILES,
    LLMConfig,
    RoleConfig,
    _build_role_configs,
    _classify_and_cooldown,
    _cooldown_key,
    _get_cooldown,
    _is_rate_limit_error,
    get_llm_with_fallback,
)
from src.verification.judge import SastJudge, _extract_json


class TestProviderRegistry:
    def test_all_recommended_providers_registered(self):
        for name in (
            "groq",
            "gemini",
            "cerebras",
            "mistral",
            "github",
            "sambanova",
            "openrouter",
            "ollama",
            "openai_compat",
        ):
            assert name in PROVIDER_PROFILES, f"missing provider: {name}"

    def test_ollama_profile_points_at_cloud_endpoint(self):
        prof = PROVIDER_PROFILES["ollama"]
        assert prof["api_type"] == "openai_compat"
        assert prof["base_url"] == "https://ollama.com/v1"
        assert prof["env_key"] == "OLLAMA_API_KEY"
        assert "cloud" in prof["default_model"]

    def test_llm_config_from_spec_parses_ollama_model(self):
        os.environ["OLLAMA_API_KEY"] = "fake"
        try:
            cfg = LLMConfig.from_spec("ollama:qwen3-coder:480b-cloud")
            assert cfg.provider == "ollama"
            assert cfg.model == "qwen3-coder:480b-cloud"
            assert cfg.api_key == "fake"
            assert cfg.base_url == "https://ollama.com/v1"
        finally:
            del os.environ["OLLAMA_API_KEY"]

    def test_profile_has_required_fields(self):
        for name, profile in PROVIDER_PROFILES.items():
            assert "api_type" in profile
            assert "env_key" in profile
            assert "default_model" in profile or name == "openai_compat"

    def test_llm_config_from_spec_parses_provider_colon_model(self):
        os.environ["GOOGLE_API_KEY"] = "fake"
        try:
            cfg = LLMConfig.from_spec("gemini:gemini-2.5-flash")
            assert cfg.provider == "gemini"
            assert cfg.model == "gemini-2.5-flash"
            assert cfg.api_key == "fake"
            assert cfg.base_url and "generativelanguage" in cfg.base_url
        finally:
            del os.environ["GOOGLE_API_KEY"]

    def test_llm_config_from_spec_uses_default_model(self):
        os.environ["CEREBRAS_API_KEY"] = "fake"
        try:
            cfg = LLMConfig.from_spec("cerebras")
            assert cfg.provider == "cerebras"
            assert cfg.model == PROVIDER_PROFILES["cerebras"]["default_model"]
        finally:
            del os.environ["CEREBRAS_API_KEY"]

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError):
            LLMConfig.from_spec("no-such-provider:foo")


class TestRoleConfig:
    def test_build_role_configs_from_env(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-google")
        monkeypatch.setenv("GROQ_API_KEY", "fake-groq")
        monkeypatch.setenv("CODING_PROVIDER", "gemini")
        monkeypatch.setenv("CODING_MODEL", "gemini-2.5-flash")
        monkeypatch.setenv("CODING_FALLBACKS", "groq:llama-3.3-70b-versatile")

        coding, judge, _ = _build_role_configs()
        assert coding is not None
        assert coding.primary.provider == "gemini"
        assert len(coding.fallbacks) == 1
        assert coding.fallbacks[0].provider == "groq"

    def test_provider_override_takes_precedence_over_coding_role(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_API_KEY", "fake-ollama")
        monkeypatch.setenv("OPENCODE_API_KEY", "fake-opencode")
        monkeypatch.setenv("CODING_PROVIDER", "opencode")
        monkeypatch.setenv("CODING_MODEL", "deepseek-v4-pro")
        monkeypatch.setenv("DEFAULT_MODEL", "llama-3.3-70b-versatile")

        coding, _, llm = _build_role_configs(provider_override="ollama")

        assert llm.provider == "ollama"
        assert llm.model == PROVIDER_PROFILES["ollama"]["default_model"]
        assert coding is not None
        assert coding.primary.provider == "ollama"
        assert coding.primary.model == PROVIDER_PROFILES["ollama"]["default_model"]

    def test_fallbacks_without_api_key_are_skipped(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-google")
        monkeypatch.setenv("CODING_PROVIDER", "gemini")
        monkeypatch.setenv("CODING_MODEL", "gemini-2.5-flash")
        monkeypatch.setenv(
            "CODING_FALLBACKS",
            "cerebras:qwen-3-coder-480b,groq:llama-3.3-70b-versatile",
        )
        monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)

        coding, _, _ = _build_role_configs()
        assert coding is not None
        assert coding.fallbacks == []

    def test_provenance_shape(self):
        primary = LLMConfig(provider="gemini", model="gemini-2.5-flash", api_key="x")
        fb = LLMConfig(provider="groq", model="llama-3.3-70b-versatile", api_key="y")
        role = RoleConfig(primary=primary, fallbacks=[fb])
        prov = role.provenance()
        assert prov["primary"] == {"provider": "gemini", "model": "gemini-2.5-flash"}
        assert prov["fallbacks"] == [{"provider": "groq", "model": "llama-3.3-70b-versatile"}]


class TestRateLimitDetection:
    def test_detects_429_status(self):
        resp = httpx.Response(status_code=429, request=httpx.Request("POST", "http://x"))
        exc = httpx.HTTPStatusError("rate limited", request=resp.request, response=resp)
        assert _is_rate_limit_error(exc) is True

    def test_detects_quota_message(self):
        exc = Exception("Resource_Exhausted: quota exceeded")
        assert _is_rate_limit_error(exc) is True

    def test_non_rate_limit_error_returns_false(self):
        assert _is_rate_limit_error(ValueError("bad input")) is False

    def test_dns_getaddrinfo_keyword_rotates(self):
        """Regression: '[Errno 11001] getaddrinfo failed' was missed by
        the keyword classifier. httpx wraps DNS failures in ConnectError
        but if a provider's HTTP layer surfaces the bare socket.gaierror
        message string we still need to rotate."""
        exc = OSError("[Errno 11001] getaddrinfo failed")
        assert _is_rate_limit_error(exc) is True

    def test_ssl_wrong_version_number_rotates(self):
        """SSL handshake glitches are transient on retry."""
        exc = OSError("[SSL: WRONG_VERSION_NUMBER] wrong version number (_ssl.c:1006)")
        assert _is_rate_limit_error(exc) is True


class TestFallbackChain:
    def setup_method(self):
        _PROVIDER_COOLDOWN.clear()

    def test_primary_succeeds_no_rotation(self):
        primary_llm = MagicMock()
        primary_llm._generate.return_value = "OK"
        fallback_llm = MagicMock()

        with patch("src.config.get_llm") as mock_get:
            mock_get.side_effect = [primary_llm, fallback_llm]
            primary = LLMConfig(provider="gemini", model="gemini-2.5-flash", api_key="k1")
            fb = LLMConfig(provider="groq", model="llama-3.3-70b-versatile", api_key="k2")
            role = RoleConfig(primary=primary, fallbacks=[fb])
            chain = get_llm_with_fallback(role)

        result = chain._generate([])
        assert result == "OK"
        fallback_llm._generate.assert_not_called()

    def test_rotates_on_rate_limit(self):
        resp = httpx.Response(status_code=429, request=httpx.Request("POST", "http://x"))
        primary_llm = MagicMock()
        primary_llm._generate.side_effect = httpx.HTTPStatusError(
            "429", request=resp.request, response=resp
        )
        fallback_llm = MagicMock()
        fallback_llm._generate.return_value = "FALLBACK_OK"

        with patch("src.config.get_llm") as mock_get:
            mock_get.side_effect = [primary_llm, fallback_llm]
            primary = LLMConfig(provider="gemini", model="gemini-2.5-flash", api_key="k1")
            fb = LLMConfig(provider="groq", model="llama-3.3-70b-versatile", api_key="k2")
            role = RoleConfig(primary=primary, fallbacks=[fb])
            chain = get_llm_with_fallback(role)

        result = chain._generate([])
        assert result == "FALLBACK_OK"
        primary_llm._generate.assert_called_once()
        fallback_llm._generate.assert_called_once()

    def test_non_rate_limit_error_propagates(self):
        primary_llm = MagicMock()
        primary_llm._generate.side_effect = ValueError("bad prompt")
        fallback_llm = MagicMock()

        with patch("src.config.get_llm") as mock_get:
            mock_get.side_effect = [primary_llm, fallback_llm]
            primary = LLMConfig(provider="gemini", model="gemini-2.5-flash", api_key="k1")
            fb = LLMConfig(provider="groq", model="llama-3.3-70b-versatile", api_key="k2")
            role = RoleConfig(primary=primary, fallbacks=[fb])
            chain = get_llm_with_fallback(role)

        with pytest.raises(ValueError):
            chain._generate([])
        fallback_llm._generate.assert_not_called()

    def test_rotates_on_read_timeout(self):
        """Regression: httpx ReadTimeout used to fall through the rotation
        classifier (it's not an HTTPStatusError and doesn't match any
        message keyword in the legacy list), so the pipeline failed the
        whole case on a stalled cloud-model call instead of trying the
        next fallback. Must now rotate (after exhausting same-endpoint
        retries -- ReadTimeout IS transient)."""
        os.environ["LLM_TRANSIENT_BACKOFF"] = "0.0"  # fast test
        primary_llm = MagicMock()
        primary_llm._generate.side_effect = httpx.ReadTimeout(
            "The read operation timed out",
            request=httpx.Request("POST", "https://ollama.com/v1/chat/completions"),
        )
        fallback_llm = MagicMock()
        fallback_llm._generate.return_value = "FALLBACK_OK"

        with patch("src.config.get_llm") as mock_get:
            mock_get.side_effect = [primary_llm, fallback_llm]
            primary = LLMConfig(provider="ollama", model="glm-5.1:cloud", api_key="k1")
            fb = LLMConfig(provider="mistral", model="codestral-latest", api_key="k2")
            role = RoleConfig(primary=primary, fallbacks=[fb])
            chain = get_llm_with_fallback(role)

        result = chain._generate([])
        assert result == "FALLBACK_OK"
        # Primary retried (initial + 2 retries) before rotation.
        assert primary_llm._generate.call_count == 3
        fallback_llm._generate.assert_called_once()

    def test_rotates_on_connect_error(self):
        os.environ["LLM_TRANSIENT_BACKOFF"] = "0.0"
        primary_llm = MagicMock()
        primary_llm._generate.side_effect = httpx.ConnectError(
            "connection refused",
            request=httpx.Request("POST", "https://ollama.com/v1/chat/completions"),
        )
        fallback_llm = MagicMock()
        fallback_llm._generate.return_value = "FALLBACK_OK"

        with patch("src.config.get_llm") as mock_get:
            mock_get.side_effect = [primary_llm, fallback_llm]
            primary = LLMConfig(provider="ollama", model="glm-5.1:cloud", api_key="k1")
            fb = LLMConfig(provider="mistral", model="codestral-latest", api_key="k2")
            role = RoleConfig(primary=primary, fallbacks=[fb])
            chain = get_llm_with_fallback(role)

        result = chain._generate([])
        assert result == "FALLBACK_OK"

    def test_retries_same_endpoint_on_transient_503_before_rotating(self):
        """Cross-model benchmark hygiene: a single transient 503 on the
        primary should NOT immediately fall back to a different model.
        Retry the same endpoint twice, then rotate only if still failing."""
        os.environ["LLM_TRANSIENT_RETRIES"] = "2"
        os.environ["LLM_TRANSIENT_BACKOFF"] = "0.0"  # no real sleep in tests

        resp_503 = httpx.Response(status_code=503, request=httpx.Request("POST", "http://x"))
        primary_llm = MagicMock()
        # Two transient 503s, then succeed on the SAME endpoint.
        primary_llm._generate.side_effect = [
            httpx.HTTPStatusError("503", request=resp_503.request, response=resp_503),
            httpx.HTTPStatusError("503", request=resp_503.request, response=resp_503),
            "PRIMARY_OK",
        ]
        fallback_llm = MagicMock()

        with patch("src.config.get_llm") as mock_get:
            mock_get.side_effect = [primary_llm, fallback_llm]
            primary = LLMConfig(provider="ollama", model="deepseek-v4-pro:cloud", api_key="k1")
            fb = LLMConfig(provider="mistral", model="codestral-latest", api_key="k2")
            role = RoleConfig(primary=primary, fallbacks=[fb])
            chain = get_llm_with_fallback(role)

        result = chain._generate([])
        assert result == "PRIMARY_OK"
        assert primary_llm._generate.call_count == 3
        fallback_llm._generate.assert_not_called()

    def test_rotates_after_transient_retries_exhausted(self):
        """If transient errors persist past the retry budget, rotate."""
        os.environ["LLM_TRANSIENT_RETRIES"] = "2"
        os.environ["LLM_TRANSIENT_BACKOFF"] = "0.0"

        resp_503 = httpx.Response(status_code=503, request=httpx.Request("POST", "http://x"))
        primary_llm = MagicMock()
        primary_llm._generate.side_effect = httpx.HTTPStatusError(
            "503", request=resp_503.request, response=resp_503
        )
        fallback_llm = MagicMock()
        fallback_llm._generate.return_value = "FALLBACK_OK"

        with patch("src.config.get_llm") as mock_get:
            mock_get.side_effect = [primary_llm, fallback_llm]
            primary = LLMConfig(provider="ollama", model="deepseek-v4-pro:cloud", api_key="k1")
            fb = LLMConfig(provider="mistral", model="codestral-latest", api_key="k2")
            role = RoleConfig(primary=primary, fallbacks=[fb])
            chain = get_llm_with_fallback(role)

        result = chain._generate([])
        assert result == "FALLBACK_OK"
        # Primary tried 3 times (initial + 2 retries) before rotating.
        assert primary_llm._generate.call_count == 3
        fallback_llm._generate.assert_called_once()

    def test_does_not_retry_same_endpoint_on_429(self):
        """429 / rate-limit / quota errors should rotate IMMEDIATELY,
        not waste retries on the same exhausted endpoint."""
        os.environ["LLM_TRANSIENT_RETRIES"] = "2"
        os.environ["LLM_TRANSIENT_BACKOFF"] = "0.0"

        resp_429 = httpx.Response(status_code=429, request=httpx.Request("POST", "http://x"))
        primary_llm = MagicMock()
        primary_llm._generate.side_effect = httpx.HTTPStatusError(
            "429", request=resp_429.request, response=resp_429
        )
        fallback_llm = MagicMock()
        fallback_llm._generate.return_value = "FALLBACK_OK"

        with patch("src.config.get_llm") as mock_get:
            mock_get.side_effect = [primary_llm, fallback_llm]
            primary = LLMConfig(provider="ollama", model="deepseek-v4-pro:cloud", api_key="k1")
            fb = LLMConfig(provider="mistral", model="codestral-latest", api_key="k2")
            role = RoleConfig(primary=primary, fallbacks=[fb])
            chain = get_llm_with_fallback(role)

        result = chain._generate([])
        assert result == "FALLBACK_OK"
        # Primary tried only ONCE -- 429 is not transient-retryable.
        primary_llm._generate.assert_called_once()
        fallback_llm._generate.assert_called_once()


class TestCooldown:
    def setup_method(self):
        _PROVIDER_COOLDOWN.clear()

    def test_429_sets_short_cooldown(self):
        cfg = LLMConfig(provider="gemini", model="gemini-2.5-flash-lite", api_key="x")
        resp = httpx.Response(status_code=429, request=httpx.Request("POST", "http://x"))
        exc = httpx.HTTPStatusError("rate limited", request=resp.request, response=resp)
        tag = _classify_and_cooldown(cfg, exc)
        assert tag == "rate_limit"
        remaining = _get_cooldown(_cooldown_key("gemini", "gemini-2.5-flash-lite"))
        assert 10.0 < remaining <= 60.0

    def test_retry_after_header_used(self):
        cfg = LLMConfig(provider="cerebras", model="qwen-235b", api_key="x")
        resp = httpx.Response(
            status_code=429,
            headers={"Retry-After": "12"},
            request=httpx.Request("POST", "http://x"),
        )
        exc = httpx.HTTPStatusError("rate limited", request=resp.request, response=resp)
        _classify_and_cooldown(cfg, exc)
        remaining = _get_cooldown(_cooldown_key("cerebras", "qwen-235b"))
        assert 10.0 <= remaining <= 13.0

    def test_404_sets_long_cooldown(self):
        cfg = LLMConfig(provider="cerebras", model="bad-model", api_key="x")
        resp = httpx.Response(status_code=404, request=httpx.Request("POST", "http://x"))
        exc = httpx.HTTPStatusError("not found", request=resp.request, response=resp)
        tag = _classify_and_cooldown(cfg, exc)
        assert tag == "not_found"
        remaining = _get_cooldown(_cooldown_key("cerebras", "bad-model"))
        assert remaining > 600.0  # effectively permanent for the session

    def test_413_context_overflow_tagged_too_large_not_rate_limit(self):
        # Groq returns 413 when a single prompt exceeds a model's per-request
        # token cap, and puts `"code": "rate_limit_exceeded"` inside the body.
        # We must classify this as permanent-for-session (too_large) rather
        # than as a recoverable rate_limit; otherwise we retry the same
        # oversized prompt after 45s and loop forever.
        cfg = LLMConfig(provider="groq", model="qwen/qwen3-32b", api_key="x")
        resp = httpx.Response(status_code=413, request=httpx.Request("POST", "http://x"))
        exc = httpx.HTTPStatusError(
            "Error code: 413 - {'error': {'message': 'Request too large for model "
            "`qwen/qwen3-32b` on tokens per minute (TPM)', 'code': 'rate_limit_exceeded'}}",
            request=resp.request,
            response=resp,
        )
        tag = _classify_and_cooldown(cfg, exc)
        assert tag == "too_large"
        remaining = _get_cooldown(_cooldown_key("groq", "qwen/qwen3-32b"))
        assert remaining > 600.0  # session-permanent so fallback takes over

    def test_context_length_exceeded_message_tagged_too_large(self):
        cfg = LLMConfig(provider="openai_compat", model="x", api_key="x")
        exc = RuntimeError(
            "openai.BadRequestError: This model's maximum context length is 8192 tokens, "
            "however your messages resulted in 10000 tokens. context_length_exceeded"
        )
        tag = _classify_and_cooldown(cfg, exc)
        assert tag == "too_large"

    def test_cooldown_skips_endpoint(self):
        """If primary is in cooldown, chain should use fallback immediately
        without making a request to the primary."""
        from unittest.mock import MagicMock, patch

        primary_llm = MagicMock()
        primary_llm._generate = MagicMock(side_effect=AssertionError("should not be called"))
        fallback_llm = MagicMock()
        fallback_llm._generate.return_value = "FB_OK"

        with patch("src.config.get_llm") as mock_get:
            mock_get.side_effect = [primary_llm, fallback_llm]
            primary = LLMConfig(provider="gemini", model="gemini-2.5-flash-lite", api_key="k1")
            fb = LLMConfig(provider="groq", model="llama-3.3-70b-versatile", api_key="k2")
            role = RoleConfig(primary=primary, fallbacks=[fb])
            chain = get_llm_with_fallback(role)

        _PROVIDER_COOLDOWN[_cooldown_key("gemini", "gemini-2.5-flash-lite")] = _time_plus(30)
        assert chain._generate([]) == "FB_OK"
        primary_llm._generate.assert_not_called()
        fallback_llm._generate.assert_called_once()


def _time_plus(seconds: float) -> float:
    import time

    return time.time() + seconds


class TestJsonHardening:
    def test_extracts_plain_json_array(self):
        assert _extract_json('[{"a": 1}]') == [{"a": 1}]

    def test_extracts_from_markdown_fence(self):
        text = 'Here is the result:\n```json\n[{"a": 1}, {"a": 2}]\n```\nDone.'
        assert _extract_json(text) == [{"a": 1}, {"a": 2}]

    def test_extracts_from_unlabeled_fence(self):
        text = '```\n[{"x": true}]\n```'
        assert _extract_json(text) == [{"x": True}]

    def test_extracts_with_trailing_text(self):
        text = '[{"v": "true_positive"}] and then some chatter'
        assert _extract_json(text) == [{"v": "true_positive"}]

    def test_falls_back_to_single_object(self):
        text = 'Here is my verdict: {"index": 0, "verdict": "false_positive"}'
        assert _extract_json(text) == {"index": 0, "verdict": "false_positive"}

    def test_returns_none_on_garbage(self):
        assert _extract_json("no json here at all") is None

    def test_strips_think_block_from_reasoning_models(self):
        text = (
            "<think>\nOkay, let me analyze each finding.\n"
            "Finding [0] looks like SQL injection.\n</think>\n"
            '[{"index": 0, "verdict": "true_positive"}]'
        )
        assert _extract_json(text) == [{"index": 0, "verdict": "true_positive"}]

    def test_sast_judge_handles_markdown_fences(self):
        judge = SastJudge(llm=MagicMock())
        response = '```json\n[{"index": 0, "verdict": "false_positive"}]\n```'
        verdicts = judge._parse_verdicts(response, count=1)
        from src.verification.models import JudgeVerdict

        assert verdicts[0] == JudgeVerdict.FALSE_POSITIVE

    def test_sast_judge_handles_single_object_response(self):
        judge = SastJudge(llm=MagicMock())
        response = '{"index": 0, "verdict": "true_positive"}'
        verdicts = judge._parse_verdicts(response, count=1)
        from src.verification.models import JudgeVerdict

        assert verdicts[0] == JudgeVerdict.TRUE_POSITIVE
