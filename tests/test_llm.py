"""Tests for the configurable LLM provider abstraction."""

from __future__ import annotations
import time
import pytest
from pydantic import ValidationError
from src.generation import (DraftResponse,GenerationPrompt)
from src.llm import (CallableLLMProvider,FakeLLMProvider,LLM,LLMConfig,LLMProviderError,LLMTimeoutError)


def make_prompt() -> GenerationPrompt:
    """Create deterministic generation prompt test data."""
    return GenerationPrompt(
        system_prompt=(
            "Answer only from the supplied repository evidence."),
        user_prompt=(
            "Where is the Request class implemented?"),
    )


def make_response() -> DraftResponse:
    """Create deterministic draft response test data."""
    return DraftResponse(
        answer=(
            "The Request class is implemented "
            "in scrapy/http/request.py."
        ),
        confidence=0.9,
    )


def make_config(timeout_seconds: float = 1.0) -> LLMConfig:
    """Create deterministic LLM configuration."""
    return LLMConfig(
        provider="test-provider",
        model="test-model",
        timeout_seconds=timeout_seconds,
    )


def test_llm_config_accepts_provider_configuration() -> None:
    """LLMConfig must represent provider-independent configuration."""
    config = make_config()

    assert config.provider == "test-provider"
    assert config.model == "test-model"
    assert config.timeout_seconds == 1.0


def test_llm_config_rejects_empty_provider() -> None:
    """Provider name must not be empty."""
    with pytest.raises(ValidationError):
        LLMConfig(
            provider="",
            model="test-model",
        )


def test_llm_config_rejects_empty_model() -> None:
    """Model name must not be empty."""
    with pytest.raises(ValidationError):
        LLMConfig(
            provider="test-provider",
            model="",
        )


def test_llm_config_rejects_non_positive_timeout() -> None:
    """Timeout must be greater than zero."""
    with pytest.raises(ValidationError):
        LLMConfig(
            provider="test-provider",
            model="test-model",
            timeout_seconds=0,
        )


def test_fake_provider_returns_deterministic_response() -> None:
    """FakeLLMProvider must return its configured response."""
    response = make_response()
    provider = FakeLLMProvider(response)

    result = provider.generate(make_prompt())

    assert result == response
    assert provider.calls == 1


def test_fake_provider_records_last_prompt() -> None:
    """FakeLLMProvider must expose the last supplied prompt."""
    provider = FakeLLMProvider(make_response())
    prompt = make_prompt()

    provider.generate(prompt)

    assert provider.last_prompt == prompt


def test_fake_provider_is_repeatable() -> None:
    """Repeated calls must return the same configured response."""
    response = make_response()
    provider = FakeLLMProvider(response)
    prompt = make_prompt()

    first = provider.generate(prompt)
    second = provider.generate(prompt)

    assert first == second
    assert provider.calls == 2


def test_callable_provider_delegates_generation() -> None:
    """CallableLLMProvider must delegate to the supplied callable."""
    response = make_response()
    received_prompts: list[GenerationPrompt] = []

    def generate(prompt: GenerationPrompt) -> DraftResponse:
        received_prompts.append(prompt)
        return response

    provider = CallableLLMProvider(generate)
    prompt = make_prompt()

    result = provider.generate(prompt)

    assert result == response
    assert received_prompts == [prompt]


def test_callable_provider_rejects_non_callable() -> None:
    """CallableLLMProvider requires a callable implementation."""
    with pytest.raises(TypeError,match="callable"):
        CallableLLMProvider(
            "not callable",  # type: ignore[arg-type]
        )


def test_callable_provider_rejects_invalid_prompt() -> None:
    """Provider boundary must require GenerationPrompt."""
    provider = CallableLLMProvider(
        lambda prompt: make_response()
    )

    with pytest.raises(TypeError,match="GenerationPrompt"):
        provider.generate(
            "invalid",  # type: ignore[arg-type]
        )


def test_callable_provider_wraps_provider_failure() -> None:
    """Unexpected provider failures must become LLMProviderError."""
    def generate(prompt: GenerationPrompt) -> DraftResponse:
        raise RuntimeError("provider unavailable")

    provider = CallableLLMProvider(generate)

    with pytest.raises(LLMProviderError,match="generation failed"):
        provider.generate(make_prompt())


def test_callable_provider_requires_draft_response() -> None:
    """Providers must return the generation contract."""
    provider = CallableLLMProvider(
        lambda prompt: "invalid response"  # type: ignore[return-value]
    )

    with pytest.raises(
        LLMProviderError,
        match="DraftResponse"):
        provider.generate(make_prompt())


def test_llm_accepts_provider_injection() -> None:
    """LLM must operate through the injected provider."""
    response = make_response()
    provider = FakeLLMProvider(response)

    llm = LLM(
        config=make_config(),
        provider=provider)

    result = llm.generate(make_prompt())

    assert result == response
    assert provider.calls == 1


def test_llm_passes_prompt_to_provider() -> None:
    """LLM must pass the GenerationPrompt unchanged."""
    provider = FakeLLMProvider(make_response())
    llm = LLM(
        config=make_config(),
        provider=provider)

    prompt = make_prompt()
    llm.generate(prompt)

    assert provider.last_prompt == prompt


def test_llm_rejects_invalid_config() -> None:
    """LLM must require LLMConfig."""
    provider = FakeLLMProvider(make_response())

    with pytest.raises(TypeError,match="LLMConfig"):
        LLM(
            config="invalid",  # type: ignore[arg-type]
            provider=provider,
        )


def test_llm_rejects_provider_without_generate() -> None:
    """LLM must require the provider interface."""
    with pytest.raises(TypeError,match="generate"):
        LLM(
            config=make_config(),
            provider=object(),  # type: ignore[arg-type]
        )


def test_llm_rejects_invalid_prompt() -> None:
    """LLM generation boundary must require GenerationPrompt."""
    provider = FakeLLMProvider(make_response())
    llm = LLM(
        config=make_config(),
        provider=provider,
    )

    with pytest.raises(TypeError,match="GenerationPrompt"):
        llm.generate(
            "invalid",  # type: ignore[arg-type]
        )


def test_llm_wraps_unexpected_provider_failure() -> None:
    """Unexpected provider exceptions must be normalized."""
    class FailingProvider:
        def generate(self, prompt: GenerationPrompt) -> DraftResponse:
            raise RuntimeError("failure")

    llm = LLM(
        config=make_config(),
        provider=FailingProvider(),
    )

    with pytest.raises(LLMProviderError,match="Configured LLM provider failed"):
        llm.generate(make_prompt())


def test_llm_preserves_llm_errors() -> None:
    """Known LLM errors must not be replaced by generic errors."""
    class FailingProvider:
        def generate(self,prompt: GenerationPrompt) -> DraftResponse:
            raise LLMProviderError("known failure")

    llm = LLM(
        config=make_config(),
        provider=FailingProvider())

    with pytest.raises(LLMProviderError,match="known failure"):
        llm.generate(make_prompt())


def test_llm_requires_draft_response_from_provider() -> None:
    """LLM must enforce the DraftResponse contract."""
    class InvalidProvider:
        def generate(self,prompt: GenerationPrompt) -> str:
            return "invalid"

    llm = LLM(
        config=make_config(),
        provider=InvalidProvider(),  # type: ignore[arg-type]
        )

    with pytest.raises( LLMProviderError, match="DraftResponse"):
        llm.generate(make_prompt())


def test_llm_timeout_is_enforced() -> None:
    """LLM must stop waiting after the configured timeout."""
    class SlowProvider:
        def generate(self,prompt: GenerationPrompt,) -> DraftResponse:
            time.sleep(0.2)
            return make_response()

    llm = LLM(
        config=make_config(
            timeout_seconds=0.05,),
        provider=SlowProvider(),
    )

    with pytest.raises(LLMTimeoutError,match="timeout"):
        llm.generate(make_prompt())


def test_llm_timeout_configuration_is_used() -> None:
    """A provider completing within the timeout must succeed."""
    class FastProvider:
        def generate(self,prompt: GenerationPrompt) -> DraftResponse:
            time.sleep(0.01)
            return make_response()

    llm = LLM(
        config=make_config(
            timeout_seconds=0.5,
        ),
        provider=FastProvider(),
    )

    result = llm.generate(make_prompt())

    assert result == make_response()


def test_fake_provider_does_not_require_external_service() -> None:
    """The deterministic provider must work without external dependencies."""
    provider = FakeLLMProvider(
        DraftResponse(
            answer="Deterministic answer.",
        )
    )

    llm = LLM(
        config=make_config(),
        provider=provider,
    )

    result = llm.generate(make_prompt())

    assert result.answer == "Deterministic answer."


def test_llm_config_is_immutable() -> None:
    """LLMConfig must remain immutable."""
    config = make_config()

    with pytest.raises(ValidationError):
        config.model = "changed"


def test_llm_provider_selection_is_configuration_only() -> None:
    """Changing provider configuration must not require LLM changes."""
    first_provider = FakeLLMProvider(
        DraftResponse(answer="Provider one.")
    )
    second_provider = FakeLLMProvider(
        DraftResponse(answer="Provider two.")
    )

    first_llm = LLM(
        config=LLMConfig(
            provider="provider-one",
            model="model-one",
        ),
        provider=first_provider,
    )

    second_llm = LLM(
        config=LLMConfig(
            provider="provider-two",
            model="model-two",
        ),
        provider=second_provider,
    )

    assert (
        first_llm.generate(make_prompt()).answer
        == "Provider one."
    )
    assert (
        second_llm.generate(make_prompt()).answer
        == "Provider two."
    )


    