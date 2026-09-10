"""Configurable LLM provider abstraction for the RAG generation stage."""

from __future__ import annotations
from concurrent.futures import (ThreadPoolExecutor,TimeoutError as FutureTimeoutError)
from typing import Callable, Protocol
from pydantic import BaseModel, ConfigDict, Field
from src.generation import DraftResponse, GenerationPrompt


class LLMConfig(BaseModel):
    """Configuration for the LLM generation boundary."""

    model_config = ConfigDict(frozen=True)

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    timeout_seconds: float = Field(
        default=60.0,
        gt=0.0)


class LLMProvider(Protocol):
    """Protocol implemented by concrete LLM providers."""

    def generate(self,prompt: GenerationPrompt) -> DraftResponse:
        """Generate a draft response from a structured prompt."""
        ...


class LLMError(RuntimeError):
    """Base error for failures at the LLM boundary."""

class LLMProviderError(LLMError):
    """Raised when an LLM provider fails."""

class LLMTimeoutError(LLMError):
    """Raised when an LLM provider exceeds its timeout."""

class CallableLLMProvider:
    """Adapter for an externally supplied LLM callable.
    This adapter allows API-backed or local-model implementations
    to be injected without coupling the core pipeline to a vendor.
    """

    def __init__(
        self,
        generate_fn: Callable[[GenerationPrompt], DraftResponse]) -> None:
        if not callable(generate_fn):
            raise TypeError(
                "generate_fn must be callable."
            )

        self._generate_fn = generate_fn

    def generate(self,prompt: GenerationPrompt) -> DraftResponse:
        """Delegate generation to the supplied implementation."""
        if not isinstance(prompt,GenerationPrompt):
            raise TypeError("prompt must be a GenerationPrompt.")

        try:
            result = self._generate_fn(prompt)
        except LLMError:
            raise
        except Exception as exc:
            raise LLMProviderError(
                "LLM provider generation failed."
            ) from exc

        if not isinstance(result,DraftResponse):
            raise LLMProviderError(
                "LLM provider must return a DraftResponse.")
        return result


class FakeLLMProvider:
    """Deterministic LLM provider for tests and local development."""

    def __init__(self,response: DraftResponse) -> None:
        if not isinstance(response,DraftResponse):
            raise TypeError(
                "response must be a DraftResponse.")

        self.response = response
        self.calls = 0
        self.last_prompt: GenerationPrompt | None = None

    def generate(self,prompt: GenerationPrompt) -> DraftResponse:
        """Return the configured response deterministically."""
        if not isinstance(prompt,GenerationPrompt):
            raise TypeError(
                "prompt must be a GenerationPrompt."
            )

        self.calls += 1
        self.last_prompt = prompt

        return self.response


class LLM:
    """Provider-independent LLM generation facade."""

    def __init__(self,config: LLMConfig,provider: LLMProvider) -> None:
        if not isinstance(config,LLMConfig):
            raise TypeError(
                "config must be an LLMConfig.")

        if not hasattr(provider, "generate"):
            raise TypeError(
                "provider must implement generate().")

        self.config = config
        self.provider = provider

    def generate(self,prompt: GenerationPrompt) -> DraftResponse:
        """Generate a draft response with bounded execution time."""
        if not isinstance(
            prompt,
            GenerationPrompt):
            raise TypeError(
                "prompt must be a GenerationPrompt.")

        executor = ThreadPoolExecutor(
            max_workers=1,
        )

        future = executor.submit(
            self._generate_from_provider,
            prompt,
        )

        try:
            return future.result(
                timeout=self.config.timeout_seconds,
            )
        except FutureTimeoutError as exc:
            future.cancel()

            raise LLMTimeoutError(
                "LLM provider exceeded the configured "
                f"timeout of {self.config.timeout_seconds} seconds."
            ) from exc
        finally:
            executor.shutdown(
                wait=False,
                cancel_futures=True,
            )

    def _generate_from_provider(self,prompt: GenerationPrompt) -> DraftResponse:
        """Invoke the configured provider and normalize failures."""
        try:
            result = self.provider.generate(prompt)
        except LLMError:
            raise
        except Exception as exc:
            raise LLMProviderError(
                "Configured LLM provider failed."
            ) from exc

        if not isinstance(result,DraftResponse):
            raise LLMProviderError(
                "Configured LLM provider must return "
                "a DraftResponse."
            )

        return result