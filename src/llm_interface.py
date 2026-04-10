"""
Unified LLM Interface for PRACTIQ

This module provides a clean interface for LLM interactions using LiteLLM.
It replaces the legacy boto3-based Bedrock implementations with a modern,
unified approach that supports multiple models and providers.

Usage:
    from llm_interface import get_default_llm, LLMConfig

    # Use default LLM
    llm = get_default_llm()
    response = llm.call(messages=[{"role": "user", "content": [{"type": "text", "text": "Hello"}]}])

    # Use custom configuration
    config = LLMConfig(model_name="claude-3-5-haiku", temperature=0.5)
    llm = UnifiedLLM(config)
    response = llm.call(messages=messages, system="You are a helpful assistant")
"""

import os
from typing import List, Dict, Optional, Any, Mapping
from dataclasses import dataclass
from loguru import logger

from litellm_helpers import (
    router_completion_with_ratelimit_retry,
    DEFAULT_ROUTER,
    convert_claude_msg_list_to_litellm_msg_list,
)


@dataclass
class LLMConfig:
    """Configuration for LLM behavior"""
    model_name: str = "gpt-4o-mini"  # Using Claude 3.5 Sonnet (Haiku 3.5 requires inference profile)
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 250
    max_tokens: int = 4096
    num_retries: int = 15
    retry_after: int = 30

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "num_retries": self.num_retries,
            "retry_after": self.retry_after,
        }


class UnifiedLLM:
    """
    Unified LLM interface that uses LiteLLM under the hood.

    This class provides a consistent API compatible with the legacy BedrockClaudeLlm
    interface while using modern LiteLLM router for improved reliability and features.

    Attributes:
        config: LLMConfig object containing model parameters
        router: LiteLLM router instance for making API calls
    """

    def __init__(self, config: Optional[LLMConfig] = None, router=None) -> None:
        """
        Initialize UnifiedLLM

        Args:
            config: Optional LLMConfig object. If None, uses default configuration.
            router: Optional LiteLLM router instance. If None, uses DEFAULT_ROUTER.
        """
        self.config = config or LLMConfig()
        self.router = router or DEFAULT_ROUTER
        self.model_id = self.config.model_name
        logger.info(f"Initialized UnifiedLLM with model: {self.model_id}")

    @property
    def _llm_type(self) -> str:
        """Return LLM type identifier"""
        return "litellm-router"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters"""
        return {"model_id": self.model_id, "config": self.config.to_dict()}

    def call(
        self,
        messages: List[Dict],
        max_new_token: Optional[int] = None,
        system: str = "",
        stop_sequences: Optional[List] = None
    ) -> str:
        """
        Call LLM with messages API format (compatible with Claude messages API)

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
                     content can be a list of dicts with 'type' and 'text' keys,
                     or a simple string
            max_new_token: Maximum tokens to generate (overrides config)
            system: System prompt string
            stop_sequences: Optional list of stop sequences (not currently supported in litellm router)

        Returns:
            String response from the model

        Example:
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello"}]
                }
            ]
            response = llm.call(messages=messages, system="You are helpful")
        """
        if stop_sequences:
            logger.warning(f"stop_sequences parameter is not fully supported in litellm router: {stop_sequences}")

        # Convert Claude message format to LiteLLM format if needed
        litellm_messages = convert_claude_msg_list_to_litellm_msg_list(messages)

        # Use provided max_new_token or fall back to config
        max_tokens = max_new_token if max_new_token is not None else self.config.max_tokens

        # Call LiteLLM router with retry logic
        response = router_completion_with_ratelimit_retry(
            model=self.config.model_name,
            messages=litellm_messages,
            router=self.router,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=max_tokens,
            num_retries=self.config.num_retries,
            retry_after=self.config.retry_after,
            system=system if system else None,
        )

        return response

    def call_single_prompt(
        self,
        prompt: str,
        max_new_token: Optional[int] = None,
        system: str = ""
    ) -> str:
        """
        Convenience method to call LLM with a single prompt string

        Args:
            prompt: User prompt string
            max_new_token: Maximum tokens to generate
            system: System prompt string

        Returns:
            String response from the model

        Example:
            response = llm.call_single_prompt("What is the capital of France?")
        """
        return self.generate(prompt=prompt, max_new_token=max_new_token, system=system)

    def generate(
        self,
        prompt: str,
        max_new_token: Optional[int] = None,
        system: str = "",
    ) -> str:
        """
        Generate response from a single prompt

        Args:
            prompt: User prompt string
            max_new_token: Maximum tokens to generate
            system: System prompt string

        Returns:
            String response from the model
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
        return self.call(messages=messages, max_new_token=max_new_token, system=system)


# Create default LLM instance with standard configuration
DEFAULT_LLM = UnifiedLLM(config=LLMConfig())


def get_default_llm() -> UnifiedLLM:
    """
    Get default LLM instance

    Returns:
        UnifiedLLM instance with default configuration
    """
    return DEFAULT_LLM


def get_llm_with_model(model_name: str, **kwargs) -> UnifiedLLM:
    """
    Get LLM instance with specific model

    Args:
        model_name: Name of the model (e.g., "claude-3-5-haiku", "llama3-1-70b")
        **kwargs: Additional configuration parameters

    Returns:
        UnifiedLLM instance configured for the specified model

    Example:
        llm = get_llm_with_model("claude-3-5-sonnet", temperature=0.7)
    """
    config = LLMConfig(model_name=model_name, **kwargs)
    return UnifiedLLM(config=config)


if __name__ == "__main__":
    # Example usage
    llm = get_default_llm()

    # Test with simple prompt
    response = llm.call_single_prompt(
        "What is the capital of France? Answer in one word.",
        system="You are a helpful assistant."
    )
    print(f"Response: {response}")

    # Test with messages format
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Write a haiku about coding"
                }
            ]
        }
    ]
    response = llm.call(messages=messages)
    print(f"Haiku: {response}")
