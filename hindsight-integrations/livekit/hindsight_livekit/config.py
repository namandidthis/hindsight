"""Global configuration for Hindsight-LiveKit integration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

DEFAULT_HINDSIGHT_API_URL = "https://api.hindsight.vectorize.io"
HINDSIGHT_API_URL_ENV = "HINDSIGHT_API_URL"
HINDSIGHT_API_KEY_ENV = "HINDSIGHT_API_KEY"

Budget = Literal["low", "mid", "high"]
TagsMatch = Literal["any", "all", "any_strict", "all_strict"]

# Voice defaults: every recall happens while a caller waits, so they favour speed.
# Hindsight documents "low" as the budget for fast, simple lookups.
DEFAULT_RECALL_BUDGET: Budget = "low"
DEFAULT_RECALL_MAX_TOKENS = 1024
DEFAULT_RECALL_TIMEOUT = 3.0
DEFAULT_RECALL_TAGS_MATCH: TagsMatch = "any"


@dataclass
class HindsightLiveKitConfig:
    """Connection and default settings for the LiveKit integration.

    Attributes:
        hindsight_api_url: URL of the Hindsight API server.
        api_key: API key for Hindsight authentication.
        recall_budget: Default recall budget level (low/mid/high).
        recall_max_tokens: Default maximum tokens for recall results.
        recall_timeout: Default seconds to wait for recall before answering without memories.
    """

    hindsight_api_url: str = DEFAULT_HINDSIGHT_API_URL
    api_key: str | None = None
    recall_budget: Budget = DEFAULT_RECALL_BUDGET
    recall_max_tokens: int = DEFAULT_RECALL_MAX_TOKENS
    recall_timeout: float = DEFAULT_RECALL_TIMEOUT


_global_config: HindsightLiveKitConfig | None = None


def configure(
    hindsight_api_url: str | None = None,
    api_key: str | None = None,
    recall_budget: Budget = DEFAULT_RECALL_BUDGET,
    recall_max_tokens: int = DEFAULT_RECALL_MAX_TOKENS,
    recall_timeout: float = DEFAULT_RECALL_TIMEOUT,
) -> HindsightLiveKitConfig:
    """Configure Hindsight connection and default settings.

    Args:
        hindsight_api_url: Hindsight API URL. Falls back to HINDSIGHT_API_URL env var,
            then Hindsight Cloud.
        api_key: API key. Falls back to HINDSIGHT_API_KEY env var.
        recall_budget: Default recall budget (low/mid/high).
        recall_max_tokens: Default max tokens for recall.
        recall_timeout: Default recall timeout in seconds.

    Returns:
        The configured HindsightLiveKitConfig.
    """
    global _global_config

    _global_config = HindsightLiveKitConfig(
        hindsight_api_url=hindsight_api_url or os.environ.get(HINDSIGHT_API_URL_ENV) or DEFAULT_HINDSIGHT_API_URL,
        api_key=api_key or os.environ.get(HINDSIGHT_API_KEY_ENV),
        recall_budget=recall_budget,
        recall_max_tokens=recall_max_tokens,
        recall_timeout=recall_timeout,
    )

    return _global_config


def get_config() -> HindsightLiveKitConfig | None:
    """Get the current global configuration."""
    return _global_config


def reset_config() -> None:
    """Reset global configuration to None."""
    global _global_config
    _global_config = None
