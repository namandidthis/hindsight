"""Shared Hindsight client resolution logic."""

from __future__ import annotations

import os
from importlib import metadata
from typing import Any

from hindsight_client import Hindsight

from .config import DEFAULT_HINDSIGHT_API_URL, HINDSIGHT_API_KEY_ENV, HINDSIGHT_API_URL_ENV, get_config

try:
    _VERSION = metadata.version("hindsight-livekit")
except metadata.PackageNotFoundError:
    _VERSION = "0.0.0"
USER_AGENT = f"hindsight-livekit/{_VERSION}"


def resolve_client(
    client: Hindsight | None,
    hindsight_api_url: str | None,
    api_key: str | None,
) -> Hindsight:
    """Resolve a Hindsight client from explicit args, global config, or env vars.

    Order: explicit arguments, then ``configure()``, then the ``HINDSIGHT_API_URL`` /
    ``HINDSIGHT_API_KEY`` env vars, then Hindsight Cloud. A LiveKit worker is usually
    configured through env vars, so they are honoured without a ``configure()`` call.
    """
    if client is not None:
        return client

    config = get_config()
    url = (
        hindsight_api_url
        or (config.hindsight_api_url if config else None)
        or os.environ.get(HINDSIGHT_API_URL_ENV)
        or DEFAULT_HINDSIGHT_API_URL
    )
    key = api_key or (config.api_key if config else None) or os.environ.get(HINDSIGHT_API_KEY_ENV)

    kwargs: dict[str, Any] = {"base_url": url, "timeout": 30.0, "user_agent": USER_AGENT}
    if key:
        kwargs["api_key"] = key
    return Hindsight(**kwargs)
