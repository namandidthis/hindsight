"""LiveKit function tools for Hindsight memory operations.

Provides a factory that creates LiveKit ``FunctionTool`` instances backed by
Hindsight's retain/recall/reflect APIs. Pass them to ``Agent(tools=[...])``.
They work with STT-LLM-TTS pipelines and realtime models.
"""

from __future__ import annotations

import logging

from hindsight_client import Hindsight
from livekit.agents import RunContext, ToolError, function_tool
from livekit.agents.llm import FunctionTool

from ._client import resolve_client
from .config import (
    DEFAULT_RECALL_BUDGET,
    DEFAULT_RECALL_MAX_TOKENS,
    DEFAULT_RECALL_TAGS_MATCH,
    Budget,
    TagsMatch,
    get_config,
)

logger = logging.getLogger(__name__)


def create_hindsight_tools(
    *,
    bank_id: str,
    client: Hindsight | None = None,
    hindsight_api_url: str | None = None,
    api_key: str | None = None,
    budget: Budget | None = None,
    max_tokens: int | None = None,
    tags: list[str] | None = None,
    recall_tags: list[str] | None = None,
    recall_tags_match: TagsMatch = DEFAULT_RECALL_TAGS_MATCH,
    recall_types: list[str] | None = None,
    retain_context: str = "voice conversation",
    include_retain: bool = True,
    include_recall: bool = True,
    include_reflect: bool = True,
) -> list[FunctionTool]:
    """Create Hindsight memory tools for a LiveKit agent.

    Returns ``FunctionTool`` instances for ``Agent(tools=[...])``. With a
    :class:`~hindsight_livekit.HindsightMemory`, share its client and bank::

        tools = create_hindsight_tools(bank_id=memory.bank_id, client=memory.client)

    Errors are raised as LiveKit ``ToolError`` so the model hears what went wrong
    and can tell the caller, instead of the call failing.

    Args:
        bank_id: The Hindsight memory bank to operate on.
        client: Pre-configured Hindsight client (preferred).
        hindsight_api_url: API URL (used if no client provided).
        api_key: API key (used if no client provided).
        budget: Recall/reflect budget level (low/mid/high). Defaults to ``"low"``.
        max_tokens: Maximum tokens for recall and reflect results.
        tags: Tags applied when storing memories via retain.
        recall_tags: Tags to filter by when searching memories.
        recall_tags_match: Tag matching mode (any/all/any_strict/all_strict).
        recall_types: Fact types to search (world, experience, observation).
        retain_context: Source label passed to Hindsight when storing memories.
        include_retain: Include the ``hindsight_retain`` (store) tool.
        include_recall: Include the ``hindsight_recall`` (search) tool.
        include_reflect: Include the ``hindsight_reflect`` (synthesize) tool. Reflect
            runs an LLM on the Hindsight server, so it is the slowest of the three.

    Returns:
        List of LiveKit FunctionTool instances.
    """
    resolved_client = resolve_client(client, hindsight_api_url, api_key)
    config = get_config()
    effective_budget: Budget = budget or (config.recall_budget if config else DEFAULT_RECALL_BUDGET)
    effective_max_tokens = max_tokens or (config.recall_max_tokens if config else DEFAULT_RECALL_MAX_TOKENS)

    tools: list[FunctionTool] = []

    if include_retain:

        @function_tool
        async def hindsight_retain(context: RunContext, content: str) -> str:
            """Store information in long-term memory so it is remembered in future calls.

            Use this when the user shares something worth remembering, such as a
            preference, a plan, or a fact about themselves.

            Args:
                content: The information to store, as a complete sentence.
            """
            try:
                await resolved_client.aretain(
                    bank_id=bank_id,
                    content=content,
                    context=retain_context,
                    tags=tags,
                )
            except Exception as e:
                logger.error(f"Hindsight retain failed: {e}")
                raise ToolError("Could not save that to memory right now.") from e
            return "Saved to memory."

        tools.append(hindsight_retain)

    if include_recall:

        @function_tool
        async def hindsight_recall(context: RunContext, query: str) -> str:
            """Search long-term memory for things the user said in earlier conversations.

            Use this when the user refers to something from the past, or when knowing
            their preferences or history would help you answer.

            Args:
                query: What to look for, as a short question or phrase.
            """
            try:
                response = await resolved_client.arecall(
                    bank_id=bank_id,
                    query=query,
                    types=recall_types,
                    budget=effective_budget,
                    max_tokens=effective_max_tokens,
                    tags=recall_tags,
                    tags_match=recall_tags_match,
                )
            except Exception as e:
                logger.error(f"Hindsight recall failed: {e}")
                raise ToolError("Could not search memory right now.") from e
            if not response.results:
                return "No relevant memories found."
            return "\n".join(f"{i}. {result.text}" for i, result in enumerate(response.results, 1))

        tools.append(hindsight_recall)

    if include_reflect:

        @function_tool
        async def hindsight_reflect(context: RunContext, query: str) -> str:
            """Answer a question by reasoning over everything in long-term memory.

            Use this for broader questions, like summarising what you know about the
            user, rather than looking up a single fact.

            Args:
                query: The question to answer from memory.
            """
            try:
                response = await resolved_client.areflect(
                    bank_id=bank_id,
                    query=query,
                    budget=effective_budget,
                    max_tokens=effective_max_tokens,
                    tags=recall_tags,
                    tags_match=recall_tags_match,
                )
            except Exception as e:
                logger.error(f"Hindsight reflect failed: {e}")
                raise ToolError("Could not reflect on memory right now.") from e
            return response.text or "No relevant memories found."

        tools.append(hindsight_reflect)

    return tools
