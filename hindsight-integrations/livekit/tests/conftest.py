"""Shared fakes for the LiveKit integration tests.

LiveKit runs for real in these tests: a real ``AgentSession`` drives the agent
through ``session.run()``. Only the LLM (a scripted fake) and the Hindsight
client (a mock) are replaced, so no servers or API keys are required.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions, llm
from livekit.agents.llm import ChatChunk, ChatContext, ChoiceDelta, FunctionToolCall

from hindsight_livekit import reset_config
from hindsight_livekit.memory import _MEMORY_MARKER


def mock_client(recall_texts: list[str] | None = None, reflect_text: str = "") -> MagicMock:
    """Return a mock Hindsight client with async recall/retain/reflect/aclose methods."""
    client = MagicMock()
    response = MagicMock()
    results = []
    for t in recall_texts or []:
        r = MagicMock()
        r.text = t
        results.append(r)
    response.results = results
    client.arecall = AsyncMock(return_value=response)
    client.aretain = AsyncMock(return_value=None)
    reflect_response = MagicMock()
    reflect_response.text = reflect_text
    client.areflect = AsyncMock(return_value=reflect_response)
    client.aclose = AsyncMock(return_value=None)
    return client


class _FakeStream(llm.LLMStream):
    def __init__(self, fake: FakeLLM, *, chat_ctx: ChatContext, tools: list[Any], conn_options: APIConnectOptions):
        super().__init__(fake, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._reply = fake.next_reply()

    async def _run(self) -> None:
        if isinstance(self._reply, FunctionToolCall):
            delta = ChoiceDelta(role="assistant", tool_calls=[self._reply])
        else:
            delta = ChoiceDelta(role="assistant", content=self._reply)
        self._event_ch.send_nowait(ChatChunk(id="fake", delta=delta))


class FakeLLM(llm.LLM):
    """LLM that returns scripted replies (text or tool calls) and records every call's context."""

    def __init__(self, replies: list[str | FunctionToolCall]) -> None:
        super().__init__()
        self._replies = list(replies)
        self.contexts: list[ChatContext] = []

    def next_reply(self) -> str | FunctionToolCall:
        return self._replies.pop(0)

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[Any] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        **kwargs: Any,
    ) -> llm.LLMStream:
        self.contexts.append(chat_ctx.copy())
        return _FakeStream(self, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options)


def messages(chat_ctx: ChatContext) -> list[tuple[str, str]]:
    return [(m.role, m.text_content or "") for m in chat_ctx.items if m.type == "message"]


def memory_messages(chat_ctx: ChatContext) -> list[str]:
    """Memory blocks injected into the conversation (not the instructions)."""
    return [text for role, text in messages(chat_ctx) if role != "system" and _MEMORY_MARKER in text]


def instructions(chat_ctx: ChatContext) -> str:
    return "\n".join(text for role, text in messages(chat_ctx) if role == "system")


def tool_outputs(chat_ctx: ChatContext) -> list[str]:
    return [item.output for item in chat_ctx.items if item.type == "function_call_output"]


@pytest.fixture(autouse=True)
def _clean_config():
    reset_config()
    yield
    reset_config()
