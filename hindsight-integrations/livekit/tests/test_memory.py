"""Tests for HindsightMemory and HindsightAgent.

All Hindsight API calls are mocked; LiveKit's AgentSession runs for real.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hindsight_client_api.exceptions import NotFoundException
from livekit.agents import AgentSession, RunContext, function_tool, llm
from livekit.agents.llm import ChatContext, FunctionToolCall

from hindsight_livekit import HindsightAgent, HindsightLiveKitError, HindsightMemory, configure
from hindsight_livekit._client import resolve_client
from hindsight_livekit.memory import _MEMORY_MARKER

from .conftest import FakeLLM, instructions, memory_messages, messages, mock_client

# ---------------------------------------------------------------------------
# Per-turn recall through a real AgentSession
# ---------------------------------------------------------------------------


class TestPerTurnRecall:
    async def test_recalled_memories_reach_the_llm(self):
        client = mock_client(["User's name is Naman", "User prefers short answers"])
        memory = HindsightMemory(bank_id="user-1", client=client, enable_session_recall=False)
        fake = FakeLLM(["Hi Naman!"])

        async with AgentSession(llm=fake) as session:
            await session.start(HindsightAgent(memory=memory, instructions="Be helpful."))
            await session.run(user_input="Do you remember my name?")

        client.arecall.assert_awaited_once()
        assert client.arecall.call_args.kwargs["query"] == "Do you remember my name?"
        assert client.arecall.call_args.kwargs["bank_id"] == "user-1"

        sent = messages(fake.contexts[0])
        memory_index = next(i for i, (_, text) in enumerate(sent) if _MEMORY_MARKER in text)
        assert sent[memory_index][0] == "assistant"
        assert "1. User's name is Naman" in sent[memory_index][1]
        assert "2. User prefers short answers" in sent[memory_index][1]
        # Memories sit directly before the user's message.
        assert sent[memory_index + 1] == ("user", "Do you remember my name?")

    async def test_memories_are_not_stored_in_session_history(self):
        client = mock_client(["User's name is Naman"])
        memory = HindsightMemory(bank_id="user-1", client=client, enable_session_recall=False)
        fake = FakeLLM(["Hi Naman!", "You said hello."])

        async with AgentSession(llm=fake) as session:
            agent = HindsightAgent(memory=memory, instructions="Be helpful.")
            await session.start(agent)
            await session.run(user_input="Hello")
            await session.run(user_input="What did I say?")

            assert memory_messages(agent.chat_ctx) == []
            assert memory_messages(session.history) == []
        # Each LLM call carries exactly one (fresh) memory block.
        assert len(memory_messages(fake.contexts[1])) == 1

    async def test_no_memories_leaves_context_unchanged(self):
        client = mock_client([])
        memory = HindsightMemory(bank_id="user-1", client=client, enable_session_recall=False)
        fake = FakeLLM(["Hello!"])

        async with AgentSession(llm=fake) as session:
            await session.start(HindsightAgent(memory=memory, instructions="Be helpful."))
            await session.run(user_input="Hi")

        assert memory_messages(fake.contexts[0]) == []

    async def test_recall_disabled(self):
        client = mock_client(["something"])
        memory = HindsightMemory(bank_id="user-1", client=client, enable_recall=False, enable_session_recall=False)
        fake = FakeLLM(["Hi."])

        async with AgentSession(llm=fake) as session:
            await session.start(HindsightAgent(memory=memory, instructions="Be helpful."))
            await session.run(user_input="Hello")

        client.arecall.assert_not_awaited()
        assert memory_messages(fake.contexts[0]) == []

    async def test_recall_runs_once_per_user_turn_with_tool_calls(self):
        client = mock_client(["User lives in Delhi"])
        memory = HindsightMemory(bank_id="user-1", client=client, enable_session_recall=False)
        fake = FakeLLM(
            [
                FunctionToolCall(name="lookup_weather", arguments='{"location": "Delhi"}', call_id="call-1"),
                "It's sunny in Delhi.",
            ]
        )

        class WeatherAgent(HindsightAgent):
            @function_tool
            async def lookup_weather(self, context: RunContext, location: str) -> str:
                """Look up the weather.

                Args:
                    location: City name.
                """
                return "sunny"

        async with AgentSession(llm=fake) as session:
            await session.start(WeatherAgent(memory=memory, instructions="Be helpful."))
            await session.run(user_input="What's the weather at home?")

        assert len(fake.contexts) == 2
        client.arecall.assert_awaited_once()
        # The follow-up call after the tool result still sees the memories.
        assert len(memory_messages(fake.contexts[1])) == 1

    async def test_recall_failure_does_not_break_the_turn(self):
        client = mock_client([])
        client.arecall = AsyncMock(side_effect=RuntimeError("connection refused"))
        memory = HindsightMemory(bank_id="user-1", client=client)
        fake = FakeLLM(["Hello!"])

        async with AgentSession(llm=fake) as session:
            await session.start(HindsightAgent(memory=memory, instructions="Be helpful."))
            await session.run(user_input="Hi")

        assert len(fake.contexts) == 1
        assert memory_messages(fake.contexts[0]) == []
        assert instructions(fake.contexts[0]) == "Be helpful."


# ---------------------------------------------------------------------------
# Session recall (on_enter) through a real AgentSession
# ---------------------------------------------------------------------------


class TestSessionRecall:
    async def test_known_facts_are_added_to_instructions(self):
        client = mock_client(["User's name is Naman", "User lives in Bangalore"])
        memory = HindsightMemory(bank_id="user-1", client=client, enable_recall=False)
        fake = FakeLLM(["Hi Naman!"])

        async with AgentSession(llm=fake) as session:
            await session.start(HindsightAgent(memory=memory, instructions="Be helpful."))
            await session.run(user_input="Hi")

        client.arecall.assert_awaited_once()
        assert client.arecall.call_args.kwargs["query"].startswith("Who is this user?")
        sent = instructions(fake.contexts[0])
        assert sent.startswith("Be helpful.\n\n<hindsight_memories>")
        assert "1. User's name is Naman" in sent
        assert "2. User lives in Bangalore" in sent
        # Session memories live in the instructions, not in the conversation.
        assert memory_messages(fake.contexts[0]) == []

    async def test_first_reply_waits_for_a_slow_lookup(self):
        """A user who speaks before on_enter finishes still gets a reply with the memories."""
        client = mock_client(["User's name is Naman"])
        response = client.arecall.return_value

        async def slow_recall(**kwargs: Any) -> MagicMock:
            await asyncio.sleep(0.3)
            return response

        client.arecall = AsyncMock(side_effect=slow_recall)
        memory = HindsightMemory(bank_id="b", client=client, enable_recall=False)
        fake = FakeLLM(["Hi Naman!"])

        async with AgentSession(llm=fake) as session:
            agent = HindsightAgent(memory=memory, instructions="Be helpful.")
            await session.start(agent)
            await session.run(user_input="Hi")  # before the 0.3s lookup has finished

        assert "User's name is Naman" in instructions(fake.contexts[0])
        assert instructions(fake.contexts[0]).count(_MEMORY_MARKER) == 1
        client.arecall.assert_awaited_once()
        # The stored history keeps the original item; only this call's copy changed.
        assert agent.instructions.count(_MEMORY_MARKER) == 1

    async def test_custom_query(self):
        client = mock_client(["fact"])
        memory = HindsightMemory(bank_id="b", client=client, enable_recall=False, session_recall_query="Open orders?")

        async with AgentSession(llm=FakeLLM(["ok"])) as session:
            await session.start(HindsightAgent(memory=memory, instructions="x"))

        assert client.arecall.call_args.kwargs["query"] == "Open orders?"

    async def test_disabled(self):
        client = mock_client(["fact"])
        memory = HindsightMemory(bank_id="b", client=client, enable_recall=False, enable_session_recall=False)
        fake = FakeLLM(["ok"])

        async with AgentSession(llm=fake) as session:
            await session.start(HindsightAgent(memory=memory, instructions="Be helpful."))
            await session.run(user_input="Hi")

        client.arecall.assert_not_awaited()
        assert instructions(fake.contexts[0]) == "Be helpful."

    async def test_reentering_does_not_stack_memories(self):
        client = mock_client(["fact one"])
        memory = HindsightMemory(bank_id="b", client=client, enable_recall=False)

        async with AgentSession(llm=FakeLLM([])) as session:
            agent = HindsightAgent(memory=memory, instructions="Base.")
            await session.start(agent)
            await agent.on_enter()  # e.g. a handoff back to this agent

            assert agent.instructions.count(_MEMORY_MARKER) == 1
            assert agent.instructions.startswith("Base.\n\n")
        # The session's lookup is reused rather than repeated.
        client.arecall.assert_awaited_once()

    async def test_attach_starts_the_lookup_before_the_session(self):
        """LiveKit doesn't hold the first reply for on_enter, so the lookup starts early."""
        client = mock_client(["fact"])
        memory = HindsightMemory(bank_id="b", client=client, enable_recall=False)

        async with AgentSession(llm=FakeLLM([])) as session:
            memory.attach(session)
            await asyncio.sleep(0.05)
            client.arecall.assert_awaited_once()

            await session.start(HindsightAgent(memory=memory, instructions="Base."))
        client.arecall.assert_awaited_once()

    async def test_attach_without_session_recall_does_not_look_up(self):
        client = mock_client(["fact"])
        memory = HindsightMemory(bank_id="b", client=client, enable_session_recall=False)
        async with AgentSession(llm=FakeLLM([])) as session:
            memory.attach(session)
            await asyncio.sleep(0)
        client.arecall.assert_not_awaited()

    async def test_subclass_on_enter_can_greet_with_memories(self):
        client = mock_client(["User's name is Naman"])
        memory = HindsightMemory(bank_id="b", client=client, enable_recall=False)
        fake = FakeLLM(["Welcome back, Naman!"])

        class Greeter(HindsightAgent):
            async def on_enter(self) -> None:
                await super().on_enter()
                self.session.generate_reply(instructions="Greet the user.")

        async with AgentSession(llm=fake) as session:
            await session.start(Greeter(memory=memory, instructions="Be helpful."))
            await asyncio.sleep(0.5)

        assert "User's name is Naman" in instructions(fake.contexts[0])


# ---------------------------------------------------------------------------
# Retain through a real AgentSession
# ---------------------------------------------------------------------------


class TestRetainSession:
    async def test_exchanges_are_appended_to_one_document(self):
        client = mock_client([])
        memory = HindsightMemory(bank_id="user-1", client=client, document_id="room-42", tags=["voice"])
        fake = FakeLLM(["Nice to meet you.", "Blue, got it."])

        async with AgentSession(llm=fake) as session:
            memory.attach(session)
            await session.start(HindsightAgent(memory=memory, instructions="Be helpful."))
            await session.run(user_input="I'm Naman")
            await session.run(user_input="My favorite color is blue")
            await memory.flush()

        assert client.aretain.await_count == 2
        first, second = (c.kwargs for c in client.aretain.call_args_list)
        for call in (first, second):
            assert call["bank_id"] == "user-1"
            assert call["document_id"] == "room-42"
            assert call["update_mode"] == "append"
            assert call["context"] == "voice conversation"
            assert call["tags"] == ["voice"]
        assert first["content"].splitlines()[0].startswith("User (")
        assert first["content"].splitlines()[0].endswith("): I'm Naman")
        assert first["content"].splitlines()[1].endswith("): Nice to meet you.")
        assert second["content"].splitlines()[0].endswith("): My favorite color is blue")
        assert second["content"].splitlines()[1].endswith("): Blue, got it.")

    async def test_retain_disabled(self):
        client = mock_client([])
        memory = HindsightMemory(bank_id="user-1", client=client, enable_retain=False)

        async with AgentSession(llm=FakeLLM(["Hi."])) as session:
            memory.attach(session)
            await session.start(HindsightAgent(memory=memory, instructions="Be helpful."))
            await session.run(user_input="Hello")
            await memory.flush()

        client.aretain.assert_not_awaited()

    async def test_works_with_a_plain_agent(self):
        """attach() only needs the session, so retain works with any Agent."""
        client = mock_client([])
        memory = HindsightMemory(bank_id="user-1", client=client)
        from livekit.agents import Agent

        async with AgentSession(llm=FakeLLM(["Hi."])) as session:
            memory.attach(session)
            await session.start(Agent(instructions="Be helpful."))
            await session.run(user_input="Hello")
            await memory.flush()

        client.aretain.assert_awaited_once()


# ---------------------------------------------------------------------------
# HindsightMemory units
# ---------------------------------------------------------------------------


class TestRecall:
    async def test_recall_passes_settings(self):
        client = mock_client(["fact"])
        memory = HindsightMemory(
            bank_id="b",
            client=client,
            recall_budget="mid",
            recall_max_tokens=512,
            recall_types=["world", "experience"],
            tags=["voice"],
            recall_tags=["user:1"],
            recall_tags_match="all_strict",
        )
        result = await memory.recall("query")

        assert result is not None and "1. fact" in result
        assert client.arecall.call_args.kwargs == {
            "bank_id": "b",
            "query": "query",
            "types": ["world", "experience"],
            "budget": "mid",
            "max_tokens": 512,
            "tags": ["user:1"],
            "tags_match": "all_strict",
        }

    async def test_voice_friendly_defaults(self):
        client = mock_client(["fact"])
        await HindsightMemory(bank_id="b", client=client).recall("q")
        kwargs = client.arecall.call_args.kwargs
        assert kwargs["budget"] == "low"
        assert kwargs["max_tokens"] == 1024
        assert kwargs["tags"] is None

    async def test_recall_timeout_returns_none(self):
        client = mock_client([])

        async def slow(**kwargs: Any) -> None:
            await asyncio.sleep(5)

        client.arecall = slow
        memory = HindsightMemory(bank_id="b", client=client, recall_timeout=0.05)
        assert await memory.recall("q") is None

    async def test_new_bank_is_not_a_warning(self, caplog: pytest.LogCaptureFixture):
        client = mock_client([])
        client.arecall = AsyncMock(side_effect=NotFoundException(status=404, reason="Not Found"))
        memory = HindsightMemory(bank_id="new-user", client=client)

        assert await memory.recall("q") is None
        assert not [r for r in caplog.records if r.levelname == "WARNING"]

    async def test_inject_without_user_message_skips_recall(self):
        client = mock_client(["fact"])
        memory = HindsightMemory(bank_id="b", client=client)
        ctx = ChatContext()
        ctx.add_message(role="system", content="instructions")

        assert await memory.inject_memories(ctx) is ctx
        client.arecall.assert_not_awaited()

    async def test_inject_does_not_mutate_input(self):
        client = mock_client(["fact"])
        memory = HindsightMemory(bank_id="b", client=client)
        ctx = ChatContext()
        ctx.add_message(role="user", content="hello")

        enriched = await memory.inject_memories(ctx)
        assert len(ctx.items) == 1
        assert len(enriched.items) == 2
        assert _MEMORY_MARKER in (enriched.items[0].text_content or "")

    async def test_revised_transcript_gets_fresh_recall(self):
        client = mock_client(["fact"])
        memory = HindsightMemory(bank_id="b", client=client)
        ctx = ChatContext()
        msg = ctx.add_message(role="user", content="book a table")

        await memory.inject_memories(ctx)
        await memory.inject_memories(ctx)
        assert client.arecall.await_count == 1

        msg.content = ["book a table for two"]
        await memory.inject_memories(ctx)
        assert client.arecall.await_count == 2


class TestRetain:
    def _event(self, role: str, text: str, created_at: float = 1_700_000_000.0) -> MagicMock:
        event = MagicMock()
        event.item = llm.ChatMessage(role=role, content=[text], created_at=created_at)
        return event

    async def test_user_line_without_reply_is_flushed_on_close(self):
        client = mock_client([])
        memory = HindsightMemory(bank_id="b", client=client, document_id="d")
        memory._on_conversation_item_added(self._event("user", "bye"))
        client.aretain.assert_not_awaited()

        await memory.aclose()
        client.aretain.assert_awaited_once()
        assert client.aretain.call_args.kwargs["content"] == "User (2023-11-14T22:13:20+00:00): bye"
        client.aclose.assert_not_awaited()  # caller owns the client

    async def test_late_user_transcript_is_stored_in_order(self):
        """Realtime models can add the user's transcript after the reply it prompted."""
        client = mock_client([])
        memory = HindsightMemory(bank_id="b", client=client)
        t = 1_700_000_000.0
        memory._on_conversation_item_added(self._event("assistant", "Hello! How can I help?", t))
        memory._on_conversation_item_added(self._event("assistant", "Bangalore is lovely.", t + 3))
        client.aretain.assert_not_awaited()  # no user line yet: keep waiting
        memory._on_conversation_item_added(self._event("user", "I live in Bangalore.", t + 1))
        await memory.flush()

        client.aretain.assert_awaited_once()
        lines = client.aretain.call_args.kwargs["content"].splitlines()
        assert [line.split(": ", 1)[1] for line in lines] == [
            "Hello! How can I help?",
            "I live in Bangalore.",
            "Bangalore is lovely.",
        ]

    async def test_non_message_items_are_ignored(self):
        client = mock_client([])
        memory = HindsightMemory(bank_id="b", client=client)
        event = MagicMock()
        event.item = llm.FunctionCall(name="f", arguments="{}", call_id="c")
        memory._on_conversation_item_added(event)
        memory._on_conversation_item_added(self._event("system", "instructions"))
        await memory.flush()
        client.aretain.assert_not_awaited()

    async def test_custom_speaker_names(self):
        client = mock_client([])
        memory = HindsightMemory(bank_id="b", client=client, user_name="Caller", agent_name="Ava")
        memory._on_conversation_item_added(self._event("user", "hi"))
        memory._on_conversation_item_added(self._event("assistant", "hello", 1_700_000_001.0))
        await memory.flush()

        lines = client.aretain.call_args.kwargs["content"].splitlines()
        assert lines[0].startswith("Caller (")
        assert lines[1].startswith("Ava (")

    async def test_appends_are_sequential(self):
        client = mock_client([])
        active = 0
        max_active = 0
        order: list[str] = []

        async def retain(**kwargs: Any) -> None:
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            order.append(kwargs["content"])
            active -= 1

        client.aretain = retain
        memory = HindsightMemory(bank_id="b", client=client)
        for i in range(3):
            memory._on_conversation_item_added(self._event("user", f"q{i}", 1_700_000_000.0 + 2 * i))
            memory._on_conversation_item_added(self._event("assistant", f"a{i}", 1_700_000_001.0 + 2 * i))
        await memory.flush()

        assert max_active == 1
        assert [c.splitlines()[0].endswith(f"q{i}") for i, c in enumerate(order)] == [True] * 3

    async def test_retain_failure_is_swallowed(self):
        client = mock_client([])
        client.aretain = AsyncMock(side_effect=RuntimeError("boom"))
        memory = HindsightMemory(bank_id="b", client=client)
        memory._on_conversation_item_added(self._event("user", "hi"))
        memory._on_conversation_item_added(self._event("assistant", "hello"))
        await memory.flush()  # does not raise

    async def test_attach_to_second_session_raises(self):
        memory = HindsightMemory(bank_id="b", client=mock_client([]))
        async with AgentSession() as first, AgentSession() as second:
            memory.attach(first)
            memory.attach(first)  # idempotent
            with pytest.raises(HindsightLiveKitError):
                memory.attach(second)


class TestClientResolution:
    def test_explicit_client_wins(self):
        client = mock_client([])
        assert resolve_client(client, "http://ignored", None) is client

    def test_uses_global_config(self):
        configure(hindsight_api_url="http://configured:8888", api_key="hsk_cfg")
        with patch("hindsight_livekit._client.Hindsight") as cls:
            resolve_client(None, None, None)
        kwargs = cls.call_args.kwargs
        assert kwargs["base_url"] == "http://configured:8888"
        assert kwargs["api_key"] == "hsk_cfg"
        assert kwargs["user_agent"].startswith("hindsight-livekit/")

    def test_uses_env_vars(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("HINDSIGHT_API_URL", "http://env:8888")
        monkeypatch.setenv("HINDSIGHT_API_KEY", "hsk_env")
        with patch("hindsight_livekit._client.Hindsight") as cls:
            resolve_client(None, None, None)
        assert cls.call_args.kwargs["base_url"] == "http://env:8888"
        assert cls.call_args.kwargs["api_key"] == "hsk_env"

    def test_defaults_to_cloud(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("HINDSIGHT_API_URL", raising=False)
        monkeypatch.delenv("HINDSIGHT_API_KEY", raising=False)
        with patch("hindsight_livekit._client.Hindsight") as cls:
            resolve_client(None, None, None)
        assert cls.call_args.kwargs["base_url"] == "https://api.hindsight.vectorize.io"
        assert "api_key" not in cls.call_args.kwargs

    async def test_owned_client_is_closed(self):
        with patch("hindsight_livekit._client.Hindsight") as cls:
            cls.return_value = mock_client([])
            memory = HindsightMemory(bank_id="b", hindsight_api_url="http://x")
        await memory.aclose()
        cls.return_value.aclose.assert_awaited_once()

    async def test_configure_defaults_apply(self):
        configure(hindsight_api_url="http://x", recall_budget="high", recall_max_tokens=2048, recall_timeout=9.0)
        client = mock_client(["f"])
        memory = HindsightMemory(bank_id="b", client=client)
        await memory.recall("q")
        assert client.arecall.call_args.kwargs["budget"] == "high"
        assert client.arecall.call_args.kwargs["max_tokens"] == 2048
        assert memory._recall_timeout == 9.0
