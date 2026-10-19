"""Live tests against a real Hindsight server (and, optionally, real LiveKit models).

Skipped unless HINDSIGHT_API_URL is set::

    docker run --rm -p 8888:8888 -e HINDSIGHT_API_LLM_PROVIDER=none ghcr.io/vectorize-io/hindsight:latest
    HINDSIGHT_API_URL=http://localhost:8888 uv run pytest tests/test_live_integration.py -v

``HINDSIGHT_API_LLM_PROVIDER=none`` stores raw chunks without fact extraction,
which is enough to check the retain → recall round trip without an LLM key.

The ``TestRealModels`` tests also need LiveKit Cloud credentials (LIVEKIT_URL,
LIVEKIT_API_KEY, LIVEKIT_API_SECRET) and run real models through LiveKit Inference:
an STT-LLM-TTS pipeline LLM and a realtime model.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest
from hindsight_client import Hindsight
from livekit.agents import AgentSession

try:
    from livekit.agents import inference
except ImportError:  # LiveKit Inference arrived in a livekit-agents 1.2.x patch release
    inference = None

from hindsight_livekit import HindsightAgent, HindsightMemory, create_hindsight_tools
from hindsight_livekit.memory import _MEMORY_MARKER

from .conftest import FakeLLM, instructions, memory_messages, tool_outputs

HINDSIGHT_API_URL = os.environ.get("HINDSIGHT_API_URL")
HAS_LIVEKIT = all(os.environ.get(k) for k in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"))
# Realtime models were added to LiveKit Inference after livekit-agents 1.2.
HAS_INFERENCE_REALTIME = inference is not None and hasattr(inference, "RealtimeModel")
requires_inference_realtime = pytest.mark.skipif(
    not HAS_INFERENCE_REALTIME, reason="this livekit-agents has no LiveKit Inference realtime models"
)

# Same guidance the README recommends for agents that use the memory tools.
MEMORY_TOOL_INSTRUCTIONS = (
    "You have long-term memory of past calls. When the user asks about themselves, "
    "their past, or their preferences, call hindsight_recall first."
)

pytestmark = pytest.mark.skipif(not HINDSIGHT_API_URL, reason="HINDSIGHT_API_URL not set")


@pytest.fixture
async def client():
    c = Hindsight(base_url=HINDSIGHT_API_URL, api_key=os.environ.get("HINDSIGHT_API_KEY"), timeout=120.0)
    yield c
    await c.aclose()


@pytest.fixture
async def bank_id(client: Hindsight):
    bank = f"livekit-test-{uuid.uuid4().hex[:8]}"
    yield bank
    try:
        await client.adelete_bank(bank)
    except Exception:
        pass


async def _wait_for_recall(client: Hindsight, bank_id: str, query: str, needle: str) -> bool:
    for _ in range(30):
        response = await client.arecall(bank_id=bank_id, query=query, budget="low", include_chunks=True)
        texts = [r.text for r in response.results] + [c.text for c in (response.chunks or {}).values()]
        if any(needle.lower() in t.lower() for t in texts):
            return True
        await asyncio.sleep(1)
    return False


async def _first_call(client: Hindsight, bank_id: str, llm: object, text: str) -> None:
    memory = HindsightMemory(bank_id=bank_id, client=client, document_id=f"{bank_id}-call-1", recall_timeout=30)
    async with AgentSession(llm=llm) as session:
        memory.attach(session)
        await session.start(HindsightAgent(memory=memory, instructions="Be helpful. Answer in one short sentence."))
        await session.run(user_input=text)
    await memory.aclose()


async def _wait_for_session_memories(agent: HindsightAgent) -> None:
    """Wait for on_enter to apply session memories (LiveKit runs it in the background)."""
    for _ in range(100):
        if _MEMORY_MARKER in agent.instructions:
            return
        await asyncio.sleep(0.1)
    raise AssertionError("session memories were never added to the instructions")


def _last_reply(result: object) -> str:
    replies = [e.item.text_content or "" for e in result.events if getattr(e, "type", "") == "message"]
    return replies[-1] if replies else ""


class TestHindsightServer:
    async def test_second_call_remembers_first_call(self, client: Hindsight, bank_id: str):
        await _first_call(
            client, bank_id, FakeLLM(["Nice to meet you!"]), "Hi, my name is Naman and I live in Bangalore."
        )
        assert await _wait_for_recall(client, bank_id, "Where does the user live?", "Bangalore")

        second = HindsightMemory(bank_id=bank_id, client=client, document_id=f"{bank_id}-call-2", recall_timeout=30)
        fake = FakeLLM(["You live in Bangalore."])
        async with AgentSession(llm=fake) as session:
            second.attach(session)
            agent = HindsightAgent(memory=second, instructions="Be helpful.")
            await session.start(agent)
            await _wait_for_session_memories(agent)
            await session.run(user_input="Where do I live?")
        await second.aclose()

        # Recalled at session start (instructions) and for the turn (conversation).
        assert "bangalore" in instructions(fake.contexts[0]).lower()
        turn_memories = memory_messages(fake.contexts[0])
        assert len(turn_memories) == 1
        assert _MEMORY_MARKER in turn_memories[0]
        assert "bangalore" in turn_memories[0].lower()

    async def test_appends_build_one_document(self, client: Hindsight, bank_id: str):
        document_id = f"{bank_id}-doc"
        memory = HindsightMemory(
            bank_id=bank_id, client=client, document_id=document_id, enable_recall=False, enable_session_recall=False
        )
        async with AgentSession(llm=FakeLLM(["Noted.", "Got it."])) as session:
            memory.attach(session)
            await session.start(HindsightAgent(memory=memory, instructions="Be helpful."))
            await session.run(user_input="My favorite color is teal.")
            await session.run(user_input="My dog is called Bruno.")
        await memory.aclose()

        document = await client.documents.get_document(bank_id=bank_id, document_id=document_id)
        text = document.original_text or ""
        assert "favorite color is teal" in text
        assert "dog is called Bruno" in text
        assert text.index("teal") < text.index("Bruno")

    async def test_tools_store_and_find(self, client: Hindsight, bank_id: str):
        from livekit.agents import Agent
        from livekit.agents.llm import FunctionToolCall

        tools = create_hindsight_tools(bank_id=bank_id, client=client, include_reflect=False)
        store = FunctionToolCall(
            name="hindsight_retain", arguments='{"content": "The user\'s anniversary is on May 5."}', call_id="c1"
        )
        async with AgentSession(llm=FakeLLM([store, "Saved."])) as session:
            await session.start(Agent(instructions="x", tools=tools))
            await session.run(user_input="Remember my anniversary is May 5.")
        assert await _wait_for_recall(client, bank_id, "When is the anniversary?", "May 5")

        find = FunctionToolCall(name="hindsight_recall", arguments='{"query": "anniversary date"}', call_id="c2")
        fake = FakeLLM([find, "It's May 5."])
        async with AgentSession(llm=fake) as session:
            await session.start(Agent(instructions="x", tools=tools))
            await session.run(user_input="When is my anniversary?")
        assert "May 5" in tool_outputs(fake.contexts[1])[0]


@pytest.mark.skipif(not HAS_LIVEKIT, reason="LiveKit Cloud credentials not set")
@pytest.mark.skipif(inference is None, reason="this livekit-agents has no LiveKit Inference")
class TestRealModels:
    async def test_pipeline_llm_remembers_across_calls(self, client: Hindsight, bank_id: str):
        llm = inference.LLM(model="google/gemma-4-31b-it")
        await _first_call(client, bank_id, llm, "Hi! I'm Naman and my golden retriever is called Bruno.")
        assert await _wait_for_recall(client, bank_id, "What is the dog's name?", "Bruno")

        memory = HindsightMemory(bank_id=bank_id, client=client, recall_timeout=30)
        async with AgentSession(llm=llm) as session:
            await session.start(HindsightAgent(memory=memory, instructions="Be helpful. Answer in one short sentence."))
            result = await session.run(user_input="What's my dog's name?")
        await memory.aclose()
        assert "bruno" in _last_reply(result).lower()

    @requires_inference_realtime
    @pytest.mark.parametrize("model", ["openai/gpt-realtime-mini", "google/gemini-live"])
    async def test_realtime_model_remembers_across_calls(self, client: Hindsight, bank_id: str, model: str):
        # Realtime models skip llm_node: memory comes from session recall (instructions).
        await _first_call(
            client,
            bank_id,
            inference.RealtimeModel(model=model, modalities=["text"]),
            "Hi! I'm Naman and I live in Bangalore.",
        )
        # The realtime model's own transcript items were retained.
        assert await _wait_for_recall(client, bank_id, "Where does the user live?", "Bangalore")

        memory = HindsightMemory(bank_id=bank_id, client=client, recall_timeout=30)
        async with AgentSession(llm=inference.RealtimeModel(model=model, modalities=["text"])) as session:
            memory.attach(session)
            agent = HindsightAgent(memory=memory, instructions="Be helpful. Answer in one short sentence.")
            await session.start(agent)
            await _wait_for_session_memories(agent)
            result = await session.run(user_input="Which city do I live in?")
        await memory.aclose()
        assert "bangalore" in _last_reply(result).lower()

    @requires_inference_realtime
    async def test_realtime_model_uses_recall_tool(self, client: Hindsight, bank_id: str):
        from livekit.agents import Agent

        await client.aretain(bank_id=bank_id, content="Naman's golden retriever is named Bruno.")
        assert await _wait_for_recall(client, bank_id, "dog name", "Bruno")

        tools = create_hindsight_tools(bank_id=bank_id, client=client, include_retain=False, include_reflect=False)
        llm = inference.RealtimeModel(model="openai/gpt-realtime-mini", modalities=["text"])
        async with AgentSession(llm=llm) as session:
            await session.start(Agent(instructions=MEMORY_TOOL_INSTRUCTIONS, tools=tools))
            result = await session.run(user_input="What's my dog's name?")
        calls = [e.item.name for e in result.events if getattr(e, "type", "") == "function_call"]
        assert calls == ["hindsight_recall"]
        assert "bruno" in _last_reply(result).lower()
