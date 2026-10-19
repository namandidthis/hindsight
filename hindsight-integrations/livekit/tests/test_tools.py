"""Tests for create_hindsight_tools.

The tools run inside a real LiveKit AgentSession: the scripted LLM calls them,
LiveKit executes them, and the result is checked in the next LLM call's context.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from livekit.agents import Agent, AgentSession, llm
from livekit.agents.llm import FunctionToolCall
from livekit.agents.llm.utils import build_legacy_openai_schema

from hindsight_livekit import configure, create_hindsight_tools

from .conftest import FakeLLM, mock_client, tool_outputs


def _names(tools: list) -> list[str]:
    # ToolContext works across LiveKit versions (1.2 returns decorated functions,
    # later versions FunctionTool objects).
    return list(llm.ToolContext(tools).function_tools)


def _call(name: str, arguments: str) -> FunctionToolCall:
    return FunctionToolCall(name=name, arguments=arguments, call_id=f"call-{name}")


async def _run_tool_call(tools: list, call: FunctionToolCall) -> FakeLLM:
    fake = FakeLLM([call, "Done."])
    async with AgentSession(llm=fake) as session:
        await session.start(Agent(instructions="Be helpful.", tools=tools))
        await session.run(user_input="Go")
    return fake


class TestToolset:
    def test_names(self):
        tools = create_hindsight_tools(bank_id="b", client=mock_client())
        assert _names(tools) == ["hindsight_retain", "hindsight_recall", "hindsight_reflect"]

    def test_include_flags(self):
        tools = create_hindsight_tools(bank_id="b", client=mock_client(), include_retain=False, include_reflect=False)
        assert _names(tools) == ["hindsight_recall"]

    def test_model_only_sees_its_own_parameters(self):
        """LiveKit injects RunContext, so the schema sent to the model has one parameter per tool."""
        tools = create_hindsight_tools(bank_id="b", client=mock_client())
        params = {
            name: list(build_legacy_openai_schema(tool)["function"]["parameters"]["properties"])
            for name, tool in llm.ToolContext(tools).function_tools.items()
        }
        assert params == {
            "hindsight_retain": ["content"],
            "hindsight_recall": ["query"],
            "hindsight_reflect": ["query"],
        }


class TestRecallTool:
    async def test_recall_results_reach_the_llm(self):
        client = mock_client(["User has a dog named Bruno", "User lives in Bangalore"])
        tools = create_hindsight_tools(bank_id="user-1", client=client, recall_tags=["voice"], recall_types=["world"])

        fake = await _run_tool_call(tools, _call("hindsight_recall", '{"query": "dog name"}'))

        client.arecall.assert_awaited_once_with(
            bank_id="user-1",
            query="dog name",
            types=["world"],
            budget="low",
            max_tokens=1024,
            tags=["voice"],
            tags_match="any",
        )
        assert tool_outputs(fake.contexts[1]) == ["1. User has a dog named Bruno\n2. User lives in Bangalore"]

    async def test_no_results(self):
        fake = await _run_tool_call(
            create_hindsight_tools(bank_id="b", client=mock_client([])),
            _call("hindsight_recall", '{"query": "anything"}'),
        )
        assert tool_outputs(fake.contexts[1]) == ["No relevant memories found."]

    async def test_failure_is_reported_to_the_llm(self):
        client = mock_client()
        client.arecall = AsyncMock(side_effect=RuntimeError("connection refused"))
        fake = await _run_tool_call(
            create_hindsight_tools(bank_id="b", client=client),
            _call("hindsight_recall", '{"query": "anything"}'),
        )
        # LiveKit passes ToolError messages to the model so it can tell the caller.
        assert tool_outputs(fake.contexts[1]) == ["Could not search memory right now."]


class TestRetainTool:
    async def test_retain(self):
        client = mock_client()
        tools = create_hindsight_tools(bank_id="user-1", client=client, tags=["voice"])

        fake = await _run_tool_call(tools, _call("hindsight_retain", '{"content": "Anniversary is on May 5."}'))

        client.aretain.assert_awaited_once_with(
            bank_id="user-1",
            content="Anniversary is on May 5.",
            context="voice conversation",
            tags=["voice"],
        )
        assert tool_outputs(fake.contexts[1]) == ["Saved to memory."]

    async def test_failure_is_reported_to_the_llm(self):
        client = mock_client()
        client.aretain = AsyncMock(side_effect=RuntimeError("boom"))
        fake = await _run_tool_call(
            create_hindsight_tools(bank_id="b", client=client),
            _call("hindsight_retain", '{"content": "x"}'),
        )
        assert tool_outputs(fake.contexts[1]) == ["Could not save that to memory right now."]


class TestReflectTool:
    async def test_reflect(self):
        client = mock_client(reflect_text="Naman is a developer in Bangalore with a dog named Bruno.")
        tools = create_hindsight_tools(bank_id="user-1", client=client, budget="mid", max_tokens=512)

        fake = await _run_tool_call(tools, _call("hindsight_reflect", '{"query": "Summarise the user"}'))

        client.areflect.assert_awaited_once_with(
            bank_id="user-1",
            query="Summarise the user",
            budget="mid",
            max_tokens=512,
            tags=None,
            tags_match="any",
        )
        assert tool_outputs(fake.contexts[1]) == ["Naman is a developer in Bangalore with a dog named Bruno."]

    async def test_empty_answer(self):
        fake = await _run_tool_call(
            create_hindsight_tools(bank_id="b", client=mock_client(reflect_text="")),
            _call("hindsight_reflect", '{"query": "q"}'),
        )
        assert tool_outputs(fake.contexts[1]) == ["No relevant memories found."]


class TestToolConfig:
    async def test_configure_defaults_apply(self):
        configure(hindsight_api_url="http://x", recall_budget="high", recall_max_tokens=2048)
        client = mock_client(["f"])
        await _run_tool_call(
            create_hindsight_tools(bank_id="b", client=client),
            _call("hindsight_recall", '{"query": "q"}'),
        )
        assert client.arecall.call_args.kwargs["budget"] == "high"
        assert client.arecall.call_args.kwargs["max_tokens"] == 2048

    def test_builds_its_own_client(self):
        with patch("hindsight_livekit._client.Hindsight") as cls:
            create_hindsight_tools(bank_id="b", hindsight_api_url="http://localhost:8888")
        assert cls.call_args.kwargs["base_url"] == "http://localhost:8888"
