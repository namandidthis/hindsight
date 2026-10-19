"""LiveKit ``Agent`` with Hindsight memory recall."""

from __future__ import annotations

from collections.abc import AsyncIterable
from typing import Any

from livekit.agents import Agent, ModelSettings, llm

from .memory import HindsightMemory


class HindsightAgent(Agent):
    """A LiveKit ``Agent`` that recalls Hindsight memories.

    Use it in place of ``Agent`` (or subclass it) and pass the session's
    :class:`HindsightMemory`::

        class Assistant(HindsightAgent):
            def __init__(self, memory: HindsightMemory) -> None:
                super().__init__(memory=memory, instructions="You are a helpful voice assistant.")

    Memory is recalled at two points:

    * **When the agent starts** (``on_enter``): what is known about the user is
      added to the agent's instructions. This works with STT-LLM-TTS pipelines and
      realtime models alike. If you override ``on_enter``, call
      ``await super().on_enter()`` first.
    * **Before each LLM reply** (``llm_node``): memories relevant to the user's
      latest message are added to a copy of the chat context for that call only.
      Realtime models don't use ``llm_node``, so this step only applies to
      STT-LLM-TTS pipelines; give realtime agents the tools from
      :func:`~hindsight_livekit.create_hindsight_tools` to look things up mid-call.
    """

    def __init__(self, *, memory: HindsightMemory, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._memory = memory
        # Instructions without recalled memories, so re-entering the agent (e.g. after
        # a handoff back to it) replaces the memory block instead of stacking another.
        self._base_instructions = self.instructions

    @property
    def memory(self) -> HindsightMemory:
        return self._memory

    async def on_enter(self) -> None:
        memories = await self._memory.recall_session_context()
        if memories is not None:
            await self.update_instructions(self._with_memories(memories))

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        model_settings: ModelSettings,
    ) -> AsyncIterable[llm.ChatChunk | str]:
        chat_ctx = await self._ensure_session_memories(chat_ctx)
        chat_ctx = await self._memory.inject_memories(chat_ctx)
        async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
            yield chunk

    def _with_memories(self, memories: str) -> str:
        return f"{self._base_instructions}\n\n{memories}"

    async def _ensure_session_memories(self, chat_ctx: llm.ChatContext) -> llm.ChatContext:
        """Make sure this LLM call sees the session memories.

        LiveKit runs ``on_enter`` in the background and doesn't hold the first reply
        for it, so a user who speaks first can get a reply built from the instructions
        as they were before ``on_enter`` added the memories. Wait for the (shared,
        already running) session recall and, if the instructions here are still the
        original ones, use the version with memories for this call.
        """
        memories = await self._memory.recall_session_context()
        if memories is None:
            return chat_ctx
        for i, item in enumerate(chat_ctx.items):
            if item.type == "message" and item.role == "system" and item.text_content == self._base_instructions:
                updated = chat_ctx.copy()
                # copy() shares item objects with the agent's history, so replace the
                # item rather than editing it.
                updated.items[i] = item.model_copy(update={"content": [self._with_memories(memories)]})
                return updated
        return chat_ctx
