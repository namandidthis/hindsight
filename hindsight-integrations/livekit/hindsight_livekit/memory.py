"""Hindsight memory for LiveKit voice agents.

``HindsightMemory`` handles long-term memory for one ``AgentSession``:

* **Session recall** — :meth:`HindsightMemory.recall_session_context` looks up what
  is known about the user once, when the agent starts. ``HindsightAgent`` adds it to
  the agent's instructions. Works with STT-LLM-TTS pipelines and realtime models.
* **Per-turn recall** — :meth:`HindsightMemory.inject_memories` searches Hindsight
  with the user's latest message and returns a copy of the chat context with the
  results inserted just before that message. ``HindsightAgent`` calls it from
  ``llm_node``, which only STT-LLM-TTS pipelines use.
* **Retain** — :meth:`HindsightMemory.attach` listens to the session's
  ``conversation_item_added`` events and appends the conversation to one Hindsight
  document per session (``update_mode="append"``).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from hindsight_client import Hindsight
from hindsight_client_api.exceptions import NotFoundException
from livekit.agents import AgentSession, CloseEvent, ConversationItemAddedEvent
from livekit.agents.llm import ChatContext, ChatMessage

from ._client import resolve_client
from .config import (
    DEFAULT_RECALL_BUDGET,
    DEFAULT_RECALL_MAX_TOKENS,
    DEFAULT_RECALL_TAGS_MATCH,
    DEFAULT_RECALL_TIMEOUT,
    Budget,
    TagsMatch,
    get_config,
)
from .errors import HindsightLiveKitError

logger = logging.getLogger(__name__)

_MEMORY_MARKER = "<hindsight_memories>"
_DEFAULT_MEMORY_PREFIX = (
    "Relevant memories about the user from past conversations. "
    "Use them only when they help answer, and never read this list out loud."
)
_DEFAULT_SESSION_RECALL_QUERY = "Who is this user? Their name, preferences, and important facts about them."


@dataclass
class _TranscriptLine:
    role: str
    created_at: float
    text: str


def _last_user_message_index(chat_ctx: ChatContext) -> int | None:
    """Return the index of the last user message in ``chat_ctx``, or None."""
    for i in range(len(chat_ctx.items) - 1, -1, -1):
        item = chat_ctx.items[i]
        if item.type == "message" and item.role == "user":
            return i
    return None


def _to_datetime(created_at: float) -> datetime:
    return datetime.fromtimestamp(created_at, tz=timezone.utc)


class HindsightMemory:
    """Persistent Hindsight memory for a single LiveKit ``AgentSession``.

    Create one per session, attach it to the session so the conversation is
    retained, and give it to a :class:`~hindsight_livekit.HindsightAgent` so
    memories are recalled::

        memory = HindsightMemory(bank_id="user-123", document_id=ctx.room.name)
        memory.attach(session)
        ctx.add_shutdown_callback(memory.aclose)
        await session.start(agent=HindsightAgent(memory=memory, instructions="..."), room=ctx.room)

    Args:
        bank_id: Hindsight memory bank to read from and write to (usually one per end user).
        client: Pre-configured Hindsight client. If omitted, one is created from
            ``hindsight_api_url``/``api_key``, then ``configure()``, then the
            ``HINDSIGHT_API_URL``/``HINDSIGHT_API_KEY`` env vars.
        hindsight_api_url: API URL (used if no client provided).
        api_key: API key (used if no client provided).
        document_id: Hindsight document that this session's transcript is appended to.
            Defaults to a random ``livekit-<uuid>`` id; the room name is a good choice.
        recall_budget: Recall budget level — ``"low"``, ``"mid"``, ``"high"``.
            Defaults to ``"low"``, which Hindsight recommends for fast lookups.
        recall_max_tokens: Maximum tokens of memories injected per recall.
        recall_timeout: Seconds to wait for recall. On timeout the agent continues
            without memories instead of leaving the caller in silence.
        recall_types: Memory types to search (``"world"``, ``"experience"``,
            ``"observation"``). ``None`` searches all of them.
        tags: Tags attached to retained content.
        recall_tags: Tags to filter recall by. Defaults to no filter.
        recall_tags_match: How ``recall_tags`` are matched (any/all/any_strict/all_strict).
        retain_context: Short label describing the source, passed to Hindsight's
            fact extraction as ``context``.
        user_name: Speaker label for the user in the retained transcript.
        agent_name: Speaker label for the agent in the retained transcript.
        enable_session_recall: Recall what is known about the user once, when the
            agent starts, and add it to the agent's instructions.
        session_recall_query: Query used for session recall.
        enable_recall: Recall memories relevant to each user message before the
            LLM replies (STT-LLM-TTS pipelines only).
        enable_retain: Store the conversation as it happens.
        memory_prefix: Instruction placed above recalled memories.
    """

    def __init__(
        self,
        bank_id: str,
        *,
        client: Hindsight | None = None,
        hindsight_api_url: str | None = None,
        api_key: str | None = None,
        document_id: str | None = None,
        recall_budget: Budget | None = None,
        recall_max_tokens: int | None = None,
        recall_timeout: float | None = None,
        recall_types: list[str] | None = None,
        tags: list[str] | None = None,
        recall_tags: list[str] | None = None,
        recall_tags_match: TagsMatch = DEFAULT_RECALL_TAGS_MATCH,
        retain_context: str = "voice conversation",
        user_name: str = "User",
        agent_name: str = "Assistant",
        enable_session_recall: bool = True,
        session_recall_query: str = _DEFAULT_SESSION_RECALL_QUERY,
        enable_recall: bool = True,
        enable_retain: bool = True,
        memory_prefix: str = _DEFAULT_MEMORY_PREFIX,
    ) -> None:
        self._bank_id = bank_id
        self._owns_client = client is None
        self._client = resolve_client(client, hindsight_api_url, api_key)
        config = get_config()
        self._document_id = document_id or f"livekit-{uuid.uuid4().hex}"
        self._recall_budget: Budget = recall_budget or (config.recall_budget if config else DEFAULT_RECALL_BUDGET)
        self._recall_max_tokens = recall_max_tokens or (
            config.recall_max_tokens if config else DEFAULT_RECALL_MAX_TOKENS
        )
        self._recall_timeout = recall_timeout or (config.recall_timeout if config else DEFAULT_RECALL_TIMEOUT)
        self._recall_types = recall_types
        self._tags = tags
        self._recall_tags = recall_tags
        self._recall_tags_match = recall_tags_match
        self._retain_context = retain_context
        self._user_name = user_name
        self._agent_name = agent_name
        self._enable_session_recall = enable_session_recall
        self._session_recall_query = session_recall_query
        self._enable_recall = enable_recall
        self._enable_retain = enable_retain
        self._memory_prefix = memory_prefix

        self._session: AgentSession | None = None
        # Transcript lines not yet sent to Hindsight. See _on_conversation_item_added
        # for when they are flushed.
        self._pending: list[_TranscriptLine] = []
        # Appends to one document must not overlap (Hindsight rejects concurrent
        # appends), so they run one at a time, in order. An asyncio.Lock is safe here
        # because a HindsightMemory belongs to one AgentSession, which runs on one loop.
        self._retain_lock = asyncio.Lock()
        self._retain_tasks: set[asyncio.Task[None]] = set()
        # Recall result for the current user message, shared by every LLM call that
        # answers it (preemptive generation, tool-call follow-ups). Keyed on message
        # id and text, so a revised transcript gets a fresh recall.
        self._recall_message_id: str | None = None
        self._recall_query: str | None = None
        self._recall_task: asyncio.Task[str | None] | None = None
        self._session_recall_task: asyncio.Task[str | None] | None = None
        self._closed = False

    @property
    def bank_id(self) -> str:
        return self._bank_id

    @property
    def document_id(self) -> str:
        return self._document_id

    @property
    def client(self) -> Hindsight:
        return self._client

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------

    async def recall(self, query: str) -> str | None:
        """Search Hindsight and return a formatted memory block, or None.

        Never raises: errors and timeouts are logged and return None so the
        conversation continues without memories.
        """
        try:
            response = await asyncio.wait_for(
                self._client.arecall(
                    bank_id=self._bank_id,
                    query=query,
                    types=self._recall_types,
                    budget=self._recall_budget,
                    max_tokens=self._recall_max_tokens,
                    tags=self._recall_tags,
                    tags_match=self._recall_tags_match,
                ),
                timeout=self._recall_timeout,
            )
        except NotFoundException:
            # A new user's bank doesn't exist until the first retain creates it.
            logger.debug(f"Hindsight bank {self._bank_id!r} not found yet (no memories to recall)")
            return None
        except asyncio.TimeoutError:
            logger.warning(f"Hindsight recall timed out after {self._recall_timeout}s (continuing without memories)")
            return None
        except Exception as e:
            logger.warning(f"Hindsight recall failed (continuing without memories): {e}")
            return None

        if not response.results:
            return None
        lines = [_MEMORY_MARKER, self._memory_prefix]
        for i, result in enumerate(response.results, 1):
            lines.append(f"{i}. {result.text}")
        lines.append("</hindsight_memories>")
        return "\n".join(lines)

    async def recall_session_context(self) -> str | None:
        """Recall what is known about the user, for the start of a session.

        Returns a formatted memory block, or None if session recall is disabled or
        nothing is found. The lookup runs once per session (``attach()`` starts it
        early) and later calls return the same result. Never raises.
        """
        if not self._enable_session_recall:
            return None
        return await asyncio.shield(self._start_session_recall())

    def _start_session_recall(self) -> asyncio.Task[str | None]:
        if self._session_recall_task is None:
            self._session_recall_task = asyncio.create_task(self.recall(self._session_recall_query))
        return self._session_recall_task

    async def inject_memories(self, chat_ctx: ChatContext) -> ChatContext:
        """Return ``chat_ctx`` with recalled memories inserted before the last user message.

        The input context is not modified, so memories never enter the session's
        stored history. Recall runs at most once per user message. If there is no
        user message or nothing relevant is found, ``chat_ctx`` is returned as is.
        """
        if not self._enable_recall:
            return chat_ctx
        index = _last_user_message_index(chat_ctx)
        if index is None:
            return chat_ctx
        user_message = chat_ctx.items[index]
        query = user_message.text_content
        if not query or not query.strip():
            return chat_ctx

        if self._recall_task is None or self._recall_message_id != user_message.id or self._recall_query != query:
            self._recall_message_id = user_message.id
            self._recall_query = query
            self._recall_task = asyncio.create_task(self.recall(query))
        # Shield so a cancelled generation (e.g. a discarded preemptive reply)
        # doesn't cancel the recall the next generation is waiting on.
        memories = await asyncio.shield(self._recall_task)
        if memories is None:
            return chat_ctx

        # LiveKit's RAG example adds looked-up context as an assistant message
        # before the user's turn (docs.livekit.io/agents/build/external-data).
        enriched = chat_ctx.copy()
        enriched.items.insert(
            index,
            ChatMessage(
                role="assistant",
                content=[memories],
                created_at=user_message.created_at - 1e-6,
            ),
        )
        return enriched

    # ------------------------------------------------------------------
    # Retain
    # ------------------------------------------------------------------

    def attach(self, session: AgentSession) -> None:
        """Retain this session's conversation in Hindsight as it happens.

        Call once, before ``session.start()``. The transcript is appended to the
        Hindsight document ``document_id`` in the background. Also starts the
        session recall early, so it is usually ready by the time the agent starts.
        """
        if self._session is session:
            return
        if self._session is not None:
            raise HindsightLiveKitError("HindsightMemory is already attached to another AgentSession")
        self._session = session
        session.on("conversation_item_added", self._on_conversation_item_added)
        session.on("close", self._on_session_close)
        # LiveKit runs Agent.on_enter in the background and doesn't hold the first
        # reply for it, so a slow session recall could miss the first turn. Starting
        # it here overlaps it with the room and model connecting in session.start().
        if self._enable_session_recall:
            self._start_session_recall()

    def _on_conversation_item_added(self, event: ConversationItemAddedEvent) -> None:
        if not self._enable_retain or self._closed:
            return
        item = event.item
        if item.type != "message" or item.role not in ("user", "assistant"):
            return
        text = item.text_content
        if not text or not text.strip():
            return

        self._pending.append(_TranscriptLine(role=item.role, created_at=item.created_at, text=text.strip()))
        # Flush once the buffer holds both sides of the conversation. With an
        # STT-LLM-TTS pipeline the user's message is always added before the reply.
        # Realtime models may add the user's transcript only after the reply it
        # prompted, so flushing on every reply would store them out of order; waiting
        # for both roles and sorting by created_at keeps the transcript in order.
        roles = {line.role for line in self._pending}
        if "user" in roles and "assistant" in roles:
            self._schedule_retain()

    def _on_session_close(self, event: CloseEvent) -> None:
        self._schedule_retain()

    def _schedule_retain(self) -> None:
        if not self._pending:
            return
        lines = sorted(self._pending, key=lambda line: line.created_at)
        self._pending = []
        content = "\n".join(self._format_line(line) for line in lines)
        task = asyncio.create_task(self._retain(content, _to_datetime(lines[-1].created_at)))
        self._retain_tasks.add(task)
        task.add_done_callback(self._retain_tasks.discard)

    def _format_line(self, line: _TranscriptLine) -> str:
        # "Name (timestamp): text" is the format Hindsight's docs use for conversations,
        # so facts are attributed to the right speaker and time.
        speaker = self._user_name if line.role == "user" else self._agent_name
        timestamp = _to_datetime(line.created_at).isoformat(timespec="seconds")
        return f"{speaker} ({timestamp}): {line.text}"

    async def _retain(self, content: str, timestamp: datetime) -> None:
        """Append ``content`` to this session's document (errors are logged and swallowed)."""
        async with self._retain_lock:
            try:
                await self._client.aretain(
                    bank_id=self._bank_id,
                    content=content,
                    timestamp=timestamp,
                    context=self._retain_context,
                    document_id=self._document_id,
                    tags=self._tags,
                    update_mode="append",
                )
            except Exception as e:
                logger.warning(f"Hindsight retain failed: {e}")

    async def flush(self) -> None:
        """Send any buffered transcript lines and wait for in-flight retains to finish."""
        self._schedule_retain()
        if self._retain_tasks:
            await asyncio.gather(*self._retain_tasks, return_exceptions=True)

    async def aclose(self) -> None:
        """Flush pending retains and close the client if this object created it.

        Register it with ``ctx.add_shutdown_callback(memory.aclose)`` so the end
        of the call is stored before the job exits.
        """
        if self._closed:
            return
        await self.flush()
        self._closed = True
        for task in (self._recall_task, self._session_recall_task):
            if task is not None and not task.done():
                task.cancel()
        if self._owns_client:
            await self._client.aclose()
