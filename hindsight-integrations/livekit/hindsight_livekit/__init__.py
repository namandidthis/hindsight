"""Hindsight-LiveKit: Persistent memory for LiveKit voice agents.

Recalls what Hindsight knows about the user when the agent starts and before
each LLM reply, and retains the conversation as it happens, one Hindsight
document per session. Optional tools let the model search and store memories
itself, which also works with realtime models.

Basic usage::

    from hindsight_livekit import HindsightAgent, HindsightMemory

    @server.rtc_session()
    async def entrypoint(ctx: JobContext):
        session = AgentSession(stt=..., llm=..., tts=...)

        memory = HindsightMemory(bank_id="user-123", document_id=ctx.room.name)
        memory.attach(session)                    # retain the conversation
        ctx.add_shutdown_callback(memory.aclose)  # flush on hang-up

        await session.start(
            agent=HindsightAgent(memory=memory, instructions="You are a helpful voice assistant."),
            room=ctx.room,
        )
"""

from .agent import HindsightAgent
from .config import (
    HindsightLiveKitConfig,
    configure,
    get_config,
    reset_config,
)
from .errors import HindsightLiveKitError
from .memory import HindsightMemory
from .tools import create_hindsight_tools

__version__ = "0.1.0"

__all__ = [
    "HindsightAgent",
    "HindsightLiveKitConfig",
    "HindsightLiveKitError",
    "HindsightMemory",
    "configure",
    "create_hindsight_tools",
    "get_config",
    "reset_config",
]
