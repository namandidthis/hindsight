"""Realtime (speech-to-speech) voice agent with Hindsight memory.

Realtime models answer straight from audio, so there is no step before each reply
where memories can be added. Instead this agent:

* recalls what is known about the user when the call starts (HindsightAgent adds it
  to the instructions), and
* gets ``hindsight_recall`` / ``hindsight_retain`` tools to look things up or save
  them mid-call.

The conversation is retained as it happens, as with the STT-LLM-TTS example.

Setup is the same as voice_agent.py::

    lk agent console examples/realtime_agent.py
    lk agent dev examples/realtime_agent.py
    lk token create --identity me --room hindsight-demo --agent hindsight-realtime-agent --join --open meet
"""

import logging
import os

from dotenv import load_dotenv
from livekit.agents import AgentServer, AgentSession, JobContext, cli, inference

from hindsight_livekit import HindsightAgent, HindsightMemory, create_hindsight_tools

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(HindsightAgent):
    def __init__(self, memory: HindsightMemory) -> None:
        super().__init__(
            memory=memory,
            # Any realtime model works: e.g. "google/gemini-live" or "xai/grok-voice".
            llm=inference.RealtimeModel(model="openai/gpt-realtime-mini"),
            # The reflect tool runs an LLM on the Hindsight server; leave it out for snappier calls.
            tools=create_hindsight_tools(bank_id=memory.bank_id, client=memory.client, include_reflect=False),
            instructions=(
                "You are a friendly voice assistant with long-term memory of past calls. "
                "When the user asks about themselves, their past, or their preferences, call "
                "hindsight_recall first. When they share something worth remembering, call "
                "hindsight_retain. Keep replies to one to three short sentences."
            ),
        )

    async def on_enter(self) -> None:
        # Recall what's known about the user first, so the greeting can use it.
        await super().on_enter()
        self.session.generate_reply(instructions="Greet the user briefly, by name if you know it.")


server = AgentServer()


@server.rtc_session(agent_name="hindsight-realtime-agent")
async def entrypoint(ctx: JobContext) -> None:
    bank_id = os.environ.get("HINDSIGHT_BANK_ID", "livekit-demo-user")
    logger.info(f"using Hindsight bank {bank_id}")

    session = AgentSession()

    memory = HindsightMemory(bank_id=bank_id, document_id=ctx.room.name)
    memory.attach(session)
    ctx.add_shutdown_callback(memory.aclose)

    await session.start(agent=Assistant(memory), room=ctx.room)
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
