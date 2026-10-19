"""STT-LLM-TTS voice agent with Hindsight memory, based on livekit-examples/agent-starter-python.

For a realtime (speech-to-speech) model, see realtime_agent.py.

Setup::

    pip install hindsight-livekit python-dotenv
    # .env.local
    LIVEKIT_URL=wss://<your-project>.livekit.cloud
    LIVEKIT_API_KEY=...
    LIVEKIT_API_SECRET=...
    HINDSIGHT_API_URL=https://api.hindsight.vectorize.io   # or http://localhost:8888
    HINDSIGHT_API_KEY=hsk_...                              # not needed for a local server
    HINDSIGHT_BANK_ID=demo-user                            # optional, defaults to livekit-demo-user

    lk agent console examples/voice_agent.py   # talk to it in your terminal
    lk agent dev examples/voice_agent.py       # connect it to LiveKit Cloud

    # then join a room with the agent dispatched to it:
    lk token create --identity me --room hindsight-demo --agent hindsight-agent --join --open meet

Tell it something about yourself, hang up, start again, and ask what it remembers.

Memory is stored per bank. This demo uses one fixed bank so every call shares memory.
A real app should use one bank per end user.
"""

import logging
import os

from dotenv import load_dotenv
from livekit.agents import AgentServer, AgentSession, JobContext, TurnHandlingOptions, cli, inference

from hindsight_livekit import HindsightAgent, HindsightMemory

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(HindsightAgent):
    def __init__(self, memory: HindsightMemory) -> None:
        super().__init__(
            memory=memory,
            llm=inference.LLM(model="google/gemma-4-31b-it"),
            instructions=(
                "You are a friendly voice assistant with long-term memory of past calls. "
                "When remembered facts are relevant, use them naturally without listing them. "
                "Respond in plain text only, in one to three short sentences."
            ),
        )

    async def on_enter(self) -> None:
        # Recall what's known about the user first, so the greeting can use it.
        await super().on_enter()
        self.session.generate_reply(instructions="Greet the user briefly, by name if you know it.")


server = AgentServer()


@server.rtc_session(agent_name="hindsight-agent")
async def entrypoint(ctx: JobContext) -> None:
    # One bank per end user. In production, derive it from your own user id (e.g. the
    # participant identity you put in the access token).
    bank_id = os.environ.get("HINDSIGHT_BANK_ID", "livekit-demo-user")
    logger.info(f"using Hindsight bank {bank_id}")

    # Same models as livekit-examples/agent-starter-python (LiveKit Inference).
    session = AgentSession(
        stt=inference.STT(model="assemblyai/universal-3-6-pro", language="en"),
        tts=inference.TTS(model="fishaudio/s2.1-pro", voice="fa4c9eb3dccc4806b382b40d61c6b10a"),
        turn_handling=TurnHandlingOptions(turn_detection=inference.TurnDetector()),
    )

    # One Hindsight document per call, so each conversation is stored as one transcript.
    memory = HindsightMemory(bank_id=bank_id, document_id=ctx.room.name)
    memory.attach(session)
    ctx.add_shutdown_callback(memory.aclose)

    await session.start(agent=Assistant(memory), room=ctx.room)
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
