# Hindsight LiveKit Integration

Persistent long-term memory for [LiveKit Agents](https://github.com/livekit/agents) voice AI via [Hindsight](https://vectorize.io/hindsight). Your agent knows what it learned in earlier calls, and stores each new conversation as it happens. Works with STT-LLM-TTS pipelines and realtime (speech-to-speech) models.

## Quick Start

```bash
pip install hindsight-livekit
```

> ✨ **Recommended: [Hindsight Cloud](https://ui.hindsight.vectorize.io)** — free tier, no self-hosting required. Sign up and grab an API key in under a minute.

```python
from livekit.agents import AgentServer, AgentSession, JobContext, inference
from hindsight_livekit import HindsightAgent, HindsightMemory

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(stt=..., tts=...)

    memory = HindsightMemory(
        bank_id="user-123",                    # one memory bank per end user
        document_id=ctx.room.name,             # one transcript per call
        hindsight_api_url="https://api.hindsight.vectorize.io",
        api_key="hsk_...",                     # or set HINDSIGHT_API_KEY
    )
    memory.attach(session)                     # retain the conversation
    ctx.add_shutdown_callback(memory.aclose)   # store the end of the call on hang-up

    await session.start(
        agent=HindsightAgent(
            memory=memory,
            llm=inference.LLM(model="google/gemma-4-31b-it"),
            instructions="You are a friendly voice assistant.",
        ),
        room=ctx.room,
    )
    await ctx.connect()
```

`HindsightAgent` is a regular LiveKit `Agent`: subclass it, add `@function_tool`s, override `on_enter` (call `await super().on_enter()` first), and so on.

Runnable agents built on [agent-starter-python](https://github.com/livekit-examples/agent-starter-python):

- [`examples/voice_agent.py`](./examples/voice_agent.py) — STT-LLM-TTS pipeline
- [`examples/realtime_agent.py`](./examples/realtime_agent.py) — realtime model with memory tools

### Self-hosting (local development)

```bash
docker run --rm -p 8888:8888 -e HINDSIGHT_API_LLM_API_KEY=$OPENAI_API_KEY ghcr.io/vectorize-io/hindsight:latest
```

```python
memory = HindsightMemory(bank_id="user-123", hindsight_api_url="http://localhost:8888")
```

See the [Hindsight installation guide](https://hindsight.vectorize.io/developer/installation) for other options. LiveKit itself can be [self-hosted](https://docs.livekit.io/transport/self-hosting/local/) too; the integration works the same either way.

## How It Works

```
memory.attach(session)
  └─ Session recall starts: "who is this user?"  (overlaps with session.start)

Agent starts (on_enter)
  └─ Known facts about the user are added to the agent's instructions

Before each LLM reply (llm_node)                       — STT-LLM-TTS pipelines only
  └─ Recall memories relevant to the user's latest message
       └─ Insert them before that message, in a copy of the chat context

As the conversation happens (conversation_item_added)
  └─ Append "User (time): …" / "Assistant (time): …" lines to the call's Hindsight document
```

| | STT-LLM-TTS pipeline | Realtime model |
|---|---|---|
| Recall when the agent starts | ✅ | ✅ |
| Recall before each reply | ✅ | — (no step before a realtime reply; use the tools) |
| Memory tools (`create_hindsight_tools`) | ✅ | ✅ |
| Retain the conversation | ✅ | ✅ (needs the model's input audio transcription for the user's side) |

**Session recall** runs once per call. `attach()` starts it, so it overlaps with LiveKit connecting the room and the model, and `HindsightAgent.on_enter` adds the result to the instructions. LiveKit runs `on_enter` in the background, so with a pipeline the first `llm_node` call also waits for the lookup and uses the instructions with memories, even if the user speaks first. A realtime model has no such step: a reply that starts before the lookup finishes (bounded by `recall_timeout`) won't include those memories, which is rare in practice since the lookup starts before the call connects.

**Per-turn recall** runs in [`llm_node`](https://docs.livekit.io/agents/build/nodes/), the step every LLM call in a pipeline goes through. That means:

- It works for spoken turns, chat text input and `session.generate_reply(user_input=...)`. The `on_user_turn_completed` hook only runs for spoken turns.
- It works with [preemptive generation](https://docs.livekit.io/agents/build/audio/#preemptive-generation): recall starts while LiveKit is still confirming the end of the user's turn, and the preemptive reply is kept. Editing the context in `on_user_turn_completed` would make LiveKit discard it.
- Memories are only added to a copy of the context for that call, so they never pile up in the session history. When the LLM calls a tool, the follow-up call reuses the same recall result.

As in LiveKit's [RAG example](https://docs.livekit.io/agents/build/external-data/), per-turn memories are added as an `assistant` message just before the user's message, wrapped in `<hindsight_memories>` tags with a note telling the model not to read them out.

**Retain** follows Hindsight's [guidance for conversations](https://hindsight.vectorize.io/developer/api/retain#retaining-a-conversation): each call is one document (`document_id`), written as `Name (timestamp): text` lines, grown with [`update_mode="append"`](https://hindsight.vectorize.io/developer/api/retain#update_mode) so Hindsight only extracts facts from the new part. Lines are sent once both the user and the agent have spoken, sorted by time, because realtime models can deliver the user's transcript after the reply it prompted. Appends are sent one at a time, in order, in the background, so they never delay a reply. `aclose()` sends whatever is left when the call ends.

**Failures never break the call.** If recall fails or takes longer than `recall_timeout`, the agent carries on without memories. If retain fails, the error is logged.

## Memory Tools

`create_hindsight_tools()` gives the model `hindsight_retain`, `hindsight_recall` and `hindsight_reflect` tools, like the other Hindsight framework integrations. They are the way for a realtime model to look things up mid-call, and work with pipelines too:

```python
from hindsight_livekit import HindsightAgent, create_hindsight_tools

agent = HindsightAgent(
    memory=memory,
    llm=inference.RealtimeModel(model="openai/gpt-realtime-mini"),
    tools=create_hindsight_tools(bank_id=memory.bank_id, client=memory.client, include_reflect=False),
    instructions=(
        "You have long-term memory of past calls. When the user asks about themselves, "
        "their past, or their preferences, call hindsight_recall first. When they share "
        "something worth remembering, call hindsight_retain."
    ),
)
```

Tell the model when to use the tools, as above: models call them far more reliably with explicit guidance. `hindsight_reflect` runs an LLM on the Hindsight server, so it is the slowest; leave it out for snappier calls. Tool errors are raised as LiveKit `ToolError`, so the model hears what went wrong and can tell the caller.

## Latency

Recall happens while a caller waits, so the defaults are tuned for voice:

- `recall_budget="low"`: Hindsight's [fast lookup](https://hindsight.vectorize.io/developer/api/recall#budget) setting.
- `recall_max_tokens=1024`: keeps the prompt small.
- `recall_timeout=3.0`: the longest a reply waits on memory before going ahead without it.

With preemptive generation turned on, most per-turn recall time overlaps with end-of-turn detection.

## Configuration

```python
HindsightMemory(
    bank_id="user-123",                  # Required: memory bank to use
    client=None,                         # Pre-built hindsight_client.Hindsight (optional)
    hindsight_api_url="...",             # Defaults to HINDSIGHT_API_URL, then Hindsight Cloud
    api_key="hsk_...",                   # Defaults to HINDSIGHT_API_KEY
    document_id=None,                    # Defaults to a random "livekit-<uuid>"
    recall_budget="low",                 # "low", "mid", or "high"
    recall_max_tokens=1024,              # Max tokens of memories per recall
    recall_timeout=3.0,                  # Seconds before carrying on without memories
    recall_types=None,                   # e.g. ["world", "experience"]; None = all
    tags=None,                           # Tags for retained content
    recall_tags=None,                    # Tags to filter recall by
    recall_tags_match="any",             # "any", "all", "any_strict", "all_strict"
    retain_context="voice conversation", # Source label that guides fact extraction
    user_name="User",                    # Speaker labels in the stored transcript
    agent_name="Assistant",
    enable_session_recall=True,          # Recall known facts when the agent starts
    session_recall_query="Who is this user? ...",
    enable_recall=True,                  # Recall before each reply (pipelines)
    enable_retain=True,                  # Store the conversation
)
```

### Global configuration

```python
from hindsight_livekit import configure

configure(
    hindsight_api_url="https://api.hindsight.vectorize.io",
    api_key="hsk_...",
    recall_budget="low",
)

memory = HindsightMemory(bank_id="user-123")
```

### Using your own `Agent` or `llm_node`

`memory.attach(session)` works with any `Agent`. To add recall to an agent that already overrides `llm_node` or `on_enter`, call the memory directly instead of subclassing `HindsightAgent`:

```python
from livekit.agents import Agent

class MyAgent(Agent):
    def __init__(self, memory: HindsightMemory) -> None:
        super().__init__(instructions="...")
        self.memory = memory

    async def on_enter(self):
        if memories := await self.memory.recall_session_context():
            await self.update_instructions(f"{self.instructions}\n\n{memories}")

    async def llm_node(self, chat_ctx, tools, model_settings):
        chat_ctx = await self.memory.inject_memories(chat_ctx)
        async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
            yield chunk
```

## Prerequisites

- `livekit-agents` 1.2 or later
- A running Hindsight instance, self-hosted or [Hindsight Cloud](https://ui.hindsight.vectorize.io)

## Running Tests

```bash
uv sync
uv run pytest tests -v
```

The unit tests run a real LiveKit `AgentSession` with a scripted LLM and a mocked Hindsight client. The live tests in `tests/test_live_integration.py` run against a real Hindsight server when `HINDSIGHT_API_URL` is set, and also against real pipeline and realtime models through LiveKit Inference when LiveKit Cloud credentials are set:

```bash
docker run --rm -p 8888:8888 -e HINDSIGHT_API_LLM_PROVIDER=none ghcr.io/vectorize-io/hindsight:latest
HINDSIGHT_API_URL=http://localhost:8888 uv run pytest tests -v

# optional: real models via LiveKit Inference
LIVEKIT_URL=wss://... LIVEKIT_API_KEY=... LIVEKIT_API_SECRET=... HINDSIGHT_API_URL=http://localhost:8888 uv run pytest tests -v
```
