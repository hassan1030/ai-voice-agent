from __future__ import annotations
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
import requests
import json

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import openai, silero

# Initialize logging
log = logging.getLogger("voice_agent")
logging.basicConfig(level=logging.INFO)

# Load environment variables
env_path = Path(__file__).resolve().parents[1] / ".env"
log.info(f"🔍 Looking for .env at: {env_path}")
load_dotenv(dotenv_path=env_path)

openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("Missing OpenAI API key")

# Load agent instructions
instructions_file = "instructions.txt"
instructions = open(instructions_file).read() if os.path.exists(instructions_file) else "You are a helpful assistant."

# Entrypoint for agent session
async def entrypoint(ctx: JobContext):
    await ctx.connect()
    log.info("✅ Connected to LiveKit room.")

    agent = Agent(instructions=instructions)
    silero_vad = silero.VAD.load()

    session = AgentSession(
        llm=openai.LLM(model="gpt-4o", api_key=openai_key),
        stt=openai.STT(api_key=openai_key),
        tts=openai.TTS(voice="shimmer", api_key=openai_key),
        vad=silero_vad,
    )

    await session.start(agent=agent, room=ctx.room)
    log.info("🗣 Agent session started.")

    await session.generate_reply(instructions="Greet the user and ask how you can help.")

    try:
        await session.run()
    finally:
        log.info("📞 Call ended or session closed. Attempting to send logs.")

        try:
            transcript = session.transcript
            caller = ctx.participant.identity if ctx.participant else "Unknown"
            session_id = ctx.job_id

            message_log = []
            for turn in transcript:
                if turn.user:
                    message_log.append({"user": turn.user})
                if turn.agent:
                    message_log.append({"agent": turn.agent})

            # Send to backend
            response = requests.post("http://127.0.0.1:8000/log", json={
                "session_id": session_id,
                "caller": caller,
                "message_log": message_log
            })
            log.info(f"📬 Sent transcript to backend. Response: {response.status_code}")

            # Write to local conversations.json
            local_log = {
                "session_id": session_id,
                "caller": caller,
                "message_log": message_log
            }

            log_file = Path(__file__).resolve().parents[1] / "conversations.json"

            if log_file.exists():
                with open(log_file, "r+") as f:
                    data = json.load(f)
                    data.append(local_log)
                    f.seek(0)
                    json.dump(data, f, indent=2)
            else:
                with open(log_file, "w") as f:
                    json.dump([local_log], f, indent=2)

            log.info("📝 Successfully wrote to conversations.json")

        except Exception as e:
            log.error(f"❌ Failed to log transcript: {e}")

# Run the agent
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, agent_name="inbound_agent"))
