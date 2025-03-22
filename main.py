# main.py
import asyncio
from fastapi import FastAPI, Request
from pydantic import BaseModel
from agent_script import initialize_agent, run_chat

app = FastAPI()

# Initialize the agent once at server startup
agent = initialize_agent()

# A Pydantic model for the request body
class ChatRequest(BaseModel):
    prompt: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    POST endpoint to interact with the agent.
    JSON body: { "prompt": "<User's question or command>" }
    """
    user_input = request.prompt
    response = await run_chat(agent, user_input)
    return {"response": response}

# Example home route to check if server is running
@app.get("/")
def read_root():
    return {"status": "OK", "message": "Welcome to the CDP Agent API!"}
