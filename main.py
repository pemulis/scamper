# main.py
import asyncio
from fastapi import FastAPI, Request
from pydantic import BaseModel
from agent_script import initialize_agent, run_chat
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# app.mount("/static", StaticFiles(directory="static"), name="static")

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

@app.get("/", response_class=HTMLResponse)
def get_index():
    with open("index.html", "r") as f:
        html_content = f.read()
    return html_content