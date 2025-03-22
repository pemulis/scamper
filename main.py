# main.py
import os
import secrets
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from agents.run import Runner  # Import Runner from the SDK
from agent_script import initialize_agent
import asyncio

app = FastAPI()

# For Basic Auth
security = HTTPBasic()
USERNAME = os.getenv("USERNAME") or "default_user"
PASSWORD = os.getenv("PASSWORD") or "default_password"

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, USERNAME)
    correct_password = secrets.compare_digest(credentials.password, PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )

# Initialize the agent once at server startup
agent = initialize_agent()

# A Pydantic model for the request body.
# `history` is a list of dicts. Each dict should have at least "role" and "content".
class ChatRequest(BaseModel):
    prompt: str
    history: list = []  # Defaults to an empty list if no history is provided.

@app.get("/", response_class=HTMLResponse)
def get_index(credentials: HTTPBasicCredentials = Depends(verify_credentials)):
    with open("index.html", "r") as f:
        html_content = f.read()
    return html_content

@app.post("/chat")
async def chat_endpoint(
    request: ChatRequest,
    credentials: HTTPBasicCredentials = Depends(verify_credentials),
):
    # Build the conversation thread:
    # If there is a history, it should already be in the format that the SDK expects.
    # We append the new user message.
    conversation = request.history + [{"role": "user", "content": request.prompt}]

    # Run the agent using the conversation context.
    # The input can be a list (of messages) or a string.
    result = await Runner.run(agent, conversation)

    # Retrieve the updated conversation history for the next turn.
    # This includes both the original history and any new messages produced by the agent.
    updated_history = result.to_input_list()

    return {"response": result.final_output, "history": updated_history}
