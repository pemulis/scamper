# main.py
import os
import secrets
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from agent_script import initialize_agent, run_chat

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# For Basic Auth
security = HTTPBasic()

# Read username/password from environment variables
USERNAME = os.getenv("USERNAME") or "default_user"
PASSWORD = os.getenv("PASSWORD") or "default_password"

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    """
    Verifies that provided Basic Auth credentials match
    the environment variables.
    """
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

# Pydantic model for the request body.
# "history" holds the conversation context as a list of messages.
class ChatRequest(BaseModel):
    prompt: str
    history: list = []

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
    # Pass both the new prompt and the conversation history
    result = await run_chat(agent, request.prompt, request.history)
    # Return the final output and the updated conversation history (to continue the thread)
    return {"response": result.final_output, "history": result.to_input_list()}
