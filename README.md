# Scamper

![Scamper, a cute but venomous Hispaniolan solenodon](scamper.png)

**Live Demo**: [Scamper on Heroku](https://scamper-e1d5d599d982.herokuapp.com/)

username: judge

password: ethglobal

Scamper is an AI agent whose goal is to **identify scam tokens** and **protect crypto investors**. It’s built with a framework that closely integrates Coinbase AgentKit and the OpenAI Agents SDK][openai-agents-sdk, providing out-of-the-box **web search**, **thread history**, and a **web API + frontend interface**.

In addition, Scamper employs a **multi-agent research loop** when asked about a token. Here’s how it works:

1. **Scamper** (the main agent) receives the user prompt (e.g., “Is XYZ token a scam?”).  
2. **Coordinator Agent** splits the request into subtopics and shares them with specialized **Researchers**.  
3. **Researchers** perform web searches, gather data, and summarize.  
4. **Coordinator** combines their reports into a concise final answer.  
5. The final response is handed back to **Scamper**, who presents it to the user.  

This mini‐ecosystem is a powerful demonstration of the **OpenAI Agents SDK** for building sophisticated multi-agent solutions with minimal boilerplate. Scamper's integration with Coinbase AgentKit means these sophisticated multi-agent flows can be used to control onchain agentic applications.

---

## Features

- **Scam Detection**: Flag suspicious tokens or tokens with red flags.  
- **Web-Search Enabled**: Researchers leverage integrated search to fetch live info.  
- **Thread History**: The conversation context is maintained for a more natural flow.  
- **Multi-Agent Orchestration**: Dynamic creation of sub-agents (Coordinator and multiple Researchers).

---

## How It Works

```text
User --> Scamper (Main Agent) --> multi_agent_research (Tool)
               |
               +--> Coordinator Agent --> (1) Subtopic "X"
               |                         \-> (2) Subtopic "Y"
               +--> Researcher Agents perform searches, short reports
               |
               +--> Final summary to user
```

- Scamper is the user-facing persona, deciding whether more in-depth research is needed.
- Coordinator breaks down the question into subtopics.
- Researchers each do a web search on their assigned subtopic, returning concise findings.
- Coordinator merges the findings into one final summary, which Scamper then sends to the user.

## Running Locally

1. Clone this repo
```
git clone https://github.com/pemulis/scamper.git
cd scamper
```

2. Install dependencies
```
pip install -r requirements.txt
```

*(Or use your preferred environment setup.)*

3. Set environment variables (optional):
```
export USERNAME="myusername"
export PASSWORD="mypassword"
```

4. Start the server (FastAPI + Uvicorn):
```
uvicorn main:app --host 0.0.0.0 --port 8000
```

5. Open http://localhost:8000 in your browser to interact with Scamper. You’ll be prompted for basic auth credentials if you configured them.

## Deploying

### Heroku
1. Create a new Heroku app and add your environment variables (`USERNAME`, `PASSWORD`, `CDP_API_KEY_NAME`, `CDP_API_KEY_PRIVATE_KEY`, `OPENAI_API_KEY` etc.) in the Heroku dashboard.

### Docker

1. Build and tag the Docker image:
```
docker build -t yourusername/scamper:latest .
```

2. (Optional) Push to Docker Hub:
```
docker push yourusername/scamper:latest
```

3. Run:
```
docker run -p 8000:8000 yourusername/scamper:latest
```

## Roadmap

- User accounts
- Saved threads
- Longer running deep research loops
- Computer using agents
- File search with vector databases
- External tools
- External agents
- Paid (or token-gated) API access capabilities
