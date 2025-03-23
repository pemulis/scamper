# Scamper

![Scamper, a cute but venomous Hispaniolan solenodon](scamper.png)

Scamper is an AI agent whose goal is to identify scam tokens and protect crypto investors. This agent was built with a new framework that more closely integrates Coinbase AgentKit and OpenAI Agents SDK, adding out-of-the-box web search functionality, thread history, and a web API and frontend interface.

In addition, Scamper creates a loop of internal agents for deeper research when asked about a token. Scamper sends the user prompt to a Coordinator, who generates research topics that are sent to Researchers. The Researchers search the web and write short reports, which are compiled by the Coordinator, and passed back to Scamper to share with the user. This is a simple but evocative demonstration of the power of the OpenAI Agents SDK, which allows you to create sophisticated agentic systems.