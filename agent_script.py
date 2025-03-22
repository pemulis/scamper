# agent_script.py
import json
import os
import asyncio
from dotenv import load_dotenv
from coinbase_agentkit import (
    AgentKit,
    AgentKitConfig,
    CdpWalletProvider,
    CdpWalletProviderConfig,
    cdp_api_action_provider,
    cdp_wallet_action_provider,
    erc20_action_provider,
    pyth_action_provider,
    wallet_action_provider,
    weth_action_provider,
)
from coinbase_agentkit_openai_agents_sdk import get_openai_agents_sdk_tools
from agents.agent import Agent
from agents.run import Runner
from agents import WebSearchTool
# from agents import FileSearchTool
# from agents import ComputerTool

load_dotenv()

# File used to persist the agent's CDP API Wallet Data
WALLET_DATA_FILE = "wallet_data.txt"

def initialize_agent():
    """Initialize and return the chatbot agent."""
    wallet_data = None
    if os.path.exists(WALLET_DATA_FILE):
        with open(WALLET_DATA_FILE) as f:
            wallet_data = f.read()

    cdp_config = CdpWalletProviderConfig(wallet_data=wallet_data) if wallet_data else None
    wallet_provider = CdpWalletProvider(cdp_config)

    agentkit = AgentKit(
        AgentKitConfig(
            wallet_provider=wallet_provider,
            action_providers=[
                cdp_api_action_provider(),
                cdp_wallet_action_provider(),
                erc20_action_provider(),
                pyth_action_provider(),
                wallet_action_provider(),
                weth_action_provider(),
            ],
        )
    )

    # Re-export wallet data in case it changed
    wallet_data_json = json.dumps(wallet_provider.export_wallet().to_dict())
    with open(WALLET_DATA_FILE, "w") as f:
        f.write(wallet_data_json)

    # Add OpenAI tools
    tools = get_openai_agents_sdk_tools(agentkit)
    tools.append(WebSearchTool())
    # tools.append(FileSearchTool())
    # tools.append(ComputerTool())

    # Create the agent
    agent = Agent(
        name="Scamper",
        instructions=(
            "You are a helpful agent that can interact onchain using the Coinbase Developer Platform AgentKit. "
            "You are empowered to interact onchain using your tools. If you ever need funds, you can request "
            "them from the faucet if you are on network ID 'base-sepolia'. If not, you can provide your wallet "
            "details and request funds from the user. Before executing your first action, get the wallet details "
            "to see what network you're on. If there is a 5XX (internal) HTTP error code, ask the user to try "
            "again later. If someone asks you to do something you can't do with your currently available tools, "
            "you must say so, and encourage them to implement it themselves using the CDP SDK + Agentkit, "
            "recommend they go to docs.cdp.coinbase.com for more information. Be concise and helpful with your "
            "responses. Refrain from restating your tools' descriptions unless it is explicitly requested. "
            "Your name is Scamper, and your goal is to help users identify scam tokens. You will be asked to "
            "evaluate whether or not tokens seem legitimate or fake, and you should give a firm answer. If you "
            "are given a web link, you should go to the website and evaluate the token based on the information "
            "you find there."
        ),
        tools=tools
    )

    return agent

async def run_chat(agent, user_input: str, history: list = None):
    """
    Run the agent with the given user input and optional conversation history.
    If a history list is provided, append the new user message to it before running the agent.
    """
    if history and isinstance(history, list) and len(history) > 0:
        # Append the new message to the existing conversation history.
        inputs = history + [{"role": "user", "content": user_input}]
    else:
        # If no history exists, use the prompt as the input.
        inputs = user_input

    # Run the agent with the provided input(s)
    output = await Runner.run(agent, inputs)
    return output
