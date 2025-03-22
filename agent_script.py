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
from agents.run import Runner, trace
from agents.tool import function_tool  # To define a Function Tool
from agents import WebSearchTool

load_dotenv()

# File used to persist the agent's CDP API Wallet Data
WALLET_DATA_FILE = "wallet_data.txt"

def _load_wallet_data():
    if os.path.exists(WALLET_DATA_FILE):
        with open(WALLET_DATA_FILE) as f:
            return f.read()
    return None

def _save_wallet_data(wallet_provider):
    wallet_data_json = json.dumps(wallet_provider.export_wallet().to_dict())
    with open(WALLET_DATA_FILE, "w") as f:
        f.write(wallet_data_json)

def create_agentkit():
    wallet_data = _load_wallet_data()
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
    _save_wallet_data(wallet_provider)
    return agentkit

def create_coordinator_agent():
    """
    The Coordinator produces one research subtopic at a time,
    and later synthesizes the final summary.
    """
    return Agent(
        name="Coordinator",
        instructions=(
            "You are the Coordinator. The user is asking about a token or some topic needing deeper research.\n\n"
            "Your responsibilities:\n"
            "1) Produce up to five subtopics, one at a time. If you have no more subtopics, say 'DONE'.\n"
            "2) For each subtopic, you only need to respond with the subtopic name. The rest of your chain of thought "
            "is hidden.\n"
            "3) Later, you will receive the research results for each subtopic and produce a final, concise summary.\n"
            "Do NOT reveal your chain-of-thought or any internal reasoning, just produce a single subtopic or 'DONE'."
        ),
        tools=[],
    )

def create_researcher_agent(agentkit, researcher_name: str):
    """
    Each researcher can have web search or other specialized tools to gather
    data about a single subtopic. They return a short summary of findings.
    """
    researcher_tools = get_openai_agents_sdk_tools(agentkit)
    researcher_tools.append(WebSearchTool())

    return Agent(
        name=researcher_name,
        instructions=(
            f"You are {researcher_name}, a specialized researcher with web search capabilities.\n"
            "Given a single subtopic, provide a concise summary of what you find. If you find nothing, say so.\n"
            "Be very succinct. Do not reveal chain-of-thought."
        ),
        tools=researcher_tools,
    )

@function_tool
async def multi_agent_research(user_prompt: str) -> str:
    """
    Perform deep multi-agent token research. Only use this tool if the user
    is asking about a token or needs advanced parallel research.

    This function:
    1) Iteratively asks a Coordinator agent for one subtopic at a time (up to 3).
       The coordinator responds with either a subtopic name or 'DONE'.
    2) For each subtopic, we spawn a "Researcher" agent to investigate it.
    3) We collect all researcher reports, then feed them back to the Coordinator
       for a final, concise summary.
    4) Returns that final summary text.

    Args:
        user_prompt: The text from the user describing the token
            (or any context that needs deeper parallel research).
    """
    print("[DEBUG] Starting multi_agent_research with user_prompt:")
    print("        ", user_prompt)

    # 1) Initialize the Coordinator and the conversation context
    agentkit = create_agentkit()
    coordinator = create_coordinator_agent()

    # We'll maintain a conversation list so the coordinator sees
    # the user prompt and the results of each subtopic's research.
    # Start the conversation by telling the coordinator about the user's prompt,
    # and that we want to find the first subtopic.
    conversation = [
        {"role": "user", "content": f"{user_prompt}\n\nPlease produce the first subtopic."}
    ]

    # We'll store (subtopic, researcher_result) pairs to feed back later
    subtopic_research_results = []

    # 2) Loop up to 3 subtopics
    for i in range(1, 4):
        print(f"[DEBUG] Requesting subtopic #{i} from Coordinator...")
        with trace(f"Coordinator: Subtopic #{i}"):
            subtopic_result = await Runner.run(coordinator, conversation)

        # The coordinator's entire message is presumably the subtopic (or 'DONE')
        subtopic_text = subtopic_result.final_output.strip()

        print(f"[DEBUG] Coordinator returned subtopic #{i}: '{subtopic_text}'")

        # If coordinator says "DONE" or produces an empty answer, break
        if not subtopic_text or subtopic_text.upper() == "DONE":
            print("[DEBUG] Coordinator indicated we are done with subtopics.")
            break

        # 2a) Create a researcher for this subtopic
        researcher_name = f"Researcher_{i}"
        print(f"[DEBUG] Creating {researcher_name} for subtopic '{subtopic_text}'")
        researcher = create_researcher_agent(agentkit, researcher_name)

        # 2b) Ask the researcher about this subtopic
        with trace(f"{researcher_name} analyzing {subtopic_text[:40]}"):
            researcher_output = await Runner.run(
                researcher,
                [{"role": "user", "content": subtopic_text}],
            )

        research_text = researcher_output.final_output
        print(f"[DEBUG] {researcher_name} returned:\n{research_text}\n")

        # Store the research result
        subtopic_research_results.append((subtopic_text, research_text))

        # 2c) Add the subtopic and the research result to the coordinator's conversation,
        # so it can incorporate that info before generating the next subtopic.
        # Then we ask for the next subtopic.
        new_messages = subtopic_result.to_input_list()  # the coordinator's prior role=assistant
        new_messages += [
            {
                "role": "user",
                "content": (
                    f"Subtopic '{subtopic_text}' is researched. Here is the result:\n\n"
                    f"{research_text}\n\n"
                    "Now produce the next subtopic. If none, say 'DONE'."
                )
            }
        ]
        conversation = new_messages

    # 3) Now we have all subtopics and their research. Time for a final summary.
    compiled_text = "Here are the subtopics and their research:\n\n"
    for idx, (subtopic, research) in enumerate(subtopic_research_results, start=1):
        compiled_text += f"{idx}. {subtopic}\nResearch: {research}\n\n"

    print("[DEBUG] Compiled research so far:\n", compiled_text)

    final_prompt = (
        f"The user prompt was:\n{user_prompt}\n\n"
        f"{compiled_text}\n"
        "Please combine these findings into one concise answer for the user."
    )

    # We can take the last coordinator conversation and append this final request
    final_conversation = conversation + [{"role": "user", "content": final_prompt}]

    print("[DEBUG] Requesting final summary from Coordinator...")
    with trace("Coordinator: Final Summary"):
        final_result = await Runner.run(coordinator, final_conversation)

    print("[DEBUG] Final summary from Coordinator:\n", final_result.final_output)
    return final_result.final_output

    """
    Perform deep multi-agent token research. Only use this tool if the user
    is asking about a token or needs advanced parallel research.

    This function:
    1) Iteratively asks a Coordinator agent for one subtopic at a time (up to 3).
       The coordinator responds with either a subtopic name or 'DONE'.
    2) For each subtopic, we spawn a "Researcher" agent to investigate it.
    3) We collect all researcher reports, then feed them back to the Coordinator
       for a final, concise summary.
    4) Returns that final summary text.

    Args:
        user_prompt: The text from the user describing the token
            (or any context that needs deeper parallel research).
    """
    # 1) Initialize the Coordinator and the conversation context
    agentkit = create_agentkit()
    coordinator = create_coordinator_agent()

    # We'll maintain a conversation list so the coordinator sees
    # the user prompt and the results of each subtopic's research.
    # Start the conversation by telling the coordinator about the user's prompt,
    # and that we want to find the first subtopic.
    conversation = [
        {"role": "user", "content": f"{user_prompt}\n\nPlease produce the first subtopic."}
    ]

    # We'll store (subtopic, researcher_result) pairs to feed back later
    subtopic_research_results = []

    # 2) Loop up to 3 subtopics
    for i in range(1, 4):
        with trace(f"Coordinator: Subtopic #{i}"):
            subtopic_result = await Runner.run(coordinator, conversation)

        # The coordinator's entire message is presumably the subtopic (or 'DONE')
        subtopic_text = subtopic_result.final_output.strip()

        # If coordinator says "DONE" or produces an empty answer, break
        if not subtopic_text or subtopic_text.upper() == "DONE":
            break

        # 2a) Create a researcher for this subtopic
        researcher = create_researcher_agent(agentkit, f"Researcher_{i}")

        # 2b) Ask the researcher about this subtopic
        with trace(f"Researcher_{i} analyzing {subtopic_text[:40]}"):
            researcher_output = await Runner.run(
                researcher,
                [{"role": "user", "content": subtopic_text}],
            )

        # Store the research result
        subtopic_research_results.append((subtopic_text, researcher_output.final_output))

        # 2c) Add the subtopic and the research result to the coordinator's conversation,
        # so it can incorporate that info before generating the next subtopic.
        # Then we ask for the next subtopic.
        new_messages = subtopic_result.to_input_list()  # the coordinator's prior role=assistant
        new_messages += [
            {
                "role": "user",
                "content": (
                    f"Subtopic '{subtopic_text}' is researched. Here is the result:\n\n"
                    f"{researcher_output.final_output}\n\n"
                    "Now produce the next subtopic. If none, say 'DONE'."
                )
            }
        ]
        conversation = new_messages

    # 3) Now we have all subtopics and their research. Time for a final summary.
    # We'll ask the coordinator to produce a single, concise summary.
    compiled_text = "Here are the subtopics and their research:\n\n"
    for idx, (subtopic, research) in enumerate(subtopic_research_results, start=1):
        compiled_text += f"{idx}. {subtopic}\nResearch: {research}\n\n"

    final_prompt = (
        f"The user prompt was:\n{user_prompt}\n\n"
        f"{compiled_text}\n"
        "Please combine these findings into one concise answer for the user."
    )

    # We can take the last coordinator conversation and append this final request
    final_conversation = conversation + [{"role": "user", "content": final_prompt}]

    with trace("Coordinator: Final Summary"):
        final_result = await Runner.run(coordinator, final_conversation)

    return final_result.final_output


def initialize_agent():
    """
    Build and return the main 'Scamper' agent that the user interacts with.
    - This agent can do onchain calls, web searches, etc.
    - It also has access to the 'multi_agent_research' tool for deeper analysis.
    """
    # Basic onchain + OpenAI tools
    agentkit = create_agentkit()
    base_tools = get_openai_agents_sdk_tools(agentkit)
    base_tools.append(WebSearchTool())

    # Our custom multi-agent tool
    all_tools = base_tools + [multi_agent_research]

    # Scamper instructions
    agent = Agent(
        name="Scamper",
        instructions=(
            "You are a helpful agent that can interact onchain using the Coinbase Developer Platform AgentKit. "
            "You are empowered to interact onchain using your tools. If you ever need funds, you can request "
            "them from the faucet if you are on network ID 'base-sepolia'. If not, you can provide your wallet "
            "details and request funds from the user. Before executing your first action, get the wallet details "
            "to see what network you're on. If there is a 5XX (internal) HTTP error code, ask the user to try "
            "again later. If someone asks you to do something you can't do with your currently available tools, "
            "you must say so, and encourage them to implement it themselves using the CDP SDK + AgentKit, "
            "recommend they go to docs.cdp.coinbase.com for more information. Be concise and helpful with your "
            "responses. Refrain from restating your tools' descriptions unless it is explicitly requested. "
            "\n\n"
            "Your name is Scamper, and your goal is to help users identify scam tokens. You will be asked to "
            "evaluate whether tokens seem legitimate or fake, and you should give a firm answer. If you are given "
            "a web link, you should go to the website and evaluate the token based on the information you find "
            "there.\n\n"
            "If you suspect the user is asking about a token or complex research is needed, call the "
            "'multi_agent_research' tool with the entire user prompt. Then incorporate the returned results."
        ),
        tools=all_tools
    )

    return agent

async def run_chat(agent: Agent, user_input: str, history: list = None):
    """
    Run the agent with the given user input and optional conversation history.
    If a history list is provided, append the new user message to it before running the agent.
    """
    if history:
        inputs = history + [{"role": "user", "content": user_input}]
    else:
        inputs = [{"role": "user", "content": user_input}]

    # Run the agent with the provided input(s)
    output = await Runner.run(agent, inputs)
    return output
