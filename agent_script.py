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
    The Coordinator produces one research subtopic (one to three sentences) at a time,
    specifically instructing the researcher to do a web search for the token or relevant details.
    After producing up to five subtopics, it says 'DONE'.
    Finally, it will incorporate the research findings into a single concise final answer.
    """
    return Agent(
        name="Coordinator",
        instructions=(
            "You are the Coordinator. The user is asking about a token or something needing deeper research.\n\n"
            "Your job is:\n"
            "1) Produce one research subtopic prompt at a time (up to 2). Each subtopic should be a short imperative instruction—"
            "1 to 3 sentences—telling the researcher what to research about the token.\n"
            "2) If you have no more subtopics, respond with 'DONE'.\n"
            "3) Later, you will receive the research results for each subtopic and produce a final, concise answer.\n\n"
            "DO NOT reveal chain-of-thought. Produce ONLY either a subtopic instruction or 'DONE' each time."
        ),
        tools=[],
    )

def create_researcher_agent(agentkit, researcher_name: str):
    """
    Each researcher can have web search or other specialized tools
    to gather data about a single subtopic. They must do a search
    based on that subtopic instruction.
    """
    researcher_tools = get_openai_agents_sdk_tools(agentkit)
    researcher_tools.append(WebSearchTool())

    return Agent(
        name=researcher_name,
        instructions=(
            f"You are {researcher_name}, a specialized researcher. "
            "You MUST perform a web search for each subtopic you receive. "
            "The subtopic will be a short imperative instruction. "
            "Use the WebSearchTool and any relevant actions to find up-to-date info. "
            "Then summarize your findings concisely. If you find nothing relevant, say so. "
            "Do not reveal chain-of-thought. Return only your final summary."
            "Do not take sources at face value, since there is false information on the Internet."
            "Be cautious and cross-reference information."
            "Remember that many tokens and coins have similar names. Make sure you research the "
            "specific token or coin you were asked about, not a coin with a similar name."
        ),
        tools=researcher_tools,
    )

@function_tool
async def multi_agent_research(user_prompt: str) -> str:
    """
    Perform deep multi-agent token research. Only use this tool if the user
    is asking about a token or needs advanced parallel research.

    This function:
    1) Iteratively asks a Coordinator agent for one subtopic at a time (up to 2).
       The coordinator responds with either a subtopic name or 'DONE'.
    2) For each subtopic, we spawn a "Researcher" agent to investigate it.
    3) We collect all researcher reports, then feed them back to the Coordinator
       for a final, concise summary.
    4) Returns that final summary text.

    Args:
        user_prompt: The text from the user describing the token
            (or any context that needs deeper parallel research).
    """
    # Using ANSI color codes for highlight:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"

    print(f"{GREEN}[DEBUG]{RESET} Starting multi_agent_research with user_prompt:")
    print(f"        {CYAN}{user_prompt}{RESET}")

    agentkit = create_agentkit()
    coordinator = create_coordinator_agent()

    conversation = [
        {"role": "user", "content": f"{user_prompt}\n\nPlease produce the first subtopic."}
    ]
    subtopic_research_results = []

    for i in range(1, 3):
        print(f"{GREEN}[DEBUG]{RESET} Requesting subtopic #{i} from Coordinator...")
        with trace(f"Coordinator: Subtopic #{i}"):
            subtopic_result = await Runner.run(coordinator, conversation)

        subtopic_text = subtopic_result.final_output.strip()

        print(f"{YELLOW}[DEBUG]{RESET} Coordinator returned subtopic #{i}: '{subtopic_text}'")

        if not subtopic_text or subtopic_text.upper() == "DONE":
            print(f"{GREEN}[DEBUG]{RESET} Coordinator indicated we are done with subtopics.")
            break

        researcher_name = f"Researcher_{i}"
        print(f"{GREEN}[DEBUG]{RESET} Creating {researcher_name} for subtopic '{subtopic_text}'")
        researcher = create_researcher_agent(agentkit, researcher_name)

        with trace(f"{researcher_name} analyzing {subtopic_text[:40]}"):
            researcher_output = await Runner.run(
                researcher,
                [{"role": "user", "content": subtopic_text}],
            )

        research_text = researcher_output.final_output
        print(f"{YELLOW}[DEBUG]{RESET} {researcher_name} returned:\n{CYAN}{research_text}{RESET}\n")

        subtopic_research_results.append((subtopic_text, research_text))

        new_messages = subtopic_result.to_input_list()
        new_messages.append({
            "role": "user",
            "content": (
                f"Subtopic '{subtopic_text}' is researched. Here is the result:\n\n"
                f"{research_text}\n\n"
                "Now produce the next subtopic. If none, say 'DONE'."
            )
        })
        conversation = new_messages

    compiled_text = "Here are the subtopics and their research:\n\n"
    for idx, (subtopic, research) in enumerate(subtopic_research_results, start=1):
        compiled_text += f"{idx}. {subtopic}\nResearch: {research}\n\n"

    print(f"{GREEN}[DEBUG]{RESET} Compiled research so far:\n{CYAN}{compiled_text}{RESET}")

    final_prompt = (
        f"The user prompt was:\n{user_prompt}\n\n"
        f"{compiled_text}\n"
        "Please combine these findings into one concise answer for the user."
    )

    final_conversation = conversation + [{"role": "user", "content": final_prompt}]

    print(f"{GREEN}[DEBUG]{RESET} Requesting final summary from Coordinator...")
    with trace("Coordinator: Final Summary"):
        final_result = await Runner.run(coordinator, final_conversation)

    print(f"{YELLOW}[DEBUG]{RESET} Final summary from Coordinator:\n{CYAN}{final_result.final_output}{RESET}")
    return final_result.final_output

def initialize_agent():
    """
    Build and return the main 'Scamper' agent that the user interacts with.
    - This agent can do onchain calls, web searches, etc.
    - It also has access to the 'multi_agent_research' tool for deeper analysis.
    """
    agentkit = create_agentkit()
    base_tools = get_openai_agents_sdk_tools(agentkit)
    base_tools.append(WebSearchTool())

    all_tools = base_tools + [multi_agent_research]

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
            "If you suspect the user is asking about a token or any other cryptocurrency, or complex research is needed, call the "
            "'multi_agent_research' tool with the entire user prompt. Then incorporate the returned results."
        ),
        tools=all_tools
    )

    return agent

async def run_chat(agent, user_input: str, history: list = None):
    if history:
        inputs = history + [{"role": "user", "content": user_input}]
    else:
        inputs = [{"role": "user", "content": user_input}]

    output = await Runner.run(agent, inputs)
    return output
