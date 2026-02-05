import logging
from datetime import datetime
from functools import partial

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END

from prompt_builder import create_prompt_builder_graph
from src.schemas import State
from src.tools import tools
from src.utils import load_config, validate_apikeys, get_model_response
from src.youtube_integration import analyze_responses

CONFIG = load_config()

prompt_builder_graph = create_prompt_builder_graph(CONFIG)

# Setup stuff

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Validate presence of API keys

load_dotenv()
validate_apikeys()
CONFIG = load_config()

# Build tools


MODELS = {m: init_chat_model(model=f"{m}:{CONFIG[f'{m.upper()}_MODEL']}",
                             temperature=CONFIG["TEMPERATURE"]).bind_tools(tools, tool_choice="web_search")
          for m in ['anthropic', 'google_genai', 'openai']}

def map_prompt_to_question(subgraph_output):
    """Map PromptBuilderState output to main State"""
    return {
        "user_question": subgraph_output.get("final_prompt", ""),
        "prompt_attributes": subgraph_output.get("prompt_attributes", {})
    }

# Build the main graph
graph = StateGraph(State)

# Setup timestamp to use it to align on same artifacts for single run
current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# Add nodes
graph.add_node("prompt_builder", prompt_builder_graph, output=map_prompt_to_question)
graph.add_node("anthropic",
               partial(get_model_response, model_provider="anthropic", current_time=current_time, models=MODELS,
                       script_config=CONFIG))
graph.add_node("openai", partial(get_model_response, model_provider="openai", current_time=current_time, models=MODELS,
                                 script_config=CONFIG))
graph.add_node("google", partial(get_model_response, model_provider="google_genai", current_time=current_time, models=MODELS,
                                 script_config=CONFIG))
graph.add_node("analyze", partial(analyze_responses, current_time=current_time))

# Add edges
# START -> prompt_builder
graph.add_edge(START, "prompt_builder")

# prompt_builder -> all three models in parallel
graph.add_edge("prompt_builder", "anthropic")
graph.add_edge("prompt_builder", "openai")
graph.add_edge("prompt_builder", "google")

# All models -> analysis
graph.add_edge("anthropic", "analyze")
graph.add_edge("openai", "analyze")
graph.add_edge("google", "analyze")

graph.add_edge("analyze", END)

app = graph.compile()
