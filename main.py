from dotenv import load_dotenv
from typing import Annotated, Dict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
import logging
from functools import partial
from src.app_messages import welcome_screen
from src.utils import generate_graph_image, load_config, validate_apikeys, validate_user_input, create_playlist_name
from src.prompts import RECOMMENDATION_PROMPT
from src.schemas import RecommendationResponse
import pandas as pd
from src.youtube_integration import YouTubePlaylistCreator
import json
from datetime import datetime

from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

TEMPERATURE = 0.8
load_dotenv()
validate_apikeys()


# Define the state
class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_question: str
    anthropic_response: str
    openai_response: str
    google_response: str
    final_answer: str
    prompt_attributes: Dict[str, str]


# Get response from any model
def get_model_response(state: State, model_provider, current_time, models) -> dict:
    """Get response with same System Message to specific Human Message for given Model Provider"""
    messages = [
        SystemMessage(content=RECOMMENDATION_PROMPT),
        HumanMessage(content=state["user_question"])
    ]

    if model_provider == "openai":
        # Use function calling method for OpenAI
        structured_llm = models[model_provider].with_structured_output(
            RecommendationResponse,
            method="function_calling"
        )
    else:
        structured_llm = models[model_provider].with_structured_output(RecommendationResponse)

    response = structured_llm.invoke(messages)

    # Generate timestamp and filename and model_outputs folder if not present

    output_dir = Path(__file__).parent / "model_outputs" / current_time

    output_dir.mkdir(exist_ok=True)

    filename = Path(__file__).parent / f"model_outputs/{current_time}/{model_provider}_response.json"

    # Dump response to JSON file
    response_dict = response.model_dump()

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(response_dict, f, indent=2, ensure_ascii=False)

    logging.info(f"{model_provider} response saved to {filename}")

    # input_tokens, output_tokens = count_tokens(model_provider, response)
    # logging.info(f"{model_provider.upper()} call summary: I: {input_tokens} input_tokens, O: {output_tokens} tokens")

    return {f"{model_provider}_response": response.model_dump()}


# Node 2: Combine responses
def analyze_responses(state: State, current_time: str) -> dict:
    """Prepare context for Google to analyze"""
    recommendations_df = pd.DataFrame()
    for model in ['anthropic', 'openai', 'google']:
        single_recommendation_df = pd.DataFrame(state[f'{model}_response']['recommendations'])
        single_recommendation_df['model'] = model
        recommendations_df = pd.concat([recommendations_df, single_recommendation_df])

    final_recommendations_df = recommendations_df.groupby(['song_title', 'artist', 'album', 'year'])['rank'].sum().reset_index()
    final_recommendations_df.columns = ['song_title', 'artist', 'album', 'year', 'total_points']
    final_recommendations_df = final_recommendations_df.sort_values(by='total_points', ascending=False)

    final_recommendations_df.to_csv(f'model_outputs/{current_time}/final_recommendations_df_{current_time}.csv')

    # Create YouTube playlist

    youtube_creator = YouTubePlaylistCreator()
    playlist_id = youtube_creator.create_playlist_from_dataframe(
        df=final_recommendations_df.head(20),  # Top 10 recommendations
        playlist_name=create_playlist_name(state['user_question']),
        song_col='song_title',
        artist_col='artist'
    )

    return {
        'final_recommendations': final_recommendations_df.to_dict(),
        'playlist_id': playlist_id
    }


# Build the graph
def build_graph(MODELS):
    workflow = StateGraph(State)

    # Setup timestamp to use it to align on same artifacts for single run
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    # Add nodes
    workflow.add_node("anthropic", partial(get_model_response, model_provider="anthropic", current_time=current_time, models=MODELS))
    workflow.add_node("openai", partial(get_model_response, model_provider="openai", current_time=current_time, models=MODELS))
    workflow.add_node("google", partial(get_model_response, model_provider="google", current_time=current_time, models=MODELS))
    workflow.add_node("analyze", partial(analyze_responses, current_time=current_time))

    # Add edges - all models run from START
    workflow.add_edge(START, "anthropic")
    workflow.add_edge(START, "openai")
    workflow.add_edge(START, "google")

    # All models  to analysis
    workflow.add_edge("anthropic", "analyze")
    workflow.add_edge("openai", "analyze")
    workflow.add_edge("google", "analyze")

    workflow.add_edge("analyze", END)

    return workflow.compile()


def main():

    CONFIG = load_config()

    welcome_screen(CONFIG['NO_OF_SONGS'])

    # Initialize models
    llm_anthropic = init_chat_model(model="anthropic:claude-haiku-4-5-20251001", temperature=CONFIG["TEMPERATURE"])
    llm_openai = init_chat_model(model="openai:gpt-4o", temperature=CONFIG["TEMPERATURE"])
    llm_google = init_chat_model(model="google_genai:gemini-pro-latest", temperature=CONFIG["TEMPERATURE"])

    MODELS = {'anthropic': llm_anthropic, 'openai': llm_openai, 'google': llm_google}

    app = build_graph(MODELS)
    generate_graph_image(app)
    song_attributes = ['genre', 'language', 'year', 'favorite_artists', 'hints']

    prompt_attributes = {}
    MAX_CHARS = CONFIG['MAX_CHARS']

    final_prompt = f"Please generate {CONFIG['NO_OF_SONGS']} song recommendations based on the following criteria: \n"

    for attribute in song_attributes:
        max_attempts = 3
        attempts = 0

        while attempts < max_attempts:
            user_input = input(f"\nPlease describe the {attribute}: ").strip()

            # Check length
            if len(user_input) > MAX_CHARS:
                print(f"⚠️  Input too long (max {MAX_CHARS} characters). Please try again.")
                attempts += 1
                continue

            # Check if empty
            if not user_input:
                print(f"⚠️  Input cannot be empty. Please try again.")
                attempts += 1
                continue

            # Validate with LLM
            print(f"🔍 Validating your input...")
            is_valid = validate_user_input(attribute, user_input)

            if is_valid:
                print(f"✅ Valid {attribute}")
                prompt_attributes[attribute] = user_input
                final_prompt += f"{attribute}: {user_input}\n"
                break
            else:
                attempts += 1
                remaining = max_attempts - attempts
                if remaining > 0:
                    print(
                        f"❌ Invalid {attribute}. Please provide a valid {attribute}. ({remaining} attempts remaining)")
                else:
                    print(f"❌ Maximum attempts reached. Using your last input anyway.")
                    prompt_attributes[attribute] = user_input
                    final_prompt += f"{attribute}: {user_input}\n"

    logging.info(f"Final prompt: {final_prompt}")

    logging.info("\n🔄 Processing your request...\n")

    # Initialize state
    initial_state = {
        "messages": [],
        "user_question": final_prompt,
        "anthropic_response": "",
        "openai_response": "",
        "google_response": "",
        "final_answer": "",
        "prompt_attributes": prompt_attributes
    }

    # Run the graph
    result = app.invoke(initial_state)

    return result
if __name__ == "__main__":
    main()

