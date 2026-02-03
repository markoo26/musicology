import json
import logging
import os
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

from src.prompts import VALIDATION_PROMPTS, RECOMMENDATION_PROMPT
from src.schemas import RecommendationResponse


def generate_graph_image(app):
    try:
        # Get the graph as PNG
        png_data = app.get_graph().draw_mermaid_png()

        # Save to file
        output_file = "langgraph_workflow.png"
        with open(output_file, "wb") as f:
            f.write(png_data)

        print(f"✅ Graph visualization saved to: {output_file}")

    except Exception as e:
        print(f"❌ Error generating PNG: {e}")


def count_tokens(model_provider, response):
    if model_provider == 'anthropic':
        input_tokens = response.response_metadata.get('usage').get('input_tokens')
        output_tokens = response.response_metadata.get('usage').get('output_tokens')
    elif model_provider == 'openai':
        input_tokens = response.response_metadata.get('token_usage').get('prompt_tokens')
        output_tokens = response.response_metadata.get('token_usage').get('completion_tokens')
    elif model_provider == 'google':
        input_tokens = response.usage_metadata['input_tokens']
        output_tokens = response.usage_metadata['output_tokens']

    return input_tokens, output_tokens


def validate_apikeys():
    for apikey in ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY', 'SPOTIPY_CLIENT_ID',
                   'SPOTIPY_CLIENT_SECRET']:
        if not os.getenv(apikey):
            raise ValueError(f"{apikey} not found in environment variables")


def validate_user_input(attribute: str, user_input: str) -> bool:
    """
    Use GPT-4o-mini to validate user input for a given attribute.
    Returns True if valid, False otherwise.
    """
    # TODO: Enable choice of the validator model
    # Separate tiny, cheap and fast model to validate user inputs.
    llm_validator = init_chat_model(model="openai:gpt-4o-mini", temperature=0.0)
    validation_prompt = VALIDATION_PROMPTS.get(attribute)

    if not validation_prompt:
        logging.warning(f"No validation prompt for attribute: {attribute}")
        return True  # If no validation prompt, accept the input

    try:
        messages = [
            SystemMessage(content=validation_prompt),
            HumanMessage(content=f"User input: {user_input}")
        ]

        response = llm_validator.invoke(messages)
        result = response.content.strip()

        logging.info(f"Validation for '{attribute}' with input '{user_input}': {result}")

        # Check if response is '1' (valid)
        return result == '1'

    except Exception as e:
        logging.error(f"Error during validation: {e}")
        return True  # On error, accept the input to not block the user


def create_playlist_name(attributes: str) -> str:
    """
    Use GPT-4o-mini to validate user input for a given attribute.
    Returns True if valid, False otherwise.
    """

    llm_title_creator = init_chat_model(model="openai:gpt-4o-mini", temperature=0.0)
    system_prompt = """You are a helpful assistant that receives as input the attributes
                    passed to a AI music recommendation system and comes up with a name for
                    a playlist setup with these songs for the user"""

    title_prompt = (f"""Based on the following attributes:
              {attributes}
              Please come up with some short and concise name for YouTube playlist
              Pick maximum 3 keywords, don't force yourself to put all of the attributes into the name""")

    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"User input: {title_prompt}")
        ]

        response = llm_title_creator.invoke(messages)
        result = response.content.strip()

        return result

    except Exception as e:
        logging.error(f"Error during validation: {e}")
        return True  # On error, accept the input to not block the user


def load_config(file_path="config.json"):
    with open(file_path, "r") as f:
        config = json.load(f)
    return config


# Get response from any model
def get_model_response(state, model_provider, current_time, models, script_config) -> dict:
    """Get response with same System Message to specific Human Message for given Model Provider"""

    breakpoint()
    messages = [
        SystemMessage(content=RECOMMENDATION_PROMPT.format(NO_OF_SONGS=script_config['NO_OF_SONGS'])),
        HumanMessage(content=state["final_prompt"])
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
    # TODO: Fix this guy
    tool_calls = response.additional_kwargs.get("tool_calls", [])

    if tool_calls:
        print("✅ Search tool was used")
        for call in tool_calls:
            print(call["name"], call["args"])
    else:
        print("❌ Search tool was NOT used")

    # Generate timestamp and filename and model_outputs folder if not present

    output_dir = Path(__file__).parent / "model_outputs" / current_time

    output_dir.mkdir(exist_ok=True)

    filename = Path(__file__).parent / f"model_outputs/{current_time}/{model_provider}_response.json"

    # Dump response to JSON file
    response_dict = response.model_dump()

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(response_dict, f, indent=2, ensure_ascii=False)

    logging.info(f"{model_provider} response saved to {filename}")

    return {f"{model_provider}_response": response.model_dump()}
