from typing import Literal

from langgraph.graph import StateGraph, END
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from src.utils import validate_user_input, load_config

CONFIG = load_config()


def create_prompt_builder_graph(config: dict):
    class PromptBuilderState(BaseModel):
        """State for the prompt building graph"""
        current_attribute_index: int = Field(default=0)
        attributes_to_collect: list[str] = Field(default_factory=lambda: config["SONG_ATTRIBUTES"])
        prompt_attributes: dict[str, str] = Field(default_factory=dict)
        validation_attempts: int = Field(default=0)
        messages: list[str] = Field(default_factory=list)
        final_prompt: str = Field(default="")
        max_attempts: int = Field(default=config["MAX_ATTEMPTS"])
        is_complete: bool = Field(default=False)

    def collect_attributes_node(state: PromptBuilderState) -> PromptBuilderState:
        """Collects and validates one attribute at a time."""
        current_idx = state.current_attribute_index
        attributes = state.attributes_to_collect

        if current_idx >= len(attributes):
            # All done - finalize prompt
            state.final_prompt = (
                f"Please generate {CONFIG['NO_OF_SONGS']} song recommendations "
                f"based on the following criteria:\n"
            )
            for attr, value in state.prompt_attributes.items():
                state.final_prompt += f"{attr}: {value}\n"
            state.is_complete = True
            return state

        current_attribute = attributes[current_idx]
        max_attempts = state.max_attempts
        attempts = state.validation_attempts

        attribute_helpers = {'mode': ['find_for_given_artists',  'find_new_artists']}

        current_attribute_helper = attribute_helpers.get(current_attribute, None)

        if current_attribute_helper:
            helper_text = f"{current_attribute_helper}"
        else:
            helper_text = ""

        # Ask for user input via interrupt
        user_input = interrupt(
            f"\nPlease describe the {current_attribute}{helper_text}: "
        )

        # Validate length
        if len(user_input) > CONFIG['MAX_CHARS']:
            state.messages.append(
                f"⚠️ Input too long (max {CONFIG['MAX_CHARS']} characters). Please try again."
            )
            state.validation_attempts = attempts + 1
            return state

        # Check if empty
        if not user_input or not user_input.strip():
            state.messages.append("⚠️ Input cannot be empty. Please try again.")
            state.validation_attempts = attempts + 1
            return state

        # Validate with LLM
        state.messages.append("🔍 Validating your input...")
        is_valid = validate_user_input(current_attribute, user_input.strip())

        if is_valid:
            state.messages.append(f"✅ Valid {current_attribute}")
            state.prompt_attributes[current_attribute] = user_input.strip()
            state.current_attribute_index = current_idx + 1
            state.validation_attempts = 0
        else:
            attempts += 1
            state.validation_attempts = attempts
            remaining = max_attempts - attempts

            if remaining > 0:
                state.messages.append(
                    f"❌ Invalid {current_attribute}. Please provide a valid {current_attribute}. "
                    f"({remaining} attempts remaining)"
                )
            else:
                state.messages.append(
                    f"❌ Maximum attempts reached. Using your last input anyway."
                )
                state.prompt_attributes[current_attribute] = user_input.strip()
                state.current_attribute_index = current_idx + 1
                state.validation_attempts = 0

        return state

    def should_continue_collecting(state: PromptBuilderState) -> Literal["collect_attributes", "end"]:
        """Router to determine if collection is complete."""
        if state.is_complete:
            return "end"
        return "collect_attributes"

    # Build the prompt builder graph

    workflow = StateGraph(PromptBuilderState)

    workflow.add_node("collect_attributes", collect_attributes_node)

    workflow.set_entry_point("collect_attributes")
    workflow.add_conditional_edges(
        "collect_attributes",
        should_continue_collecting,
        {
            "collect_attributes": "collect_attributes",
            "end": END
        }
    )

    return workflow.compile()


graph = create_prompt_builder_graph(CONFIG)
