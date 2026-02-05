from typing import Annotated, Dict
from typing import List

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, NotRequired

from pydantic import BaseModel, Field


class MusicRecommendation(BaseModel):
    rank: int = Field(description="Ranking from 1-10, where 10 is the strongest recommendation")
    song_title: str = Field(description="Title of the recommended song")
    artist: str = Field(description="Artist or band name")
    album: str = Field(description="Album name")
    year: int = Field(description="Year of release")
    reason: str = Field(description="Why this song matches the user's criteria")


class RecommendationResponse(BaseModel):
    recommendations: List[MusicRecommendation] = Field(
        description="List of exactly 10 music recommendations, ranked from strongest (10) to weakest (1)"
    )


class State(TypedDict):
    # LangGraph reducer field
    messages: Annotated[List, add_messages]

    # Core inputs / outputs
    user_question: NotRequired[str]
    final_prompt: NotRequired[str]
    final_answer: NotRequired[str]

    # Model responses
    anthropic_response: NotRequired[str]
    openai_response: NotRequired[str]
    google_genai_response: NotRequired[str] #TODO: See if it's fixed now

    # Prompt-building state
    prompt_attributes: NotRequired[Dict[str, str]]
    attributes_to_collect: List[str]
    current_attribute_index: NotRequired[int]

    # Control / validation
    validation_attempts: NotRequired[int]
    max_attempts: int
    is_complete: NotRequired[bool]

class SongRecommendationState(TypedDict):
    current_attribute_index: int
    attributes_to_collect: list[str]
    prompt_attributes: dict[str, str]
    current_user_input: str
    validation_attempts: int
    validation_error: str | None
    final_prompt: str
    max_attempts: int
