from typing import Annotated, Dict
from typing import List

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

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
    messages: Annotated[list, add_messages] = Field(default = [])
    user_question: str = Field(default = '')
    anthropic_response: str = Field(default = '')
    openai_response: str = Field(default = '')
    google_response: str = Field(default = '')
    final_answer: str = Field(default = '')
    prompt_attributes: Dict[str, str] = Field(default = '')
    current_attribute_index: int = Field(default=0)
    attributes_to_collect: list[str]
    validation_attempts: int = Field(default=0)
    final_prompt: str = Field(default="")
    max_attempts: int
    is_complete: bool = Field(default=False)

class SongRecommendationState(TypedDict):
    current_attribute_index: int
    attributes_to_collect: list[str]
    prompt_attributes: dict[str, str]
    current_user_input: str
    validation_attempts: int
    validation_error: str | None
    final_prompt: str
    max_attempts: int
