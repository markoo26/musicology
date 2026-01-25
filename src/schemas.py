from typing import Annotated, Dict
from typing import List

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


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
    messages: Annotated[list, add_messages]
    user_question: str
    anthropic_response: str
    openai_response: str
    google_response: str
    final_answer: str
    prompt_attributes: Dict[str, str]
