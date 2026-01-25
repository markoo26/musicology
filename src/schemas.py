from typing import List

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