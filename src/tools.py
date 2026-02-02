from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import Tool
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# DuckDuckGo Search
search_tool = DuckDuckGoSearchRun()

# Wikipedia
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# Spotify


# Spotify (requires SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET in .env)
def spotify_search(query: str) -> str:
    """Search Spotify for songs, artists, or albums"""
    try:
        sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials())
        results = sp.search(q=query, limit=5, type='track,artist')

        output = []

        # Format tracks
        if results['tracks']['items']:
            output.append("Tracks:")
            for track in results['tracks']['items']:
                artists = ', '.join([artist['name'] for artist in track['artists']])
                output.append(f"  - {track['name']} by {artists} ({track['album']['name']})")

        # Format artists
        if results['artists']['items']:
            output.append("\nArtists:")
            for artist in results['artists']['items']:
                genres = ', '.join(artist['genres'][:3]) if artist['genres'] else 'N/A'
                output.append(f"  - {artist['name']} (Genres: {genres})")

        return '\n'.join(output) if output else "No results found"
    except Exception as e:
        return f"Spotify search error: {str(e)}"

tools = [
    Tool(
        name="web_search",
        description="Search the web for real, existing songs and artists. Use for general web information.",
        func=search_tool.run,
    ),
    Tool(
        name="wikipedia",
        description="Search Wikipedia for detailed information about artists, bands, albums, or music history.",
        func=wikipedia.run,
    ),
    Tool(
        name="spotify_search",
        description="Search Spotify for songs, artists, and albums. Returns real music data including track names, artists, and genres. Best for finding actual songs and verifying they exist.",
        func=spotify_search,
    )
]