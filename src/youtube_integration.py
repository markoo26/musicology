
import logging
import os
import pickle

import pandas as pd
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from src.schemas import State
from src.utils import create_playlist_name

SCOPES = ['https://www.googleapis.com/auth/youtube.force-ssl']


class YouTubePlaylistCreator:
    def __init__(self, api_key=None, client_secrets_file='client_secrets.json'):
        """
        Initialize YouTube API client
        api_key: For search-only operations (no playlist creation)
        client_secrets_file: For OAuth operations (playlist creation)
        """
        self.api_key = api_key
        self.client_secrets_file = client_secrets_file
        self.youtube = None

    def authenticate(self):
        """Authenticate using OAuth 2.0 for playlist creation"""
        creds = None

        # Token file stores the user's access and refresh tokens
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)

        # If there are no valid credentials, let the user log in
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.client_secrets_file, SCOPES)
                creds = flow.run_local_server(port=0)

            # Save the credentials for the next run
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)

        self.youtube = build('youtube', 'v3', credentials=creds)
        logging.info("YouTube API authenticated successfully")

    def search_video(self, song_title, artist):
        """Search for a video by song title and artist"""
        try:
            search_query = f"{song_title} {artist}"

            request = self.youtube.search().list(
                part='snippet',
                q=search_query,
                type='video',
                maxResults=1,
                videoCategoryId='10'  # Music category
            )

            response = request.execute()

            if response['items']:
                video_id = response['items'][0]['id']['videoId']
                video_title = response['items'][0]['snippet']['title']
                logging.info(f"Found: {video_title} (ID: {video_id})")
                return video_id
            else:
                logging.warning(f"No video found for: {search_query}")
                return None

        except Exception as e:
            logging.error(f"Error searching for {song_title} by {artist}: {e}")
            return None

    def create_playlist(self, title, description=""):
        """Create a new YouTube playlist"""
        try:
            request = self.youtube.playlists().insert(
                part='snippet,status',
                body={
                    'snippet': {
                        'title': title,
                        'description': description
                    },
                    'status': {
                        'privacyStatus': 'private'  # or 'public', 'unlisted'
                    }
                }
            )

            response = request.execute()
            playlist_id = response['id']
            logging.info(f"Playlist created: {title} (ID: {playlist_id})")
            return playlist_id

        except Exception as e:
            logging.error(f"Error creating playlist: {e}")
            return None

    def add_video_to_playlist(self, playlist_id, video_id):
        """Add a video to a playlist"""
        try:
            request = self.youtube.playlistItems().insert(
                part='snippet',
                body={
                    'snippet': {
                        'playlistId': playlist_id,
                        'resourceId': {
                            'kind': 'youtube#video',
                            'videoId': video_id
                        }
                    }
                }
            )

            response = request.execute()
            logging.info(f"Video {video_id} added to playlist {playlist_id}")
            return True

        except Exception as e:
            logging.error(f"Error adding video to playlist: {e}")
            return False

    def create_playlist_from_dataframe(self, df, playlist_name,
                                       song_col='song_title',
                                       artist_col='artist',
                                       description="AI-generated music recommendations"):
        """
        Create a YouTube playlist from a pandas DataFrame

        Parameters:
        - df: DataFrame with song recommendations
        - playlist_name: Name for the new playlist
        - song_col: Column name containing song titles
        - artist_col: Column name containing artist names
        """
        if not self.youtube:
            self.authenticate()

        # Create the playlist
        playlist_id = self.create_playlist(playlist_name, description)

        if not playlist_id:
            logging.error("Failed to create playlist")
            return None

        added_count = 0
        failed_songs = []

        # Search and add each song
        for idx, row in df.iterrows():
            song_title = row[song_col]
            artist = row[artist_col]

            print(f"Searching for: {song_title} by {artist}...")
            video_id = self.search_video(song_title, artist)

            if video_id:
                if self.add_video_to_playlist(playlist_id, video_id):
                    added_count += 1
                else:
                    failed_songs.append(f"{song_title} - {artist}")
            else:
                failed_songs.append(f"{song_title} - {artist}")

        print(f"\n✅ Playlist created successfully!")
        print(f"📊 Added {added_count}/{len(df)} songs")
        print(f"🔗 Playlist URL: https://www.youtube.com/playlist?list={playlist_id}")

        if failed_songs:
            print(f"\n⚠️  Failed to add {len(failed_songs)} songs:")
            for song in failed_songs:
                print(f"  - {song}")

        return playlist_id

def analyze_responses(state: State, current_time: str) -> dict:
    """Prepare context for Google to analyze"""
    recommendations_df = pd.DataFrame()
    for model in ['anthropic', 'openai', 'google_genai']:
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