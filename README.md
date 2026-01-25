# Musicology

The goal of the app is to provide music recommendations powered by not one, but ensemble of three LLMs from popular providers:

- OpenAI,
- Anthropic,
- Google.

Similar to music competitions those three LLMs vote for recommendations and assign points to the songs. Afterwards they are
added up and the songs with the highest points across all LLMs are recommended to the user.

# Tech stuff

App is implemented using **langgraph** framework with following simple graph.

```mermaid
flowchart TD
    A[Step 1: Build Dynamic Prompt<br/>Inputs: genre, language, year,<br/>favorite_artists, hints]

    A --> A1[Step 1.1: ChatGPT-4 Mini<br/>Loopback Input Validation]
    A1 --> A

    A --> B1[Step 2.1: Ask OpenAI LLM]
    A --> B2[Step 2.2: Ask Anthropic LLM]
    A --> B3[Step 2.3: Ask Gemini LLM]

    B1 --> C[Step 3: Summarize Votes<br/>pd.DataFrame]
    B2 --> C
    B3 --> C

    C --> D[Step 4: Generate Playlist<br/>Using YouTube API]
    D --> E[Return Playlist]
```

# Installation

1. Clone the repository
   ```
   gh repo clone markoo26/musicology
   ```
   
2. Setup venv and install dependencies
    ```
   uv venv
   venv activate
   uv pip install .
   ```

3. Setup environment variables for LLM providers. Refer to `.env.example` for required variables.
4. Setup `client_secrets.json` to access the Youtube API.
5. Run python script
   ```
   python main.py
   ```
# TODO
1. Use search tools instead of creating some random titles
