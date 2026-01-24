def welcome_screen(n_songs):
    print("=" * 60)
    print("Multi-LLM Music Recommendation system")
    print("=" * 60)
    print(f"\nThis system uses 3 AI models to generate you {n_songs} proposals of songs to listen:")
    print("\nEach of them is scored from 1 to 10 and passed as Structured Output to form a Pandas DataFrame")
    print("\nPoints from all LLMs are aggregated and top 10 songs across all models are fed to the user")
    print("\nType 'quit' or 'exit' to stop.\n")

