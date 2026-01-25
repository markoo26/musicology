import os

from dotenv import load_dotenv

load_dotenv()


def list_anthropic_models():
    """List available Anthropic Claude models"""
    try:
        from anthropic import Anthropic

        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        print("\n" + "=" * 70)
        print("📦 ANTHROPIC MODELS")
        print("=" * 70)

        # List models
        models = client.models.list()

        for model in models.data:
            print(f"  • {model.id}")
            if hasattr(model, 'display_name'):
                print(f"    Display: {model.display_name}")

        print(f"\nTotal: {len(models.data)} models")

    except ImportError:
        print("❌ anthropic package not installed. Run: pip install anthropic")
    except Exception as e:
        print(f"❌ Error: {e}")


def list_openai_models():
    """List available OpenAI models"""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        print("\n" + "=" * 70)
        print("📦 OPENAI MODELS")
        print("=" * 70)

        # List models
        models = client.models.list()

        # Filter to just chat models (gpt-*)
        chat_models = [m for m in models.data if m.id.startswith('gpt')]

        for model in sorted(chat_models, key=lambda x: x.id):
            print(f"  • {model.id}")

        print(f"\nTotal chat models: {len(chat_models)}")

    except ImportError:
        print("❌ openai package not installed. Run: pip install openai")
    except Exception as e:
        print(f"❌ Error: {e}")


def list_google_models():
    """List available Google models"""
    try:
        import google.generativeai as genai

        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

        print("\n" + "=" * 70)
        print("📦 GOOGLE MODELS")
        print("=" * 70)

        # List models
        models = genai.list_models()

        # Filter to generative models
        generative_models = [m for m in models if 'generateContent' in m.supported_generation_methods]

        for model in generative_models:
            print(f"  • {model.name.replace('models/', '')}")
            if hasattr(model, 'display_name'):
                print(f"    Display: {model.display_name}")

        print(f"\nTotal generative models: {len(generative_models)}")

    except ImportError:
        print("❌ google-generativeai package not installed. Run: pip install google-generativeai")
    except Exception as e:
        print(f"❌ Error: {e}")


def show_common_models():
    """Show commonly used models for reference"""
    print("\n" + "=" * 70)
    print("💡 COMMONLY USED MODELS (for init_chat_model)")
    print("=" * 70)

    print("\n🔹 ANTHROPIC (Cost-effective → Premium)")
    print("  • anthropic:claude-haiku-4-5-20251001       (Fastest, cheapest)")
    print("  • anthropic:claude-sonnet-4-20250514        (Balanced)")
    print("  • anthropic:claude-opus-4-20250514          (Most capable)")

    print("\n🔹 OPENAI")
    print("  • openai:gpt-4o-mini                        (Cheapest)")
    print("  • openai:gpt-4o                             (Latest)")
    print("  • openai:gpt-4-turbo                        (Fast)")

    print("\n🔹 GOOGLE")
    print("  • google_genai:gemini-1.5-flash             (Fast, cheap)")
    print("  • google_genai:gemini-1.5-pro               (More capable)")
    print("  • google_genai:gemini-2.0-flash-exp         (Experimental)")

    print("\n" + "=" * 70)


def main():
    import sys

    print("=" * 70)
    print("AVAILABLE AI MODELS")
    print("=" * 70)

    if len(sys.argv) > 1:
        provider = sys.argv[1].lower()

        if provider == "anthropic":
            list_anthropic_models()
        elif provider == "openai":
            list_openai_models()
        elif provider == "google":
            list_google_models()
        else:
            print(f"Unknown provider: {provider}")
            print("Available: anthropic, openai, google")
    else:
        # List all
        list_anthropic_models()
        list_openai_models()
        list_google_models()
        show_common_models()

    print("\n💡 Usage: python list_models.py [provider]")
    print("   Example: python list_models.py anthropic")


if __name__ == "__main__":
    main()