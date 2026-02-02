import json
import sys
from pathlib import Path

import pandas as pd


def preview_json(file_path: str) -> None:
    """Load and display JSON file contents"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"\n{'=' * 60}")
        print(f"JSON File: {file_path}")
        print(f"{'=' * 60}\n")

        # Pretty print the JSON
        print(json.dumps(data, indent=2, ensure_ascii=False))

        # If it contains recommendations, show summary
        if isinstance(data, dict) and 'recommendations' in data:
            print(f"\n{'=' * 60}")
            print(f"Summary: {len(data['recommendations'])} recommendations found")
            print(f"{'=' * 60}\n")

    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON file - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error reading JSON file: {e}")
        sys.exit(1)


def preview_csv(file_path: str) -> None:
    """Load and display CSV file as pandas DataFrame"""
    try:
        df = pd.read_csv(file_path)

        print(f"\n{'=' * 60}")
        print(f"CSV File: {file_path}")
        print(f"{'=' * 60}\n")

        print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")

        # Display the DataFrame
        print(df.to_string(index=False))

        print(f"\n{'=' * 60}")
        print(f"Data Types:")
        print(f"{'=' * 60}")
        print(df.dtypes)

    except pd.errors.EmptyDataError:
        print(f"❌ Error: CSV file is empty")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error reading CSV file: {e}")
        sys.exit(1)


def main():
    if len(sys.argv) != 2:
        print("Usage: python preview_recommendations.py <file_path>")
        print("\nExamples:")
        print("  python preview_recommendations.py model_outputs/anthropic_response_2024_01_15.json")
        print("  python preview_recommendations.py final_recommendations_df_2024_01_15.csv")
        sys.exit(1)

    file_path = sys.argv[1]

    # Check if file exists
    if not Path(file_path).exists():
        print(f"❌ Error: File '{file_path}' not found")
        sys.exit(1)

    # Determine file type and preview accordingly
    file_extension = Path(file_path).suffix.lower()

    if file_extension == '.json':
        preview_json(file_path)
    elif file_extension == '.csv':
        preview_csv(file_path)
    else:
        print(f"❌ Error: Unsupported file type '{file_extension}'")
        print("Supported formats: .json, .csv")
        sys.exit(1)


if __name__ == "__main__":
    main()