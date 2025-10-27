import os
import sys
import json
from openai import OpenAI

"""
Check available models for the OPENAI_API_KEY in the environment.
Usage:
  export OPENAI_API_KEY="sk-..."
  python scripts/check_models.py

The script prints a categorized list of models (embeddings, chat/generation, other).
"""

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY not set in environment", file=sys.stderr)
    sys.exit(2)

client = OpenAI(api_key=api_key)

try:
    resp = client.models.list()
    models = getattr(resp, "data", None) or (resp.get("data") if isinstance(resp, dict) else None)
    if not models:
        print("No models found or unexpected response:\n", resp)
        sys.exit(0)

    embedding_models = []
    chat_models = []
    other_models = []

    for m in models:
        mid = getattr(m, "id", None) or (m.get("id") if isinstance(m, dict) else None)
        if not mid:
            continue
        low = mid.lower()
        if "embed" in low or "embedding" in low:
            embedding_models.append(mid)
        elif any(x in low for x in ("gpt", "claude", "gemini", "llama", "qwen", "grok", "nova", "o3", "o4")) or "chat" in low:
            chat_models.append(mid)
        else:
            other_models.append(mid)

    out = {
        "total_models": len(models),
        "embeddings": sorted(set(embedding_models)),
        "chat_or_generation": sorted(set(chat_models)),
        "other": sorted(set(other_models)),
    }

    print(json.dumps(out, indent=2, ensure_ascii=False))

    # Print quick recommendation
    preferred = [m for m in out["embeddings"] if m.startswith("openai.")]
    if preferred:
        print("\nRecommended embedding model(s) available:")
        for m in preferred:
            print(" -", m)
    elif out["embeddings"]:
        print("\nEmbedding models available (no openai.* prefix found):")
        for m in out["embeddings"]:
            print(" -", m)
    else:
        print("\nNo embedding models detected for this API key.")

except Exception as e:
    print("ERROR while listing models:", e, file=sys.stderr)
    sys.exit(1)
