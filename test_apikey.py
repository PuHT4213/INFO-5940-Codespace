#!/usr/bin/env python3
"""test_apikey.py

Small utility to verify whether a TAVILY_API_KEY is present and usable.
Usage:
    python test_apikey.py [--key KEY] [--query QUERY]

Exit codes:
    0 - success (API key valid and search returned a response)
    1 - API call failed (exception raised)
    2 - API key not provided
    3 - unexpected response structure
"""

import os
import sys
import argparse


from dotenv import load_dotenv

load_dotenv()
def main() -> int:
    parser = argparse.ArgumentParser(description="Test Tavily API key by performing a small search.")
    parser.add_argument("--key", "-k", help="Tavily API key (override env var)")
    parser.add_argument("--query", "-q", default="test", help="Search query to use for the test")
    args = parser.parse_args()
    
    api_key = args.key or os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("TAVILY_API_KEY not set. Provide with --key or set environment variable TAVILY_API_KEY.")
        return 2

    try:
        # Import here to surface a clear error if the package is missing
        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)
        print("Calling Tavily search with query:", args.query)
        resp = client.search(args.query, max_results=1)

        # Basic, defensive checks of response structure
        if isinstance(resp, dict):
            results = resp.get("results")
            if results is not None:
                print("Success: API call returned results (length = {}).".format(len(results)))
                if results:
                    # print a short preview
                    first = results[0]
                    title = first.get("title") or first.get("name") or "(no title)"
                    content = first.get("content") or "(no content)"
                    print("Preview:")
                    print("- {}: {}".format(title, content[:300]))
                return 0
            else:
                print("Unexpected response: missing 'results' key in response:\n", resp)
                return 3
        else:
            print("Unexpected response type:", type(resp), resp)
            return 3

    except Exception as e:
        # Try to make the error message concise and helpful
        print("API call failed with exception:")
        print(repr(e))
        # If this looks like an auth error, give a hint
        msg = str(e).lower()
        if "401" in msg or "auth" in msg or "api key" in msg:
            print("Authentication error likely: check that TAVILY_API_KEY is correct and has not expired.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
