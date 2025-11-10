# utils.py
import os
import pickle
import requests
from datetime import datetime

API_URL = "https://november7-730026606190.europe-west1.run.app/messages"
CACHE_FILE = "data_cache.pkl"

def fetch_messages_from_api(limit_per_page: int = 100):
    """
    Fetches all messages from the public API using pagination.
    Each page returns up to 100 items.
    """
    print("Fetching messages from the public API (with pagination)...")

    all_items = []
    skip = 0

    while True:
        url = f"{API_URL}/?skip={skip}&limit={limit_per_page}"
        headers = {"accept": "application/json"}
        response = requests.get(url, headers=headers)

        if response.status_code == 403:
            print("⚠️  Hit API page/permission limit — stopping early.")
            break
        response.raise_for_status()

        data = response.json()
        items = data.get("items", [])
        all_items.extend(items)

        print(f"Fetched {len(items)} items (total so far: {len(all_items)})")
        if len(items) < limit_per_page:
            break

        skip += limit_per_page

    print(f"✅ Done! Total messages fetched: {len(all_items)}")
    return all_items

def load_messages(force_refresh: bool = False):
    """
    Loads messages either from cache or from the API.
    - If cache exists and is recent, load from cache.
    - Otherwise, fetch new data and cache it.
    """
    if not force_refresh and os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            messages = pickle.load(f)
        print(f"Loaded {len(messages)} messages from cache.")
        return messages

    messages = fetch_messages_from_api()
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(messages, f)
    print(f"Fetched {len(messages)} messages and cached to {CACHE_FILE}.")
    return messages