# utils.py
import os
import pickle
import requests
from datetime import datetime, timedelta

API_URL = "https://november7-730026606190.europe-west1.run.app/messages"
CACHE_FILE = "data_cache.pkl"
CACHE_MAX_AGE_HOURS = 12  # auto-refresh cache after this many hours


def fetch_messages_from_api(limit_per_page: int = 100, max_pages: int = None):
    """
    Fetch messages from the public API using pagination.
    Each page returns up to `limit_per_page` items.
    
    Args:
        limit_per_page: Number of items to fetch per request (default 100).
                        Can be increased (e.g., 200 or 300) but use caution with memory.
        max_pages: Optional limit on how many pages to fetch total.
                   Useful for testing (e.g., max_pages=5).
    """
    print(f"📡 Fetching messages from API in pages of {limit_per_page}...")

    all_items = []
    skip = 0
    page = 0

    while True:
        url = f"{API_URL}/?skip={skip}&limit={limit_per_page}"
        headers = {"accept": "application/json"}

        response = requests.get(url, headers=headers)
        if response.status_code == 403:
            print("⚠️ Hit API permission limit — stopping early.")
            break

        response.raise_for_status()
        data = response.json()
        items = data.get("items", [])

        if not items:
            break

        # Normalize and clean message fields
        for item in items:
            item["message"] = (item.get("message") or "").strip()
            item["user_name"] = (item.get("user_name") or "").strip()

        all_items.extend(items)
        print(f"📦 Page {page + 1}: fetched {len(items)} items (total {len(all_items)})")

        if len(items) < limit_per_page:
            break

        page += 1
        skip += limit_per_page
        if max_pages and page >= max_pages:
            print(f"🛑 Stopping after {page} pages as per max_pages={max_pages}.")
            break

    print(f"✅ Done! Total messages fetched: {len(all_items)}")
    return all_items


def is_cache_fresh() -> bool:
    """
    Returns True if the cache exists and is newer than CACHE_MAX_AGE_HOURS.
    """
    if not os.path.exists(CACHE_FILE):
        return False
    file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(CACHE_FILE))
    return file_age < timedelta(hours=CACHE_MAX_AGE_HOURS)


def load_messages(force_refresh: bool = False, limit_per_page: int = 100, max_pages: int = None):
    """
    Load messages from cache (if fresh) or fetch from API.
    
    Args:
        force_refresh: Whether to skip cache and fetch new data.
        limit_per_page: Number of items per API request.
        max_pages: Optional limit for debugging or memory control.
    """
    if not force_refresh and is_cache_fresh():
        try:
            with open(CACHE_FILE, "rb") as f:
                messages = pickle.load(f)
            print(f"💾 Loaded {len(messages)} messages from cache.")
            return messages
        except Exception as e:
            print(f"⚠️ Cache read error ({e}) — refetching...")

    messages = fetch_messages_from_api(limit_per_page=limit_per_page, max_pages=max_pages)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(messages, f)
    print(f"💾 Cached {len(messages)} messages to {CACHE_FILE}.")
    return messages