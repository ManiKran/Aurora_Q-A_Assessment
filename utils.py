# utils.py
import os
import pickle
import requests
from datetime import datetime, timedelta

API_URL = "https://november7-730026606190.europe-west1.run.app/messages"
CACHE_FILE = "data_cache.pkl"
CACHE_MAX_AGE_HOURS = 12  # auto-refresh cache after this many hours


def fetch_messages_from_api(limit_total: int = 3349):
    """
    Fetch all messages in a single API request.
    Attempts to pull all messages by setting limit=3349.
    Falls back gracefully if the API enforces smaller limits.
    """
    print(f"📡 Fetching all messages from API with limit={limit_total} (single request)...")

    url = f"{API_URL}/?skip=0&limit={limit_total}"
    headers = {"accept": "application/json"}
    response = requests.get(url, headers=headers)

    if response.status_code in (404, 405):
        print("⚠️ Pagination not supported — retrying base endpoint without params.")
        response = requests.get(API_URL, headers=headers)

    if response.status_code == 402:
        raise RuntimeError("💰 API quota reached (HTTP 402 Payment Required).")

    response.raise_for_status()
    data = response.json()
    items = data.get("items", data)

    # Normalize message text and user names
    for item in items:
        item["message"] = (item.get("message") or "").strip()
        item["user_name"] = (item.get("user_name") or "").strip()

    print(f"✅ Done! Total messages fetched: {len(items)}")
    return items


def is_cache_fresh() -> bool:
    """Returns True if cache exists and is newer than CACHE_MAX_AGE_HOURS."""
    if not os.path.exists(CACHE_FILE):
        return False
    file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(CACHE_FILE))
    return file_age < timedelta(hours=CACHE_MAX_AGE_HOURS)


def load_messages(force_refresh: bool = False, limit_per_page: int = 3349):
    """
    Load messages from cache (if fresh) or fetch from API.
    """
    if not force_refresh and is_cache_fresh():
        try:
            with open(CACHE_FILE, "rb") as f:
                messages = pickle.load(f)
            print(f"💾 Loaded {len(messages)} messages from cache.")
            return messages
        except Exception as e:
            print(f"⚠️ Cache read error ({e}) — refetching...")

    # 🚀 Fetch all messages (no artificial cap)
    messages = fetch_messages_from_api(limit_total=limit_per_page)

    # Save to cache
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(messages, f)
    print(f"💾 Cached {len(messages)} messages to {CACHE_FILE}.")

    return messages