import os
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

# ──────────────────────────────
# Setup
# ──────────────────────────────
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found. Please set it in your .env file.")

client = OpenAI(api_key=api_key)
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ──────────────────────────────
# Helper: format timestamps
# ──────────────────────────────
def format_timestamp(ts):
    """Convert ISO timestamps to a readable short format."""
    if not ts:
        return ""
    if isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            return ts
    return ts.strftime("%Y-%m-%d %H:%M")

# ──────────────────────────────
# Context builder
# ──────────────────────────────
def build_context(context_messages, max_chars=3500):
    """
    Build readable message context.
    Keeps latest messages last (chronological order).
    """
    # ensure chronological (oldest → newest)
    sorted_msgs = sorted(
        context_messages,
        key=lambda m: m.get("timestamp") or datetime.min
    )

    seen = set()
    formatted = []
    for msg in sorted_msgs:
        text = msg.get("text", "").strip()
        user = msg.get("user_name", "Unknown")
        ts = format_timestamp(msg.get("timestamp"))
        entry = f"[{ts}] {user}: {text}"
        if text and entry not in seen:
            formatted.append(entry)
            seen.add(entry)

    context_text = "\n".join(formatted)

    if len(context_text) > max_chars:
        print("⚠️ Trimming long context for token safety.")
        context_text = context_text[-max_chars:]  # keep most recent part

    return context_text

# ──────────────────────────────
# Generate contextual answer
# ──────────────────────────────
def generate_answer(question, context_messages):
    """
    Produces a grounded answer using tiered logic:
      1️⃣ Direct fact if clearly found.
      2️⃣ Approximation with “I don’t have the exact information…” if inferred.
      3️⃣ “I don’t have any information…” if unrelated or missing.
    """
    # No context at all → immediate fallback
    if not context_messages:
        return "I don’t have any information about the question you asked."

    # Build compact context
    context_text = build_context(context_messages)

    # Stronger, explicit reasoning instructions
    system_prompt = (
        "You are a precise and factual AI assistant analyzing member messages.\n\n"
        "You will be provided with time-stamped conversation snippets (context) and a question.\n"
        "Use them to infer an answer according to the following exact hierarchy:\n\n"
        "1️⃣ If the answer is explicitly stated in the context, give a short, confident factual answer.\n"
        "2️⃣ If you can reasonably infer or approximate from the context but it’s not directly stated, "
        "begin your response with: 'I don’t have the exact information for this, but based on the available context, ...'\n"
        "   → Then clearly explain your reasoning.\n"
        "3️⃣ If no relevant context exists, respond exactly with: "
        "'I don’t have any information about the question you asked.'\n\n"
        "Additional requirements:\n"
        "- Never hallucinate or assume details that are not implied.\n"
        "- Keep answers under 80 words unless absolutely necessary.\n"
        "- Do not repeat or quote the messages directly; summarize concisely."
    )

    # Build user message
    user_prompt = (
        f"Here are the context messages (oldest → newest):\n\n"
        f"{context_text}\n\n"
        f"User Question:\n{question}\n\n"
        f"Now answer strictly following the hierarchy above.\n"
        f"If the information is uncertain, be transparent about it.\n\n"
        f"Answer:"
    )

    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=350,
        )

        answer = response.choices[0].message.content.strip()

        # Normalize fallbacks for safety
        lower = answer.lower()
        if "i don’t have any information" in lower:
            return "I don’t have any information about the question you asked."
        if "i don't have any information" in lower:
            return "I don’t have any information about the question you asked."

        if not answer:
            return "I don’t have any information about the question you asked."

        return answer

    except Exception as e:
        print("❌ OpenAI API error:", str(e))
        return "Sorry, something went wrong while generating the answer."