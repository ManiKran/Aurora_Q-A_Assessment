# llm.py
import os
from openai import OpenAI
from dotenv import load_dotenv

# ──────────────────────────────
# Load environment and initialize client
# ──────────────────────────────
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found. Please set it in your .env file.")

client = OpenAI(api_key=api_key)

# ──────────────────────────────
# Generate Answer
# ──────────────────────────────
def generate_answer(question, context_messages):
    """
    Generate an answer using an LLM, grounded in the provided messages.
    """
    if not context_messages:
        return "I couldn’t find any relevant messages to answer that question."

    # Build the context string
    context_text = "\n".join(
        [f"{msg['user_name']}: {msg['text']}" for msg in context_messages]
    )

    prompt = f"""
You are a helpful assistant that answers questions about members based only on the context below.
Each message is written by a member.

Context:
{context_text}

Question:
{question}

If the answer cannot be found in the context, reply with:
"I don’t have enough information to answer that."

Answer:
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        answer = response.choices[0].message.content.strip()
        return answer

    except Exception as e:
        print("❌ Error from OpenAI API:", e)
        return "Sorry, something went wrong while generating the answer."