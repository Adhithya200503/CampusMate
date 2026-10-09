"""
LLM client — sends prompts to an OpenAI-compatible API.
"""

import requests
from typing import List, Dict


def call_llm(system_prompt: str, user_message: str,
             api_key: str, model: str, base_url: str) -> str:
    """
    Call an OpenAI-compatible chat completions endpoint.
    Returns the assistant's reply text.
    """
    if not api_key or api_key == "your_api_key_here":
        return (
            "[ERROR] LLM API key not configured. "
            "Please set LLM_API_KEY in your .env file."
        )

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.3,
        "max_tokens": 1024,
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except requests.exceptions.Timeout:
        return "[ERROR] LLM request timed out. Please try again."
    except requests.exceptions.HTTPError as e:
        return f"[ERROR] LLM API error: {e} — {resp.text[:200]}"
    except Exception as e:
        return f"[ERROR] LLM request failed: {e}"


def build_prompt(question: str, chunks: List[Dict]) -> str:
    """Build the user prompt with retrieved context."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Source {i}: {chunk['source']}, Page {chunk['page']}]\n{chunk['text']}"
        )

    context = "\n\n".join(context_parts)
    return (
        f"Context from college documents:\n\n{context}\n\n"
        f"Student's question: {question}\n\n"
        f"Answer the question using ONLY the context above. "
        f"Cite sources as [Source N] for each fact."
    )
