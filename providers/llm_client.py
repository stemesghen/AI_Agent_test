import os
import requests
import json

def call_llm(messages):
    # Prefer Azure if available
    if os.getenv("AZURE_OPENAI_API_KEY"):
        url = f"{os.getenv('AZURE_OPENAI_ENDPOINT')}/openai/deployments/{os.getenv('AZURE_OPENAI_DEPLOYMENT')}/chat/completions?api-version={os.getenv('AZURE_OPENAI_API_VERSION')}"
        headers = {
            "api-key": os.getenv("AZURE_OPENAI_API_KEY"),
            "Content-Type": "application/json",
        }
    # Otherwise use OpenAI public API
    elif os.getenv("OPENAI_API_KEY"):
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
            "Content-Type": "application/json",
        }
    else:
        raise RuntimeError("No LLM API key found (Azure or OpenAI).")

    data = {
        "model": os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
        "messages": messages,
        "temperature": 0,
    }

    r = requests.post(url, headers=headers, data=json.dumps(data))
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

