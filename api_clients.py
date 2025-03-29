import requests
import sys
from rich import print as rprint
from .config import PERPLEXITY_API_KEY
from typing import Optional, Dict, Any

PERPLEXITY_API_URL = 'https://api.perplexity.ai/chat/completions'

def fetch_perplexity_info(topic: str) -> Optional[str]:
    if not PERPLEXITY_API_KEY:
        rprint("[red]Error: PERPLEXITY_API_KEY is not set. Please configure it in the .config module.[/red]")
        sys.exit(1)

    headers = {
        'Authorization': f'Bearer {PERPLEXITY_API_KEY}',
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    payload = {
        'model': 'llama-3-sonar-large-32k-online',
        'messages': [
            {"role": "system", "content": "You are an AI assistant. Provide a concise, factual summary."},
            {"role": "user", "content": topic}
        ],
        'max_tokens': 600
    }

    try:
        response = requests.post(PERPLEXITY_API_URL, headers=headers, json=payload, timeout=45)
        response.raise_for_status()
        data = response.json()
        try:
            return data['choices'][0]['message']['content']
        except (KeyError, IndexError):
            rprint("[yellow]Warning: Unexpected response format from Perplexity API.[/yellow]")
            return None
    except requests.exceptions.RequestException as e:
        rprint(f"[red]Error: Failed to fetch data from Perplexity API. Details: {e}[/red]")
        return None