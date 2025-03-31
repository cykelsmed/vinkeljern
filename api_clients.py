"""
API client module for Vinkeljernet project.

This module provides functions to interact with external APIs such as
Perplexity for information retrieval and OpenAI for generating angles.
"""

# At the top of your file:
try:
    import aiohttp
except ImportError:
    print("Error: aiohttp package not found. Installing now...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "aiohttp==3.9.1"])
    import aiohttp

import requests
import sys
import json
from rich import print as rprint
from typing import Optional, Dict, Any, List
from openai import OpenAI

# Try to import from newer style API structure, fall back to older style if needed
try:
    from openai.types.error import APIError, APIConnectionError, RateLimitError
except ImportError:
    # Fall back to older OpenAI SDK structure
    try:
        from openai.error import APIError, APIConnectionError, RateLimitError
    except ImportError:
        # Define our own error classes as a last resort
        class APIError(Exception):
            pass
        class APIConnectionError(Exception):
            pass
        class RateLimitError(Exception):
            pass

from config import PERPLEXITY_API_KEY, OPENAI_API_KEY
from models import RedaktionelDNA
from prompt_engineering import construct_angle_prompt, parse_angles_from_response
import asyncio
from cache_manager import cached_api
from retry_manager import retry_with_circuit_breaker, CircuitOpenError, MaxRetriesExceededError

from error_handling import (
    retry_with_backoff, 
    safe_execute_async, 
    APIKeyMissingError, 
    SSLVerificationError, 
    APIConnectionError, 
    APIResponseError,
    display_api_response_error,
    log_info, log_warning, log_error
)

class SecurityWarning(Warning):
    """Custom warning for security-related issues."""
    pass

PERPLEXITY_API_URL = 'https://api.perplexity.ai/chat/completions'

async def create_secure_api_session(
    dev_mode: bool = False,
    timeout: int = 45,
    retries: int = 3,
    backoff_factor: float = 0.5
) -> aiohttp.ClientSession:
    """
    Creates a secure aiohttp client session with proper SSL handling and retry logic.
    """
    import ssl
    import warnings
    import aiohttp
    from aiohttp import TCPConnector, ClientTimeout
    
    # Always use Python's built-in CA certificates
    import certifi
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    
    if dev_mode:
        warnings.warn(
            "⚠️ WARNING: SSL verification disabled! This is insecure and should ONLY be used in development.",
            category=SecurityWarning
        )
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
    
    # Configure timeout
    timeout_config = ClientTimeout(total=timeout)
    
    # Create connector with SSL context
    connector = TCPConnector(ssl=ssl_context)
    
    # Create session
    return aiohttp.ClientSession(
        connector=connector,
        timeout=timeout_config
    )

@cached_api(ttl=3600)  # Cache for 1 hour
@retry_with_circuit_breaker(
    max_retries=3,
    initial_backoff=1.0,
    backoff_factor=2.0,
    exceptions=[aiohttp.ClientError, asyncio.TimeoutError, ConnectionError],
    circuit_name="perplexity_api"
)
@safe_execute_async(fallback_return=None)
async def fetch_topic_information(topic: str, dev_mode: bool = False) -> Optional[str]:
    """
    Fetch information about a topic using the Perplexity API asynchronously.
    
    Args:
        topic: The news topic to fetch information about
        dev_mode: If True, disables SSL verification (development only!)
        
    Returns:
        Optional[str]: The information retrieved or None if failed
    """
    if not PERPLEXITY_API_KEY:
        raise APIKeyMissingError("Perplexity API nøgle mangler. Sørg for at have en PERPLEXITY_API_KEY i din .env fil.")
    
    # Debug API key (mask most of it)
    key_preview = PERPLEXITY_API_KEY[:6] + "..." + PERPLEXITY_API_KEY[-4:] if len(PERPLEXITY_API_KEY) > 10 else "Invalid key format"
    log_info(f"Using Perplexity API key starting with {key_preview}")

    log_info(f"Indhenter information om emnet \"{topic}\" via Perplexity...")
    rprint(f"[blue]Indhenter information om emnet \"{topic}\" via Perplexity...[/blue]")

    # Add this right before making the request:
    masked_key = PERPLEXITY_API_KEY[:5] + "..." + PERPLEXITY_API_KEY[-3:] if len(PERPLEXITY_API_KEY) > 8 else "INVALID"
    log_info(f"Using Perplexity API key (masked): {masked_key}")
    log_info(f"Request URL: {PERPLEXITY_API_URL}")

    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "sonar",  # This is supported in the free tier
        "messages": [
            {"role": "system", "content": "Du er en professionel journalist, der skal give koncis og faktabaseret information om et aktuelt nyhedsemne. Giv grundige, men velstrukturerede fakta, baggrund og kontekst. Inkluder omhyggeligt datoer, tal og faktuelle detaljer, der er relevante for emnet. Undgå at udelade væsentlig information."},
            {"role": "user", "content": f"Giv mig en grundig, men velstruktureret oversigt over den aktuelle situation vedrørende følgende nyhedsemne: {topic}. Inkluder relevante fakta, baggrund, kontekst og eventuelle nylige udviklinger. Vær præcis og faktabaseret."}
        ],
        "max_tokens": 1000,
        "temperature": 0.2,
        "top_p": 0.9,
        "return_images": False,
        "return_related_questions": False
    }

    try:
        # Use our secure session creation function
        async with await create_secure_api_session(dev_mode=dev_mode) as session:
            async with session.post(PERPLEXITY_API_URL, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    log_error(f"Full error response: {error_text}")
                    rprint(f"[bold red]Full API error:[/bold red] {error_text}")
                    
                    try:
                        error_json = json.loads(error_text)
                        error_message = error_json.get('error', {}).get('message', error_text)
                        log_error(f"API Error message: {error_message}")
                    except json.JSONDecodeError:
                        error_message = error_text
                    
                    log_error(f"Perplexity API Error: Status {response.status}: {error_message}")
                    rprint(f"[bold red]Perplexity API Error:[/bold red] Status {response.status}: {error_message}")
                    display_api_response_error(response)
                    return None
                data = await response.json()
                try:
                    return data['choices'][0]['message']['content']
                except (KeyError, IndexError):
                    log_warning("Unexpected response format from Perplexity API.")
                    rprint("[yellow]Warning: Unexpected response format from Perplexity API.[/yellow]")
                    return None
    except SSLVerificationError as e:
        log_error(f"SSL Error: {e}")
        rprint(f"[bold red]SSL Error:[/bold red] {e}")
        rprint("[yellow]Try running with --dev-mode if this is a development environment.[/yellow]")
        return None
    except APIConnectionError as e:
        log_error(f"Connection Error: {e}")
        rprint(f"[bold red]Connection Error:[/bold red] {e}")
        return None
    except APIResponseError as e:
        log_error(f"API Error: Status {e.status}: {e.message}")
        rprint(f"[bold red]API Error:[/bold red] Status {e.status}: {e.message}")
        return None
    except Exception as e:
        log_error(f"Request Error: {e}")
        rprint(f"[bold red]Request Error:[/bold red] {e}")
        return None

import json
from typing import Any, Dict, List

def generate_angles(emne: str, topic_info: str, profile: Any, bypass_cache: bool = False) -> List[Dict[str, Any]]:
    """
    Generate news angles for the given topic and profile.
    
    This function calls the external API to generate angles. If the result
    is returned as a string, we attempt to parse it as JSON. The function
    always returns a list of dictionaries.
    
    Example of correct output (list of dict):
      [
        {
          "overskrift": "Eksempelvinkel",
          "beskrivelse": "Denne vinkel handler om...",
          "nyhedskriterier": ["konflikt", "identifikation"],
          ... 
        },
        ...
      ]
    
    Example of incorrect output (single string):
      "Dette er ikke et korrekt JSON-format..."
    
    Args:
        emne: The news topic.
        topic_info: Background information on the topic.
        profile: The editorial DNA profile.
        bypass_cache: if True, bypass local cache.
        
    Returns:
        A list of angle dictionaries.
    """
    # Call your API here. For example:
    result = call_angle_api(emne, topic_info, profile, bypass_cache)
    
    # If result is a string, attempt to parse it as JSON.
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except Exception as ex:
            print(f"Fejl ved parsing af API-svar til JSON: {ex}")
            result = []  # Fallback to empty list if parsing fejler.
    
    # Make sure we return a list
    if not isinstance(result, list):
        print("Advarsel: Genererede vinkler er ikke i listeformat. Forsøger at konvertere...")
        result = [result] if isinstance(result, dict) else []
    
    return result

# Dummy implementation of the API call. Replace with actual implementation.
def call_angle_api(emne: str, topic_info: str, profile: Any, bypass_cache: bool) -> Any:
    # For demonstration, this function might sometimes return a string.
    # Replace with your actual API integration.
    # For example, it may return a JSON string:
    api_response = """
    [
        {
            "overskrift": "Kunstens magt i demokratiet",
            "beskrivelse": "En dybdegående vinkel på, hvordan kunst påvirker demokratiet.",
            "nyhedskriterier": ["identifikation", "konflikt"],
            "begrundelse": "Denne vinkel belyser sammenhængen mellem kunst og samfund.",
            "startSpørgsmål": ["Hvordan påvirker kunsten demokratiet?"]
        }
    ]
    """
    return api_response