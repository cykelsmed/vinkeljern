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

from config import PERPLEXITY_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY
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

def generate_angles(emne: str, topic_info: str, profile: Any, bypass_cache: bool = False) -> List[Dict[str, Any]]:
    """
    Generate news angles for the given topic and profile.
    
    This function directly handles OpenAI API calls to generate angles due to issues
    with the decorators adding unsupported parameters. It processes the response
    and returns a list of structured angle dictionaries.
    
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
    
    Args:
        emne: The news topic.
        topic_info: Background information on the topic.
        profile: The editorial DNA profile.
        bypass_cache: if True, bypass local cache (not used in this implementation).
        
    Returns:
        A list of angle dictionaries.
        
    Raises:
        ValueError: If angles cannot be generated
    """
    # Check for API key
    if not OPENAI_API_KEY:
        raise ValueError(
            "OpenAI API-nøgle mangler. Sørg for at have en OPENAI_API_KEY i din .env fil."
        )
    
    try:
        # Import directly to ensure we're using the most recent version
        from openai import OpenAI
        
        # Create a simple client with just the API key
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Convert profile into strings for prompt construction
        principper = "\n".join([f"- {p}" for p in profile.kerneprincipper])
        nyhedskriterier = "\n".join([f"- {k}: {v}" for k, v in profile.nyhedsprioritering.items()])
        fokusomrader = "\n".join([f"- {f}" for f in profile.fokusOmrader])
        nogo_omrader = "\n".join([f"- {n}" for n in profile.noGoOmrader]) if profile.noGoOmrader else "Ingen"
        
        # Create the prompt
        prompt = construct_angle_prompt(
            emne,
            topic_info,
            principper,
            profile.tone_og_stil,
            fokusomrader,
            nyhedskriterier,
            nogo_omrader
        )
        
        # Log the request
        log_info(f"Sender anmodning til Claude API for emnet '{emne}'")
        
        # Call the Anthropic Claude API instead of OpenAI
        import requests
        import os
        
        # Use API key from config
        claude_api_key = ANTHROPIC_API_KEY
        
        if not claude_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY mangler i miljøvariablerne. Sørg for at tilføje denne til din .env fil."
            )
        
        # Prepare system and user message with Claude's format
        claude_messages = [
            {"role": "system", "content": "Du er en erfaren journalist med ekspertise i at udvikle kreative og relevante nyhedsvinkler."},
            {"role": "user", "content": prompt}
        ]
        
        # Claude API call
        claude_response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": claude_api_key,
                "anthropic-version": "2023-06-01"
            },
            json={
                "model": "claude-3-opus-20240229",
                "max_tokens": 2500,
                "temperature": 0.7,
                "system": "Du er en erfaren journalist med ekspertise i at udvikle kreative og relevante nyhedsvinkler.",
                "messages": [{"role": "user", "content": prompt}],
            }
        )
        
        # Parse Claude response
        if claude_response.status_code != 200:
            log_error(f"Claude API fejl: {claude_response.status_code}: {claude_response.text}")
            raise ValueError(f"Claude API fejl: {claude_response.status_code}")
            
        response_data = claude_response.json()
        response_text = response_data['content'][0]['text']
        angles = parse_angles_from_response(response_text)
        
        # Log success
        log_info(f"Genereret {len(angles)} vinkler succesfuldt")
        
        # Make sure we return a list
        if not isinstance(angles, list):
            if isinstance(angles, dict):
                # Single angle in dict format
                angles = [angles]
            else:
                log_error(f"Uventet format: {type(angles)}")
                raise ValueError(f"Uventet format: {type(angles)}. Forventede en liste eller dict.")
        
        # Add perplexity information to each angle if available
        if topic_info and isinstance(topic_info, str):
            try:
                # Extract first 1000 chars to keep it concise
                perplexity_extract = topic_info[:1000] + ("..." if len(topic_info) > 1000 else "")
                for angle in angles:
                    if isinstance(angle, dict):
                        angle['perplexityInfo'] = perplexity_extract
            except Exception as e:
                # Log but don't fail if we can't add perplexity info
                log_warning(f"Kunne ikke tilføje perplexity information: {e}")
        
        return angles
        
    except Exception as e:
        log_error(f"Uventet fejl ved generering af vinkler: {e}")
        raise ValueError(
            f"Uventet fejl ved generering af vinkler: {e}. "
            "Kontakt venligst support hvis problemet fortsætter."
        )

# Temporarily removed decorators to debug the issue
def call_angle_api(emne: str, topic_info: str, profile: Any, bypass_cache: bool = False) -> Any:
    """
    Call the OpenAI API to generate news angles based on the topic and editorial profile.
    
    Args:
        emne: The news topic
        topic_info: Background information on the topic
        profile: The editorial DNA profile
        bypass_cache: If True, bypass local cache
        
    Returns:
        Generated angles or raises an exception if API call fails
    """
    if not OPENAI_API_KEY:
        raise ValueError(
            "OpenAI API-nøgle mangler. Sørg for at have en OPENAI_API_KEY i din .env fil."
        )
    
    try:
        # Import directly to ensure we're using the most recent version
        from openai import OpenAI
        
        # Use minimal client initialization
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        log_info("DEBUG - Creating OpenAI client with only api_key parameter")
        
        # Convert profile into strings for prompt construction
        principper = "\n".join([f"- {p}" for p in profile.kerneprincipper])
        nyhedskriterier = "\n".join([f"- {k}: {v}" for k, v in profile.nyhedsprioritering.items()])
        fokusomrader = "\n".join([f"- {f}" for f in profile.fokusOmrader])
        nogo_omrader = "\n".join([f"- {n}" for n in profile.noGoOmrader]) if profile.noGoOmrader else "Ingen"
        
        # Create the prompt
        prompt = construct_angle_prompt(
            emne,
            topic_info,
            principper,
            profile.tone_og_stil,
            fokusomrader,
            nyhedskriterier,
            nogo_omrader
        )
        
        # Log the request (sanitized for privacy)
        log_info(f"Sender anmodning til OpenAI API for emnet '{emne}'")
        
        # Call the OpenAI API with minimal parameters
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Du er en erfaren journalist med ekspertise i at udvikle kreative og relevante nyhedsvinkler."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2500
        )
        
        # Extract the response text
        response_text = response.choices[0].message.content
        
        # Parse the angles from the response
        angles = parse_angles_from_response(response_text)
        
        # Log success
        log_info(f"Genereret {len(angles)} vinkler succesfuldt")
        
        return angles
        
    except Exception as e:
        log_error(f"Uventet fejl ved generering af vinkler: {e}")
        raise ValueError(
            f"Uventet fejl ved generering af vinkler: {e}. "
            "Kontakt venligst support hvis problemet fortsætter."
        )