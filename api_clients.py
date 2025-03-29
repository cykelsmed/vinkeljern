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
    
    # Configure SSL context
    ssl_context = ssl.create_default_context()
    
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
    
    # Create session without the problematic retry options
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

    log_info(f"Indhenter information om emnet \"{topic}\" via Perplexity...")
    rprint(f"[blue]Indhenter information om emnet \"{topic}\" via Perplexity...[/blue]")

    headers = {
        'Authorization': f'Bearer {PERPLEXITY_API_KEY}',
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    payload = {
        'model': 'llama-3-sonar-large-32k-online',
        'messages': [
            {"role": "system", "content": "Du er en professionel journalist, der skal give koncis og faktabaseret information om et aktuelt nyhedsemne. Giv grundige, men velstrukturerede fakta, baggrund og kontekst. Inkluder omhyggeligt datoer, tal og faktuelle detaljer, der er relevante for emnet. Undgå at udelade væsentlig information."},
            {"role": "user", "content": f"Giv mig en grundig, men velstruktureret oversigt over den aktuelle situation vedrørende følgende nyhedsemne: {topic}. Inkluder relevante fakta, baggrund, kontekst og eventuelle nylige udviklinger. Vær præcis og faktabaseret."}
        ],
        'max_tokens': 1200,
        'temperature': 0.2
    }

    try:
        # Use our secure session creation function
        async with await create_secure_api_session(dev_mode=dev_mode) as session:
            async with session.post(PERPLEXITY_API_URL, headers=headers, json=payload) as response:
                if response.status != 200:
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

@cached_api(ttl=7200)  # Cache for 2 hours
@retry_with_circuit_breaker(
    max_retries=2,
    initial_backoff=2.0,
    backoff_factor=3.0,
    exceptions=[ConnectionError, TimeoutError, APIError],  # Changed from openai.APIError to APIError
    circuit_name="openai_api"
)
def generate_angles(topic: str, topic_info: Optional[str], profile: RedaktionelDNA) -> List[Dict[str, Any]]:
    """
    Generate news angles based on topic information and editorial profile using OpenAI.
    
    Args:
        topic: The news topic
        topic_info: Information about the topic (from Perplexity)
        profile: RedaktionelDNA profile object
        bypass_cache: If True, forces a fresh API call (default: False)
    
    Returns:
        list: List of generated angle objects
    """
    if not OPENAI_API_KEY:
        rprint("[red]Error: OPENAI_API_KEY is not set. Please configure it in the .env file.[/red]")
        return []
    
    rprint(f"[blue]Genererer vinkler for \"{topic}\" baseret på profilen...[/blue]")
    
    # Handle None topic_info with a fallback message
    if topic_info is None:
        rprint("[yellow]Advarsel: Kunne ikke hente baggrundsinformation. Genererer vinkler med begrænset kontekst.[/yellow]")
        topic_info = f"Emnet handler om {topic}. Ingen yderligere baggrundsinformation tilgængelig."
    
    try:
        # Initialize OpenAI client without proxies parameter
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Format profile data for prompt
        nyhedskriterier = ", ".join([f"{k}" for k in profile.nyhedsprioritering.keys()])
        principper = "\n".join([f"- {list(p.keys())[0]}: {list(p.values())[0]}" for p in profile.kerneprincipper])
        fokus = "\n".join([f"- {focus}" for focus in profile.fokusområder])
        nogo = ", ".join([f"{area}" for area in profile.nogo_områder])
        
        prompt = construct_angle_prompt(
            topic=topic,
            topic_info=topic_info,
            principper=principper,
            tone_og_stil=profile.tone_og_stil,
            fokusområder=fokus,
            nyhedskriterier=nyhedskriterier,
            nogo_områder=nogo
        )
        
        response = client.chat.completions.create(
            model="gpt-4",  # Or whichever model you have access to
            messages=[
                {"role": "system", "content": "Du er en erfaren journalist med ekspertise i at udvikle nyhedsvinkler. Returner et JSON array med vinkler."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        # Extract the JSON content
        content = response.choices[0].message.content
        
        try:
            angles = parse_angles_from_response(content)
            if angles:
                rprint(f"[green]✓[/green] Genereret {len(angles)} vinkler")
                return angles
            else:
                rprint("[red]Fejl: Kunne ikke parse vinkler fra API-svaret.[/red]")
                return []
        except Exception as e:
            rprint(f"[red]Fejl ved parsing af vinkler: {e}[/red]")
            # Fallback attempt to parse JSON directly
            try:
                raw_data = json.loads(content)
                if isinstance(raw_data, list):
                    return raw_data  # Return directly if it's already a list
                elif isinstance(raw_data, dict) and "angles" in raw_data:
                    return raw_data["angles"]  # Some responses might wrap angles in an object
                elif isinstance(raw_data, dict):
                    # Try to identify if keys are numeric (sometimes OpenAI returns indexed objects)
                    if all(key.isdigit() for key in raw_data.keys() if key):
                        return [raw_data[k] for k in sorted(raw_data.keys(), key=lambda x: int(x) if x.isdigit() else 0)]
                    else:
                        # Last resort: treat the entire object as one angle
                        return [raw_data]
                else:
                    rprint("[red]Uventet format på API-svaret.[/red]")
                    return []
            except json.JSONDecodeError:
                rprint("[red]API-svaret indeholder ikke gyldig JSON.[/red]")
                return []
    
    except Exception as e:
        rprint(f"[bold red]Fejl ved generering af vinkler:[/bold red] {str(e)}")
        return []