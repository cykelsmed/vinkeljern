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
from typing import Optional, Dict, Any, List, Tuple, Callable
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

@cached_api(ttl=7200)  # Cache for 2 hours
@retry_with_circuit_breaker(
    max_retries=2,
    initial_backoff=1.0,
    backoff_factor=2.0,
    exceptions=[requests.RequestException, asyncio.TimeoutError, ConnectionError],
    circuit_name="perplexity_source_suggestions"
)
@safe_execute_async(fallback_return=None)
async def generate_expert_source_suggestions(
    topic: str, 
    angle_headline: str, 
    angle_description: str, 
    dev_mode: bool = False,
    bypass_cache: bool = False,
    progress_callback: Optional[Callable[[int], None]] = None
) -> Optional[Dict[str, Any]]:
    """
    Generate expert and source suggestions for a specific news angle.
    
    Args:
        topic: The main news topic
        angle_headline: The headline of the specific angle
        angle_description: Description of the angle
        dev_mode: If True, disables SSL verification (development only!)
        bypass_cache: If True, ignore cached results
        progress_callback: Optional callback function to report progress (0-100)
        
    Returns:
        Optional[Dict]: Dictionary with experts and sources or None if failed
          Format: {
            "experts": [{"name": str, "title": str, "organization": str, "expertise": str}],
            "sources": [{"name": str, "type": str, "description": str, "url": str}],
            "statistics": [{"description": str, "source": str, "relevance": str}]
          }
    """
    if not PERPLEXITY_API_KEY:
        raise APIKeyMissingError("Perplexity API nøgle mangler. Sørg for at have en PERPLEXITY_API_KEY i din .env fil.")
    
    log_info(f"Genererer ekspertforslag til vinkel: \"{angle_headline}\"")
    
    # Update progress if callback provided
    if progress_callback:
        await progress_callback(10)

    # Construct the prompt for expert source suggestions
    prompt = f"""
    For nyhedsemnet "{topic}" med den specifikke vinkel:
    
    Overskrift: {angle_headline}
    Beskrivelse: {angle_description}
    
    Giv forslag til følgende tre kategorier (streng formateret som JSON):
    
    1. EKSPERTER: Find 2-3 danske eksperter der kunne interviewes om vinklen. For hver ekspert, angiv:
       - navn (fuldt navn)
       - titel (f.eks. "Professor i økonomi")
       - organisation (deres universitet, forskningsinstitution, virksomhed, etc.)
       - ekspertise (kort beskrivelse af deres faglige relevans for vinklen)
    
    2. KILDER: Identificer 2-3 relevante rapporter, undersøgelser eller andre kilder til at underbygge vinklen. For hver kilde, angiv:
       - navn (titel på kilden)
       - type (rapport, undersøgelse, database, osv.)
       - beskrivelse (1-2 sætninger om hvad kilden indeholder relevant for vinklen)
       - url (hvis du kender et specifikt link, ellers "N/A")
    
    3. STATISTIK: Foreslå 1-2 statistikker eller datakilder som kunne underbygge vinklen:
       - beskrivelse (hvad statistikken viser)
       - kilde (hvor data kommer fra)
       - relevans (1 sætning om hvorfor denne statistik er relevant for vinklen)
    
    Returner svaret formateret som et JSON objekt med ovenstående felter. Vær meget specifikk og konkret, undgå vage eller generiske forslag. Giv kun realistiske, faktiske eksperter og kilder (ikke fiktion).
    
    Eksempel på JSON-format (her skal du erstatte med faktisk indhold baseret på vinkel og emne):
    
    ```json
    {
      "eksperter": [
        {
          "navn": "Peter Jensen",
          "titel": "Professor i klimaforandringer",
          "organisation": "Københavns Universitet",
          "ekspertise": "Fokus på havniveaustigninger og kystområder"
        }
      ],
      "kilder": [
        {
          "navn": "DMI's klimarapport 2023",
          "type": "Årsrapport",
          "beskrivelse": "Detaljeret analyse af klimaændringer i Danmark",
          "url": "N/A"
        }
      ],
      "statistik": [
        {
          "beskrivelse": "Havniveaustigning langs danske kyster 2010-2023",
          "kilde": "Danmarks Statistik",
          "relevans": "Dokumenterer den accelererende stigning i havniveau"
        }
      ]
    }
    ```
    
    Vær sikker på at eksperterne er relevante for det specifikke vinkel, ikke kun for det overordnede emne. Returner KUN det rene JSON-objekt uden forklaringer eller indledende tekst.
    """

    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }

    # Create a system prompt that emphasizes factual, concrete suggestions
    system_prompt = """Du er en researcher for en dansk journalist, der skal foreslå reelle, faktiske eksperter, kilder og statistikker til en kommende artikel. 
    Dine forslag skal være:
    1. Konkrete og specifikke - ikke vage eller generelle
    2. Faktuelle - brug kun reelle eksperter og organisationer der faktisk eksisterer i Danmark
    3. Relevante - præcist målrettet vinklen og ikke kun det overordnede emne
    4. Forskellige - undgå overlap i de foreslåede eksperter/kilder
    5. Formateret præcist som JSON uden yderligere forklaringer
    
    Hvis du er i tvivl om en ekspert, kilde eller statistik, så prioritér kvalitet frem for kvantitet. Hellere færre, men meget relevante forslag."""

    payload = {
        "model": "sonar",  # Using the model with the most updated information 
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 800,
        "temperature": 0.2,  # Low temperature for factual responses
        "top_p": 0.9,
        "return_images": False,
        "return_related_questions": False
    }

    # Update progress
    if progress_callback:
        await progress_callback(30)

    try:
        # Use our secure session creation function
        async with await create_secure_api_session(dev_mode=dev_mode) as session:
            async with session.post(PERPLEXITY_API_URL, headers=headers, json=payload) as response:
                # Update progress
                if progress_callback:
                    await progress_callback(70)
                
                if response.status != 200:
                    error_text = await response.text()
                    log_error(f"Perplexity API Error for expert suggestions: {response.status}: {error_text[:200]}")
                    return None
                
                data = await response.json()
                
                # Update progress
                if progress_callback:
                    await progress_callback(90)
                
                try:
                    result = data['choices'][0]['message']['content']
                    
                    # Extract just the JSON part from the response
                    import re
                    
                    # Try to find JSON content within code blocks
                    json_match = re.search(r'```json\s*(.*?)\s*```', result, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                    else:
                        # If no code block, try to find content that looks like JSON
                        json_str = re.search(r'(\{.*\})', result, re.DOTALL)
                        if json_str:
                            json_str = json_str.group(1)
                        else:
                            # Just use the whole content as last resort
                            json_str = result
                    
                    # Parse the JSON string
                    parsed_data = json.loads(json_str)
                    
                    # Normalize field names to English for consistency in the code
                    normalized_data = {
                        "experts": parsed_data.get("eksperter", []),
                        "sources": parsed_data.get("kilder", []),
                        "statistics": parsed_data.get("statistik", [])
                    }
                    
                    # Final progress update
                    if progress_callback:
                        await progress_callback(100)
                    
                    return normalized_data
                    
                except (KeyError, IndexError, json.JSONDecodeError) as e:
                    log_warning(f"Failed to parse expert suggestions: {str(e)}")
                    log_warning(f"Raw response: {result[:200]}...")
                    return None
                    
    except Exception as e:
        log_error(f"Error generating expert suggestions: {str(e)}")
        return None

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
async def fetch_topic_information(topic: str, dev_mode: bool = False, bypass_cache: bool = False, progress_callback=None) -> Optional[str]:
    """
    Fetch information about a topic using the Perplexity API asynchronously.
    
    Args:
        topic: The news topic to fetch information about
        dev_mode: If True, disables SSL verification (development only!)
        bypass_cache: If True, ignore cached results
        progress_callback: Optional callback function to report progress (0-100)
        
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
    
    # Update progress if callback provided
    if progress_callback:
        await progress_callback(25)

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
            # Update progress before making the request
            if progress_callback:
                await progress_callback(40)
                
            async with session.post(PERPLEXITY_API_URL, headers=headers, json=payload) as response:
                # Update progress after receiving response
                if progress_callback:
                    await progress_callback(75)
                    
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
                
                # Update progress after parsing response
                if progress_callback:
                    await progress_callback(90)
                    
                try:
                    result = data['choices'][0]['message']['content']
                    
                    # Final progress update
                    if progress_callback:
                        await progress_callback(100)
                        
                    return result
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

async def generate_angles(
    topic: str,
    topic_info: str,
    profile: RedaktionelDNA,
    bypass_cache: bool = False
) -> List[Dict[str, Any]]:
    """
    Genererer nyhedsvinkler baseret på emne, baggrundsinformation og profil.
    """
    if not OPENAI_API_KEY:
        raise ValueError(
            "OpenAI API-nøgle mangler. Sørg for at have en OPENAI_API_KEY i din .env fil."
        )
    
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        principper = "\n".join([f"- {p}" for p in profile.kerneprincipper])
        nyhedskriterier = "\n".join([f"- {k}: {v}" for k, v in profile.nyhedsprioritering.items()])
        fokusomrader = "\n".join([f"- {f}" for f in profile.fokusOmrader])
        nogo_omrader = "\n".join([f"- {n}" for n in profile.noGoOmrader]) if profile.noGoOmrader else "Ingen"
        
        prompt = construct_angle_prompt(
            topic,
            topic_info,
            principper,
            profile.tone_og_stil,
            fokusomrader,
            nyhedskriterier,
            nogo_omrader
        )
        
        response = await client.send_request(prompt, temperature=0.7)
        angles = parse_angles_from_response(response)
        
        if not angles:
            log_error(f"Failed to parse any angles for topic: {topic}")
        
        return angles
    except Exception as e:
        log_error(f"Error generating angles: {str(e)}")
        return []

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