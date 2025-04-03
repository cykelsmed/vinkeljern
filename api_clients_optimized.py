"""
API client module for Vinkeljernet project (optimized version).

This module provides optimized functions to interact with external APIs such as
Perplexity for information retrieval and Claude/OpenAI for generating angles.
"""

import aiohttp
import asyncio
import json
import time
import logging
import requests
from typing import Optional, Dict, Any, List, Callable, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from models import RedaktionelDNA
from prompt_engineering import construct_angle_prompt, parse_angles_from_response
from cache_manager import cached_api, get_cache_stats, optimize_cache
from retry_manager import retry_with_circuit_breaker, CircuitOpenError, MaxRetriesExceededError
from error_handling import (
    retry_with_backoff, 
    safe_execute_async, 
    APIKeyMissingError, 
    APIConnectionError
)

# Config imports
try:
    from config import (
        PERPLEXITY_API_KEY, 
        OPENAI_API_KEY, 
        ANTHROPIC_API_KEY, 
        PERPLEXITY_TIMEOUT,
        ANTHROPIC_TIMEOUT, 
        OPENAI_TIMEOUT,
        USE_STREAMING,
        MAX_CONCURRENT_REQUESTS
    )
except ImportError:
    # Default values if not in config
    from config import PERPLEXITY_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY
    PERPLEXITY_TIMEOUT = 60  # seconds
    ANTHROPIC_TIMEOUT = 90  # seconds
    OPENAI_TIMEOUT = 60  # seconds
    USE_STREAMING = False
    MAX_CONCURRENT_REQUESTS = 5

# Configure logging
logger = logging.getLogger("vinkeljernet.api")

# API endpoints
PERPLEXITY_API_URL = 'https://api.perplexity.ai/chat/completions'
ANTHROPIC_API_URL = 'https://api.anthropic.com/v1/messages'
OPENAI_API_URL = 'https://api.openai.com/v1/chat/completions'

# Connection pool for API requests
_connection_pool = None
_session_pool = {}

def get_connection_pool(max_workers=MAX_CONCURRENT_REQUESTS):
    """Get the global thread pool executor for API requests."""
    global _connection_pool
    if _connection_pool is None:
        _connection_pool = ThreadPoolExecutor(max_workers=max_workers)
    return _connection_pool

async def get_aiohttp_session(key=None, timeout=60):
    """Get or create an aiohttp session from the pool."""
    global _session_pool
    
    if key is None:
        key = 'default'
        
    if key not in _session_pool or _session_pool[key].closed:
        # Configure timeout
        timeout_config = aiohttp.ClientTimeout(total=timeout)
        
        # Create connector with SSL context
        import ssl
        import certifi
        
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        connector = aiohttp.TCPConnector(ssl=ssl_context, limit=MAX_CONCURRENT_REQUESTS)
        
        _session_pool[key] = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout_config,
            raise_for_status=False
        )
        
    return _session_pool[key]

async def close_all_sessions():
    """Close all aiohttp sessions in the pool."""
    global _session_pool
    
    for key, session in list(_session_pool.items()):
        if not session.closed:
            await session.close()
    
    _session_pool.clear()

@cached_api(ttl=3600)  # Cache for 1 hour
@retry_with_circuit_breaker(
    max_retries=3,
    initial_backoff=1.0,
    backoff_factor=2.0,
    exceptions=[aiohttp.ClientError, asyncio.TimeoutError, ConnectionError],
    circuit_name="perplexity_api"
)
@safe_execute_async(fallback_return=None)
async def fetch_topic_information(
    topic: str, 
    dev_mode: bool = False, 
    bypass_cache: bool = False,
    progress_callback=None,
    detailed: bool = False
) -> Optional[str]:
    """
    Fetch information about a topic using the Perplexity API asynchronously with optimized performance.
    
    Args:
        topic: The news topic to fetch information about
        dev_mode: If True, disables SSL verification (development only!)
        bypass_cache: If True, ignore cached results
        progress_callback: Optional callback function to report progress (0-100)
        detailed: If True, get more comprehensive information
        
    Returns:
        Optional[str]: The information retrieved or None if failed
    """
    if not PERPLEXITY_API_KEY:
        raise APIKeyMissingError("Perplexity API nøgle mangler. Sørg for at have en PERPLEXITY_API_KEY i din .env fil.")
    
    # Update progress if callback provided
    if progress_callback:
        await progress_callback(25)

    # Improved prompts for better performance
    if detailed:
        # More comprehensive prompt for detailed report with specific structure
        user_prompt = f"""
        Giv en grundig og velstruktureret analyse af emnet '{topic}' med følgende sektioner:
        
        # OVERSIGT
        En kort 3-5 linjers opsummering af emnet, der dækker det mest centrale.
        
        # BAGGRUND
        Relevant historisk kontekst og udvikling indtil nu. Inkluder vigtige begivenheder og milepæle med præcise datoer.
        
        # AKTUEL STATUS
        Den nuværende situation med fokus på de seneste udviklinger. Beskriv præcist hvad der sker lige nu og hvorfor det er vigtigt.
        
        # NØGLETAL
        Konkrete statistikker, data og fakta relateret til emnet. Inkluder tal, procenter og, hvis muligt, kilder til informationen.
        
        # PERSPEKTIVER
        De forskellige synspunkter og holdninger til emnet fra forskellige aktører og interessenter. Præsenter de forskellige sider objektivt.
        
        # RELEVANS FOR DANMARK
        Hvordan emnet specifikt relaterer til eller påvirker Danmark og danskerne. Inkluder lokale eksempler når det er relevant.
        
        # FREMTIDSUDSIGTER
        Forventede eller mulige fremtidige udviklinger og tendenser baseret på de aktuelle fakta.
        
        # KILDER
        En liste over 3-5 pålidelige danske kilder, hvor man kan finde yderligere information om emnet.
        
        Formatér svaret med tydelige overskrifter for hver sektion. Sørg for at information er så faktabaseret og objektiv som muligt.
        """
        max_tokens = 1500
    else:
        # Optimized standard prompt for faster response
        user_prompt = f"""
        Giv en koncis og velstruktureret oversigt over følgende nyhedsemne: '{topic}'.
        
        Din oversigt skal indeholde:
        
        # OVERSIGT
        En kort 2-3 linjers sammenfatning af, hvad emnet handler om.
        
        # AKTUEL STATUS
        Den nuværende situation og hvorfor emnet er relevant lige nu.
        
        # NØGLETAL
        2-3 vigtige fakta, statistikker eller tal relateret til emnet.
        
        # PERSPEKTIVER
        Kort opsummering af forskellige synspunkter på emnet.
        
        Hold svaret faktuelt og præcist med vigtige datoer og konkrete detaljer.
        """
        max_tokens = 800  # Reduced token count for faster response

    # Optimized system prompt for better performance
    system_prompt = """Du er en erfaren dansk journalist med ekspertise i at fremstille komplekse emner på en struktureret og faktabaseret måde. Din opgave er at give pålidelig og velstruktureret information om aktuelle nyhedsemner. Vær koncis og fokuseret."""

    payload = {
        "model": "sonar",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "top_p": 0.85,
        "return_images": False,
        "return_related_questions": False
    }

    # Update progress if callback provided
    if progress_callback:
        await progress_callback(40)

    # Get session from pool
    session = await get_aiohttp_session(key="perplexity", timeout=PERPLEXITY_TIMEOUT)
    
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    
    start_time = time.time()
    
    try:
        # Make API request with timeout
        async with session.post(PERPLEXITY_API_URL, json=payload, headers=headers) as response:
            # Update progress
            if progress_callback:
                await progress_callback(70)
                
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"Perplexity API error: status={response.status}, response={error_text}")
                return None
                
            response_data = await response.json()
            
            # Update progress
            if progress_callback:
                await progress_callback(90)
                
            # Extract content
            content = response_data['choices'][0]['message']['content']
            
            # Record latency
            latency = time.time() - start_time
            logger.info(f"Perplexity API request completed in {latency:.2f} seconds")
            
            # Final progress update
            if progress_callback:
                await progress_callback(100)
                
            return content
    except Exception as e:
        logger.error(f"Error fetching topic information: {str(e)}")
        # Attempt to close and recreate the session on error
        try:
            if "perplexity" in _session_pool:
                await _session_pool["perplexity"].close()
                del _session_pool["perplexity"]
        except:
            pass
        raise

@cached_api(ttl=86400)  # Cache for 24 hours - sources don't change often
async def fetch_source_suggestions(
    topic: str,
    bypass_cache: bool = False
) -> Optional[str]:
    """
    Fetch source suggestions for a topic using Claude API.
    
    Args:
        topic: The news topic to find sources for
        bypass_cache: If True, ignore cached results
        
    Returns:
        Optional[str]: Source suggestions or None if failed
    """
    if not ANTHROPIC_API_KEY:
        logger.warning("Anthropic API key missing, cannot fetch source suggestions")
        return None
        
    # Optimized prompt for source suggestions
    prompt = f"""
    Baseret på emnet '{topic}', giv en kort liste med 3-5 relevante og troværdige danske kilder, 
    som en journalist kunne bruge til research. Inkluder officielle hjemmesider, forskningsinstitutioner, 
    eksperter og organisationer. Formater som en simpel punktopstilling med korte beskrivelser på dansk.
    Hold dit svar under 250 ord og fokuser på de mest pålidelige kilder.
    """
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01"
    }
    
    payload = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 500,
        "temperature": 0.2,
        "system": "Du er en hjælpsom researchassistent med stort kendskab til troværdige danske kilder. Du svarer altid på dansk.",
        "messages": [{"role": "user", "content": prompt}],
    }
    
    # Get session from pool
    session = await get_aiohttp_session(key="anthropic_sources", timeout=30)  # Shorter timeout for this simpler task
    
    try:
        async with session.post(ANTHROPIC_API_URL, json=payload, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"Anthropic API error (sources): status={response.status}, response={error_text}")
                return None
                
            response_data = await response.json()
            return response_data['content'][0]['text']
            
    except Exception as e:
        logger.error(f"Error fetching source suggestions: {str(e)}")
        return None

async def generate_editorial_considerations(
    topic: str, 
    profile_name: str, 
    angles: List[Dict],
    bypass_cache: bool = False
) -> str:
    """
    Generate editorial considerations for the given angles via Claude API.
    
    Args:
        topic: The news topic
        profile_name: Name of the editorial profile used
        angles: List of generated angles
        bypass_cache: If True, ignore cached results
        
    Returns:
        str: Editorial considerations text
    """
    # Create cache key for this specific request
    cache_key = f"editorial_considerations_{topic}_{profile_name}_{len(angles)}"
    
    # Check cache
    if not bypass_cache:
        from cache_manager import load_from_cache
        cached_result = load_from_cache(cache_key)
        if cached_result is not None:
            logger.info("Using cached editorial considerations")
            return cached_result
    
    if not ANTHROPIC_API_KEY:
        return "Kunne ikke generere redaktionelle overvejelser: Manglende API-nøgle."
    
    # Format angles for the prompt - only include essential information to reduce token usage
    formatted_angles = ""
    for i, angle in enumerate(angles, 1):
        headline = angle.get('overskrift', 'Ingen overskrift')
        description = angle.get('beskrivelse', 'Ingen beskrivelse')
        criteria = angle.get('nyhedskriterier', [])
        criteria_str = ", ".join(criteria) if criteria else "Ingen angivne"
        
        formatted_angles += f"Vinkel {i}: {headline}\n"
        formatted_angles += f"Beskrivelse: {description}\n"
        formatted_angles += f"Nyhedskriterier: {criteria_str}\n\n"
    
    # Optimized prompt for editorial considerations
    prompt = f"""
    Som erfaren nyhedsredaktør, giv en kortfattet redaktionel analyse af følgende vinkelforslag til emnet "{topic}" med henblik på "{profile_name}" profilen:
    
    {formatted_angles}
    
    Giv en saglig og konstruktiv analyse der omfatter:
    
    1. JOURNALISTISKE STYRKER: Hvilke vinkler er særligt stærke og hvorfor?
    2. REDAKTIONELLE MULIGHEDER: Hvilke journalistiske formater ville passe godt?
    3. PRIORITERING: Hvilke 2-3 vinkler bør prioriteres højest?
    
    Vær kortfattet og specifik med direkte reference til de angivne vinkler.
    """
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01"
    }
    
    payload = {
        "model": "claude-3-haiku-20240307",  # Using a smaller, faster model
        "max_tokens": 1000,
        "temperature": 0.2,
        "system": "Du er en erfaren redaktør på et dansk nyhedsmedie med ekspertise i journalistiske vinkler.",
        "messages": [{"role": "user", "content": prompt}],
    }
    
    # Get session from pool
    session = await get_aiohttp_session(key="anthropic_editorial", timeout=45)
    
    try:
        async with session.post(ANTHROPIC_API_URL, json=payload, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"Anthropic API error (editorial): status={response.status}, response={error_text}")
                return "Kunne ikke generere redaktionelle overvejelser. Der opstod en fejl ved kontakt til AI-systemet."
                
            response_data = await response.json()
            result = response_data['content'][0]['text']
            
            # Cache the result
            from cache_manager import save_to_cache
            save_to_cache(cache_key, result, 86400)  # Cache for 24 hours
            
            return result
            
    except Exception as e:
        logger.error(f"Error generating editorial considerations: {str(e)}")
        return f"Kunne ikke generere redaktionelle overvejelser: {str(e)}"

async def _stream_claude_response(prompt, system_prompt=""):
    """Stream response from Claude API and assemble it incrementally."""
    if not ANTHROPIC_API_KEY:
        raise APIKeyMissingError("Anthropic API nøgle mangler.")
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "claude-3-opus-20240229-streaming-v1"  # Enable streaming
    }
    
    payload = {
        "model": "claude-3-opus-20240229",
        "max_tokens": 2500,
        "temperature": 0.7,
        "stream": True,
        "system": system_prompt or "Du er en erfaren journalist med ekspertise i at udvikle kreative og relevante nyhedsvinkler.",
        "messages": [{"role": "user", "content": prompt}],
    }
    
    # Get session
    session = await get_aiohttp_session(key="anthropic_stream", timeout=ANTHROPIC_TIMEOUT)
    
    try:
        async with session.post(ANTHROPIC_API_URL, json=payload, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"Claude API streaming error: {response.status}, {error_text}")
                raise APIConnectionError(f"Claude API fejl: {response.status}")
            
            # Process streaming response
            full_response = ""
            async for line in response.content:
                line = line.decode('utf-8').strip()
                if not line or line == "data: [DONE]":
                    continue
                    
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        if 'delta' in data and 'text' in data['delta']:
                            chunk = data['delta']['text']
                            full_response += chunk
                    except Exception as e:
                        logger.error(f"Error parsing streaming response line: {e}")
            
            return full_response
            
    except Exception as e:
        logger.error(f"Error in Claude API streaming: {e}")
        raise APIConnectionError(f"Fejl ved forbindelse til Claude API: {str(e)}")

async def process_generation_request(
    topic: str,
    profile: RedaktionelDNA,
    bypass_cache: bool = False,
    progress_callback = None
) -> List[Dict[str, Any]]:
    """
    Process an angle generation request with optimized parallel API calls.
    
    Args:
        topic: News topic to generate angles for
        profile: Editorial DNA profile to use
        bypass_cache: If True, ignore cached results
        progress_callback: Optional callback function for progress updates
        
    Returns:
        List[Dict]: Generated angles
    """
    # Update progress
    if progress_callback:
        await progress_callback(10)
    
    # Convert profile into strings for prompt construction
    principper = "\n".join([f"- {p}" for p in profile.kerneprincipper])
    nyhedskriterier = "\n".join([f"- {k}: {v}" for k, v in profile.nyhedsprioritering.items()])
    fokusomrader = "\n".join([f"- {f}" for f in profile.fokusOmrader])
    nogo_omrader = "\n".join([f"- {n}" for n in profile.noGoOmrader]) if profile.noGoOmrader else "Ingen"
    
    # Launch background tasks in parallel
    # 1. Get topic information
    # 2. Get source suggestions
    background_info_task = asyncio.create_task(
        fetch_topic_information(topic, bypass_cache=bypass_cache, progress_callback=progress_callback)
    )
    
    # Update progress
    if progress_callback:
        await progress_callback(20)
    
    # Wait for background info which is needed for the prompt
    topic_info = await background_info_task
    
    if not topic_info:
        topic_info = f"Emnet handler om {topic}. Ingen yderligere baggrundsinformation tilgængelig."
    
    # Create the prompt
    prompt = construct_angle_prompt(
        topic,
        topic_info,
        principper,
        profile.tone_og_stil,
        fokusomrader,
        nyhedskriterier,
        nogo_omrader
    )
    
    # Update progress
    if progress_callback:
        await progress_callback(40)
    
    # Generate angles with Claude API (with optimized streaming if enabled)
    try:
        if USE_STREAMING:
            response_text = await _stream_claude_response(prompt)
        else:
            # Non-streaming version
            headers = {
                "Content-Type": "application/json",
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01"
            }
            
            payload = {
                "model": "claude-3-opus-20240229",
                "max_tokens": 2500,
                "temperature": 0.7,
                "system": "Du er en erfaren journalist med ekspertise i at udvikle kreative og relevante nyhedsvinkler.",
                "messages": [{"role": "user", "content": prompt}],
            }
            
            # Get session
            session = await get_aiohttp_session(key="anthropic", timeout=ANTHROPIC_TIMEOUT)
            
            async with session.post(ANTHROPIC_API_URL, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Claude API error: {response.status}, {error_text}")
                    raise APIConnectionError(f"Claude API fejl: {response.status}")
                    
                response_data = await response.json()
                response_text = response_data['content'][0]['text']
        
        # Update progress
        if progress_callback:
            await progress_callback(70)
            
        # Parse angles from response
        angles = parse_angles_from_response(response_text)
        
        # Sources task (can run in parallel while parsing angles)
        source_task = asyncio.create_task(
            fetch_source_suggestions(topic, bypass_cache=bypass_cache)
        )
        
        if not angles:
            logger.error("No angles parsed from response")
            return []
            
        # Add background info to each angle
        perplexity_extract = topic_info[:1000] + ("..." if len(topic_info) > 1000 else "")
        for angle in angles:
            if isinstance(angle, dict):
                angle['perplexityInfo'] = perplexity_extract
        
        # Update progress
        if progress_callback:
            await progress_callback(85)
        
        # Wait for the source suggestions to complete
        try:
            source_text = await source_task
            if source_text:
                for angle in angles:
                    if isinstance(angle, dict):
                        angle['kildeForslagInfo'] = source_text
        except Exception as e:
            logger.error(f"Error getting source suggestions: {e}")
            # Continue without source suggestions if they fail
        
        # Update progress
        if progress_callback:
            await progress_callback(95)
        
        # Filter and rank angles (we'll optimize this function separately)
        from angle_processor import filter_and_rank_angles
        ranked_angles = filter_and_rank_angles(angles, profile, 5)
        
        # Final progress update
        if progress_callback:
            await progress_callback(100)
            
        return ranked_angles
        
    except Exception as e:
        logger.error(f"Error generating angles: {e}")
        raise

class APIPerformanceMetrics:
    """Track performance metrics for API calls."""
    
    def __init__(self):
        self.requests = 0
        self.successes = 0
        self.failures = 0
        self.total_latency = 0
        self.max_latency = 0
        self.min_latency = float('inf')
        self.cache_hits = 0
        self.cache_misses = 0
        self.start_time = time.time()
        
    def record_request(self, success: bool, latency: float, cache_hit: bool = False):
        """Record a request with its result and latency."""
        self.requests += 1
        
        if success:
            self.successes += 1
        else:
            self.failures += 1
            
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
            
        if not cache_hit:  # Only count latency for actual API calls
            self.total_latency += latency
            self.max_latency = max(self.max_latency, latency)
            self.min_latency = min(self.min_latency, latency)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        uptime = time.time() - self.start_time
        
        avg_latency = self.total_latency / max(1, (self.requests - self.cache_hits))
        success_rate = (self.successes / max(1, self.requests)) * 100
        cache_hit_rate = (self.cache_hits / max(1, self.requests)) * 100
        
        return {
            "uptime_seconds": uptime,
            "uptime_formatted": f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m {int(uptime % 60)}s",
            "total_requests": self.requests,
            "successful_requests": self.successes,
            "failed_requests": self.failures,
            "success_rate": f"{success_rate:.1f}%",
            "average_latency_ms": int(avg_latency * 1000),
            "min_latency_ms": int(self.min_latency * 1000) if self.min_latency != float('inf') else 0,
            "max_latency_ms": int(self.max_latency * 1000),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": f"{cache_hit_rate:.1f}%"
        }

# Global metrics instance
metrics = APIPerformanceMetrics()

def get_performance_metrics() -> Dict[str, Any]:
    """
    Get combined performance metrics for the API client.
    
    Returns:
        Dict containing API and cache performance metrics
    """
    api_metrics = metrics.get_metrics()
    cache_metrics = get_cache_stats()
    
    from retry_manager import get_circuit_stats
    circuit_metrics = get_circuit_stats()
    
    return {
        "api": api_metrics,
        "cache": cache_metrics,
        "circuits": circuit_metrics
    }

async def initialize_api_client():
    """Initialize the API client, warming up connections."""
    # Pre-create sessions for each API
    await get_aiohttp_session(key="perplexity", timeout=PERPLEXITY_TIMEOUT)
    await get_aiohttp_session(key="anthropic", timeout=ANTHROPIC_TIMEOUT)
    await get_aiohttp_session(key="anthropic_sources", timeout=30)
    await get_aiohttp_session(key="anthropic_editorial", timeout=45)
    await get_aiohttp_session(key="anthropic_stream", timeout=ANTHROPIC_TIMEOUT)
    
    # Initialize thread pool
    get_connection_pool()
    
    # Check cache size and clean up if needed
    from cache_manager import _check_cache_size
    await _check_cache_size()
    
    logger.info("API client initialized and ready")

async def shutdown_api_client():
    """Properly shut down the API client, closing connections."""
    # Close all sessions
    await close_all_sessions()
    
    # Shut down thread pool
    if _connection_pool is not None:
        _connection_pool.shutdown(wait=True)
        
    logger.info("API client shut down successfully")