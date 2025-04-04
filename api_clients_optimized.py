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
                return f"Fejl ved hentning af emne-information: Status {response.status}"
            
            try:
                response_data = await response.json()
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in Perplexity API response: {e}")
                return "Fejl: Modtog ikke gyldigt svar fra API'et"
            
            # Update progress
            if progress_callback:
                await progress_callback(90)
                
            # Extract content with error handling
            try:
                content = response_data['choices'][0]['message']['content']
                if not content:
                    logger.error("Empty content in Perplexity API response for topic info")
                    return "Ingen information tilgængelig. Der var et problem med API-svaret."
            except (KeyError, IndexError, TypeError) as e:
                logger.error(f"Failed to extract content from Perplexity API response: {e}")
                return f"Kunne ikke hente information: {str(e)}"
            
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
    progress_callback = None,
    include_expert_sources: bool = True,
    include_knowledge_distillate: bool = True,
) -> List[Dict[str, Any]]:
    """
    Process an angle generation request with optimized parallel API calls.
    
    Args:
        topic: News topic to generate angles for
        profile: Editorial DNA profile to use
        bypass_cache: If True, ignore cached results
        progress_callback: Optional callback function for progress updates
        include_expert_sources: If True, generate expert source suggestions for each angle
        include_knowledge_distillate: If True, generate a knowledge distillate from background info
        
    Returns:
        List[Dict]: Generated angles with additional information
    """
    # Update progress
    if progress_callback:
        await progress_callback(5)
    
    # Convert profile into strings for prompt construction
    principper = "\n".join([f"- {p}" for p in profile.kerneprincipper])
    nyhedskriterier = "\n".join([f"- {k}: {v}" for k, v in profile.nyhedsprioritering.items()])
    fokusomrader = "\n".join([f"- {f}" for f in profile.fokusOmrader])
    nogo_omrader = "\n".join([f"- {n}" for n in profile.noGoOmrader]) if profile.noGoOmrader else "Ingen"
    
    # Launch background tasks in parallel
    # 1. Get topic information
    background_info_task = asyncio.create_task(
        fetch_topic_information(topic, bypass_cache=bypass_cache, progress_callback=progress_callback)
    )
    
    # Update progress
    if progress_callback:
        await progress_callback(15)
    
    # Wait for background info which is needed for the prompt
    topic_info = await background_info_task
    
    if not topic_info:
        topic_info = f"Emnet handler om {topic}. Ingen yderligere baggrundsinformation tilgængelig."
    
    # Update progress
    if progress_callback:
        await progress_callback(20)
    
    # Launch knowledge distillate task in parallel (can run while generating angles)
    knowledge_distillate_task = None
    if include_knowledge_distillate:
        try:
            knowledge_distillate_task = asyncio.create_task(
                generate_knowledge_distillate(
                    topic_info=topic_info,
                    topic=topic,
                    bypass_cache=bypass_cache
                )
            )
        except Exception as e:
            logger.error(f"Failed to start knowledge distillate task: {e}")
            # Continue without knowledge distillate if task creation fails
    
    # Create the prompt for angle generation
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
        await progress_callback(30)
    
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
            await progress_callback(50)
            
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
            await progress_callback(60)
        
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
        
        # Filter and rank angles
        from angle_processor import filter_and_rank_angles
        ranked_angles = filter_and_rank_angles(angles, profile, 5)
        
        # Update progress
        if progress_callback:
            await progress_callback(70)
        
        # Get knowledge distillate result if the task was started
        knowledge_distillate = None
        if knowledge_distillate_task:
            try:
                knowledge_distillate = await knowledge_distillate_task
                # Add knowledge distillate to each angle
                if knowledge_distillate:
                    # Check if the result has an error field, and log it but still attach the partial result
                    if isinstance(knowledge_distillate, dict) and "error" in knowledge_distillate:
                        logger.warning(f"Knowledge distillate contains error: {knowledge_distillate.get('error')}")
                        # Still attach the result with the error field, as it contains empty arrays
                        # This prevents null reference errors in the UI
                        
                    for angle in ranked_angles:
                        if isinstance(angle, dict):
                            angle['videnDistillat'] = knowledge_distillate
            except Exception as e:
                logger.error(f"Error generating knowledge distillate: {e}")
                # Create a safe fallback distillate with error information
                fallback_distillate = {
                    "error": f"Kunne ikke generere videndistillat: {str(e)}",
                    "key_statistics": [],
                    "key_claims": [],
                    "perspectives": [],
                    "important_dates": []
                }
                # Still add the fallback to prevent UI errors
                for angle in ranked_angles:
                    if isinstance(angle, dict):
                        angle['videnDistillat'] = fallback_distillate
        
        # Update progress
        if progress_callback:
            await progress_callback(75)
        
        # Generate expert source suggestions for each angle if requested
        if include_expert_sources:
            expert_sources_tasks = []
            
            # Start expert source suggestions tasks for each angle
            for i, angle in enumerate(ranked_angles):
                if not isinstance(angle, dict):
                    continue
                    
                # Only process top angles to avoid too many API calls
                if i >= 3:  # Limit to top 3 angles
                    break
                    
                try:
                    # Extract headline and description for the expert source suggestion
                    headline = angle.get('overskrift', f"Vinkel om {topic}")
                    description = angle.get('beskrivelse', "")
                    
                    # Create task for expert source suggestions
                    task = asyncio.create_task(
                        generate_expert_source_suggestions(
                            topic=topic,
                            angle_headline=headline,
                            angle_description=description,
                            bypass_cache=bypass_cache
                        )
                    )
                    
                    # Store task with its angle index
                    expert_sources_tasks.append((i, task))
                except Exception as e:
                    logger.error(f"Failed to start expert source suggestions task for angle {i}: {e}")
            
            # Update progress
            if progress_callback:
                await progress_callback(85)
            
            # Wait for all expert source tasks to complete
            for i, task in expert_sources_tasks:
                try:
                    expert_sources = await task
                    if expert_sources and i < len(ranked_angles):
                        # Check if there's an error in the expert sources
                        if isinstance(expert_sources, dict) and "error" in expert_sources:
                            logger.warning(f"Expert sources for angle {i} contains error: {expert_sources.get('error')}")
                            # Still attach it as it has empty arrays for experts, institutions, etc.
                        
                        # Add expert sources to the angle
                        ranked_angles[i]['ekspertKilder'] = expert_sources
                except Exception as e:
                    logger.error(f"Error generating expert sources for angle {i}: {e}")
                    # Create a fallback expert sources structure with error information
                    fallback_sources = {
                        "experts": [],
                        "institutions": [],
                        "data_sources": [],
                        "error": f"Kunne ikke generere ekspertkilder: {str(e)}"
                    }
                    # Still add the fallback to prevent UI errors
                    if i < len(ranked_angles):
                        ranked_angles[i]['ekspertKilder'] = fallback_sources
        
        # Update progress
        if progress_callback:
            await progress_callback(95)
        
        # Add metadata to angles about which features were included
        for angle in ranked_angles:
            if isinstance(angle, dict):
                angle['harVidenDistillat'] = include_knowledge_distillate and 'videnDistillat' in angle
                angle['harEkspertKilder'] = include_expert_sources and 'ekspertKilder' in angle
        
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
    await get_aiohttp_session(key="anthropic_knowledge", timeout=60)
    await get_aiohttp_session(key="perplexity_experts", timeout=PERPLEXITY_TIMEOUT)
    
    # Initialize thread pool
    get_connection_pool()
    
    # Check cache size and clean up if needed
    from cache_manager import _check_cache_size
    await _check_cache_size()
    
    logger.info("API client initialized and ready")

@cached_api(ttl=7200)  # Cache for 2 hours
@retry_with_circuit_breaker(
    max_retries=3,
    initial_backoff=1.0,
    backoff_factor=2.0,
    exceptions=[aiohttp.ClientError, asyncio.TimeoutError, ConnectionError],
    circuit_name="anthropic_knowledge_distillate"
)
@safe_execute_async(fallback_return=None)
async def generate_knowledge_distillate(
    topic_info: str,
    topic: str,
    bypass_cache: bool = False,
    progress_callback: Optional[Callable] = None
) -> Optional[Dict[str, Any]]:
    """
    Generate a knowledge distillate from background information using Claude.
    
    Args:
        topic_info: The background information about the topic
        topic: The topic name/subject
        bypass_cache: If True, ignore cached results
        progress_callback: Optional callback function to report progress (0-100)
        
    Returns:
        Optional[Dict]: Structured knowledge distillate or None if failed
    """
    if not ANTHROPIC_API_KEY:
        raise APIKeyMissingError("Anthropic API nøgle mangler. Sørg for at have en ANTHROPIC_API_KEY i din .env fil.")
    
    # Update progress if callback provided
    if progress_callback:
        await progress_callback(25)
    
    # Optimized prompt for extracting structured knowledge
    system_prompt = """Du er en analytisk AI-forsker specialiseret i at uddrage og organisere den vigtigste viden fra baggrundsinformation. 
Din opgave er at skabe et videndistillat i form af et velstruktureret, kompakt format optimeret til journalistisk brug.
Du skal være yderst præcis og faktuel, og kun inkludere information der er direkte understøttet af kildematerialet.
Dit output skal være kortfattet, struktureret og fokuseret på de mest relevante fakta."""

    user_prompt = f"""
    Analyser følgende baggrundsinformation om emnet '{topic}' og skab et videndistillat.
    
    BAGGRUNDSINFORMATION:
    {topic_info}
    
    Formater dit svar som et JSON-objekt med følgende felter:
    
    1. "noegletal": En liste med de 3-5 vigtigste statistikker/tal fra materialet. Hvert nøgletal har følgende struktur:
       - "tal": Det faktiske tal/statistik
       - "beskrivelse": Kort beskrivelse af hvad tallet repræsenterer
       - "kilde": Kilden hvis angivet
    
    2. "centralePaastand": En liste med 3-5 centrale påstande/fakta fra materialet. Hver påstand har følgende struktur:
       - "paastand": Den faktiske påstand/faktum
       - "kilde": Kilden hvis angivet
    
    3. "vinkler": En liste med 3-4 forskellige perspektiver på emnet. Hvert perspektiv har følgende struktur:
       - "vinkel": Kort beskrivelse af perspektivet
       - "aktør": Hvem der repræsenterer dette perspektiv
    
    4. "datoer": En liste med 2-3 vigtige datoer relateret til emnet. Hver dato har følgende struktur:
       - "dato": Den specifikke dato (format: YYYY-MM-DD eller beskrivelse hvis præcis dato ikke er kendt)
       - "begivenhed": Hvad der skete på denne dato
       - "betydning": Kort beskrivelse af hvorfor denne dato er vigtig
    
    Brug KUN information der findes i baggrundsmaterialet. Hvis der mangler information til et af felterne, inkluder en kort liste med de felter der er tilgængelige. Svar med det rene JSON-objekt, uden forklarende tekst eller indledning.
    """
    
    # Update progress if callback provided
    if progress_callback:
        await progress_callback(40)
    
    # Get session from pool
    session = await get_aiohttp_session(key="anthropic_knowledge", timeout=60)
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01"
    }
    
    payload = {
        "model": "claude-3-opus-20240229",
        "max_tokens": 1500,
        "temperature": 0.1,  # Lower temperature for more deterministic results
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    
    start_time = time.time()
    
    try:
        # Make API request with timeout
        async with session.post(ANTHROPIC_API_URL, json=payload, headers=headers) as response:
            # Update progress
            if progress_callback:
                await progress_callback(70)
                
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"Anthropic API error (knowledge distillate): status={response.status}, response={error_text}")
                return {"error": f"API Error: Status {response.status}", "key_statistics": [], "key_claims": [], "perspectives": [], "important_dates": []}
            
            try:    
                response_data = await response.json()
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in Anthropic API response: {e}")
                return {"error": "Response is not valid JSON", "key_statistics": [], "key_claims": [], "perspectives": [], "important_dates": []}
            
            # Update progress
            if progress_callback:
                await progress_callback(85)
                
            # Extract content with error handling
            try:
                content = response_data['content'][0]['text']
                if not content:
                    logger.error("Empty text content in Anthropic API response")
                    return {"error": "Empty content received from API", "key_statistics": [], "key_claims": [], "perspectives": [], "important_dates": []}
            except (KeyError, IndexError, TypeError) as e:
                logger.error(f"Failed to extract content from Anthropic API response: {e}")
                return {"error": f"Failed to extract content: {str(e)}", "key_statistics": [], "key_claims": [], "perspectives": [], "important_dates": []}
            
            # Use the enhanced JSON parser for robust parsing
            from json_parser import safe_parse_json
            
            # Define expected structure for knowledge distillate
            expected_format = {
                "key_statistics": [],  # or "noegletal": []
                "key_claims": [],      # or "centralePaastand": []
                "perspectives": [],    # or "vinkler": []
                "important_dates": []  # or "datoer": []
            }
            
            # Parse the content using the robust parser
            distillate = safe_parse_json(
                content,
                context="knowledge distillate",
                fallback={
                    "error": "Failed to parse knowledge distillate",
                    "key_statistics": [],
                    "key_claims": [],
                    "perspectives": [],
                    "important_dates": []
                }
            )
            
            # Normalize field names if they use Danish variants
            if "noegletal" in distillate and "key_statistics" not in distillate:
                distillate["key_statistics"] = distillate.pop("noegletal")
            if "centralePaastand" in distillate and "key_claims" not in distillate:
                distillate["key_claims"] = distillate.pop("centralePaastand")
            if "vinkler" in distillate and "perspectives" not in distillate:
                distillate["perspectives"] = distillate.pop("vinkler")
            if "datoer" in distillate and "important_dates" not in distillate:
                distillate["important_dates"] = distillate.pop("datoer")
            
            # Ensure all required fields exist
            for field in expected_format:
                if field not in distillate:
                    distillate[field] = []
            
            # Record latency
            latency = time.time() - start_time
            logger.info(f"Knowledge distillate generation completed in {latency:.2f} seconds")
            
            # Final progress update
            if progress_callback:
                await progress_callback(100)
                
            return distillate
    except Exception as e:
        logger.error(f"Error generating knowledge distillate: {str(e)}")
        # Attempt to close and recreate the session on error
        try:
            if "anthropic_knowledge" in _session_pool:
                await _session_pool["anthropic_knowledge"].close()
                del _session_pool["anthropic_knowledge"]
        except:
            pass
        raise

@cached_api(ttl=43200)  # Cache for 12 hours - expert sources don't change often
@retry_with_circuit_breaker(
    max_retries=3,
    initial_backoff=1.0,
    backoff_factor=2.0,
    exceptions=[aiohttp.ClientError, asyncio.TimeoutError, ConnectionError],
    circuit_name="perplexity_expert_sources"
)
@safe_execute_async(fallback_return=None)
async def generate_expert_source_suggestions(
    topic: str,
    angle_headline: str,
    angle_description: str,
    bypass_cache: bool = False,
    progress_callback: Optional[Callable] = None
) -> Optional[Dict[str, Any]]:
    """
    Generate detailed suggestions for expert sources and institutions for a specific news angle.
    
    Args:
        topic: The general news topic
        angle_headline: The headline of the specific angle
        angle_description: Description of the news angle
        bypass_cache: If True, ignore cached results
        progress_callback: Optional callback function to report progress (0-100)
        
    Returns:
        Optional[Dict]: Structured dictionary with expert sources, institutions, and data sources
    """
    if not PERPLEXITY_API_KEY:
        raise APIKeyMissingError("Perplexity API nøgle mangler. Sørg for at have en PERPLEXITY_API_KEY i din .env fil.")
    
    # Update progress if callback provided
    if progress_callback:
        await progress_callback(20)
    
    # Comprehensive system prompt for expert source suggestions
    system_prompt = """Du er en specialiseret research-assistent med dyb kendskab til ekspertkilder i Danmark. 
Din opgave er at identificere og foreslå reelle, navngivne eksperter, organisationer og datakilder til journalistisk arbejde.
Du skal fokusere på at give konkrete, specifikke forslag der er direkte relevante for den angivne vinkel.
Prioriter danske eksperter og institutioner, og inkluder kun internationale kilder hvis de er særligt relevante.
Vær yderst præcis med navne, titler og organisationer. Undgå hypotetiske eller generiske forslag."""

    # User prompt with specific instructions for structured output
    user_prompt = f"""
    Find konkrete ekspertkilder, institutioner og datakilder til følgende journalistiske vinkel:
    
    EMNE: {topic}
    OVERSKRIFT: {angle_headline}
    BESKRIVELSE: {angle_description}
    
    Formater dit svar som et JSON-objekt med følgende felter:
    
    1. "eksperter": En liste med 4-6 konkrete ekspertkilder, hver med følgende struktur:
       - "navn": Ekspertens fulde navn (SKAL være en reel person)
       - "titel": Ekspertens titel og rolle
       - "organisation": Arbejdsplads eller tilknytning
       - "ekspertise": Kort beskrivelse af relevant ekspertise
       - "kontakt": Kontaktinformation, hvis tilgængelig (email, telefon, eller henvisning til institution)
       - "relevans": Kort forklaring på hvorfor denne ekspert er relevant for vinklen
    
    2. "institutioner": En liste med 3-5 relevante organisationer, hver med følgende struktur:
       - "navn": Organisationens fulde navn
       - "type": Type af organisation (universitet, myndighed, NGO, etc.)
       - "relevans": Hvorfor denne organisation er relevant
       - "kontaktperson": Navngiven kontaktperson/presseansvarlig hvis kendt, ellers "Presseafdeling"
       - "kontakt": Kontaktinformation til organisationen/presseafdelingen
    
    3. "datakilder": En liste med 2-4 specifikke datakilder, hver med følgende struktur:
       - "titel": Titel på rapport, database eller datasæt
       - "udgiver": Organisation der har udgivet eller vedligeholder datakilden
       - "beskrivelse": Kort beskrivelse af hvilke data der findes her
       - "link": Link til datakilden, hvis tilgængelig
       - "senest_opdateret": Hvornår datakilden sidst blev opdateret (hvis kendt)
    
    VIGTIGE KRAV:
    - Alle eksperter SKAL være reelle personer med korrekte titler og tilknytninger
    - Undgå generiske beskrivelser som "en ekspert i [emne]" - navngiv konkrete personer
    - Organisationer skal være reelle, eksisterende institutioner
    - For kilder med tilknytning til universiteter, angiv specifikt institut/afdeling
    - Prioriter diversitet i både køn og institutionel tilknytning
    - Inkluder både akademiske eksperter, praktikere og relevante myndighedspersoner
    - Sørg for at alle forslag er direkte relevante for den konkrete vinkel, ikke bare det overordnede emne
    - Inkluder kun kontaktoplysninger du er sikker på er korrekte (ellers angiv "Ikke offentligt tilgængelig")
    
    Svar udelukkende med det rene JSON-objekt, uden forklarende tekst eller indledning.
    """
    
    # Update progress if callback provided
    if progress_callback:
        await progress_callback(40)
    
    # Get session from pool
    session = await get_aiohttp_session(key="perplexity_experts", timeout=PERPLEXITY_TIMEOUT)
    
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "sonar",  # Using Perplexity's most up-to-date model for current information
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 2000,
        "temperature": 0.2,  # Lower temperature for more deterministic results
        "top_p": 0.85,
        "return_images": False,
        "return_related_questions": False
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
                logger.error(f"Perplexity API error (expert sources): status={response.status}, response={error_text}")
                return {
                    "experts": [],
                    "institutions": [],
                    "data_sources": [],
                    "error": f"API Error: Status {response.status}"
                }
            
            try:
                response_data = await response.json()
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in Perplexity API response: {e}")
                return {
                    "experts": [],
                    "institutions": [],
                    "data_sources": [],
                    "error": "Response is not valid JSON"
                }
            
            # Update progress
            if progress_callback:
                await progress_callback(85)
                
            # Extract content with error handling
            try:
                content = response_data['choices'][0]['message']['content']
                if not content:
                    logger.error("Empty content in Perplexity API response")
                    return {
                        "experts": [],
                        "institutions": [],
                        "data_sources": [],
                        "error": "Empty content received from API"
                    }
            except (KeyError, IndexError, TypeError) as e:
                logger.error(f"Failed to extract content from Perplexity API response: {e}")
                return {
                    "experts": [],
                    "institutions": [],
                    "data_sources": [],
                    "error": f"Failed to extract content: {str(e)}"
                }
            
            # Use the enhanced JSON parser for robust parsing
            from json_parser import safe_parse_json
            
            # Define expected structure for expert sources
            expected_format = {
                "experts": [],         # or "eksperter": []
                "institutions": [],    # or "institutioner": []
                "data_sources": []     # or "datakilder": []
            }
            
            # Parse the content using the robust parser
            source_suggestions = safe_parse_json(
                content,
                context="expert source suggestions",
                fallback={
                    "error": "Failed to parse expert sources",
                    "experts": [],
                    "institutions": [],
                    "data_sources": [],
                    "raw_response": content[:300] + ("..." if len(content) > 300 else "")
                }
            )
            
            # Normalize field names if they use Danish variants
            if "eksperter" in source_suggestions and "experts" not in source_suggestions:
                source_suggestions["experts"] = source_suggestions.pop("eksperter")
            if "institutioner" in source_suggestions and "institutions" not in source_suggestions:
                source_suggestions["institutions"] = source_suggestions.pop("institutioner")
            if "datakilder" in source_suggestions and "data_sources" not in source_suggestions:
                source_suggestions["data_sources"] = source_suggestions.pop("datakilder")
            
            # Ensure all required fields exist
            for field in expected_format:
                if field not in source_suggestions:
                    source_suggestions[field] = []
            
            # Record latency
            latency = time.time() - start_time
            logger.info(f"Expert source suggestions generated in {latency:.2f} seconds")
            
            # Final progress update
            if progress_callback:
                await progress_callback(100)
                
            return source_suggestions
    except Exception as e:
        logger.error(f"Error generating expert source suggestions: {str(e)}")
        # Attempt to close and recreate the session on error
        try:
            if "perplexity_experts" in _session_pool:
                await _session_pool["perplexity_experts"].close()
                del _session_pool["perplexity_experts"]
        except:
            pass
        raise

async def shutdown_api_client():
    """Properly shut down the API client, closing connections."""
    # Close all sessions
    await close_all_sessions()
    
    # Shut down thread pool
    if _connection_pool is not None:
        _connection_pool.shutdown(wait=True)
        
    logger.info("API client shut down successfully")