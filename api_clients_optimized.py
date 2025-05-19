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
import re
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
    ANTHROPIC_TIMEOUT = 180  # seconds
    OPENAI_TIMEOUT = 60  # seconds
    USE_STREAMING = False
    MAX_CONCURRENT_REQUESTS = 5

# Configure logging
logger = logging.getLogger("vinkeljernet.api")

# API endpoints
# Fjern Perplexity API URL og nøgle, brug kun Anthropic
# PERPLEXITY_API_URL = 'https://api.perplexity.ai/chat/completions'
ANTHROPIC_API_URL = 'https://api.anthropic.com/v1/messages'
OPENAI_API_URL = 'https://api.openai.com/v1/chat/completions'

# Connection pool for API requests
_connection_pool = None
_session_pool = {}
_initialization_logged = False  # Track if initialization has been logged

def get_connection_pool(max_workers=MAX_CONCURRENT_REQUESTS):
    """Get the global thread pool executor for API requests."""
    global _connection_pool
    if (_connection_pool is None):
        _connection_pool = ThreadPoolExecutor(max_workers=max_workers)
    return _connection_pool

async def get_aiohttp_session(key=None, timeout=60):
    """Get or create an aiohttp session from the pool."""
    global _session_pool
    
    if key is None:
        key = 'default'
    
    try:    
        # Check if the session exists and is not closed
        if key not in _session_pool or _session_pool[key].closed:
            # First make sure we have a running event loop
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                logger.warning(f"No running event loop found when creating session for {key}. Creating new loop.")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Configure timeout
            timeout_config = aiohttp.ClientTimeout(total=timeout)
            
            # Create connector with SSL context
            import ssl
            import certifi
            
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            connector = aiohttp.TCPConnector(
                ssl=ssl_context, 
                limit=MAX_CONCURRENT_REQUESTS,
                force_close=False,  # Don't force close connections
                enable_cleanup_closed=True  # Enable cleanup of closed connections
            )
            
            # Create a new session with the connector
            try:
                _session_pool[key] = aiohttp.ClientSession(
                    connector=connector,
                    timeout=timeout_config,
                    raise_for_status=False
                )
                logger.debug(f"Created new aiohttp session for key: {key}")
            except RuntimeError as e:
                if "Event loop is closed" in str(e):
                    logger.warning(f"Event loop was closed when creating session for {key}. Creating new loop.")
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    _session_pool[key] = aiohttp.ClientSession(
                        connector=connector,
                        timeout=timeout_config,
                        raise_for_status=False
                    )
                    logger.debug(f"Created new aiohttp session with new loop for key: {key}")
                else:
                    raise
        
        return _session_pool[key]
    except Exception as e:
        logger.error(f"Error creating session for key {key}: {str(e)}")
        # Return a new session outside the pool as fallback
        try:
            # Try to get the current loop again
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            timeout_config = aiohttp.ClientTimeout(total=timeout)
            connector = aiohttp.TCPConnector(limit=10)
            new_session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout_config,
                raise_for_status=False
            )
            logger.warning(f"Using fallback session for key {key} (not in pool)")
            return new_session
        except Exception as inner_e:
            logger.error(f"Failed to create fallback session: {str(inner_e)}")
            raise RuntimeError(f"Cannot create aiohttp session: {str(e)}, then: {str(inner_e)}")

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
    circuit_name="claude_topic_info"
)
@safe_execute_async(fallback_return=None)
async def fetch_topic_information(
    topic: str, 
    dev_mode: bool = False, 
    bypass_cache: bool = False,
    progress_callback=None,
    detailed: bool = False
) -> Optional[dict]:
    """
    Fetch information about a topic using the Claude API asynchronously.
    
    Args:
        topic: The news topic to fetch information about
        dev_mode: If True, disables SSL verification (development only!)
        bypass_cache: If True, ignore cached results
        progress_callback: Optional callback function to report progress (0-100)
        detailed: If True, get more comprehensive information
        
    Returns:
        Optional[dict]: The information retrieved or None if failed
    """
    if not ANTHROPIC_API_KEY:
        raise APIKeyMissingError("Claude (Anthropic) API nøgle mangler. Sørg for at have en ANTHROPIC_API_KEY i din .env fil.")
    
    # Update progress if callback provided
    if progress_callback:
        await progress_callback(25)

    # Prompt til Claude
    if detailed:
        user_prompt = f"""
        Giv en grundig og velstruktureret analyse af emnet '{topic}' med følgende sektioner:
        \n# OVERSIGT\nEn kort 3-5 linjers opsummering af emnet, der dækker det mest centrale.
        \n# BAGGRUND\nRelevant historisk kontekst og udvikling indtil nu. Inkluder vigtige begivenheder og milepæle med præcise datoer.
        \n# AKTUEL STATUS\nDen nuværende situation med fokus på de seneste udviklinger. Beskriv præcist hvad der sker lige nu og hvorfor det er vigtigt.
        \n# NØGLETAL\nKonkrete statistikker, data og fakta relateret til emnet. Inkluder tal, procenter og, hvis muligt, kilder til informationen.
        \n# PERSPEKTIVER\nDe forskellige synspunkter og holdninger til emnet fra forskellige aktører og interessenter. Præsenter de forskellige sider objektivt.
        \n# RELEVANS FOR DANMARK\nHvordan emnet specifikt relaterer til eller påvirker Danmark og danskerne. Inkluder lokale eksempler når det er relevant.
        \n# FREMTIDSUDSIGTER\nForventede eller mulige fremtidige udviklinger og tendenser baseret på de aktuelle fakta.
        \n# KILDER\nEn liste over 3-5 pålidelige danske kilder, hvor man kan finde yderligere information om emnet.\nFormatér svaret med tydelige overskrifter for hver sektion. Sørg for at information er så faktabaseret og objektiv som muligt.
        """
        max_tokens = 1500
    else:
        user_prompt = f"""
        Giv en koncis og velstruktureret oversigt over følgende nyhedsemne: '{topic}'.
        \nDin oversigt skal indeholde:
        \n# OVERSIGT\nEn kort 2-3 linjers sammenfatning af, hvad emnet handler om.
        \n# AKTUEL STATUS\nDen nuværende situation og hvorfor emnet er relevant lige nu.
        \n# NØGLETAL\n2-3 vigtige fakta, statistikker eller tal relateret til emnet.
        \n# PERSPEKTIVER\nKort opsummering af forskellige synspunkter på emnet.\nHold svaret faktuelt og præcist med vigtige datoer og konkrete detaljer.
        """
        max_tokens = 800

    system_prompt = """Du er en erfaren dansk journalist med ekspertise i at fremstille komplekse emner på en struktureret og faktabaseret måde. Din opgave er at give pålidelig og velstruktureret information om aktuelle nyhedsemner. Vær koncis og fokuseret."""

    payload = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": user_prompt}
        ]
    }

    if progress_callback:
        await progress_callback(40)

    session = await get_aiohttp_session(key="anthropic_topic_info", timeout=60)
    headers = {
        "Content-Type": "application/json",
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01"
    }
    start_time = time.time()
    try:
        async with session.post(ANTHROPIC_API_URL, json=payload, headers=headers) as response:
            if progress_callback:
                await progress_callback(70)
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"Claude API error: status={response.status}, response={error_text}")
                return f"Fejl ved hentning af emne-information: Status {response.status}"
            try:
                response_data = await response.json()
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in Claude API response: {e}")
                return "Fejl: Modtog ikke gyldigt svar fra API'et"
            if progress_callback:
                await progress_callback(90)
            try:
                content = response_data['content'][0]['text']
                if not content:
                    logger.error("Empty content in Claude API response for topic info")
                    return "Ingen information tilgængelig. Der var et problem med API-svaret."
            except (KeyError, IndexError, TypeError) as e:
                logger.error(f"Failed to extract content from Claude API response: {e}")
                return f"Kunne ikke hente information: {str(e)}"
            # Ingen kildeudtræk - returner kun tekst
            latency = time.time() - start_time
            logger.info(f"Claude API request completed in {latency:.2f} seconds")
            if progress_callback:
                await progress_callback(100)
            return {"text": content, "sources": {}}
    except Exception as e:
        logger.error(f"Error fetching topic information: {str(e)}")
        try:
            if "anthropic_topic_info" in _session_pool:
                await _session_pool["anthropic_topic_info"].close()
                del _session_pool["anthropic_topic_info"]
        except:
            pass
        raise

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
        "max_tokens": 1200,
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
    Process a generation request with the given args using the optimized API client.
    
    Args:
        topic: The topic to generate angles for
        profile: The editorial DNA profile
        bypass_cache: If True, bypass the cache
        progress_callback: Optional callback for progress updates
        include_expert_sources: If True, include expert sources
        include_knowledge_distillate: If True, include knowledge distillate
        
    Returns:
        List[Dict[str, Any]]: The generated angles
    """
    try:
        # Update progress - check if progress_callback is callable
        if progress_callback and callable(progress_callback):
            await progress_callback(5)
        
        # Convert profile into strings for prompt construction
        principper = "\n".join([f"- {p}" for p in profile.kerneprincipper])
        nyhedskriterier = "\n".join([f"- {k}: {v}" for k, v in profile.nyhedsprioritering.items()])
        fokusomrader = "\n".join([f"- {f}" for f in profile.fokusOmrader])
        nogo_omrader = "\n".join([f"- {n}" for n in profile.noGoOmrader]) if profile.noGoOmrader else "Ingen"
        
        # 1. Get topic information first since we need it for knowledge distillate
        if progress_callback and callable(progress_callback):
            await progress_callback(10)
            
        topic_info = await fetch_topic_information(
            topic, 
            bypass_cache=bypass_cache, 
            progress_callback=progress_callback if callable(progress_callback) else None
        )
        
        if not topic_info:
            topic_info = f"Emnet handler om {topic}. Ingen yderligere baggrundsinformation tilgængelig."
        
        # Update progress - check if progress_callback is callable
        if progress_callback and callable(progress_callback):
            await progress_callback(25)
        
        # 2. Generate knowledge distillate BEFORE generating angles
        knowledge_distillate = None
        knowledge_distillate_task = None
        if include_knowledge_distillate:
            try:
                if progress_callback and callable(progress_callback):
                    await progress_callback(30)
                    
                knowledge_distillate_task = asyncio.create_task(generate_knowledge_distillate(
                    topic_info=topic_info,
                    topic=topic,
                    bypass_cache=bypass_cache,
                    progress_callback=progress_callback if callable(progress_callback) else None
                ))
                
                if progress_callback and callable(progress_callback):
                    await progress_callback(40)
                    
            except Exception as e:
                logger.error(f"Failed to create knowledge distillate task: {e}")
                # Continue without knowledge distillate
        
        # 3. Incorporate knowledge distillate into the angle generation prompt
        knowledge_distillate_text = ""
        if knowledge_distillate:
            # Format the knowledge distillate for inclusion in the prompt
            knowledge_distillate_text = "VIDENDISTILLAT OM EMNET:\n\n"
            
            # Add main points if available
            if "hovedpunkter" in knowledge_distillate and knowledge_distillate["hovedpunkter"]:
                knowledge_distillate_text += "Hovedpunkter:\n"
                for point in knowledge_distillate["hovedpunkter"][:5]:  # Limit to 5
                    knowledge_distillate_text += f"- {point}\n"
                knowledge_distillate_text += "\n"
                
            # Add key statistics if available
            if "noegletal" in knowledge_distillate and knowledge_distillate["noegletal"]:
                knowledge_distillate_text += "Nøgletal:\n"
                for stat in knowledge_distillate["noegletal"][:3]:  # Limit to 3
                    if isinstance(stat, dict):
                        tal = stat.get("tal", "")
                        beskrivelse = stat.get("beskrivelse", "")
                        knowledge_distillate_text += f"- {tal}: {beskrivelse}\n"
                    else:
                        knowledge_distillate_text += f"- {stat}\n"
                knowledge_distillate_text += "\n"
                
            # Add key claims if available
            if "centralePaastand" in knowledge_distillate and knowledge_distillate["centralePaastand"]:
                knowledge_distillate_text += "Centrale påstande:\n"
                for claim in knowledge_distillate["centralePaastand"][:3]:  # Limit to 3
                    if isinstance(claim, dict):
                        paastand = claim.get("paastand", "")
                        knowledge_distillate_text += f"- {paastand}\n"
                    else:
                        knowledge_distillate_text += f"- {claim}\n"
                knowledge_distillate_text += "\n"
        
        # Create the prompt for angle generation, now including knowledge distillate
        output_format = "array"
        prompt = construct_angle_prompt(
            topic,
            topic_info,
            principper,
            profile.tone_og_stil,
            fokusomrader,
            nyhedskriterier,
            nogo_omrader,
            additional_context=knowledge_distillate_text,
            output_format=output_format
        )
        
        # Update progress - check if progress_callback is callable
        if progress_callback and callable(progress_callback):
            await progress_callback(50)
        
        # 4. Generate angles with Claude API
        response_text = None
        session = None
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
                    "max_tokens": 4096,
                    "temperature": 0.7,
                    "system": "Du er en erfaren journalist med ekspertise i at udvikle kreative og relevante nyhedsvinkler.",
                    "messages": [{"role": "user", "content": prompt}],
                }
                session = await get_aiohttp_session(key="anthropic", timeout=ANTHROPIC_TIMEOUT)
                try:
                    async with session.post(ANTHROPIC_API_URL, json=payload, headers=headers) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            logger.error(f"Claude API error: {response.status}, {error_text}")
                            raise APIConnectionError(f"Claude API fejl: {response.status}")
                        response_data = await response.json()
                        logger.warning(f"Claude API response_data (angles): {json.dumps(response_data, ensure_ascii=False)[:2000]}")
                        # Log finish_reason hvis tilgængelig
                        finish_reason = None
                        if 'stop_reason' in response_data:
                            finish_reason = response_data['stop_reason']
                        elif 'content' in response_data and response_data['content'] and 'stop_reason' in response_data['content'][0]:
                            finish_reason = response_data['content'][0]['stop_reason']
                        elif 'content' in response_data and response_data['content'] and 'finish_reason' in response_data['content'][0]:
                            finish_reason = response_data['content'][0]['finish_reason']
                        if finish_reason:
                            log_level = logging.INFO if finish_reason == 'end_turn' else logging.WARNING
                            logger.log(log_level, f"Claude API finish_reason: {finish_reason}")
                            if finish_reason in ('max_tokens', 'length'):
                                logger.error("Claude API response was truncated due to max_tokens. Overvej at generere færre elementer pr. kald eller brug JSON Lines.")
                        response_text = response_data['content'][0]['text'] if 'content' in response_data and response_data['content'] and 'text' in response_data['content'][0] else None
                        if not response_text:
                            logger.error(f"Claude API response_data (no text found, angles): {json.dumps(response_data, ensure_ascii=False)[:1000]}")
                except Exception as api_exc:
                    import traceback
                    logger.error(f"Exception under Claude API-kald (angles): {type(api_exc).__name__}: {api_exc}\n{traceback.format_exc()}")
                    try:
                        error_text = await response.text()
                        logger.error(f"Claude API raw response.text: {error_text[:1000]}")
                    except Exception:
                        pass
                    raise

            # Log hele Claude-svaret før parsing
            logger.debug(f"Raw Claude response: {response_text}")

            # Update progress - check if progress_callback is callable
            if progress_callback and callable(progress_callback):
                await progress_callback(60)

            # Parse angles from response
            angles = parse_angles_from_response(response_text)
            # Hvis parsing fejler eller trunkering mistænkes, prøv JSON Lines fallback
            if not angles or (len(angles) == 1 and 'Fejl' in angles[0].get('overskrift', '')):
                jsonl_objs = []
                for l in (response_text or '').splitlines():
                    l = l.strip()
                    if l.startswith('{') and l.endswith('}'):
                        try:
                            obj = json.loads(l)
                            jsonl_objs.append(obj)
                        except Exception:
                            continue
                if jsonl_objs:
                    angles = jsonl_objs
            # Gem fejl-vinkler hvis parsing fejler
            error_angles = []
            if angles and len(angles) == 1 and (
                'Fejl' in angles[0].get('overskrift', '') or 'Ingen vinkler' in angles[0].get('overskrift', '')
            ):
                error_angles = angles.copy()

            if not angles or len(angles) == 0:
                logger.error(f"No angles parsed from response. Raw response text (first 200 chars): {response_text[:200]}")
                return [{
                    "overskrift": "Fejl under generering af vinkler",
                    "beskrivelse": "Kunne ikke generere vinkler fra AI-svaret. Se log for detaljer.",
                    "nyhedskriterier": ["aktualitet"],
                    "error": "No angles could be parsed from the response"
                }]

            # Add background info to each angle
            perplexity_extract = topic_info["text"][:1000] + ("..." if len(topic_info["text"]) > 1000 else "")
            for angle in angles:
                if isinstance(angle, dict):
                    angle['perplexityInfo'] = perplexity_extract
            
            # Update progress - check if progress_callback is callable
            if progress_callback and callable(progress_callback):
                await progress_callback(70)
            
            # Filter and rank angles
            from angle_processor import filter_and_rank_angles
            ranked_angles = filter_and_rank_angles(angles, profile, 5)

            # Hvis alle vinkler filtreres væk, returnér fejl-vinklerne hvis de findes
            if (not ranked_angles or len(ranked_angles) == 0):
                logger.error("No angles left after filtering")
                if error_angles:
                    return error_angles
                return [{
                    "overskrift": "Ingen vinkler efter filtrering",
                    "beskrivelse": "Filtreringen fjernede alle vinkler. Dette kan skyldes at de genererede vinkler ikke passede til profilen.",
                    "nyhedskriterier": ["aktualitet"],
                    "error": "All angles were filtered out"
                }]
            
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
                        
                        # Make sure the knowledge distillate is properly attached to each angle
                        for angle in ranked_angles:
                            if isinstance(angle, dict):
                                # Directly attach the knowledge distillate
                                angle['videnDistillat'] = knowledge_distillate
                                # Set the flag to indicate the angle has a knowledge distillate
                                angle['harVidenDistillat'] = True
                except Exception as e:
                    logger.error(f"Error generating knowledge distillate: {e}")
                    # Create a safe fallback distillate with error information
                    fallback_distillate = {
                        "error": f"Kunne ikke generere videndistillat: {str(e)}",
                        "hovedpunkter": [],
                        "noegletal": [],
                        "centralePaastand": [],
                        "vinkler": [],
                        "datoer": []
                    }
                    # Still add the fallback to prevent UI errors
                    for angle in ranked_angles:
                        if isinstance(angle, dict):
                            angle['videnDistillat'] = fallback_distillate
                            angle['harVidenDistillat'] = True
            
            # Update progress - check if progress_callback is callable
            if progress_callback and callable(progress_callback):
                await progress_callback(80)
            
            # Generate expert source suggestions for each angle if requested
            if include_expert_sources:
                expert_sources_tasks = []
                # Start expert source suggestions tasks for each angle (ingen begrænsning på antal)
                for i, angle in enumerate(ranked_angles):
                    if not isinstance(angle, dict):
                        continue
                    # Brug altid faktiske felter, ikke defaults
                    headline = angle.get('overskrift', f"Vinkel om {topic}")
                    description = angle.get('beskrivelse', '')
                    if description.strip().lower().startswith('ingen beskrivelse'):
                        description = ''
                    rationale = angle.get('begrundelse', '')
                    # Byg prompt med alle relevante felter
                    task = asyncio.create_task(
                        generate_expert_source_suggestions(
                            topic=topic,
                            angle_headline=headline,
                            angle_description=description,
                            bypass_cache=bypass_cache,
                            progress_callback=None,
                            rationale=rationale if rationale else None
                        )
                    )
                    expert_sources_tasks.append((i, task))
                # Update progress - check if progress_callback is callable
                if progress_callback and callable(progress_callback):
                    await progress_callback(90)
                # Wait for all expert source tasks to complete og map korrekt
                for i, task in expert_sources_tasks:
                    try:
                        expert_sources = await task
                        if expert_sources and i < len(ranked_angles):
                            if isinstance(expert_sources, dict) and "error" in expert_sources:
                                logger.warning(f"Expert sources for angle {i} contains error: {expert_sources.get('error')}")
                            ranked_angles[i]['ekspertKilder'] = expert_sources
                            ranked_angles[i]['harEkspertKilder'] = True
                    except Exception as e:
                        logger.error(f"Error generating expert sources for angle {i}: {str(e)}")
                        fallback_sources = {
                            "experts": [],
                            "institutions": [],
                            "data_sources": [],
                            "error": f"Kunne ikke generere ekspertkilder: {str(e)}"
                        }
                        if i < len(ranked_angles):
                            ranked_angles[i]['ekspertKilder'] = fallback_sources
                            ranked_angles[i]['harEkspertKilder'] = True
            
            # Add metadata to angles about which features were included
            for angle in ranked_angles:
                if isinstance(angle, dict):
                    angle['harVidenDistillat'] = include_knowledge_distillate and 'videnDistillat' in angle
                    angle['harEkspertKilder'] = include_expert_sources and 'ekspertKilder' in angle
            
            # Final progress update - check if progress_callback is callable
            if progress_callback and callable(progress_callback):
                await progress_callback(100)
                
            return ranked_angles
            
        except Exception as e:
            logger.error(f"Error generating angles: {str(e)}")
            # Return a minimal result with error information instead of raising
            return [{
                "overskrift": f"Fejl under vinkelgenerering: {str(e)}",
                "beskrivelse": "Der opstod en fejl under generering af vinkler. Se logs for detaljer.",
                "nyhedskriterier": ["aktualitet"],
                "error": str(e)
            }]
    except Exception as e:
        logger.error(f"Unexpected error in process_generation_request: {str(e)}")
        # Return a minimal result with error information
        return [{
            "overskrift": f"Uventet fejl: {str(e)}",
            "beskrivelse": "Der opstod en uventet fejl. Se logs for detaljer.",
            "nyhedskriterier": ["aktualitet"],
            "error": str(e)
        }]

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
    global _initialization_logged
    
    # Pre-create sessions for each API
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
    
    if not _initialization_logged:
        logger.info("API client initialized and ready")
        _initialization_logged = True

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
    This optimized version reduces token usage and improves performance.
    
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
    
    # Update progress if callback provided - check if it's actually callable
    if progress_callback and callable(progress_callback):
        await progress_callback(25)
    
    # Extract just the key information from the topic_info to reduce token usage
    # Truncate to maximum 3000 characters to reduce token consumption
    truncated_topic_info = topic_info
    if len(topic_info) > 3000:
        # Try to find section markers to make a cleaner truncation
        sections = ["## OVERSIGT", "## AKTUEL STATUS", "## NØGLETAL", "## PERSPEKTIVER"]
        found_sections = []
        
        for section in sections:
            if section in topic_info:
                found_sections.append((section, topic_info.find(section)))
        
        found_sections.sort(key=lambda x: x[1])  # Sort by position
        
        if found_sections:
            # Take just the first 3 sections we found
            content_to_include = []
            for i, (section, pos) in enumerate(found_sections[:3]):
                next_pos = found_sections[i+1][1] if i+1 < len(found_sections) else len(topic_info)
                section_content = topic_info[pos:next_pos]
                # Limit each section to 1000 chars max
                if len(section_content) > 1000:
                    section_content = section_content[:997] + "..."
                content_to_include.append(section_content)
            truncated_topic_info = "\n".join(content_to_include)
        else:
            # No sections found, do a simple truncation
            truncated_topic_info = topic_info[:2997] + "..."
    
    # Optimized system prompt - shorter but focused on the key task
    system_prompt = """Du er en analytisk AI der uddrag de vigtigste fakta fra tekst. 
Skab et kompakt, struktureret videndistillat med fokus på de mest relevante fakta for journalistisk brug.
Vær yderst præcis og faktuel, brug kun information der er direkte nævnt i teksten."""

    # Optimized user prompt - simplified and more focused
    user_prompt = f"""
    Analyser denne information om '{topic}' og lav et videndistillat med de vigtigste fakta.
    
    BAGGRUNDSINFORMATION:
    {truncated_topic_info}
    
    Formater dit svar som et JSON-objekt med følgende nøgler (brug PRÆCIST disse danske feltnavne):
    1. "hovedpunkter": En liste med 4 vigtige punkter fra materialet.
    2. "noegletal": En liste med 3 nøglestal, hvert med "tal", "beskrivelse" og "kilde" (hvis angivet).
    3. "centralePaastand": En liste med 3 vigtige påstande, hver med "paastand" og "kilde" (hvis angivet).
    
    Brug KUN information fra baggrundsmaterialet. Svar udelukkende med et rent JSON-objekt, uden forklaringer.
    """
    
    # Update progress if callback provided
    if progress_callback and callable(progress_callback):
        await progress_callback(40)
    
    # Get session from pool - use a smaller model and tighter timeout for better performance
    session = await get_aiohttp_session(key="anthropic_knowledge", timeout=45)
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01"
    }
    
    # Use the smaller, faster Haiku model instead of Opus for knowledge distillation
    payload = {
        "model": "claude-3-haiku-20240307",  # Smaller, faster model
        "max_tokens": 1000,  # Reduced token limit
        "temperature": 0.1,  # Lower temperature for more deterministic results
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    
    start_time = time.time()
    
    try:
        # Make API request with timeout
        async with session.post(ANTHROPIC_API_URL, json=payload, headers=headers) as response:
            # Update progress - check if progress_callback is callable
            if progress_callback and callable(progress_callback):
                await progress_callback(70)
                
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"Anthropic API error (knowledge distillate): status={response.status}, response={error_text}")
                return {
                    "hovedpunkter": [],
                    "noegletal": [],
                    "centralePaastand": [], 
                    "error": f"API Error: Status {response.status}"
                }
            
            try:    
                response_data = await response.json()
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in Anthropic API response: {e}")
                return {
                    "hovedpunkter": [],
                    "noegletal": [], 
                    "centralePaastand": [],
                    "error": "Response is not valid JSON"
                }
            
            # Update progress - check if progress_callback is callable
            if progress_callback and callable(progress_callback):
                await progress_callback(85)
                
            # Extract content with error handling
            try:
                content = ""
                if 'content' in response_data and len(response_data['content']) > 0:
                    content = response_data['content'][0].get('text', '')
                if not content:
                    logger.error("Empty text content in Anthropic API response")
                    return {
                        "hovedpunkter": [],
                        "noegletal": [], 
                        "centralePaastand": [],
                        "error": "Empty content received from API"
                    }
            except (KeyError, IndexError, TypeError) as e:
                logger.error(f"Failed to extract content from Anthropic API response: {e}")
                return {
                    "hovedpunkter": [],
                    "noegletal": [], 
                    "centralePaastand": [],
                    "error": f"Failed to extract content: {str(e)}"
                }
            
            # Add after line 1322, before parsing the content
            logger.debug(f"Raw knowledge distillate response: {content[:1000]}...")
            
            # Use the enhanced JSON parser for robust parsing
            from json_parser import safe_parse_json
            
            # Define default distillate structure - simpler with fewer fields
            fallback_distillate = {
                "hovedpunkter": [],
                "noegletal": [], 
                "centralePaastand": []
            }
            
            # Extra precaution: Check if content actually contains JSON
            if not ('{' in content and '}' in content):
                logger.error(f"Content does not appear to contain JSON: {content[:100]}...")
                return {
                    **fallback_distillate,
                    "error": "Response did not contain valid JSON formatting"
                }
                
            # Parse the content using the robust parser
            try:
                distillate = safe_parse_json(
                    content,
                    context="knowledge distillate",
                    fallback=fallback_distillate
                )
                
                if not isinstance(distillate, dict):
                    logger.error(f"Parsed content is not a dictionary: {type(distillate)}")
                    return fallback_distillate
            except Exception as e:
                logger.error(f"Error during JSON parsing: {e}")
                return {
                    **fallback_distillate,
                    "error": f"JSON parsing error: {str(e)}"
                }
            
            # Normalize field names if they use English variants
            field_mappings = {
                "main_points": "hovedpunkter",
                "key_statistics": "noegletal",
                "key_claims": "centralePaastand"
            }
            
            for eng_field, dk_field in field_mappings.items():
                if eng_field in distillate and dk_field not in distillate:
                    distillate[dk_field] = distillate.pop(eng_field)
            
            # Ensure all required fields exist with proper types
            for field in fallback_distillate:
                if field not in distillate:
                    distillate[field] = []
                elif not isinstance(distillate[field], list):
                    logger.warning(f"Field {field} is not a list, converting to empty list")
                    distillate[field] = []
            
            # Record latency
            latency = time.time() - start_time
            logger.info(f"Knowledge distillate generation completed in {latency:.2f} seconds")
            
            # Final progress update - check if progress_callback is callable
            if progress_callback and callable(progress_callback):
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
        
        # Return a valid fallback structure
        return {
            "hovedpunkter": [],
            "noegletal": [],
            "centralePaastand": [],
            "error": f"Exception during API call: {str(e)}"
        }

@cached_api(ttl=43200)  # Cache for 12 hours - expert sources don't change often
@retry_with_circuit_breaker(
    max_retries=3,
    initial_backoff=1.0,
    backoff_factor=2.0,
    exceptions=[aiohttp.ClientError, asyncio.TimeoutError, ConnectionError],
    circuit_name="anthropic_expert_sources"
)
@safe_execute_async(fallback_return=None)
async def generate_expert_source_suggestions(
    topic: str,
    angle_headline: str,
    angle_description: str,
    bypass_cache: bool = False,
    progress_callback: Optional[Callable] = None,
    rationale: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Generate detailed suggestions for expert sources and institutions for a specific news angle using Claude API.
    """
    # Definer system_prompt i starten af funktionen for at sikre, at den altid er tilgængelig
    system_prompt = """Du er en erfaren journalist med specialviden om ekspertkilder, organisationer og datakilder i Danmark. 
Din opgave er at identificere de mest relevante og pålidelige eksperter og institutioner til specifikke journalistiske vinkler.

Returner altid et velformateret JSON-objekt med følgende struktur:
{
  "eksperter": [
    {
      "navn": "Jens Hansen",
      "titel": "Professor i miljøret",
      "organisation": "Københavns Universitet, Juridisk Fakultet",
      "ekspertise": "Specialiseret i miljølovgivning og bæredygtighedsregulering",
      "kontakt": "jens.hansen@jur.ku.dk",
      "relevans": "Kan belyse de juridiske aspekter af kompostregulering i byområder"
    }
  ],
  "institutioner": [
    {
      "navn": "Miljøstyrelsen",
      "type": "Offentlig myndighed",
      "relevans": "Ansvarlig for regulering af affaldssortering og miljøgodkendelser",
      "kontaktperson": "Anne Nielsen, Presseafdeling",
      "kontakt": "presse@mst.dk"
    }
  ],
  "datakilder": [
    {
      "titel": "Affaldsstatistik 2023",
      "udgiver": "Danmarks Statistik",
      "beskrivelse": "Årlig rapport med data om affaldshåndtering fordelt på typer og kommuner",
      "link": "https://www.dst.dk/affaldsstatistik2023",
      "senest_opdateret": "Marts 2023"
    }
  ]
}

Fokuser på at finde reelle personer med relevant ekspertise, faktiske institutioner og pålidelige datakilder, der specifikt relaterer til den journalistiske vinkel. Sørg for diversitet i køn og institutionel tilknytning blandt eksperterne."""

    if not ANTHROPIC_API_KEY:
        raise APIKeyMissingError("Anthropic API nøgle mangler. Sørg for at have en ANTHROPIC_API_KEY i din .env fil.")

    if progress_callback:
        await progress_callback(20)

    rationale_str = f"\nBEGRUNDELSE: {rationale}" if rationale else ""
    user_prompt = (
        "Find konkrete ekspertkilder, institutioner og datakilder til følgende journalistiske vinkel:"
        f"\nEMNE: {topic}"
        f"\nOVERSKRIFT: {angle_headline}"
        f"\nBESKRIVELSE: {angle_description}"
        f"{rationale_str}"
        "\nFormater dit svar som et JSON-objekt med følgende felter:"
        "\n1. \"eksperter\": En liste med 4-6 konkrete ekspertkilder, hver med følgende struktur:"
        "\n   - \"navn\": Ekspertens fulde navn (SKAL være en reel person)"
        "\n   - \"titel\": Ekspertens titel og rolle"
        "\n   - \"organisation\": Arbejdsplads eller tilknytning"
        "\n   - \"ekspertise\": Kort beskrivelse af relevant ekspertise"
        "\n   - \"kontakt\": Kontaktinformation, hvis tilgængelig (email, telefon, eller henvisning til institution)"
        "\n   - \"relevans\": Kort forklaring på hvorfor denne ekspert er relevant for vinklen"
        "\n2. \"institutioner\": En liste med 3-5 relevante organisationer, hver med følgende struktur:"
        "\n   - \"navn\": Organisationens fulde navn"
        "\n   - \"type\": Type af organisation (universitet, myndighed, NGO, etc.)"
        "\n   - \"relevans\": Hvorfor denne organisation er relevant"
        "\n   - \"kontaktperson\": Navngiven kontaktperson/presseansvarlig hvis kendt, ellers \"Presseafdeling\""
        "\n   - \"kontakt\": Kontaktinformation til organisationen/presseafdelingen"
        "\n3. \"datakilder\": En liste med 2-4 specifikke datakilder, hver med følgende struktur:"
        "\n   - \"titel\": Titel på rapport, database eller datasæt"
        "\n   - \"udgiver\": Organisation der har udgivet eller vedligeholder datakilden"
        "\n   - \"beskrivelse\": Kort beskrivelse af hvilke data der findes her"
        "\n   - \"link\": Link til datakilden, hvis tilgængelig"
        "\n   - \"senest_opdateret\": Hvornår datakilden sidst blev opdateret (hvis kendt)"
        "\nVIGTIGE KRAV:"
        "\n- Alle eksperter SKAL være reelle personer med korrekte titler og tilknytninger"
        "\n- Undgå generiske beskrivelser som \"en ekspert i [emne]\" - navngiv konkrete personer"
        "\n- Organisationer skal være reelle, eksisterende institutioner"
        "\n- For kilder med tilknytning til universiteter, angiv specifikt institut/afdeling"
        "\n- Prioriter diversitet i både køn og institutionel tilknytning"
        "\n- Inkluder både akademiske eksperter, praktikere og relevante myndighedspersoner"
        "\n- Sørg for at alle forslag er direkte relevante for den konkrete vinkel, ikke bare det overordnede emne"
        "\n- Inkluder kun kontaktoplysninger du er sikker på er korrekte (ellers angiv \"Ikke offentligt tilgængelig\")"
        "\nSvar udelukkende med det rene JSON-objekt, uden forklarende tekst eller indledning."
    )

    if progress_callback:
        await progress_callback(40)

    headers = {
        "Content-Type": "application/json",
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01"
    }

    payload = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 4096,
        "temperature": 0.2,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt + "\n\nReturnér udelukkende gyldig JSON. Ingen forklaringer, ingen tekst udenfor JSON."}],
    }

    session = await get_aiohttp_session(key="anthropic_expert_sources", timeout=45)

    try:
        async with session.post(ANTHROPIC_API_URL, json=payload, headers=headers) as response:
            raw_bytes = await response.read()
            logger.debug(f"Raw Claude API response bytes: {raw_bytes[:500]}{'...' if len(raw_bytes) > 500 else ''}")
            try:
                raw_text = raw_bytes.decode('utf-8')
            except UnicodeDecodeError:
                raw_text = raw_bytes.decode('utf-8', errors='replace')
            logger.debug(f"Raw Claude API response text: {raw_text[:500]}{'...' if len(raw_text) > 500 else ''}")
            # Reset response for json parsing
            import io
            response._body = raw_bytes
            response.content._cursor = 0
            if response.status != 200:
                error_text = raw_text
                logger.error(f"Claude API error: status={response.status}, response={error_text}")
                return f"Fejl ved hentning af emne-information: Status {response.status}"
            try:
                response_data = await response.json()
            except Exception as e:
                logger.error(f"Invalid JSON in Claude API response: {e}. Preview: {raw_text[:500]}{'...' if len(raw_text) > 500 else ''}")
                return "Fejl: Modtog ikke gyldigt svar fra API'et"
            logger.warning(f"Claude API response_data (expert sources): {json.dumps(response_data, ensure_ascii=False)[:2000]}")
            content = response_data['content'][0]['text'] if 'content' in response_data and response_data['content'] and 'text' in response_data['content'][0] else None
            if not content:
                logger.error(f"Anthropic API response_data (no text found): {json.dumps(response_data, ensure_ascii=False)[:1000]}")
                return {
                    "eksperter": [],
                    "institutioner": [],
                    "datakilder": [],
                    "error": "No content returned from Claude API"
                }
    
            # Robust JSON parsing
            from json_parser import safe_parse_json
            expected_format = {
                "eksperter": [],
                "institutioner": [],
                "datakilder": []
            }
            expert_suggestions = safe_parse_json(
                content,
                context="expert source suggestions",
                fallback={
                    "error": "Failed to parse expert sources",
                    "eksperter": [],
                    "institutioner": [],
                    "datakilder": [],
                    "raw_response": content[:300] + ("..." if len(content) > 300 else "")
                }
            )
            # Ensure all required fields exist
            for field in expected_format:
                if field not in expert_suggestions:
                    expert_suggestions[field] = []
            if progress_callback:
                await progress_callback(100)
            return expert_suggestions
    except Exception as e:
        logger.error(f"Error generating expert source suggestions: {str(e)}")
        return {
            "eksperter": [],
            "institutioner": [],
            "datakilder": [],
            "error": f"Exception: {str(e)}"
        }

async def shutdown_api_client():
    """Properly shut down the API client, closing connections."""
    # Close all sessions
    await close_all_sessions()
    
    # Shut down thread pool
    if _connection_pool is not None:
        _connection_pool.shutdown(wait=True)
        
    logger.info("API client shut down successfully")

async def generate_angles(
    topic: str,
    profile: RedaktionelDNA,
    bypass_cache: bool = False,
    progress_callback = None,
    detailed: bool = False
) -> List[Dict[str, Any]]:
    """
    Generate angles for a news topic based on an editorial profile.
    
    Args:
        topic: News topic to generate angles for
        profile: Editorial DNA profile to use
        bypass_cache: If True, ignore cached results
        progress_callback: Optional callback function for progress updates
        detailed: If True, generate more detailed angles
        
    Returns:
        List[Dict]: Generated angles with additional information
    """
    logger.info("generate_angles called - redirecting to process_generation_request")
    
    try:
        # Ensure profile is a RedaktionelDNA object
        if not isinstance(profile, RedaktionelDNA):
            logger.error(f"Invalid profile type in generate_angles: {type(profile)}. Expected RedaktionelDNA object.")
            raise TypeError(f"Profile must be a RedaktionelDNA object, not {type(profile)}")
            
        # Call the new implementation
        return await process_generation_request(
            topic=topic,
            profile=profile,
            bypass_cache=bypass_cache,
            progress_callback=progress_callback if callable(progress_callback) else None,
            include_expert_sources=True,
            include_knowledge_distillate=True
        )
    except Exception as e:
        logger.error(f"Error in generate_angles: {str(e)}")
        # Return a simplified error result
        return [{
            "overskrift": f"Fejl under vinkelgenerering: {str(e)}",
            "beskrivelse": "Der opstod en fejl under generering af vinkler. Se logs for detaljer.",
            "nyhedskriterier": ["aktualitet"],
            "error": str(e)
        }]

async def fallback_angle_generator(
    topic: str,
    profile: RedaktionelDNA,
    bypass_cache: bool = False
) -> List[Dict[str, Any]]:
    """
    Genererer simplere vinkler når den primære vinkelgenerering fejler.
    Denne funktion bruger en enklere prompt og en mindre model for hurtigere svar og højere pålidelighed.
    
    Args:
        topic: Nyhedsemnet at generere vinkler for
        profile: Redaktionel DNA-profil (kun grundlæggende info bruges)
        bypass_cache: Hvis True, ignorer cachede resultater
        
    Returns:
        List[Dict]: Genererede vinkler med basal information
    """
    logger.info(f"Fallback vinkelgenerering aktiveret for emne: {topic}")
    
    if not ANTHROPIC_API_KEY and not OPENAI_API_KEY:
        logger.error("Ingen API nøgler tilgængelige for fallback vinkelgenerering")
        return [{
            "overskrift": "Fejl: Ingen API nøgler tilgængelige",
            "beskrivelse": "Der kunne ikke genereres vinkler, da API nøgler mangler.",
            "nyhedskriterier": ["aktualitet"],
            "begrundelse": "Teknisk fejl: Manglende API nøgler",
            "startSpørgsmål": ["Hvorfor er vinklen interessant?"]
        }]
        
    # Opret en enklere prompt med minimal kontekst
    # Undgå komplekse JSON-strukturer der kan fejle
    simple_prompt = f"""
    Generer 3 journalistiske vinkler på emnet: "{topic}".

    Lav korte, simple vinkler med følgende struktur i et gyldigt JSON-array:
    [
      {{
        "overskrift": "Kort og præcis overskrift",
        "beskrivelse": "2-3 sætninger der beskriver vinklen",
        "nyhedskriterier": ["relevante nyhedskriterier"],
        "begrundelse": "Kort begrundelse for vinklens relevans",
        "startSpørgsmål": ["Et startspørgsmål til vinklen"]
      }},
      ...flere vinkler...
    ]
    
    Følgende er særligt vigtigt for mediets profil:
    - Tone: {profile.tone_og_stil[:100]}
    - Fokus: {", ".join(profile.fokusOmrader[:3]) if profile.fokusOmrader else "Generelt"}
    
    Dit svar skal KUN være et gyldigt JSON-array med præcis den struktur jeg har angivet. Intet andet.
    """
    
    try:
        # Prøv først med Claude-3-Haiku (hurtigere og mere stabil)
        if ANTHROPIC_API_KEY:
            logger.info("Bruger Claude-3-Haiku til fallback vinkelgenerering")
            
            headers = {
                "Content-Type": "application/json",
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01"
            }
            
            payload = {
                "model": "claude-3-haiku-20240307",  # Mindre, hurtigere model
                "max_tokens": 1200,
                "temperature": 0.2,  # Lavere temperatur for mere forudsigelige resultater
                "system": "Du er en erfaren journalist, der er ekspert i at generere korte, klare vinkler på nyhedsemner. Du skal ALTID svare med et gyldigt JSON-array og intet andet.",
                "messages": [{"role": "user", "content": simple_prompt}],
            }
            
            # Få session fra pool - brug kortere timeout
            session = await get_aiohttp_session(key="anthropic_fallback", timeout=30)
            
            async with session.post(ANTHROPIC_API_URL, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Anthropic API fejl (fallback): status={response.status}, response={error_text}")
                    # Fortsæt til OpenAI forsøg hvis tilgængelig
                else:
                    response_data = await response.json()
                    response_text = response_data['content'][0]['text']
                    
                    # Brug den forbedrede safe_parse_json fra json_parser.py
                    from json_parser import enhanced_safe_parse_json
                    
                    # Definer forventet struktur for enklere fallback parsing
                    expected_structure = {
                        "overskrift": "Fejl under generering af vinkler",
                        "beskrivelse": "Kunne ikke generere detaljer",
                        "nyhedskriterier": ["aktualitet"],
                        "begrundelse": "Teknisk fejl",
                        "startSpørgsmål": ["Hvorfor er emnet relevant?"]
                    }
                    
                    # Få både resultat og debug info
                    parsed_result, debug_info = enhanced_safe_parse_json(
                        response_text, 
                        "fallback vinkelgenerering",
                        fallback=[{
                            "overskrift": f"Fallback vinkel for {topic}",
                            "beskrivelse": "Automatisk genereret simpel vinkel da detaljeret generering fejlede",
                            "nyhedskriterier": ["aktualitet"],
                            "begrundelse": "Teknisk fallback generering",
                            "startSpørgsmål": ["Hvorfor er emnet relevant?"]
                        }],
                        return_debug_info=True
                    )
                    
                    if debug_info["success"]:
                        logger.info(f"Fallback vinkelgenerering lykkedes med {debug_info['repair_technique'] or 'direkte parsing'}")
                        
                        # Normalisering for at sikre at vi har et array af vinkler
                        if isinstance(parsed_result, dict):
                            parsed_result = [parsed_result]
                        
                        # Validering og normalisering af vinkler
                        from json_parser import validate_angles
                        angles = validate_angles(parsed_result)
                        
                        if angles:
                            logger.info(f"Genererede {len(angles)} simple vinkler via fallback")
                            return angles
                        
        # Fallback til OpenAI hvis Claude fejlede eller ikke er tilgængelig
        if OPENAI_API_KEY:
            logger.info("Bruger OpenAI til fallback vinkelgenerering")
            
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            openai_payload = {
                "model": "gpt-3.5-turbo",  # Bruger 3.5-turbo for balance mellem hastighed/kvalitet
                "messages": [
                    {"role": "system", "content": "Du er en erfaren journalist, der er ekspert i at generere korte, klare vinkler på nyhedsemner. Du skal ALTID svare med et gyldigt JSON-array og intet andet."},
                    {"role": "user", "content": simple_prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 1000
            }
            
            # Få session fra pool - brug kortere timeout
            session = await get_aiohttp_session(key="openai_fallback", timeout=30)
            
            async with session.post(OPENAI_API_URL, json=openai_payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"OpenAI API fejl (fallback): status={response.status}, response={error_text}")
                else:
                    response_data = await response.json()
                    response_text = response_data['choices'][0]['message']['content']
                    
                    # Brug enhanced_safe_parse_json igen
                    from json_parser import enhanced_safe_parse_json
                    parsed_result, debug_info = enhanced_safe_parse_json(
                        response_text, 
                        "OpenAI fallback vinkelgenerering",
                        return_debug_info=True
                    )
                    
                    if debug_info["success"]:
                        logger.info(f"OpenAI fallback vinkelgenerering lykkedes med {debug_info['repair_technique'] or 'direkte parsing'}")
                        
                        # Normalisering for at sikre at vi har et array af vinkler
                        if isinstance(parsed_result, dict):
                            parsed_result = [parsed_result]
                        
                        # Validering og normalisering af vinkler
                        from json_parser import validate_angles
                        angles = validate_angles(parsed_result)
                        
                        if angles:
                            logger.info(f"Genererede {len(angles)} simple vinkler via OpenAI fallback")
                            return angles
                        
        # Hvis begge metoder fejlede, generer helt basale vinkler
        logger.warning("Alle fallback metoder fejlede, genererer basis-vinkler")
        return [
            {
                "overskrift": f"Udforskning af {topic}",
                "beskrivelse": f"En grundlæggende analyse af {topic} og dets betydning for Danmark.",
                "nyhedskriterier": ["aktualitet", "væsentlighed"],
                "begrundelse": "Emnet er aktuelt og relevant for en bred målgruppe",
                "startSpørgsmål": ["Hvordan påvirker dette emne danskerne?"]
            },
            {
                "overskrift": f"Eksperter vurderer konsekvenserne af {topic}",
                "beskrivelse": f"Fagfolk udtaler sig om de potentielle konsekvenser af {topic} på kort og lang sigt.",
                "nyhedskriterier": ["væsentlighed", "aktualitet"],
                "begrundelse": "Ekspertvurderinger giver dybde og troværdighed til dækningen",
                "startSpørgsmål": ["Hvilke konsekvenser ser eksperterne?"]
            },
            {
                "overskrift": f"5 ting du skal vide om {topic}",
                "beskrivelse": f"Et overblik over de vigtigste aspekter af {topic} forklaret på en letforståelig måde.",
                "nyhedskriterier": ["identifikation", "væsentlighed"],
                "begrundelse": "Formidling af komplekst emne i tilgængeligt format",
                "startSpørgsmål": ["Hvad er det vigtigste at forstå om emnet?"]
            }
        ]
    except Exception as e:
        logger.error(f"Uventet fejl i fallback_angle_generator: {str(e)}")
        # Returner en enkelt fejl-vinkel
        return [{
            "overskrift": f"Nyheder om {topic}",
            "beskrivelse": f"Denne vinkel blev automatisk genereret efter en teknisk fejl. Generer venligst nye vinkler.",
            "nyhedskriterier": ["aktualitet"],
            "begrundelse": f"Automatisk genereret efter fejl: {str(e)}",
            "startSpørgsmål": ["Hvad er de vigtigste aspekter af denne nyhed?"]
        }]

async def generate_optimized_angles(
    topic: str,
    profile: RedaktionelDNA,
    complexity: int = 3,
    bypass_cache: bool = False,
    progress_callback: Optional[Callable] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Genererer vinkler med optimeret performance, caching og fejlhåndtering.
    
    Denne funktion implementerer:
    1. Intelligent caching baseret på emne+profil-kombination
    2. Optimeret prompt engineering for hurtigere svar
    3. Avancerede timeout og retry-strategier
    4. Delresultat-lagring til recovery
    
    Args:
        topic: Nyhedsemnet
        profile: Den redaktionelle profil
        complexity: Kompleksiteten af forespørgslen (1-5)
        bypass_cache: Om cache skal ignoreres
        progress_callback: Valgfri callback-funktion til statusopdateringer
        
    Returns:
        Tuple[List, Dict]: Liste af genererede vinkler og metadata
    """
    import time
    import uuid
    import logging
    from datetime import datetime
    
    # Import vores optimerede moduler
    from enhanced_cache import (
        load_topic_profile_result,
        cache_topic_profile_result,
        PartialResult,
        intelligent_cache_key,
        determine_ttl_strategy,
        record_request
    )
    from enhanced_timeout import (
        TimeoutConfig, 
        RetryStrategy,
        ProgressTracker,
        with_timeout,
        with_retry,
        AdaptiveTimeout
    )
    from prompt_engineering import OptimizedPromptEngineering
    from error_handling import DetailedAngleError, generate_fallback_response
    
    # Opsætning
    logger = logging.getLogger("vinkeljernet.api_client")
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    # Adaptive timeouts baseret på tidligere kald
    adaptive_timeout = AdaptiveTimeout(
        base_timeout=TimeoutConfig.ANGLE_GENERATION,
        min_timeout=30.0,
        max_timeout=180.0
    )
    
    # Tracker til at følge fremskridt
    progress = ProgressTracker(total_steps=100, callback=progress_callback)
    await progress.update(5, "Starter vinkelgenerering")
    
    # Metadata der skal returneres med resultatet
    metadata = {
        "request_id": request_id,
        "topic": topic,
        "profile_id": profile.id if hasattr(profile, "id") else "unknown",
        "timestamp": datetime.now().isoformat(),
        "complexity": complexity
    }
    
    try:
        # 1. Check cache først
        if not bypass_cache:
            await progress.update(10, "Tjekker cache")
            cached_result = await load_topic_profile_result(
                "generate_angles", topic, profile.id if hasattr(profile, "id") else "unknown", 
                complexity=complexity
            )
            
            if cached_result:
                logger.info(f"Serving cached angles for topic '{topic}', profile {profile.id if hasattr(profile, 'id') else 'unknown'}")
                execution_time = time.time() - start_time
                
                # Registrer cache hit
                record_request(
                    topic, 
                    profile.id if hasattr(profile, "id") else "unknown", 
                    was_cached=True, 
                    execution_time=execution_time
                )
                
                # Opdater metadata
                metadata["source"] = "cache"
                metadata["execution_time"] = execution_time
                
                await progress.update(100, "Færdig (fra cache)")
                
                return cached_result, metadata
        
        # 2. Opret delresultat til recovery
        partial_result = PartialResult(request_id, total_steps=5)
        partial_result.add_result("metadata", metadata)
        
        # 3. Saml baggrundsinfo til prompten
        await progress.update(15, "Samler baggrundsinformation")
        
        # Konverter profil til dict format for OptimizedPromptEngineering
        profile_dict = profile.dict() if hasattr(profile, "dict") else {
            "kerneprincipper": profile.kerneprincipper,
            "tone_og_stil": profile.tone_og_stil,
            "fokusOmrader": profile.fokusOmrader,
            "nyhedsprioritering": profile.nyhedsprioritering,
            "noGoOmrader": profile.noGoOmrader
        }
        
        # Baggrundsinformation - enten fra topic_info eller bare emnet selv
        topic_info = topic
        
        # 4. Opret en optimeret prompt
        await progress.update(25, "Genererer optimeret prompt")
        try:
            prompt = OptimizedPromptEngineering.create_efficient_angle_prompt(
                topic=topic,
                topic_info=topic_info,
                profile=profile_dict
            )
            partial_result.add_result("prompt", prompt)
        except Exception as e:
            logger.warning(f"Fejl ved oprettelse af optimeret prompt: {str(e)}, falder tilbage til standard prompt")
            # Fallback til simplere prompt hvis optimeret prompt fejler
            from prompt_engineering import construct_angle_prompt
            
            kerneprincipper = "\n".join([f"- {p}" for p in profile.kerneprincipper[:5]]) if hasattr(profile, "kerneprincipper") and profile.kerneprincipper else "Ingen specificerede principper"
            tone_og_stil = profile.tone_og_stil if hasattr(profile, "tone_og_stil") else "Professionel tone"
            fokusområder = "\n".join([f"- {f}" for f in profile.fokusOmrader[:5]]) if hasattr(profile, "fokusOmrader") and profile.fokusOmrader else "Generelle nyhedsområder"
            nyhedskriterier = "\n".join([f"- {k}: {v}" for k, v in list(profile.nyhedsprioritering.items())[:5]]) if hasattr(profile, "nyhedsprioritering") and profile.nyhedsprioritering else "Aktualitet, væsentlighed, identifikation"
            no_go_områder = "\n".join([f"- {n}" for n in profile.noGoOmrader[:3]]) if hasattr(profile, "noGoOmrader") and profile.noGoOmrader else "Ingen specificerede no-go områder"
            
            prompt = construct_angle_prompt(
                topic=topic,
                topic_info=topic_info,
                principper=kerneprincipper,
                tone_og_stil=tone_og_stil,
                fokusområder=fokusområder,
                nyhedskriterier=nyhedskriterier,
                nogo_områder=no_go_områder
            )
            partial_result.add_result("prompt", prompt)
        
        # 5. Generer vinkler via API
        await progress.update(40, "Kontakter LLM API")
        
        # Beregn timeout baseret på kompleksitet
        timeout = adaptive_timeout.get_timeout()
        logger.info(f"Bruger timeout på {timeout} sekunder for kompleksitet {complexity}")
        
        try:
            # Brug Claude API med timeout og retry
            async def call_claude_api():
                """Wrapper omkring Claude API kald med timeout og retry"""
                nonlocal prompt
                
                if not ANTHROPIC_API_KEY:
                    raise DetailedAngleError(
                        "Claude API nøgle mangler", 
                        error_type="api_error", 
                        user_friendly_message="Claude API nøgle er ikke konfigureret"
                    )
                
                headers = {
                    "Content-Type": "application/json",
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01"
                }
                
                payload = {
                    "model": "claude-3-opus-20240229",  # Brug en passende model
                    "max_tokens": 4000,
                    "temperature": 0.4,  # Lavere temperatur for mere konsistent JSON
                    "system": "Du er en erfaren journalist, der er ekspert i at generere journalistiske vinkler på nyhedsemner. Du svarer altid med et velformateret JSON-array og aldrig med forklarende tekst. Først analyserer du emnet grundigt og sikrer at vinkler er relevante og har en god journalistisk kvalitet.",
                    "messages": [{"role": "user", "content": prompt}],
                }
                
                session = await get_aiohttp_session(key="claude_vinkel", timeout=timeout + 10)
                
                async with session.post(ANTHROPIC_API_URL, json=payload, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Claude API error: {response.status}, {error_text}")
                        raise APIConnectionError(f"Claude API fejl: {response.status}")
                    response_data = await response.json()
                    # Log altid hele response_data
                    logger.warning(f"Claude API response_data (angles): {json.dumps(response_data, ensure_ascii=False)[:2000]}")
                    # Log finish_reason hvis tilgængelig
                    finish_reason = None
                    if 'stop_reason' in response_data:
                        finish_reason = response_data['stop_reason']
                    elif 'content' in response_data and response_data['content'] and 'stop_reason' in response_data['content'][0]:
                        finish_reason = response_data['content'][0]['stop_reason']
                    elif 'content' in response_data and response_data['content'] and 'finish_reason' in response_data['content'][0]:
                        finish_reason = response_data['content'][0]['finish_reason']
                    if finish_reason:
                        log_level = logging.INFO if finish_reason == 'end_turn' else logging.WARNING
                        logger.log(log_level, f"Claude API finish_reason: {finish_reason}")
                        if finish_reason in ('max_tokens', 'length'):
                            logger.error("Claude API response was truncated due to max_tokens. Overvej at generere færre elementer pr. kald eller brug JSON Lines.")
                    response_text = response_data['content'][0]['text'] if 'content' in response_data and response_data['content'] and 'text' in response_data['content'][0] else None
                    if not response_text:
                        logger.error(f"Claude API response_data (no text found, angles): {json.dumps(response_data, ensure_ascii=False)[:1000]}")

            # Eksekver API kald med timeout og retry
            api_response = await with_timeout(
                with_retry(
                    call_claude_api,
                    max_retries=2,
                    base_delay=2.0,
                    backoff_strategy=RetryStrategy.EXPONENTIAL_BACKOFF
                ),
                timeout=timeout,
                fallback_result=None
            )
            
            if not api_response:
                raise DetailedAngleError(
                    "API kald timeout",
                    error_type="timeout",
                    user_friendly_message="Vinkelgenerering tog for lang tid",
                    suggestions=["Prøv med en mindre kompleks forespørgsel", "Prøv igen senere"]
                )
            
            # Gem det rå API svar
            partial_result.add_result("raw_response", api_response)
            
            # 6. Parse JSON-svar
            await progress.update(70, "Parser API svar")
            
            from json_parser import enhanced_safe_parse_json
            from prompt_engineering import parse_angles_from_response
            
            angles_data, debug_info = enhanced_safe_parse_json(
                api_response, 
                context="angle generation",
                return_debug_info=True
            )
            # Hvis partial extraction bruges, log tabt data
            if debug_info.get("partial_extraction"):
                lost_segments = debug_info.get("unparsed_segments", [])
                for seg in lost_segments:
                    logger.error(f"Partial JSON extraction: Tabt segment: '{seg[:100]}{'...' if len(seg) > 100 else ''}'")
            
            if not debug_info["success"]:
                # Hvis parsing fejlede, prøv fallback parsing
                logger.warning("Primær JSON parsing fejlede, prøver fallback parsing")
                angles = parse_angles_from_response(api_response)
                
                if not angles or len(angles) < 3:
                    # Hvis begge parsing-metoder fejler, brug vores fallback generator
                    logger.warning("Både primær og fallback parsing fejlede, bruger fallback vinkelgenerering")
                    angles = await fallback_angle_generator(topic, profile, bypass_cache=True)
                    metadata["fallback_used"] = True
            else:
                # Brug de parsede vinkler
                if isinstance(angles_data, list):
                    angles = angles_data
                elif isinstance(angles_data, dict) and "vinkler" in angles_data:
                    angles = angles_data["vinkler"]
                else:
                    angles = [angles_data]  # Enkelt vinkel
                    
                # Valider antal vinkler
                if not angles or len(angles) < 3:
                    logger.warning(f"Kun {len(angles) if angles else 0} vinkler genereret, bruger fallback")
                    fallback_angles = await fallback_angle_generator(topic, profile, bypass_cache=True)
                    angles.extend(fallback_angles)
                    metadata["partial_fallback_used"] = True
                    
            # Gem de endelige vinkler
            partial_result.add_result("angles", angles)
            
            # Opdatere adaptive timeout med succesfuldt kald
            call_duration = time.time() - start_time
            adaptive_timeout.record_call(call_duration, True)
            
            # 7. Cache resultatet
            await progress.update(90, "Gemmer resultater")
            execution_time = time.time() - start_time
            metadata["execution_time"] = execution_time
            
            if len(angles) > 0:
                await cache_topic_profile_result(
                    "generate_angles", 
                    topic, 
                    profile.id if hasattr(profile, "id") else "unknown", 
                    angles,
                    complexity=complexity,
                    execution_time=execution_time
                )
                
                # Registrer succesfuldt kald
                record_request(
                    topic, 
                    profile.id if hasattr(profile, "id") else "unknown", 
                    was_cached=False, 
                    execution_time=execution_time
                )
            
            # Mark som færdig
            partial_result.mark_complete()
            await progress.update(100, "Færdig")
            
            metadata["source"] = "api"
            metadata["angle_count"] = len(angles)
            
            return angles, metadata
            
        except Exception as e:
            # Gem fejlen
            error_message = str(e)
            logger.error(f"Fejl under vinkelgenerering: {error_message}")
            partial_result.add_error("api_call", error_message)
            
            # Opdater adaptive timeout med fejlet kald
            call_duration = time.time() - start_time
            adaptive_timeout.record_call(call_duration, False)
            
            # Tjek om vi allerede har delvist resultat med vinkler
            if "angles" in partial_result.results and partial_result.results["angles"]:
                logger.info("Returnerer delvist resultat med vinkler fra før fejl")
                angles = partial_result.results["angles"]
                metadata["source"] = "partial_result"
                metadata["partial_error"] = error_message
            else:
                # Fallback til enklere vinkler
                logger.info("Bruger fallback vinkelgenerering")
                angles = await fallback_angle_generator(topic, profile, bypass_cache=True)
                metadata["source"] = "fallback"
                metadata["error"] = error_message
            
            # Registrer fejlet kald
            execution_time = time.time() - start_time
            metadata["execution_time"] = execution_time
            record_request(
                topic, 
                profile.id if hasattr(profile, "id") else "unknown", 
                was_cached=False, 
                execution_time=execution_time,
                status="error"
            )
            
            await progress.update(100, "Færdig (med fejl)")
            return angles, metadata
            
    except Exception as e:
        # Håndter uventede fejl
        error_message = str(e)
        logger.error(f"Uventet fejl i generate_optimized_angles: {error_message}")
        
        execution_time = time.time() - start_time
        metadata["execution_time"] = execution_time
        metadata["fatal_error"] = error_message
        
        # Sidste udvej: Returner helt basale vinkler
        angles = [
            {
                "overskrift": f"Nyheder om {topic}",
                "beskrivelse": f"En generel dækning af {topic} med fokus på de vigtigste aspekter.",
                "begrundelse": "Genereret efter fejl i vinkelgenererings-systemet",
                "nyhedskriterier": ["aktualitet"],
                "startSpørgsmål": ["Hvad er de vigtigste aspekter af dette emne?"]
            },
            {
                "overskrift": f"Baggrund og analyse: {topic}",
                "beskrivelse": f"En dybdegående analyse af {topic} og dets konsekvenser.",
                "begrundelse": "Genereret efter fejl i vinkelgenererings-systemet",
                "nyhedskriterier": ["væsentlighed"],
                "startSpørgsmål": ["Hvilke konsekvenser har dette emne?"]
            },
            {
                "overskrift": f"Hvordan påvirker {topic} almindelige mennesker?",
                "beskrivelse": f"En undersøgelse af {topic} fra et borger-perspektiv.",
                "begrundelse": "Genereret efter fejl i vinkelgenererings-systemet",
                "nyhedskriterier": ["identifikation"],
                "startSpørgsmål": ["Hvordan påvirker dette emne almindelige borgere?"]
            }
        ]
        
        await progress.update(100, "Færdig (med fatal fejl)")
        return angles, metadata