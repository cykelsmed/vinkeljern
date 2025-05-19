"""
Wrapper module for API clients that abstracts the actual implementation.
This allows switching between different implementations (optimized, mock, etc.)
while maintaining the same interface.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List, Callable
from models import RedaktionelDNA  # Add missing import

# Client state management
_main_event_loop = None
_initialization_logged = False  # Track if initialization has been logged

# Import the actual implementation
try:
    from api_clients_optimized import (
        fetch_topic_information,
        process_generation_request,
        generate_angles,
        generate_knowledge_distillate,
        generate_expert_source_suggestions,
        generate_editorial_considerations,
        initialize_api_client,
        shutdown_api_client,
        get_performance_metrics,
        optimize_cache
    )
except ImportError:
    from api_clients import (
        fetch_topic_information,
        process_generation_request,
        generate_angles,
        generate_knowledge_distillate,
        generate_expert_source_suggestions,
        generate_editorial_considerations,
        initialize_api_client,
        shutdown_api_client,
        get_performance_metrics,
        optimize_cache
    )

# Re-export the implementation
__all__ = [
    'fetch_topic_information',
    'process_generation_request',
    'generate_angles',
    'generate_knowledge_distillate',
    'generate_expert_source_suggestions',
    'generate_editorial_considerations',
    'initialize_api_client',
    'shutdown_api_client',
    'get_performance_metrics',
    'optimize_cache',
    'ensure_event_loop'
]

# Logger setup
logger = logging.getLogger("vinkeljernet.api_wrapper")

# Always use the optimized client for better performance
USE_OPTIMIZED_CLIENT = True

def get_implementation():
    """
    Get the appropriate implementation based on configuration.
    
    Returns:
        Module: Either the optimized or original API client module
    """
    global _initialization_logged
    
    if USE_OPTIMIZED_CLIENT:
        try:
            import api_clients_optimized
            if not _initialization_logged:
                logger.info("Using optimized API client implementation")
                _initialization_logged = True
            return api_clients_optimized
        except ImportError:
            import api_clients
            if not _initialization_logged:
                logger.warning("Optimized API client not available, using original implementation")
                _initialization_logged = True
            return api_clients
    else:
        import api_clients
        if not _initialization_logged:
            logger.info("Using original API client implementation")
            _initialization_logged = True
        return api_clients

def ensure_event_loop():
    """Ensures that there is a running event loop, creating one if necessary."""
    global _main_event_loop
    
    try:
        # Try to get the current running loop
        loop = asyncio.get_running_loop()
        if _main_event_loop is None:
            _main_event_loop = loop
        return loop
    except RuntimeError:
        # No running loop, create a new one
        if _main_event_loop is None or _main_event_loop.is_closed():
            _main_event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(_main_event_loop)
            logger.info("Created new event loop as no running loop was found")
        return _main_event_loop

# ====== Main API Interface Functions ======

async def fetch_topic_information(
    topic: str, 
    dev_mode: bool = False, 
    bypass_cache: bool = False,
    progress_callback = None,
    detailed: bool = False
) -> Optional[str]:
    """
    Fetch information about a topic using the appropriate API client.
    
    Args:
        topic: The news topic to fetch information about
        dev_mode: If True, disables SSL verification (development only!)
        bypass_cache: If True, ignore cached results
        progress_callback: Optional callback function to report progress (0-100)
        detailed: If True, get more comprehensive information
        
    Returns:
        Optional[str]: The information retrieved or None if failed
    """
    # Ensure we have a valid event loop
    ensure_event_loop()
    
    implementation = get_implementation()
    
    # Check if the implementation has the updated signature with 'detailed'
    if detailed and hasattr(implementation, 'fetch_topic_information') and 'detailed' in implementation.fetch_topic_information.__code__.co_varnames:
        return await implementation.fetch_topic_information(
            topic,
            dev_mode=dev_mode,
            bypass_cache=bypass_cache,
            progress_callback=progress_callback,
            detailed=detailed
        )
    else:
        # Fall back to standard call without detailed parameter
        return await implementation.fetch_topic_information(
            topic,
            dev_mode=dev_mode,
            bypass_cache=bypass_cache,
            progress_callback=progress_callback
        )

async def fetch_source_suggestions(
    topic: str,
    bypass_cache: bool = False
) -> Optional[str]:
    """
    Fetch source suggestions for a topic.
    
    Args:
        topic: The news topic to find sources for
        bypass_cache: If True, ignore cached results
        
    Returns:
        Optional[str]: Source suggestions or None if failed
    """
    ensure_event_loop()
    implementation = get_implementation()
    
    # Check if the implementation has this function
    if hasattr(implementation, 'fetch_source_suggestions'):
        return await implementation.fetch_source_suggestions(topic, bypass_cache=bypass_cache)
    else:
        # Fall back to None if not implemented
        logger.warning("fetch_source_suggestions not available in current implementation")
        return None

async def generate_angles(
    emne: str, 
    topic_info: str, 
    profile: RedaktionelDNA, 
    bypass_cache: bool = False
) -> List[Dict[str, Any]]:
    """
    Generate news angles for the given topic and profile.
    
    Args:
        emne: The news topic.
        topic_info: Background information on the topic.
        profile: The editorial DNA profile.
        bypass_cache: if True, bypass local cache.
        
    Returns:
        A list of angle dictionaries.
    """
    ensure_event_loop()
    implementation = get_implementation()
    
    if hasattr(implementation, 'generate_angles'):
        logger.debug(f"Profile type before calling implementation: {type(profile)}")
        if not isinstance(profile, RedaktionelDNA):
            logger.error(f"Invalid profile type: {type(profile)}. Expected RedaktionelDNA object.")
            raise TypeError(f"Profile must be a RedaktionelDNA object, not {type(profile)}")
            
        # Check the function signature to determine how to call it
        if implementation.generate_angles.__code__.co_argcount >= 5:  # New signature (topic, profile, bypass_cache, progress_callback, detailed)
            # For newer implementation with different signature
            return await implementation.generate_angles(
                topic=emne,
                profile=profile,
                bypass_cache=bypass_cache
            )
        else:
            # For older implementation
            return await implementation.generate_angles(emne, topic_info, profile, bypass_cache)
    else:
        raise NotImplementedError("generate_angles function not found in API client implementation")

async def generate_editorial_considerations(
    topic: str, 
    profile_name: str, 
    angles: List[Dict],
    bypass_cache: bool = False
) -> str:
    """
    Generate editorial considerations for the given angles.
    
    Args:
        topic: The news topic
        profile_name: Name of the editorial profile used
        angles: List of generated angles
        bypass_cache: If True, ignore cached results
        
    Returns:
        str: Editorial considerations text
    """
    ensure_event_loop()
    implementation = get_implementation()
    
    # Check if the implementation has this function
    if hasattr(implementation, 'generate_editorial_considerations'):
        return await implementation.generate_editorial_considerations(
            topic, 
            profile_name, 
            angles, 
            bypass_cache=bypass_cache
        )
    else:
        # Return a basic message if not implemented
        return "Redaktionelle overvejelser er ikke tilgængelige i denne version."

async def process_generation_request(
    topic: str,
    profile: RedaktionelDNA,
    bypass_cache: bool = False,
    progress_callback = None,
    include_expert_sources: bool = True,
    include_knowledge_distillate: bool = True
) -> List[Dict[str, Any]]:
    """
    Process an angle generation request with optional optimizations.
    
    Args:
        topic: News topic to generate angles for
        profile: Editorial DNA profile to use
        bypass_cache: If True, ignore cached results
        progress_callback: Optional callback function for progress updates
        include_expert_sources: If True, generate expert source suggestions for each angle
        include_knowledge_distillate: If True, generate a knowledge distillate from background info
        
    Returns:
        List[Dict]: Generated angles
    """
    ensure_event_loop()
    implementation = get_implementation()
    
    # Check if the implementation has the optimized version with new parameters
    if hasattr(implementation, 'process_generation_request'):
        # Check if the implementation supports the new parameters
        impl_function = implementation.process_generation_request
        param_names = impl_function.__code__.co_varnames[:impl_function.__code__.co_argcount]
        
        if 'include_expert_sources' in param_names and 'include_knowledge_distillate' in param_names:
            return await implementation.process_generation_request(
                topic, 
                profile, 
                bypass_cache=bypass_cache, 
                progress_callback=progress_callback,
                include_expert_sources=include_expert_sources,
                include_knowledge_distillate=include_knowledge_distillate
            )
        else:
            # Fall back to the standard call if the implementation doesn't support the new parameters
            return await implementation.process_generation_request(
                topic, 
                profile, 
                bypass_cache=bypass_cache, 
                progress_callback=progress_callback
            )
    else:
        # Fall back to standard implementation combining individual steps
        if progress_callback:
            await progress_callback(10)
            
        # Get topic info
        topic_info = await fetch_topic_information(
            topic, 
            bypass_cache=bypass_cache, 
            progress_callback=progress_callback
        )
        
        if progress_callback:
            await progress_callback(40)
            
        if not topic_info:
            topic_info = f"Emnet handler om {topic}. Ingen yderligere baggrundsinformation tilgængelig."
        
        # Generate angles
        angles = await generate_angles(topic, topic_info, profile, bypass_cache)
        
        if progress_callback:
            await progress_callback(80)
            
        # Get source suggestions if available
        try:
            source_text = await fetch_source_suggestions(topic, bypass_cache=bypass_cache)
            if source_text:
                for angle in angles:
                    if isinstance(angle, dict):
                        angle['kildeForslagInfo'] = source_text
        except:
            pass
            
        if progress_callback:
            await progress_callback(90)
            
        # Filter and rank angles
        from angle_processor import filter_and_rank_angles
        ranked_angles = filter_and_rank_angles(angles, profile, 5)
        
        if progress_callback:
            await progress_callback(100)
            
        return ranked_angles

def get_performance_metrics() -> Dict[str, Any]:
    """
    Get combined performance metrics for the API client.
    
    Returns:
        Dict containing API and cache performance metrics
    """
    implementation = get_implementation()
    
    if hasattr(implementation, 'get_performance_metrics'):
        return implementation.get_performance_metrics()
    else:
        # Basic metrics if not available
        from cache_manager import get_cache_stats
        from retry_manager import get_circuit_stats
        
        return {
            "api": {
                "note": "Advanced API metrics not available in current implementation"
            },
            "cache": get_cache_stats(),
            "circuits": get_circuit_stats()
        }

async def initialize_api_client():
    """Initialize the API client for optimal performance."""
    # Ensure we have a valid event loop
    ensure_event_loop()
    
    implementation = get_implementation()
    
    if hasattr(implementation, 'initialize_api_client'):
        await implementation.initialize_api_client()
    else:
        if not _initialization_logged:
            logger.info("API client initialization not available in current implementation")
            _initialization_logged = True

async def shutdown_api_client():
    """Properly shut down the API client, closing connections."""
    implementation = get_implementation()
    
    if hasattr(implementation, 'shutdown_api_client'):
        await implementation.shutdown_api_client()
    else:
        logger.info("API client shutdown not available in current implementation")