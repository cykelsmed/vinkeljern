"""
Resilient API module for Vinkeljernet.

Integrates the fault-tolerant architecture with the existing Vinkeljernet components.
Provides resilient wrappers for core API functionalities.
"""

import asyncio
import logging
import time
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Callable, Optional, Tuple, Union, Awaitable

from fault_tolerance import (
    FaultTolerantService,
    FaultTolerantAngleGenerator,
    VinkeljernetAppStatus,
    get_app_status,
    with_circuit_breaker,
    CircuitBreaker,
    CircuitOpenError,
    ServiceDegradedException
)

# Configure logging
logger = logging.getLogger("vinkeljernet.resilient_api")

# Get the global app status instance
app_status = get_app_status()

# Create fault-tolerant services for different API providers
openai_service = FaultTolerantService(
    name="openai",
    cache_ttl=3600 * 24,  # 24 hours
    failure_threshold=3,
    recovery_timeout=60
)

anthropic_service = FaultTolerantService(
    name="anthropic",
    cache_ttl=3600 * 24,
    failure_threshold=3,
    recovery_timeout=60
)

perplexity_service = FaultTolerantService(
    name="perplexity",
    cache_ttl=3600 * 24,
    failure_threshold=3,
    recovery_timeout=60
)

# Create a fault-tolerant angle generator
angle_generator = FaultTolerantAngleGenerator(
    service_name="angle_generator",
    cache_ttl=3600 * 24,
    min_acceptable_angles=3,
    use_generic_fallbacks=True
)

# Register components with app status
app_status.register_component("openai", openai_service)
app_status.register_component("anthropic", anthropic_service)
app_status.register_component("perplexity", perplexity_service)
app_status.register_component("angle_generator", angle_generator)

async def resilient_generate_angles(
    topic: str,
    profile: Any,
    generate_angles_func: Callable,
    bypass_cache: bool = False,
    topic_info: str = None,
    **kwargs
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Resilient wrapper for angle generation that ensures fault tolerance.
    
    Args:
        topic: Topic to generate angles for
        profile: Editorial profile
        generate_angles_func: Function to generate angles
        bypass_cache: Whether to bypass cache
        topic_info: Optional topic information
        **kwargs: Additional arguments for the generate function
        
    Returns:
        Tuple of (angles, metadata)
    """
    try:
        # Use the fault-tolerant angle generator
        return await angle_generator.generate_angles(
            topic=topic,
            profile=profile,
            generate_func=generate_angles_func,
            bypass_cache=bypass_cache,
            topic_info=topic_info,
            **kwargs
        )
    except Exception as e:
        logger.error(f"Error in resilient_generate_angles: {str(e)}")
        # Create generic fallback angles as absolute last resort
        fallback_angles = angle_generator._create_simple_fallback_angles(topic)
        return fallback_angles, {
            "error": str(e),
            "source": "emergency_fallback",
            "degraded": True,
            "fallback_used": True
        }

async def resilient_get_expert_sources(
    topic: str,
    angle_headline: str,
    angle_description: str,
    source_func: Callable,
    bypass_cache: bool = False
) -> Dict[str, Any]:
    """
    Resilient wrapper for expert sources suggestions.
    
    Args:
        topic: Main topic
        angle_headline: Headline of the angle
        angle_description: Description of the angle
        source_func: Function to get expert sources
        bypass_cache: Whether to bypass cache
        
    Returns:
        Dictionary with expert sources
    """
    # Create a cache key
    hash_key = hashlib.md5(f"{topic}_{angle_headline}".encode('utf-8')).hexdigest()
    cache_key = f"sources_{hash_key}"
    
    try:
        # Use FaultTolerantService directly
        sources_service = FaultTolerantService(name="sources")
        
        sources, metadata = await sources_service.call(
            func=source_func,
            cache_key=cache_key,
            use_cache=not bypass_cache,
            fallback={
                "experts": [],
                "institutions": [],
                "data_sources": [],
                "_fallback": True
            },
            args=[],
            kwargs={
                "topic": topic,
                "angle_headline": angle_headline,
                "angle_description": angle_description,
                "bypass_cache": bypass_cache
            }
        )
        
        if metadata.get("fallback_used") and sources.get("_fallback"):
            # Add generic placeholders for fallback sources
            if not sources.get("experts"):
                sources["experts"] = [
                    {"name": "Relevant fagekspert", "title": "Forsker inden for området", "organization": "Universitet eller forskningsinstitution"}
                ]
                
            if not sources.get("institutions"):
                sources["institutions"] = [
                    {"name": "Relevant brancheorganisation", "type": "Interesseorganisation"}
                ]
        
        return sources
        
    except Exception as e:
        logger.error(f"Error in resilient_get_expert_sources: {str(e)}")
        # Return minimal fallback structure
        return {
            "experts": [],
            "institutions": [],
            "data_sources": [],
            "error": str(e),
            "_fallback": True
        }

@with_circuit_breaker("topic_info")
async def resilient_get_topic_info(
    topic: str,
    get_info_func: Callable,
    bypass_cache: bool = False
) -> str:
    """
    Resilient wrapper for topic information retrieval.
    
    Args:
        topic: Topic to get information for
        get_info_func: Function to get topic information
        bypass_cache: Whether to bypass cache
        
    Returns:
        Topic information as string
    """
    # Prepare cache key
    cache_key = f"topic_info_{topic.lower().replace(' ', '_')}"
    
    try:
        # Use FaultTolerantService
        info_service = FaultTolerantService(name="topic_info")
        
        info, metadata = await info_service.call(
            func=get_info_func,
            cache_key=cache_key,
            use_cache=not bypass_cache,
            fallback=f"Emnet handler om {topic}.",
            args=[topic],
            kwargs={"bypass_cache": bypass_cache}
        )
        
        return info or f"Emnet handler om {topic}."
        
    except Exception as e:
        logger.error(f"Error in resilient_get_topic_info: {str(e)}")
        return f"Emnet handler om {topic}."

class ResilientAPIClient:
    """
    A resilient API client wrapper that adds fault tolerance to various API clients
    in the Vinkeljernet system.
    """
    
    def __init__(self, provider_name: str, api_client: Any):
        """
        Initialize a resilient API client wrapper.
        
        Args:
            provider_name: Name of the API provider (e.g., 'openai', 'anthropic')
            api_client: The original API client to wrap
        """
        self.provider_name = provider_name
        self.original_client = api_client
        self.service = get_service(provider_name)
        
    async def call_with_fallback(
        self, 
        method_name: str, 
        fallback_value: Any = None,
        cache_key: str = None,
        use_cache: bool = True,
        *args, 
        **kwargs
    ) -> Any:
        """
        Call an API method with fault tolerance.
        
        Args:
            method_name: Name of the method to call on the original client
            fallback_value: Value to return if the call fails
            cache_key: Optional cache key for caching responses
            use_cache: Whether to use cache
            *args, **kwargs: Arguments to pass to the method
            
        Returns:
            The result from the API call or fallback value
        """
        if not hasattr(self.original_client, method_name):
            raise AttributeError(f"Method {method_name} not found in {self.provider_name} client")
        
        method = getattr(self.original_client, method_name)
        
        result, metadata = await self.service.call(
            func=method,
            cache_key=cache_key,
            use_cache=use_cache,
            fallback=fallback_value,
            args=args,
            kwargs=kwargs
        )
        
        return result

def get_resilient_client(api_client: Any, provider_name: str) -> ResilientAPIClient:
    """
    Get a resilient wrapper for an API client.
    
    Args:
        api_client: Original API client
        provider_name: Name of the API provider
        
    Returns:
        A resilient API client wrapper
    """
    return ResilientAPIClient(provider_name, api_client)

# Get a service by name
def get_service(name: str) -> FaultTolerantService:
    """Get a fault-tolerant service by name"""
    from fault_tolerance import get_service as ft_get_service
    return ft_get_service(name)

# Health check function for API endpoints
def api_health_check(service_name: str, check_func: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Perform a health check on an API service.
    
    Args:
        service_name: Name of the service
        check_func: Optional function to check the service
        
    Returns:
        Health check result
    """
    result = {
        "service": service_name,
        "healthy": True,
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        # Check if service has a circuit breaker and it's open
        circuit = CircuitBreaker.get(service_name)
        
        if circuit and circuit.is_open:
            result["healthy"] = False
            result["severity"] = "warning"
            result["message"] = f"Circuit breaker for {service_name} is open"
            return result
            
        # If a check function is provided, run it
        if check_func and callable(check_func):
            check_result = check_func()
            if not check_result.get("success", False):
                result["healthy"] = False
                result["severity"] = "error"
                result["message"] = check_result.get("message", "API check failed")
                
    except Exception as e:
        result["healthy"] = False
        result["severity"] = "error"
        result["message"] = f"Health check error: {str(e)}"
    
    return result

# Register health checks
app_status.register_health_check(lambda: api_health_check("openai"))
app_status.register_health_check(lambda: api_health_check("anthropic"))
app_status.register_health_check(lambda: api_health_check("perplexity"))

async def monitor_system_health(check_interval: int = 60):
    """
    Continuously monitor system health and take actions if needed.
    
    Args:
        check_interval: Interval between checks in seconds
    """
    while True:
        status = app_status.run_health_check()
        
        # Log if system is degraded
        if app_status.is_degraded():
            logger.warning(f"System is in degraded mode. Issues: {len(status['issues'])}")
            
            # Enable degraded mode for all caches if system is degraded
            for service_name in app_status.get_degraded_services():
                service = get_service(service_name)
                if hasattr(service, 'cache'):
                    service.cache.set_degraded_mode(True)
                    logger.info(f"Enabled degraded mode for {service_name} cache")
        
        # Check if any services have recovered
        elif status["overall_health"] == "healthy":
            # Reset degraded mode for all caches if system is healthy
            for service_name, service_info in status["services"].items():
                service = get_service(service_name)
                if hasattr(service, 'cache') and service.cache.degraded_mode:
                    service.cache.set_degraded_mode(False)
                    logger.info(f"Disabled degraded mode for {service_name} cache")
        
        await asyncio.sleep(check_interval)

# Utility function to start the monitoring in the background
def start_health_monitoring():
    """Start the health monitoring in the background"""
    asyncio.create_task(monitor_system_health())
    logger.info("Started system health monitoring")