"""
Retry management module for Vinkeljernet project.

This module provides retry functionality with exponential backoff and circuit breaker
patterns to handle transient failures in API calls.
"""

import time
import random
import logging
import functools
import asyncio
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import traceback
import socket
import aiohttp
import http.client
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vinkeljernet.retry")

# Type variables for function annotations
T = TypeVar('T')
AsyncFunc = Callable[..., Any]
SyncFunc = Callable[..., Any]

class CircuitState(Enum):
    """State enum for the circuit breaker."""
    CLOSED = "closed"      # Circuit is closed, requests are allowed
    OPEN = "open"          # Circuit is open, requests are blocked
    HALF_OPEN = "half-open"  # Circuit is testing if service is back


@dataclass
class CircuitBreakerStats:
    """Statistics for a circuit breaker."""
    success_count: int = 0
    failure_count: int = 0
    consecutive_failures: int = 0
    last_failure_time: Optional[datetime] = None
    total_retry_count: int = 0

# Common HTTP status codes that indicate transient failures
RETRYABLE_HTTP_STATUS_CODES = {
    408,  # Request Timeout
    429,  # Too Many Requests
    500,  # Internal Server Error
    502,  # Bad Gateway
    503,  # Service Unavailable
    504,  # Gateway Timeout
}

# Dictionary to track API performance by endpoint
api_performance_metrics = {
    "endpoints": {},  # endpoint -> {success_rate, avg_latency, error_count, etc.}
    "global": {
        "total_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "total_retries": 0,
        "avg_latency": 0,
        "total_latency": 0,
    }
}

# Track recent failures for adaptive backoff
recent_failures = {}  # endpoint -> [timestamp1, timestamp2, ...]
MAX_RECENT_FAILURES = 10  # Keep track of up to 10 most recent failures per endpoint

class CircuitBreakerRegistry:
    """Registry to manage circuit breakers by service name."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CircuitBreakerRegistry, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the registry."""
        self.breakers: Dict[str, CircuitBreaker] = {}
        self.stats: Dict[str, CircuitBreakerStats] = {}
    
    def register(self, name: str, breaker: 'CircuitBreaker') -> None:
        """Register a circuit breaker."""
        self.breakers[name] = breaker
        self.stats[name] = CircuitBreakerStats()
    
    def get_breaker(self, name: str) -> Optional['CircuitBreaker']:
        """Get a circuit breaker by name."""
        return self.breakers.get(name)
    
    def get_stats(self, name: str) -> CircuitBreakerStats:
        """Get statistics for a circuit breaker."""
        if name not in self.stats:
            self.stats[name] = CircuitBreakerStats()
        return self.stats[name]
    
    def record_success(self, name: str) -> None:
        """Record a successful call."""
        stats = self.get_stats(name)
        stats.success_count += 1
        stats.consecutive_failures = 0
    
    def record_failure(self, name: str) -> None:
        """Record a failed call."""
        stats = self.get_stats(name)
        stats.failure_count += 1
        stats.consecutive_failures += 1
        stats.last_failure_time = datetime.now()
    
    def record_retry(self, name: str) -> None:
        """Record a retry attempt."""
        stats = self.get_stats(name)
        stats.total_retry_count += 1


class CircuitBreaker:
    """
    Circuit breaker implementation to prevent excessive retries on failing services.
    
    The circuit breaker pattern prevents repeated calls to services that are failing,
    which can make an outage worse by continuing to bombard the failing service.
    """
    
    def __init__(
        self, 
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_max_calls: int = 3
    ):
        """
        Initialize a circuit breaker.
        
        Args:
            name: Unique name for this circuit breaker
            failure_threshold: Number of consecutive failures before opening circuit
            recovery_timeout: Seconds to wait before trying again
            half_open_max_calls: Number of test calls allowed in half-open state
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.state = CircuitState.CLOSED
        self.last_state_change = datetime.now()
        self.half_open_calls = 0
        
        # Register with the registry
        CircuitBreakerRegistry().register(name, self)
        
    def allow_request(self) -> bool:
        """
        Check if a request should be allowed.
        
        Returns:
            bool: True if request is allowed, False otherwise
        """
        registry = CircuitBreakerRegistry()
        stats = registry.get_stats(self.name)
        
        if self.state == CircuitState.CLOSED:
            return True
        
        elif self.state == CircuitState.OPEN:
            # Check if recovery timeout has elapsed
            elapsed = datetime.now() - self.last_state_change
            if elapsed.total_seconds() >= self.recovery_timeout:
                # Transition to half-open
                logger.info(f"Circuit {self.name} transitioning from OPEN to HALF-OPEN")
                self.state = CircuitState.HALF_OPEN
                self.last_state_change = datetime.now()
                self.half_open_calls = 0
                return True
            return False
            
        elif self.state == CircuitState.HALF_OPEN:
            # Allow limited number of test requests
            if self.half_open_calls < self.half_open_max_calls:
                self.half_open_calls += 1
                return True
            return False
        
        return True
    
    def record_success(self) -> None:
        """Record a successful call."""
        registry = CircuitBreakerRegistry()
        registry.record_success(self.name)
        
        if self.state == CircuitState.HALF_OPEN:
            # If enough successful calls in half-open state, close the circuit
            if self.half_open_calls >= self.half_open_max_calls:
                logger.info(f"Circuit {self.name} transitioning from HALF-OPEN to CLOSED")
                self.state = CircuitState.CLOSED
                self.last_state_change = datetime.now()
    
    def record_failure(self) -> None:
        """Record a failed call."""
        registry = CircuitBreakerRegistry()
        registry.record_failure(self.name)
        stats = registry.get_stats(self.name)
        
        if self.state == CircuitState.CLOSED:
            # Open the circuit if threshold is reached
            if stats.consecutive_failures >= self.failure_threshold:
                logger.warning(
                    f"Circuit {self.name} opening after {stats.consecutive_failures} "
                    f"consecutive failures"
                )
                self.state = CircuitState.OPEN
                self.last_state_change = datetime.now()
        
        elif self.state == CircuitState.HALF_OPEN:
            # Any failure in half-open state opens the circuit again
            logger.warning(f"Circuit {self.name} returning to OPEN state after failure in HALF-OPEN")
            self.state = CircuitState.OPEN
            self.last_state_change = datetime.now()


def retry_with_circuit_breaker(
    max_retries: int = 3,
    initial_backoff: float = 1.0, 
    max_backoff: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    exceptions: List[Type[Exception]] = None,
    circuit_name: Optional[str] = None,
    failure_threshold: int = 5,
    recovery_timeout: int = 60
) -> Callable:
    """
    Decorator that retries a function with exponential backoff on specified exceptions
    and implements circuit breaker pattern to prevent excessive retries.
    
    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        initial_backoff: Initial backoff time in seconds (default: 1.0)
        max_backoff: Maximum backoff time in seconds (default: 60.0)
        backoff_factor: Factor to increase backoff each retry (default: 2.0)
        jitter: Add random jitter to backoff time to prevent thundering herd (default: True)
        exceptions: List of exceptions to catch and retry on
                   (default: [ConnectionError, TimeoutError])
        circuit_name: Name for the circuit breaker (defaults to function name)
        failure_threshold: Number of failures before circuit opens (default: 5)
        recovery_timeout: Seconds to wait before trying again (default: 60)
    
    Returns:
        The decorated function
    """
    if exceptions is None:
        exceptions = [ConnectionError, TimeoutError]
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Use function name as circuit name if not provided
        nonlocal circuit_name
        if circuit_name is None:
            circuit_name = func.__qualname__
            
        # Create or get circuit breaker
        registry = CircuitBreakerRegistry()
        if registry.get_breaker(circuit_name) is None:
            CircuitBreaker(
                name=circuit_name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout
            )
            
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            circuit = registry.get_breaker(circuit_name)
            retries = 0
            
            # Remove 'proxies' parameter if present to avoid errors with newer OpenAI API
            if 'proxies' in kwargs:
                kwargs.pop('proxies')
            
            while True:
                # Check if circuit allows the request
                if not circuit.allow_request():
                    error_msg = f"Circuit {circuit_name} is OPEN - requests blocked"
                    logger.warning(error_msg)
                    raise CircuitOpenError(error_msg)
                
                try:
                    # DEBUG: Log the kwargs before calling the function
                    logger.info(f"DEBUG - retry_manager sync_wrapper kwargs: {kwargs}")
                    
                    # Ensure 'proxies' parameter is not in kwargs
                    if 'proxies' in kwargs:
                        logger.warning(f"DEBUG - Found and removed 'proxies' in sync_wrapper: {kwargs['proxies']}")
                        kwargs.pop('proxies')
                    
                    result = func(*args, **kwargs)
                    circuit.record_success()
                    return result
                except tuple(exceptions) as e:
                    retries += 1
                    if retries > max_retries:
                        logger.error(f"Max retries ({max_retries}) exceeded for {func.__name__}")
                        circuit.record_failure()
                        raise MaxRetriesExceededError(f"Maximum retries ({max_retries}) exceeded") from e
                    
                    # Calculate backoff time with exponential increase
                    wait_time = min(
                        initial_backoff * (backoff_factor ** (retries - 1)),
                        max_backoff
                    )
                    
                    # Add jitter if enabled (±15%)
                    if jitter:
                        wait_time = wait_time * random.uniform(0.85, 1.15)
                    
                    registry.record_retry(circuit_name)
                    logger.warning(
                        f"Retry {retries}/{max_retries} for {func.__name__} "
                        f"after {wait_time:.1f}s. Error: {e}"
                    )
                    
                    time.sleep(wait_time)
                except Exception as e:
                    # Non-retryable exception
                    circuit.record_failure()
                    raise
        
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            circuit = registry.get_breaker(circuit_name)
            retries = 0
            
            # Remove 'proxies' parameter if present to avoid errors with newer OpenAI API
            if 'proxies' in kwargs:
                kwargs.pop('proxies')
            
            while True:
                # Check if circuit allows the request
                if not circuit.allow_request():
                    error_msg = f"Circuit {circuit_name} is OPEN - requests blocked"
                    logger.warning(error_msg)
                    raise CircuitOpenError(error_msg)
                
                try:
                    # DEBUG: Log the kwargs before calling the function
                    logger.info(f"DEBUG - retry_manager async_wrapper kwargs: {kwargs}")
                    
                    # Ensure 'proxies' parameter is not in kwargs
                    if 'proxies' in kwargs:
                        logger.warning(f"DEBUG - Found and removed 'proxies' in async_wrapper: {kwargs['proxies']}")
                        kwargs.pop('proxies')
                    
                    result = await func(*args, **kwargs)
                    circuit.record_success()
                    return result
                except tuple(exceptions) as e:
                    retries += 1
                    if retries > max_retries:
                        logger.error(f"Max retries ({max_retries}) exceeded for {func.__name__}")
                        circuit.record_failure()
                        raise MaxRetriesExceededError(f"Maximum retries ({max_retries}) exceeded") from e
                    
                    # Calculate backoff time with exponential increase
                    wait_time = min(
                        initial_backoff * (backoff_factor ** (retries - 1)),
                        max_backoff
                    )
                    
                    # Add jitter if enabled (±15%)
                    if jitter:
                        wait_time = wait_time * random.uniform(0.85, 1.15)
                    
                    registry.record_retry(circuit_name)
                    logger.warning(
                        f"Retry {retries}/{max_retries} for {func.__name__} "
                        f"after {wait_time:.1f}s. Error: {e}"
                    )
                    
                    await asyncio.sleep(wait_time)
                except Exception as e:
                    # Non-retryable exception
                    circuit.record_failure()
                    raise
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


class RetryError(Exception):
    """Base class for retry-related exceptions."""
    pass


class MaxRetriesExceededError(RetryError):
    """Exception raised when maximum retries have been exceeded."""
    pass


class CircuitOpenError(RetryError):
    """Exception raised when a request is rejected because circuit is open."""
    pass


# Utility functions for debugging and monitoring
def get_circuit_stats(circuit_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Get statistics for circuit breakers.
    
    Args:
        circuit_name: Optional name of circuit to get stats for.
                      If None, returns stats for all circuits.
    
    Returns:
        Dict containing circuit breaker statistics
    """
    registry = CircuitBreakerRegistry()
    result = {}
    
    if circuit_name:
        if circuit_name in registry.stats:
            breaker = registry.get_breaker(circuit_name)
            stats = registry.get_stats(circuit_name)
            result[circuit_name] = {
                "state": breaker.state.value if breaker else "unknown",
                "success_count": stats.success_count,
                "failure_count": stats.failure_count,
                "consecutive_failures": stats.consecutive_failures,
                "last_failure": stats.last_failure_time.isoformat() if stats.last_failure_time else None,
                "total_retries": stats.total_retry_count
            }
    else:
        for name in registry.stats:
            breaker = registry.get_breaker(name)
            stats = registry.get_stats(name)
            result[name] = {
                "state": breaker.state.value if breaker else "unknown",
                "success_count": stats.success_count,
                "failure_count": stats.failure_count,
                "consecutive_failures": stats.consecutive_failures,
                "last_failure": stats.last_failure_time.isoformat() if stats.last_failure_time else None,
                "total_retries": stats.total_retry_count
            }
    
    return result


def reset_circuit(circuit_name: str) -> bool:
    """
    Manually reset a circuit breaker to CLOSED state.
    
    Args:
        circuit_name: Name of the circuit to reset
        
    Returns:
        bool: True if reset was successful, False otherwise
    """
    registry = CircuitBreakerRegistry()
    breaker = registry.get_breaker(circuit_name)
    
    if breaker:
        logger.info(f"Manually resetting circuit {circuit_name} to CLOSED state")
        breaker.state = CircuitState.CLOSED
        breaker.last_state_change = datetime.now()
        stats = registry.get_stats(circuit_name)
        stats.consecutive_failures = 0
        return True
    
    return False

class AdaptiveRetryStrategy:
    """
    Implements an adaptive retry strategy that adjusts backoff parameters
    based on the current API performance and error patterns.
    """
    
    def __init__(self, 
                endpoint: str,
                base_max_retries: int = 3,
                base_initial_backoff: float = 1.0,
                base_max_backoff: float = 60.0):
        self.endpoint = endpoint
        self.base_max_retries = base_max_retries
        self.base_initial_backoff = base_initial_backoff
        self.base_max_backoff = base_max_backoff
        
        # Initialize endpoint metrics if not exists
        if endpoint not in api_performance_metrics["endpoints"]:
            api_performance_metrics["endpoints"][endpoint] = {
                "success_count": 0,
                "failure_count": 0,
                "retry_count": 0,
                "avg_latency": 0,
                "total_latency": 0,
                "success_rate": 1.0,  # Start optimistically
                "rate_limit_hits": 0,
                "last_success": None,
                "last_failure": None,
                "error_types": {}
            }
    
    def get_retry_params(self) -> dict:
        """
        Get adjusted retry parameters based on current performance metrics.
        
        Returns:
            dict with retry parameters
        """
        metrics = api_performance_metrics["endpoints"].get(
            self.endpoint, 
            {"success_rate": 1.0, "rate_limit_hits": 0}
        )
        
        # Adjust max retries based on success rate
        # Lower success rates mean we might need more retries
        success_rate = metrics.get("success_rate", 1.0)
        rate_limit_hits = metrics.get("rate_limit_hits", 0)
        
        # Calculate adjusted parameters
        if success_rate < 0.5:
            # Poor success rate - increase retries and backoff
            max_retries = min(self.base_max_retries + 2, 6)  # Max 6 retries
            initial_backoff = min(self.base_initial_backoff * 2, 5.0)
            max_backoff = min(self.base_max_backoff * 1.5, 120.0)
        elif rate_limit_hits > 5:
            # Frequent rate limiting - increase backoff dramatically
            max_retries = self.base_max_retries
            initial_backoff = min(self.base_initial_backoff * 3, 10.0)
            max_backoff = min(self.base_max_backoff * 2, 240.0)
        else:
            # Normal operation
            max_retries = self.base_max_retries
            initial_backoff = self.base_initial_backoff
            max_backoff = self.base_max_backoff
        
        return {
            "max_retries": max_retries,
            "initial_backoff": initial_backoff,
            "max_backoff": max_backoff
        }
    
    def update_metrics(self, 
                      success: bool, 
                      latency: float,
                      error_type: str = None,
                      status_code: int = None) -> None:
        """
        Update performance metrics for this endpoint.
        
        Args:
            success: Whether the call was successful
            latency: Request latency in seconds
            error_type: Type of error if request failed
            status_code: HTTP status code if applicable
        """
        metrics = api_performance_metrics["endpoints"][self.endpoint]
        
        # Update endpoint metrics
        if success:
            metrics["success_count"] += 1
            metrics["last_success"] = datetime.now()
        else:
            metrics["failure_count"] += 1
            metrics["last_failure"] = datetime.now()
            
            # Track error types
            if error_type:
                metrics["error_types"][error_type] = metrics["error_types"].get(error_type, 0) + 1
            
            # Track rate limiting hits
            if status_code == 429:
                metrics["rate_limit_hits"] += 1
                
            # Track recent failures for this endpoint
            if self.endpoint not in recent_failures:
                recent_failures[self.endpoint] = []
                
            recent_failures[self.endpoint].append(datetime.now())
            if len(recent_failures[self.endpoint]) > MAX_RECENT_FAILURES:
                recent_failures[self.endpoint].pop(0)  # Remove oldest
        
        # Update latency metrics
        total_requests = metrics["success_count"] + metrics["failure_count"]
        if total_requests > 1:
            # Update average latency using weighted average
            prev_avg = metrics["avg_latency"]
            metrics["avg_latency"] = ((prev_avg * (total_requests - 1)) + latency) / total_requests
        else:
            metrics["avg_latency"] = latency
            
        metrics["total_latency"] += latency
        
        # Calculate success rate
        metrics["success_rate"] = metrics["success_count"] / total_requests if total_requests > 0 else 1.0
        
        # Update global metrics
        global_metrics = api_performance_metrics["global"]
        global_metrics["total_requests"] += 1
        if success:
            global_metrics["successful_requests"] += 1
        else:
            global_metrics["failed_requests"] += 1
            
        global_metrics["total_latency"] += latency
        global_metrics["avg_latency"] = (
            global_metrics["total_latency"] / global_metrics["total_requests"]
        )

@contextmanager
def measure_latency():
    """Context manager to measure code execution time."""
    start_time = time.time()
    yield
    latency = time.time() - start_time
    return latency

def is_transient_exception(exception: Exception) -> bool:
    """
    Determine if an exception is likely transient and should be retried.
    
    Args:
        exception: The exception to check
        
    Returns:
        bool: True if the exception is likely transient
    """
    # Network-related errors
    if isinstance(exception, (ConnectionError, TimeoutError, socket.timeout, 
                             socket.error, ConnectionRefusedError)):
        return True
        
    # HTTP-related errors
    if isinstance(exception, (http.client.HTTPException, aiohttp.ClientError)):
        return True
        
    # Check for status code in HTTP responses
    if hasattr(exception, 'status_code') and exception.status_code in RETRYABLE_HTTP_STATUS_CODES:
        return True
        
    if hasattr(exception, 'response') and hasattr(exception.response, 'status_code'):
        if exception.response.status_code in RETRYABLE_HTTP_STATUS_CODES:
            return True
    
    # Handle specific API errors
    error_str = str(exception).lower()
    transient_keywords = [
        'timeout', 'too many requests', 'rate limit', 'capacity', 
        'overloaded', 'temporarily unavailable', 'retry', 'connection'
    ]
    
    return any(keyword in error_str for keyword in transient_keywords)

def classify_error(exception: Exception) -> str:
    """
    Classify an exception into a general error category.
    
    Args:
        exception: The exception to classify
        
    Returns:
        str: Error category
    """
    if isinstance(exception, (ConnectionError, socket.error, ConnectionRefusedError)):
        return "connection_error"
    elif isinstance(exception, (TimeoutError, socket.timeout)):
        return "timeout"
    elif hasattr(exception, 'status_code') and exception.status_code == 429:
        return "rate_limited"
    elif hasattr(exception, 'response') and hasattr(exception.response, 'status_code') and exception.response.status_code == 429:
        return "rate_limited"
    elif "rate limit" in str(exception).lower():
        return "rate_limited"
    elif "timeout" in str(exception).lower():
        return "timeout"
    elif "capacity" in str(exception).lower() or "overloaded" in str(exception).lower():
        return "service_overloaded"
    else:
        return "other"

def adaptive_retry_with_circuit_breaker(
    base_max_retries: int = 3,
    base_initial_backoff: float = 1.0, 
    base_max_backoff: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    exceptions: List[Type[Exception]] = None,
    circuit_name: Optional[str] = None,
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
    endpoint_id: Optional[str] = None
) -> Callable:
    """
    Enhanced decorator that implements adaptive retry strategy with exponential backoff
    and circuit breaker pattern. This version uses API performance metrics to
    dynamically adjust retry parameters.
    
    Args:
        base_max_retries: Base maximum number of retry attempts (will be adjusted)
        base_initial_backoff: Base initial backoff time in seconds (will be adjusted)
        base_max_backoff: Base maximum backoff time in seconds (will be adjusted)
        backoff_factor: Factor to increase backoff each retry
        jitter: Add random jitter to backoff time to prevent thundering herd
        exceptions: List of exceptions to catch and retry on
        circuit_name: Name for the circuit breaker (defaults to function name)
        failure_threshold: Number of failures before circuit opens
        recovery_timeout: Seconds to wait before trying again
        endpoint_id: Optional identifier for the API endpoint
    
    Returns:
        The decorated function
    """
    if exceptions is None:
        exceptions = [Exception]  # Retry on any exception by default
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Use function name as circuit name if not provided
        nonlocal circuit_name
        if circuit_name is None:
            circuit_name = func.__qualname__
            
        # Create endpoint ID if not provided
        nonlocal endpoint_id
        if endpoint_id is None:
            endpoint_id = circuit_name
            
        # Create or get circuit breaker
        registry = CircuitBreakerRegistry()
        if registry.get_breaker(circuit_name) is None:
            CircuitBreaker(
                name=circuit_name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout
            )
            
        # Create adaptive retry strategy
        strategy = AdaptiveRetryStrategy(
            endpoint=endpoint_id,
            base_max_retries=base_max_retries,
            base_initial_backoff=base_initial_backoff,
            base_max_backoff=base_max_backoff
        )
            
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            circuit = registry.get_breaker(circuit_name)
            retries = 0
            
            # Get adjusted retry parameters based on current metrics
            retry_params = strategy.get_retry_params()
            max_retries = retry_params["max_retries"]
            initial_backoff = retry_params["initial_backoff"]
            max_backoff = retry_params["max_backoff"]
            
            # Remove 'proxies' parameter if present to avoid errors
            if 'proxies' in kwargs:
                kwargs.pop('proxies')
            
            while True:
                # Check if circuit allows the request
                if not circuit.allow_request():
                    error_msg = f"Circuit {circuit_name} is OPEN - requests blocked"
                    logger.warning(error_msg)
                    raise CircuitOpenError(error_msg)
                
                start_time = time.time()
                success = False
                try:
                    result = func(*args, **kwargs)
                    success = True
                    latency = time.time() - start_time
                    
                    # Record metrics and success
                    strategy.update_metrics(success=True, latency=latency)
                    circuit.record_success()
                    return result
                    
                except tuple(exceptions) as e:
                    latency = time.time() - start_time
                    error_type = classify_error(e)
                    
                    # Check if this is a transient error that should be retried
                    if not is_transient_exception(e):
                        logger.info(f"Non-transient error {type(e).__name__} encountered, not retrying: {e}")
                        circuit.record_failure()
                        strategy.update_metrics(
                            success=False, 
                            latency=latency,
                            error_type=error_type
                        )
                        raise
                    
                    retries += 1
                    if retries > max_retries:
                        logger.error(f"Max retries ({max_retries}) exceeded for {func.__name__}")
                        circuit.record_failure()
                        strategy.update_metrics(
                            success=False, 
                            latency=latency,
                            error_type=error_type
                        )
                        raise MaxRetriesExceededError(f"Maximum retries ({max_retries}) exceeded") from e
                    
                    # Check for rate limiting and adjust accordingly
                    status_code = None
                    if hasattr(e, 'status_code'):
                        status_code = e.status_code
                    elif hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                        status_code = e.response.status_code
                        
                    # Record the failure but don't update circuit yet
                    strategy.update_metrics(
                        success=False, 
                        latency=latency,
                        error_type=error_type,
                        status_code=status_code
                    )
                    
                    # Special handling for rate limiting (429)
                    if status_code == 429 or error_type == "rate_limited":
                        # Rate limited - use longer backoff
                        wait_time = min(
                            initial_backoff * (backoff_factor ** retries),
                            max_backoff * 2  # Double the max backoff for rate limiting
                        )
                    else:
                        # Calculate backoff time with exponential increase
                        wait_time = min(
                            initial_backoff * (backoff_factor ** (retries - 1)),
                            max_backoff
                        )
                    
                    # Add jitter if enabled (±15%)
                    if jitter:
                        wait_time = wait_time * random.uniform(0.85, 1.15)
                    
                    registry.record_retry(circuit_name)
                    logger.warning(
                        f"Retry {retries}/{max_retries} for {func.__name__} "
                        f"after {wait_time:.1f}s. Error: {error_type} - {e}"
                    )
                    
                    time.sleep(wait_time)
                except Exception as e:
                    # Non-retryable exception
                    latency = time.time() - start_time
                    circuit.record_failure()
                    strategy.update_metrics(
                        success=False, 
                        latency=latency,
                        error_type="unhandled"
                    )
                    raise
        
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            circuit = registry.get_breaker(circuit_name)
            retries = 0
            
            # Get adjusted retry parameters based on current metrics
            retry_params = strategy.get_retry_params()
            max_retries = retry_params["max_retries"]
            initial_backoff = retry_params["initial_backoff"]
            max_backoff = retry_params["max_backoff"]
            
            # Remove 'proxies' parameter if present to avoid errors
            if 'proxies' in kwargs:
                kwargs.pop('proxies')
            
            while True:
                # Check if circuit allows the request
                if not circuit.allow_request():
                    error_msg = f"Circuit {circuit_name} is OPEN - requests blocked"
                    logger.warning(error_msg)
                    raise CircuitOpenError(error_msg)
                
                start_time = time.time()
                success = False
                try:
                    result = await func(*args, **kwargs)
                    success = True
                    latency = time.time() - start_time
                    
                    # Record metrics and success
                    strategy.update_metrics(success=True, latency=latency)
                    circuit.record_success()
                    return result
                    
                except tuple(exceptions) as e:
                    latency = time.time() - start_time
                    error_type = classify_error(e)
                    
                    # Check if this is a transient error that should be retried
                    if not is_transient_exception(e):
                        logger.info(f"Non-transient error {type(e).__name__} encountered, not retrying: {e}")
                        circuit.record_failure()
                        strategy.update_metrics(
                            success=False, 
                            latency=latency,
                            error_type=error_type
                        )
                        raise
                    
                    retries += 1
                    if retries > max_retries:
                        logger.error(f"Max retries ({max_retries}) exceeded for {func.__name__}")
                        circuit.record_failure()
                        strategy.update_metrics(
                            success=False, 
                            latency=latency,
                            error_type=error_type
                        )
                        raise MaxRetriesExceededError(f"Maximum retries ({max_retries}) exceeded") from e
                    
                    # Check for rate limiting and adjust accordingly
                    status_code = None
                    if hasattr(e, 'status_code'):
                        status_code = e.status_code
                    elif hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                        status_code = e.response.status_code
                        
                    # Record the failure but don't update circuit yet
                    strategy.update_metrics(
                        success=False, 
                        latency=latency,
                        error_type=error_type,
                        status_code=status_code
                    )
                    
                    # Special handling for rate limiting (429)
                    if status_code == 429 or error_type == "rate_limited":
                        # Rate limited - use longer backoff
                        wait_time = min(
                            initial_backoff * (backoff_factor ** retries),
                            max_backoff * 2  # Double the max backoff for rate limiting
                        )
                    else:
                        # Calculate backoff time with exponential increase
                        wait_time = min(
                            initial_backoff * (backoff_factor ** (retries - 1)),
                            max_backoff
                        )
                    
                    # Add jitter if enabled (±15%)
                    if jitter:
                        wait_time = wait_time * random.uniform(0.85, 1.15)
                    
                    registry.record_retry(circuit_name)
                    logger.warning(
                        f"Retry {retries}/{max_retries} for {func.__name__} "
                        f"after {wait_time:.1f}s. Error: {error_type} - {e}"
                    )
                    
                    await asyncio.sleep(wait_time)
                except Exception as e:
                    # Non-retryable exception
                    latency = time.time() - start_time
                    circuit.record_failure()
                    strategy.update_metrics(
                        success=False, 
                        latency=latency,
                        error_type="unhandled"
                    )
                    raise
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator

def get_api_performance_metrics() -> Dict[str, Any]:
    """
    Get API performance metrics across all endpoints.
    
    Returns:
        Dict containing API performance metrics
    """
    return api_performance_metrics

def reset_api_metrics(endpoint: Optional[str] = None) -> None:
    """
    Reset API performance metrics.
    
    Args:
        endpoint: Optional endpoint to reset metrics for.
               If None, resets metrics for all endpoints.
    """
    global api_performance_metrics
    
    if endpoint:
        if endpoint in api_performance_metrics["endpoints"]:
            api_performance_metrics["endpoints"][endpoint] = {
                "success_count": 0,
                "failure_count": 0,
                "retry_count": 0,
                "avg_latency": 0,
                "total_latency": 0,
                "success_rate": 1.0,
                "rate_limit_hits": 0,
                "last_success": None,
                "last_failure": None,
                "error_types": {}
            }
    else:
        # Reset all metrics
        api_performance_metrics = {
            "endpoints": {},
            "global": {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "total_retries": 0,
                "avg_latency": 0,
                "total_latency": 0,
            }
        }