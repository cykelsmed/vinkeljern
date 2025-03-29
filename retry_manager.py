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
            
            while True:
                # Check if circuit allows the request
                if not circuit.allow_request():
                    error_msg = f"Circuit {circuit_name} is OPEN - requests blocked"
                    logger.warning(error_msg)
                    raise CircuitOpenError(error_msg)
                
                try:
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
            
            while True:
                # Check if circuit allows the request
                if not circuit.allow_request():
                    error_msg = f"Circuit {circuit_name} is OPEN - requests blocked"
                    logger.warning(error_msg)
                    raise CircuitOpenError(error_msg)
                
                try:
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