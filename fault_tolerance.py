"""
Fault tolerance module for Vinkeljernet.

This module provides fault tolerance mechanisms including:
1. Circuit breaker pattern
2. Fault-tolerant service wrappers
3. Health monitoring
4. Service degradation handling
5. Automatic recovery

Usage:
    # Wrap a function with a circuit breaker
    @CircuitBreaker(name="my_service", failure_threshold=3)
    def my_service_function(arg1, arg2):
        # This function will be protected by the circuit breaker
        return external_api_call(arg1, arg2)
    
    # Create a fault-tolerant service
    service = FaultTolerantService(
        name="my_service",
        service_function=my_function,
        fallback_function=my_fallback,
        circuit_breaker=CircuitBreaker(name="my_service_circuit")
    )
    
    # Use the service
    result = service.execute(arg1, arg2)
"""

import os
import json
import time
import logging
import threading
import functools
import traceback
import concurrent.futures
import hashlib
import asyncio
import inspect
from enum import Enum
from typing import Dict, List, Any, Callable, Optional, TypeVar, Union, Generic, Type, Set, Tuple, Awaitable, cast
from datetime import datetime, timedelta
from functools import wraps

# Type variables for generic functions
T = TypeVar('T')
U = TypeVar('U')
F = TypeVar('F', bound=Callable[..., Any])

# Set up logging
logger = logging.getLogger("vinkeljernet.fault_tolerance")

# Global registry of services and circuit breakers
_services = {}
_circuit_breakers = {}

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"      # Circuit is tripped, requests fail fast
    HALF_OPEN = "half_open"  # Testing if service is healthy again


class CircuitBreaker:
    """
    Implementation of the Circuit Breaker pattern.
    
    Prevents cascading failures by failing fast when a service is unhealthy.
    Allows for self-healing by periodically testing the service.
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        reset_timeout_seconds: int = 60,
        half_open_max_calls: int = 1,
        exclude_exceptions: List[Type[Exception]] = None,
        callback_on_open: Callable[[], None] = None,
        callback_on_close: Callable[[], None] = None
    ):
        """
        Initialize a new circuit breaker.
        
        Args:
            name: Unique name for this circuit breaker
            failure_threshold: Number of failures before circuit opens
            reset_timeout_seconds: Seconds to wait before trying half-open state
            half_open_max_calls: Number of test calls to allow in half-open state
            exclude_exceptions: List of exception types to NOT count as failures
            callback_on_open: Function to call when circuit opens
            callback_on_close: Function to call when circuit closes
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout_seconds = reset_timeout_seconds
        self.half_open_max_calls = half_open_max_calls
        self.exclude_exceptions = exclude_exceptions or []
        self.callback_on_open = callback_on_open
        self.callback_on_close = callback_on_close
        
        # State variables
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._successful_calls = 0
        self._last_failure_time = None
        self._last_failure_exception = None
        self._half_open_calls = 0
        self._total_calls = 0
        self._successful_calls = 0
        self._state_change_time = datetime.now()
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Register this circuit breaker in the global registry
        _circuit_breakers[name] = self
    
    @property
    def state(self) -> str:
        """Get the current state of the circuit as a string"""
        return self._state.value
    
    @property
    def is_open(self) -> bool:
        """Check if the circuit is open (requests will fail fast)"""
        return self._state == CircuitState.OPEN
    
    @property
    def is_closed(self) -> bool:
        """Check if the circuit is closed (normal operation)"""
        return self._state == CircuitState.CLOSED
    
    @property
    def is_half_open(self) -> bool:
        """Check if the circuit is half-open (testing recovery)"""
        return self._state == CircuitState.HALF_OPEN
    
    @property
    def failures(self) -> int:
        """Get the current failure count"""
        return self._failures
    
    @property
    def failure_rate(self) -> str:
        """Get the current failure rate as a percentage string"""
        if self._total_calls == 0:
            return "0%"
        rate = (self._failures / self._total_calls) * 100
        return f"{rate:.1f}%"
    
    @property
    def last_failure(self) -> Optional[str]:
        """Get the timestamp of the last failure, or None if no failures"""
        if self._last_failure_time:
            return self._last_failure_time.isoformat()
        return None
        
    def record_success(self) -> None:
        """Record a successful call through this circuit"""
        with self._lock:
            self._total_calls += 1
            self._successful_calls += 1
            
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls += 1
                
                # If we've had enough successful test calls, close the circuit
                if self._half_open_calls >= self.half_open_max_calls:
                    self._change_state(CircuitState.CLOSED)
                    if self.callback_on_close:
                        try:
                            self.callback_on_close()
                        except Exception as e:
                            logger.warning(f"Error in on_close callback for circuit '{self.name}': {e}")
    
    def record_failure(self, exception: Exception) -> None:
        """Record a failed call through this circuit"""
        # Check if this exception type should be ignored
        for excluded_type in self.exclude_exceptions:
            if isinstance(exception, excluded_type):
                logger.debug(f"Circuit '{self.name}' ignoring excluded exception: {type(exception).__name__}")
                return
        
        with self._lock:
            self._total_calls += 1
            self._failures += 1
            self._last_failure_time = datetime.now()
            self._last_failure_exception = exception
            
            # If in half-open state, go back to open on any failure
            if self._state == CircuitState.HALF_OPEN:
                self._change_state(CircuitState.OPEN)
                return
            
            # If in closed state and threshold reached, open circuit
            if self._state == CircuitState.CLOSED and self._failures >= self.failure_threshold:
                self._change_state(CircuitState.OPEN)
                if self.callback_on_open:
                    try:
                        self.callback_on_open()
                    except Exception as e:
                        logger.warning(f"Error in on_open callback for circuit '{self.name}': {e}")
    
    def allow_request(self) -> bool:
        """
        Check if a request should be allowed through the circuit.
        
        Returns:
            bool: True if the request should proceed, False to short-circuit
        """
        with self._lock:
            # Always allow if closed
            if self._state == CircuitState.CLOSED:
                return True
            
            # If open, check if it's time to try half-open
            if self._state == CircuitState.OPEN:
                timeout_expired = (
                    self._last_failure_time and
                    datetime.now() - self._last_failure_time > timedelta(seconds=self.reset_timeout_seconds)
                )
                
                if timeout_expired:
                    self._change_state(CircuitState.HALF_OPEN)
                    self._half_open_calls = 0
                    return True
                
                return False
            
            # If half-open, only allow limited test calls
            if self._state == CircuitState.HALF_OPEN:
                return self._half_open_calls < self.half_open_max_calls
            
            # Should never reach here
            return False
    
    def _change_state(self, new_state: CircuitState) -> None:
        """Change the circuit state and log the transition"""
        old_state = self._state
        self._state = new_state
        self._state_change_time = datetime.now()
        
        # Reset failure count when closing the circuit
        if new_state == CircuitState.CLOSED:
            self._failures = 0
        
        logger.info(f"Circuit '{self.name}' state changed: {old_state.value} -> {new_state.value}")
    
    def reset(self) -> None:
        """Reset the circuit breaker to closed state"""
        with self._lock:
            self._failures = 0
            self._half_open_calls = 0
            self._state = CircuitState.CLOSED
            logger.info(f"Circuit '{self.name}' manually reset to CLOSED")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about this circuit breaker"""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failures": self._failures,
                "failure_threshold": self.failure_threshold,
                "failure_rate": self.failure_rate,
                "last_failure": self.last_failure,
                "total_calls": self._total_calls,
                "successful_calls": self._successful_calls,
                "reset_timeout_seconds": self.reset_timeout_seconds,
                "state_change_time": self._state_change_time.isoformat()
            }
    
    def __call__(self, func):
        """
        Decorator to wrap a function with this circuit breaker.
        
        Usage:
            @CircuitBreaker(name="my_service", failure_threshold=3)
            def my_function():
                # Function will be protected by circuit breaker
                pass
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not self.allow_request():
                raise CircuitOpenError(
                    f"Circuit '{self.name}' is OPEN - failing fast", 
                    circuit=self
                )
            
            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure(e)
                raise
        
        return wrapper
    
    @classmethod
    def get(cls, name: str) -> Optional['CircuitBreaker']:
        """Get a circuit breaker by name from the global registry"""
        return _circuit_breakers.get(name)
    
    @classmethod
    def get_all_stats(cls) -> Dict[str, Dict[str, Any]]:
        """Get stats for all circuit breakers"""
        return {name: breaker.get_stats() for name, breaker in _circuit_breakers.items()}
    
    @classmethod
    def reset_all(cls) -> None:
        """Reset all registered circuit breakers"""
        for circuit in _circuit_breakers.values():
            circuit.reset()


class CircuitOpenError(Exception):
    """Exception raised when a circuit is open"""
    
    def __init__(self, message: str, circuit: CircuitBreaker = None):
        super().__init__(message)
        self.circuit = circuit


class ServiceDegradedException(Exception):
    """Raised when a service is degraded but may still provide limited functionality."""
    def __init__(self, message: str, service_name: str, severity: str = "medium"):
        self.service_name = service_name
        self.severity = severity  # "low", "medium", "high"
        self.timestamp = datetime.now().isoformat()
        super().__init__(message)


class RetryPolicy:
    """
    Policy for retrying operations that can fail transiently.
    
    Supports different backoff strategies and configurable limits.
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay_ms: int = 100,
        max_delay_ms: int = 10000,
        backoff_factor: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: List[Type[Exception]] = None
    ):
        """
        Initialize a retry policy.
        
        Args:
            max_retries: Maximum number of retries before giving up
            base_delay_ms: Initial delay between retries in milliseconds
            max_delay_ms: Maximum delay between retries in milliseconds
            backoff_factor: Multiplier for delay after each retry
            jitter: Whether to add randomness to delay to prevent thundering herd
            retryable_exceptions: List of exception types that should trigger a retry
        """
        self.max_retries = max_retries
        self.base_delay_ms = base_delay_ms
        self.max_delay_ms = max_delay_ms
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions or []
    
    def get_delay_ms(self, attempt: int) -> int:
        """
        Calculate the delay for a specific retry attempt.
        
        Args:
            attempt: The current retry attempt (0-based)
            
        Returns:
            int: Delay in milliseconds
        """
        import random
        
        delay = self.base_delay_ms * (self.backoff_factor ** attempt)
        delay = min(delay, self.max_delay_ms)
        
        if self.jitter:
            # Add up to 25% random jitter
            jitter_amount = delay * 0.25
            delay = delay - jitter_amount + (random.random() * jitter_amount * 2)
        
        return int(delay)
    
    def should_retry(self, exception: Exception) -> bool:
        """
        Determine if an exception should trigger a retry.
        
        Args:
            exception: The exception that was raised
            
        Returns:
            bool: True if should retry, False otherwise
        """
        if not self.retryable_exceptions:
            return True
        
        for ex_type in self.retryable_exceptions:
            if isinstance(exception, ex_type):
                return True
        
        return False


class FaultTolerantCache:
    """
    Fault-tolerant cache system that can be used to store and retrieve results
    from API calls, with support for graceful degradation and local fallback.
    
    This class works together with CircuitBreaker to improve the system's 
    robustness during API failures and network issues.
    """
    
    def __init__(
        self, 
        name: str,
        storage_dir: str = "./.cache",
        ttl: int = 3600,  # 1 hour
        emergency_ttl: int = 86400 * 7,  # 7 days for fallback
        allow_stale: bool = True,
    ):
        """
        Initialize a fault-tolerant cache.
        
        Args:
            name: Cache category name
            storage_dir: Directory for cached files
            ttl: Time-to-live in seconds for normal cache entries
            emergency_ttl: Longer TTL for emergency situations (during failures)
            allow_stale: Whether stale cache may be used during failures
        """
        self.name = name
        self.storage_dir = os.path.join(storage_dir, name)
        self.ttl = ttl
        self.emergency_ttl = emergency_ttl
        self.allow_stale = allow_stale
        self.degraded_mode = False
        self.stats = {
            "hits": 0,
            "misses": 0,
            "emergency_hits": 0,
            "stale_hits": 0,
            "stores": 0,
            "errors": 0,
        }
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.storage_dir):
            try:
                os.makedirs(self.storage_dir)
                logger.info(f"Created cache directory: {self.storage_dir}")
            except Exception as e:
                logger.error(f"Failed to create cache directory: {str(e)}")
                # Continue even if we can't create directory - we can still function without cache
    
    def _get_key_path(self, key: str) -> str:
        """Convert a cache key to a file path"""
        # Hash the key to ensure a valid filename
        hashed_key = hashlib.md5(key.encode('utf-8')).hexdigest()
        return os.path.join(self.storage_dir, f"{hashed_key}.json")
    
    def _is_valid(self, metadata: dict) -> bool:
        """Check if a cache entry is still valid based on TTL"""
        now = time.time()
        created = metadata.get("created", 0)
        ttl = metadata.get("ttl", self.ttl)
        
        return now - created < ttl
    
    def _is_emergency_valid(self, metadata: dict) -> bool:
        """Check if a cache entry is valid in emergency situations (longer TTL)"""
        now = time.time()
        created = metadata.get("created", 0)
        emergency_ttl = metadata.get("emergency_ttl", self.emergency_ttl)
        
        return now - created < emergency_ttl
    
    def get(self, key: str, default: Any = None, emergency: bool = False) -> Tuple[Any, dict]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            default: Default value if key doesn't exist
            emergency: Whether we're in an emergency situation (allow stale cache)
            
        Returns:
            Tuple of (cache_value, metadata)
        """
        cache_path = self._get_key_path(key)
        
        try:
            if not os.path.exists(cache_path):
                self.stats["misses"] += 1
                return default, {"exists": False}
            
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            metadata = cache_data.get("metadata", {})
            value = cache_data.get("value", default)
            
            # Check if cache is valid
            if emergency or self.degraded_mode:
                # In emergency situations use stale cache if allowed
                if self._is_valid(metadata):
                    self.stats["hits"] += 1
                    return value, metadata
                elif self._is_emergency_valid(metadata):
                    self.stats["emergency_hits"] += 1
                    metadata["stale"] = True
                    return value, metadata
                else:
                    # Even for emergency situations the cache is too old
                    self.stats["misses"] += 1
                    return default, {"exists": False, "too_old": True}
            elif self._is_valid(metadata):
                self.stats["hits"] += 1
                return value, metadata
            else:
                self.stats["misses"] += 1
                return default, {"exists": True, "expired": True}
                
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Error reading from cache ({key}): {str(e)}")
            return default, {"error": str(e)}
    
    def set(self, key: str, value: Any, metadata: dict = None) -> bool:
        """
        Store a value in the cache.
        
        Args:
            key: Cache key
            value: Value to store
            metadata: Additional metadata
            
        Returns:
            bool: Whether the operation succeeded
        """
        cache_path = self._get_key_path(key)
        
        try:
            # Prepare metadata
            meta = metadata or {}
            meta.update({
                "created": time.time(),
                "ttl": meta.get("ttl", self.ttl),
                "emergency_ttl": meta.get("emergency_ttl", self.emergency_ttl),
                "key": key
            })
            
            # Create cache data structure
            cache_data = {
                "value": value,
                "metadata": meta
            }
            
            # Write to cache file
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, default=str)
                
            self.stats["stores"] += 1
            return True
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Error writing to cache ({key}): {str(e)}")
            return False
    
    def invalidate(self, key: str) -> bool:
        """Invalidate a specific cache entry."""
        try:
            cache_path = self._get_key_path(key)
            if os.path.exists(cache_path):
                os.remove(cache_path)
                return True
            return False
        except Exception as e:
            logger.error(f"Error invalidating cache ({key}): {str(e)}")
            return False
    
    def clear(self) -> int:
        """
        Clear the entire cache.
        
        Returns:
            int: Number of deleted cache files
        """
        import glob
        
        try:
            pattern = os.path.join(self.storage_dir, "*.json")
            cache_files = glob.glob(pattern)
            
            for file_path in cache_files:
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.error(f"Error removing cache file {file_path}: {str(e)}")
            
            return len(cache_files)
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            return 0
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        import glob
        
        try:
            pattern = os.path.join(self.storage_dir, "*.json")
            cache_files = glob.glob(pattern)
            
            # Add cache file information
            self.stats.update({
                "file_count": len(cache_files),
                "degraded_mode": self.degraded_mode,
            })
            
            # Calculate hit rate
            total_requests = self.stats["hits"] + self.stats["misses"] + self.stats["emergency_hits"] + self.stats["stale_hits"]
            hit_rate = 0.0
            if total_requests > 0:
                hit_rate = (self.stats["hits"] + self.stats["emergency_hits"] + self.stats["stale_hits"]) / total_requests * 100
                
            self.stats["hit_rate"] = f"{hit_rate:.1f}%"
            return self.stats
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
            return self.stats
    
    def set_degraded_mode(self, enabled: bool) -> None:
        """Set degraded mode (use stale cache if necessary)"""
        self.degraded_mode = enabled
        logger.warning(f"Cache '{self.name}' degraded mode: {enabled}")


async def run_with_timeout(coro, timeout):
    """Run a coroutine with a timeout"""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        raise TimeoutError(f"Operation timed out after {timeout} seconds")


class FaultTolerantService:
    """
    A fault-tolerant service class that combines CircuitBreaker and FaultTolerantCache
    to create robust API calls with graceful degradation.
    """
    
    def __init__(
        self,
        name: str,
        cache_ttl: int = 3600,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        cache_dir: str = "./.cache",
    ):
        """
        Initialize a fault-tolerant service.
        
        Args:
            name: Service name (used for both cache and circuit breaker)
            cache_ttl: Cache time-to-live in seconds
            failure_threshold: Number of failures before circuit opens
            recovery_timeout: Seconds before circuit tries to close again
            cache_dir: Directory for cache
        """
        self.name = name
        
        # Create circuit breaker
        if name not in _circuit_breakers:
            self.circuit = CircuitBreaker(
                name=name, 
                failure_threshold=failure_threshold,
                reset_timeout_seconds=recovery_timeout
            )
        else:
            self.circuit = _circuit_breakers[name]
        
        # Create cache
        self.cache = FaultTolerantCache(
            name=name,
            storage_dir=cache_dir,
            ttl=cache_ttl
        )
        
        # Service status
        self.healthy = True
        self.last_success = datetime.now()
        self.last_error = None
        self.last_error_message = None
        
        # Register this service
        _services[name] = self
        
        logger.info(f"Initialized fault-tolerant service: {name}")
    
    async def call(
        self,
        func: Callable,
        cache_key: str = None,
        use_cache: bool = True,
        store_cache: bool = True,
        fallback: Any = None,
        metadata: dict = None,
        args: list = None,
        kwargs: dict = None,
        timeout: int = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Execute an API call with circuit breaking and caching.
        
        Args:
            func: Function to execute
            cache_key: Key for caching (None disables caching)
            use_cache: Whether to use cache
            store_cache: Whether to cache the result
            fallback: Fallback value if call fails
            metadata: Additional metadata for caching
            args: Arguments for the function
            kwargs: Keyword arguments for the function
            timeout: Timeout in seconds
            
        Returns:
            Tuple of (response, metadata)
        """
        args = args or []
        kwargs = kwargs or {}
        
        response = None
        result_metadata = {
            "source": None,
            "cached": False,
            "stale": False,
            "degraded": False,
            "error": None,
        }
        
        # Check if we're using cache
        if use_cache and cache_key:
            # Try to get from cache
            cached_response, cache_meta = self.cache.get(
                key=cache_key, 
                emergency=not self.circuit.is_closed
            )
            
            if "exists" in cache_meta and cache_meta.get("exists", False) != False:
                response = cached_response
                result_metadata["source"] = "cache"
                result_metadata["cached"] = True
                result_metadata["cache_metadata"] = cache_meta
                result_metadata["stale"] = cache_meta.get("stale", False)
                
                # If we're in circuit open, return cache even if stale
                if self.circuit.is_open and "stale" in cache_meta:
                    logger.warning(f"Service {self.name} is degraded, returning stale cached result")
                    result_metadata["degraded"] = True
                    return response, result_metadata
        
        # If we don't have a cached result, try to call the API
        if response is None:
            try:
                # Check if circuit allows requests
                if not self.circuit.allow_request():
                    raise CircuitOpenError(f"Circuit {self.name} is open", self.circuit)
                
                # Execute the function with timeout if specified
                if inspect.iscoroutinefunction(func):
                    if timeout:
                        response = await run_with_timeout(func(*args, **kwargs), timeout)
                    else:
                        response = await func(*args, **kwargs)
                else:
                    if timeout:
                        # Use concurrent.futures for timeout with sync functions
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(func, *args, **kwargs)
                            response = future.result(timeout=timeout)
                    else:
                        response = func(*args, **kwargs)
                
                # Record success in circuit breaker
                self.circuit.record_success()
                
                # Update status
                self.healthy = True
                self.last_success = datetime.now()
                result_metadata["source"] = "api"
                
                # Cache the result if necessary
                if store_cache and cache_key:
                    self.cache.set(cache_key, response, metadata)
                
            except CircuitOpenError as e:
                # Circuit is open, use fallback
                logger.warning(f"Circuit {self.name} is open: {str(e)}")
                response = fallback
                result_metadata["source"] = "fallback"
                result_metadata["error"] = f"Circuit open: {str(e)}"
                result_metadata["degraded"] = True
                
            except (TimeoutError, asyncio.TimeoutError) as e:
                # Timeout occurred
                self.healthy = False
                self.last_error = datetime.now()
                self.last_error_message = f"Timeout after {timeout} seconds"
                self.circuit.record_failure(e)
                
                logger.error(f"Timeout in service {self.name}: {str(e)}")
                response = fallback
                result_metadata["source"] = "fallback"
                result_metadata["error"] = self.last_error_message
                result_metadata["degraded"] = True
                
            except Exception as e:
                # Other errors
                self.healthy = False
                self.last_error = datetime.now()
                self.last_error_message = str(e)
                self.circuit.record_failure(e)
                
                logger.error(f"Error in service {self.name}: {str(e)}")
                response = fallback
                result_metadata["source"] = "fallback"
                result_metadata["error"] = str(e)
                result_metadata["degraded"] = True
                
                # If cache is in degraded mode, try again with emergency flag
                if cache_key:
                    cached_response, cache_meta = self.cache.get(
                        key=cache_key, 
                        emergency=True
                    )
                    
                    if cached_response is not None:
                        response = cached_response
                        result_metadata["source"] = "emergency_cache"
                        result_metadata["cached"] = True
                        result_metadata["degraded"] = True
                        result_metadata["stale"] = True
        
        return response, result_metadata
    
    def get_status(self) -> Dict[str, Any]:
        """Get status for this service"""
        return {
            "name": self.name,
            "healthy": self.healthy,
            "circuit": self.circuit.state,
            "last_success": self.last_success.isoformat(),
            "last_error": self.last_error.isoformat() if self.last_error else None,
            "last_error_message": self.last_error_message,
            "cache_stats": self.cache.get_stats(),
            "circuit_stats": self.circuit.get_stats()
        }
    
    def reset(self) -> None:
        """Reset service status and circuit breaker"""
        self.circuit.reset()
        self.healthy = True
        self.last_error = None
        self.last_error_message = None
        logger.info(f"Service {self.name} reset")


class FaultTolerantComponent:
    """Base class for fault-tolerant components with common functionality"""
    
    def __init__(self, name: str, failure_threshold: int = 3):
        self.name = name
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.successful_calls = 0
        self.total_calls = 0
        self.last_error = None
        self.healthy = True
        
    def record_success(self):
        """Record a successful operation"""
        self.successful_calls += 1
        self.total_calls += 1
        self.failures = 0
        self.healthy = True
        
    def record_failure(self, error):
        """Record a failed operation"""
        self.failures += 1
        self.total_calls += 1
        self.last_error = error
        
        if self.failures >= self.failure_threshold:
            self.healthy = False
            
    async def call_method(self, method, *args, **kwargs):
        """Call a method with fault tracking"""
        try:
            if inspect.iscoroutinefunction(method):
                result = await method(*args, **kwargs)
            else:
                result = method(*args, **kwargs)
                
            self.record_success()
            return result
        except Exception as e:
            self.record_failure(e)
            raise
    
    def get_stats(self):
        """Get component statistics"""
        success_rate = 0
        if self.total_calls > 0:
            success_rate = (self.successful_calls / self.total_calls) * 100
            
        return {
            "name": self.name,
            "healthy": self.healthy,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failures": self.failures,
            "success_rate": f"{success_rate:.1f}%",
            "last_error": str(self.last_error) if self.last_error else None
        }


class FaultTolerantAngleGenerator(FaultTolerantComponent):
    """
    A fault-tolerant angle generator ensuring users always get usable results,
    even if API calls fail or other components don't work as expected.
    
    Implements strategy #1 from the desired points - ensuring that even if 
    a component fails, the user still gets a usable result.
    """
    
    def __init__(
        self,
        service_name: str = "angle_generator",
        cache_ttl: int = 3600 * 24,  # 24 hours
        failure_threshold: int = 3,
        recovery_timeout: int = 60,
        min_acceptable_angles: int = 3,
        cache_dir: str = "./.cache/angles",
        use_generic_fallbacks: bool = True
    ):
        """
        Initialize a fault-tolerant angle generator.
        
        Args:
            service_name: Name for service registration
            cache_ttl: Cache lifetime in seconds
            failure_threshold: Number of failures before circuit breaker opens
            recovery_timeout: Time before circuit breaker tests recovery
            min_acceptable_angles: Minimum number of angles for result to be considered useful
            cache_dir: Directory for cache
            use_generic_fallbacks: Whether to use generic fallback angles on failure
        """
        super().__init__(name=service_name, failure_threshold=failure_threshold)
        self.service_name = service_name
        self.min_acceptable_angles = min_acceptable_angles
        self.use_generic_fallbacks = use_generic_fallbacks
        
        # Create fault tolerant service
        self.service = FaultTolerantService(
            name=service_name,
            cache_ttl=cache_ttl,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            cache_dir=cache_dir
        )
        
        # Stats
        self.successful_generations = 0
        self.partial_successful_generations = 0
        self.failed_generations = 0
        self.fallback_used_count = 0
        
        logger.info(f"Initialized fault-tolerant angle generator: {service_name}")
    
    def _create_cache_key(self, topic: str, profile_id: str) -> str:
        """Generate a unique cache key based on topic and profile"""
        return f"angle_{profile_id}_{topic.lower().replace(' ', '_')}"
    
    def _create_simple_fallback_angles(self, topic: str) -> List[Dict[str, Any]]:
        """Generate simple fallback angles when all else fails."""
        fallback_angles = [
            {
                "overskrift": f"Aktuel status på {topic}",
                "beskrivelse": "En opdateret analyse af den nuværende situation og udvikling.",
                "begrundelse": "Aktualitet er et vigtigt nyhedskriterie, og denne vinkel giver læserne et overblik over den seneste udvikling.",
                "nyhedskriterier": ["aktualitet", "væsentlighed"],
                "startSpørgsmål": ["Hvad er den seneste udvikling i sagen?", "Hvordan ser situationen ud lige nu?"],
                "_fallback": True
            },
            {
                "overskrift": f"Eksperter vurderer konsekvenserne af {topic}",
                "beskrivelse": "Forskellige eksperters vurdering af mulige konsekvenser på kort og lang sigt.",
                "begrundelse": "Ekspertvurderinger giver dybde og troværdighed til historien, samt hjælper med at forstå perspektiverne.",
                "nyhedskriterier": ["væsentlighed", "aktualitet"],
                "startSpørgsmål": ["Hvilke konsekvenser forudser eksperterne?", "Hvordan kan dette påvirke samfundet?"],
                "_fallback": True
            },
            {
                "overskrift": f"Det skal du vide om {topic} som borger",
                "beskrivelse": "Praktisk guide til hvordan borgere forholder sig til situationen og hvilke handlemuligheder de har.",
                "begrundelse": "Denne servicejournalistik hjælper læserne med at forstå, hvordan emnet påvirker deres hverdag direkte.",
                "nyhedskriterier": ["identifikation", "væsentlighed"],
                "startSpørgsmål": ["Hvordan påvirker dette den almindelige borger?", "Hvilke handlemuligheder har man som borger?"],
                "_fallback": True
            },
            {
                "overskrift": f"Baggrund: Derfor er {topic} vigtigt netop nu",
                "beskrivelse": "En uddybende forklaring af konteksten og hvorfor emnet er relevant på nuværende tidspunkt.",
                "begrundelse": "Denne baggrundsartikel giver dybde og perspektiv til historien, hvilket hjælper læserne med at forstå emnets væsentlighed.",
                "nyhedskriterier": ["væsentlighed", "aktualitet"],
                "startSpørgsmål": ["Hvorfor er dette emne vigtigt lige nu?", "Hvad er den historiske kontekst for dette emne?"],
                "_fallback": True
            },
            {
                "overskrift": f"5 spørgsmål og svar om {topic}",
                "beskrivelse": "De mest presserende spørgsmål og svar om emnet i et letforståeligt format.",
                "begrundelse": "Q&A-formatet er en effektiv måde at formidle kompleks information på en tilgængelig måde, hvilket øger læserengagementet.",
                "nyhedskriterier": ["identifikation", "forståelse"],
                "startSpørgsmål": ["Hvad er de mest almindelige misforståelser om emnet?", "Hvilke spørgsmål stiller folk oftest om dette emne?"],
                "_fallback": True
            }
        ]
        return fallback_angles
    
    async def generate_angles(
        self,
        topic: str,
        profile: Any,
        generate_func: Callable,
        bypass_cache: bool = False,
        topic_info: str = None,
        *args,
        **kwargs
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Generate angles with fault tolerance, fallbacks, and caching.
        
        Args:
            topic: Topic to generate angles for
            profile: Editorial DNA profile
            generate_func: Function that generates angles
            bypass_cache: Whether to ignore cache
            topic_info: Background information about the topic
            args, kwargs: Additional arguments to generate_func
            
        Returns:
            Tuple of (angles, metadata)
        """
        # Create metadata for the result
        result_metadata = {
            "source": None,
            "cached": False,
            "degraded": False,
            "partial_result": False,
            "fallback_used": False,
            "error": None,
            "angle_count": 0,
            "requested_topic": topic
        }
        
        # Prepare cache key
        profile_id = getattr(profile, "id", None) or getattr(profile, "navn", "unknown_profile")
        cache_key = None if bypass_cache else self._create_cache_key(topic, profile_id)
        
        try:
            # Prepare parameters for generate_func
            kwargs["topic"] = topic
            kwargs["profile"] = profile
            
            if topic_info:
                kwargs["topic_info"] = topic_info
            
            # Define fallback value
            fallback_angles = []
            if self.use_generic_fallbacks:
                fallback_angles = self._create_simple_fallback_angles(topic)
                
            # Call service with circuit breaker and cache
            angles, call_metadata = await self.service.call(
                func=generate_func,
                cache_key=cache_key,
                use_cache=not bypass_cache,
                store_cache=True,
                fallback=fallback_angles,
                metadata={"topic": topic, "profile_id": profile_id},
                args=args,
                kwargs=kwargs
            )
            
            # Update metadata from service call
            result_metadata.update(call_metadata)
            
            # Validate result - make sure we have angles and they're in the right format
            if not angles or not isinstance(angles, list):
                logger.warning(f"Generate angles returned invalid result: {type(angles)}")
                
                if fallback_angles:
                    angles = fallback_angles
                    result_metadata["fallback_used"] = True
                    result_metadata["source"] = "internal_fallback"
                    self.fallback_used_count += 1
                else:
                    angles = []
            
            # Check if we have enough angles
            angle_count = len(angles)
            result_metadata["angle_count"] = angle_count
            
            if angle_count < self.min_acceptable_angles:
                # Not enough angles - add fallback angles to reach minimum
                if fallback_angles:
                    # Only add the fallback angles we need
                    missing_count = self.min_acceptable_angles - angle_count
                    if missing_count > 0:
                        angles.extend(fallback_angles[:missing_count])
                        result_metadata["partial_result"] = True
                        result_metadata["fallback_used"] = True
                        result_metadata["source"] = result_metadata.get("source", "api") + "_with_fallback"
                        self.partial_successful_generations += 1
                    else:
                        # We don't need fallbacks
                        self.successful_generations += 1
                else:
                    # No fallbacks available - return what we have
                    if angle_count > 0:
                        result_metadata["partial_result"] = True
                        self.partial_successful_generations += 1
                    else:
                        self.failed_generations += 1
            else:
                # Sufficient number of angles
                self.successful_generations += 1
            
            return angles, result_metadata
            
        except Exception as e:
            logger.error(f"Error in FaultTolerantAngleGenerator.generate_angles: {str(e)}")
            self.failed_generations += 1
            
            # Return fallback angles on failure
            if fallback_angles:
                result_metadata["fallback_used"] = True
                result_metadata["error"] = str(e)
                result_metadata["source"] = "error_fallback"
                result_metadata["degraded"] = True
                result_metadata["angle_count"] = len(fallback_angles)
                self.fallback_used_count += 1
                return fallback_angles, result_metadata
            else:
                # If we don't have fallbacks, raise the error
                raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about angle generator"""
        total_generations = (
            self.successful_generations + 
            self.partial_successful_generations + 
            self.failed_generations
        )
        
        stats = {
            "service_name": self.service_name,
            "total_generations": total_generations,
            "successful_generations": self.successful_generations,
            "partial_successful_generations": self.partial_successful_generations,
            "failed_generations": self.failed_generations,
            "fallback_used_count": self.fallback_used_count,
        }
        
        # Calculate success rate
        if total_generations > 0:
            full_success_rate = (self.successful_generations / total_generations) * 100
            any_success_rate = ((self.successful_generations + self.partial_successful_generations) 
                              / total_generations) * 100
                              
            stats["full_success_rate"] = f"{full_success_rate:.1f}%"
            stats["any_success_rate"] = f"{any_success_rate:.1f}%"
        
        # Add service stats
        stats["service_status"] = self.service.get_status()
        
        return stats


# Get a service by name
def get_service(name: str) -> FaultTolerantService:
    """Get a fault-tolerant service by name"""
    if name not in _services:
        _services[name] = FaultTolerantService(name)
    return _services[name]


def get_all_services_status() -> Dict[str, Dict[str, Any]]:
    """Get status for all services"""
    return {name: service.get_status() for name, service in _services.items()}


def with_circuit_breaker(name: str, **circuit_kwargs) -> Callable[[F], F]:
    """
    Decorator to wrap a function with a circuit breaker.
    
    Args:
        name: Name for the circuit breaker
        **circuit_kwargs: Additional arguments for CircuitBreaker constructor
        
    Returns:
        Decorated function that respects circuit breaker state
    """
    def decorator(func: F) -> F:
        # Ensure the circuit breaker exists
        if name not in _circuit_breakers:
            CircuitBreaker(name, **circuit_kwargs)
        
        circuit = _circuit_breakers[name]
        
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not circuit.allow_request():
                raise CircuitOpenError(f"Circuit '{name}' is OPEN - failing fast", circuit)
            
            try:
                result = await func(*args, **kwargs)
                circuit.record_success()
                return result
            except Exception as e:
                circuit.record_failure(e)
                raise
        
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not circuit.allow_request():
                raise CircuitOpenError(f"Circuit '{name}' is OPEN - failing fast", circuit)
            
            try:
                result = func(*args, **kwargs)
                circuit.record_success()
                return result
            except Exception as e:
                circuit.record_failure(e)
                raise
        
        if inspect.iscoroutinefunction(func):
            return cast(F, async_wrapper)
        return cast(F, sync_wrapper)
    
    return decorator