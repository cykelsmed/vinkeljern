"""
Enhanced timeout and retry module for Vinkeljernet project.

This module provides advanced timeout handling and retry strategies 
for API requests, particularly for LLM calls that may take longer time.
"""

import time
import asyncio
import logging
import functools
from typing import Any, Callable, Optional, TypeVar, Dict, List, Tuple
from datetime import datetime
import aiohttp
import random

# Configure logging
logger = logging.getLogger("vinkeljernet.timeout")

# Type variables
T = TypeVar('T')
AsyncFunc = Callable[..., Any]

# Timeout configuration
class TimeoutConfig:
    """Configurable timeouts for different request types."""
    
    # Standard timeouts (in seconds)
    DEFAULT = 60
    KNOWLEDGE_QUERY = 40
    ANGLE_GENERATION = 90
    EXPERT_SOURCES = 30
    BACKGROUND_INFO = 25
    EDITORIAL_FEEDBACK = 20
    
    # Timeout factors for fallback strategies
    PRIMARY_FACTOR = 1.0     # Normal timeout
    EXTEND_FACTOR = 1.5      # Extended timeout for retry
    FALLBACK_FACTOR = 0.7    # Shorter timeout for fallback service
    URGENT_FACTOR = 0.5      # Shorter timeout for urgent requests
    
    @staticmethod
    def get_timeout(request_type: str, factor: float = 1.0) -> float:
        """
        Get timeout duration for request type.
        
        Args:
            request_type: Type of request 
            factor: Multiplier for standard timeout
            
        Returns:
            float: Timeout in seconds
        """
        timeouts = {
            "default": TimeoutConfig.DEFAULT,
            "knowledge_query": TimeoutConfig.KNOWLEDGE_QUERY,
            "angle_generation": TimeoutConfig.ANGLE_GENERATION,
            "expert_sources": TimeoutConfig.EXPERT_SOURCES,
            "background_info": TimeoutConfig.BACKGROUND_INFO,
            "editorial_feedback": TimeoutConfig.EDITORIAL_FEEDBACK
        }
        
        base_timeout = timeouts.get(request_type.lower(), TimeoutConfig.DEFAULT)
        return base_timeout * factor

class RetryStrategy:
    """Smart retry strategies for different types of failures."""
    
    # Maximum retry attempts
    MAX_RETRIES = 3
    
    # Backoff factors
    LINEAR_BACKOFF = "linear"     # Retry after n, 2n, 3n seconds
    EXPONENTIAL_BACKOFF = "expo"  # Retry after n, n^2, n^3 seconds
    FIBONACCI_BACKOFF = "fib"     # Retry after n, n, 2n, 3n, 5n seconds
    
    # Jitter settings
    JITTER_RANGE = 0.2  # ±20% randomness to avoid thundering herd
    
    @staticmethod
    def calculate_delay(attempt: int, base_delay: float, strategy: str) -> float:
        """
        Calculate delay before next retry.
        
        Args:
            attempt: Current attempt number (1-based)
            base_delay: Base delay in seconds
            strategy: Backoff strategy name
            
        Returns:
            float: Delay in seconds
        """
        # Get raw delay based on strategy
        if strategy == RetryStrategy.LINEAR_BACKOFF:
            raw_delay = base_delay * attempt
        elif strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            raw_delay = base_delay * (2 ** (attempt - 1))
        elif strategy == RetryStrategy.FIBONACCI_BACKOFF:
            # Calculate Fibonacci number for this attempt
            a, b = 1, 1
            for _ in range(attempt - 1):
                a, b = b, a + b
            raw_delay = base_delay * a
        else:
            # Default to linear
            raw_delay = base_delay * attempt
        
        # Add jitter (±JITTER_RANGE)
        jitter = 1.0 + random.uniform(-RetryStrategy.JITTER_RANGE, RetryStrategy.JITTER_RANGE)
        final_delay = raw_delay * jitter
        
        return final_delay

    @staticmethod
    def should_retry(error: Exception, attempt: int) -> bool:
        """
        Determine if a request should be retried.
        
        Args:
            error: The exception that occurred
            attempt: Current attempt number
            
        Returns:
            bool: True if should retry
        """
        # Never retry after max attempts
        if attempt >= RetryStrategy.MAX_RETRIES:
            return False
        
        # Check specific error types
        if isinstance(error, asyncio.TimeoutError):
            # Always retry timeouts
            return True
        elif isinstance(error, aiohttp.ClientError):
            # Retry network-related errors
            return True
        elif isinstance(error, aiohttp.ServerConnectionError):
            # Retry server connection errors
            return True
        elif isinstance(error, aiohttp.ClientResponseError):
            # Retry on certain status codes
            if hasattr(error, 'status'):
                # Retry on 429 (too many requests), 502/503/504 (server errors)
                return error.status in [429, 502, 503, 504]
            return False
        
        # Default to not retry
        return False

class ProgressTracker:
    """Track and report progress for multi-stage operations."""
    
    def __init__(self, total_steps: int = 100, callback: Optional[Callable] = None):
        """
        Initialize progress tracker.
        
        Args:
            total_steps: Total number of steps (default: 100 for percentage)
            callback: Optional callback function to report progress
        """
        self.total_steps = total_steps
        self.current = 0
        self.callback = callback
        self.start_time = time.time()
        self.last_update = self.start_time
        self.checkpoints = []  # List of (step, time) tuples
        
    async def update(self, step: int, message: str = "") -> None:
        """
        Update progress.
        
        Args:
            step: Current step (0-total_steps)
            message: Optional progress message
        """
        # Ensure step is within bounds
        self.current = max(0, min(step, self.total_steps))
        now = time.time()
        self.last_update = now
        
        # Record checkpoint
        self.checkpoints.append((self.current, now))
        
        # Call the callback if provided
        if self.callback and callable(self.callback):
            if asyncio.iscoroutinefunction(self.callback):
                await self.callback(self.current, self.total_steps, message)
            else:
                self.callback(self.current, self.total_steps, message)
    
    def get_progress_percentage(self) -> float:
        """Get progress as a percentage."""
        if self.total_steps <= 0:
            return 100.0
        return (self.current / self.total_steps) * 100.0
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time
    
    def estimate_remaining_time(self) -> Optional[float]:
        """
        Estimate remaining time based on progress.
        
        Returns:
            Optional[float]: Estimated seconds remaining or None if can't estimate
        """
        if not self.checkpoints or self.current <= 0 or self.current >= self.total_steps:
            return None
        
        # Use recent checkpoints for more accurate estimate
        recent_points = self.checkpoints[-min(5, len(self.checkpoints)):]
        if len(recent_points) < 2:
            return None
            
        # Calculate rate of progress (steps/second)
        first_step, first_time = recent_points[0]
        last_step, last_time = recent_points[-1]
        
        time_diff = last_time - first_time
        steps_diff = last_step - first_step
        
        if time_diff <= 0 or steps_diff <= 0:
            return None
            
        rate = steps_diff / time_diff
        
        # Estimate remaining time
        remaining_steps = self.total_steps - self.current
        if rate > 0:
            return remaining_steps / rate
        
        return None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with progress info."""
        remaining = self.estimate_remaining_time()
        
        return {
            "progress": self.get_progress_percentage(),
            "current_step": self.current,
            "total_steps": self.total_steps,
            "elapsed_seconds": self.get_elapsed_time(),
            "estimated_remaining_seconds": remaining,
            "estimated_completion_time": (
                datetime.fromtimestamp(time.time() + remaining).isoformat() 
                if remaining is not None else None
            )
        }

async def with_timeout(
    func: AsyncFunc,
    *args: Any,
    timeout: float,
    fallback_result: Any = None,
    timeout_handler: Optional[Callable] = None,
    **kwargs: Any
) -> Any:
    """
    Execute an async function with timeout and fallback.
    
    Args:
        func: Async function to execute
        *args: Arguments to pass to the function
        timeout: Timeout in seconds
        fallback_result: Result to return on timeout
        timeout_handler: Optional function to call on timeout
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Any: Function result or fallback
    """
    try:
        # Run function with timeout
        return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"Function {func.__name__} timed out after {timeout} seconds")
        
        # Call timeout handler if provided
        if timeout_handler:
            if asyncio.iscoroutinefunction(timeout_handler):
                await timeout_handler(func.__name__, timeout)
            else:
                timeout_handler(func.__name__, timeout)
        
        return fallback_result

async def with_retry(
    func: AsyncFunc,
    *args: Any,
    max_retries: int = RetryStrategy.MAX_RETRIES,
    base_delay: float = 1.0,
    backoff_strategy: str = RetryStrategy.EXPONENTIAL_BACKOFF,
    **kwargs: Any
) -> Any:
    """
    Execute an async function with retry capability.
    
    Args:
        func: Async function to execute
        *args: Arguments to pass to the function
        max_retries: Maximum retry attempts
        base_delay: Base delay between retries in seconds
        backoff_strategy: Strategy for calculating delays
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Any: Function result
        
    Raises:
        Exception: Last error if all retries fail
    """
    attempt = 0
    last_error = None
    
    while attempt < max_retries:
        attempt += 1
        
        try:
            if attempt == 1:
                # First attempt
                logger.debug(f"Executing {func.__name__}")
            else:
                # Retry
                logger.info(f"Retry {attempt}/{max_retries} for {func.__name__}")
                
            # Execute the function
            result = await func(*args, **kwargs)
            
            # If successful, return the result
            return result
            
        except Exception as e:
            last_error = e
            
            # Check if we should retry
            if not RetryStrategy.should_retry(e, attempt):
                logger.warning(f"Not retrying {func.__name__} after error: {e}")
                raise
                
            # Calculate delay before retry
            if attempt < max_retries:
                delay = RetryStrategy.calculate_delay(
                    attempt, base_delay, backoff_strategy
                )
                
                logger.info(f"Will retry {func.__name__} in {delay:.2f}s after error: {e}")
                await asyncio.sleep(delay)
    
    # If we get here, we've exhausted retries
    logger.error(f"All {max_retries} retries failed for {func.__name__}")
    if last_error:
        raise last_error
    
    raise RuntimeError(f"All {max_retries} retries failed for {func.__name__}")

def with_timeout_decorator(
    timeout: float,
    fallback_result: Any = None,
    timeout_handler: Optional[Callable] = None
) -> Callable:
    """
    Decorator to add timeout to an async function.
    
    Args:
        timeout: Timeout in seconds
        fallback_result: Result to return on timeout
        timeout_handler: Optional function to call on timeout
        
    Returns:
        Callable: Decorated function
    """
    def decorator(func: AsyncFunc) -> AsyncFunc:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await with_timeout(
                func, 
                *args, 
                timeout=timeout, 
                fallback_result=fallback_result, 
                timeout_handler=timeout_handler,
                **kwargs
            )
        return wrapper
    return decorator

def with_retry_decorator(
    max_retries: int = RetryStrategy.MAX_RETRIES,
    base_delay: float = 1.0,
    backoff_strategy: str = RetryStrategy.EXPONENTIAL_BACKOFF
) -> Callable:
    """
    Decorator to add retry capability to an async function.
    
    Args:
        max_retries: Maximum retry attempts
        base_delay: Base delay between retries in seconds
        backoff_strategy: Strategy for calculating delays
        
    Returns:
        Callable: Decorated function
    """
    def decorator(func: AsyncFunc) -> AsyncFunc:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await with_retry(
                func,
                *args,
                max_retries=max_retries,
                base_delay=base_delay,
                backoff_strategy=backoff_strategy,
                **kwargs
            )
        return wrapper
    return decorator

class AdaptiveTimeout:
    """
    Adaptive timeout manager that adjusts based on request history.
    
    This class learns from past API call durations and adjusts timeouts
    dynamically to optimize performance while minimizing timeouts.
    """
    
    def __init__(self, 
                 base_timeout: float = TimeoutConfig.DEFAULT,
                 min_timeout: float = 10.0,
                 max_timeout: float = 120.0,
                 history_size: int = 50):
        """
        Initialize adaptive timeout.
        
        Args:
            base_timeout: Initial timeout in seconds
            min_timeout: Minimum timeout allowed
            max_timeout: Maximum timeout allowed
            history_size: Number of past calls to track
        """
        self.base_timeout = base_timeout
        self.min_timeout = min_timeout
        self.max_timeout = max_timeout
        self.history_size = history_size
        
        # Call history: list of (duration, success) tuples
        self.history: List[Tuple[float, bool]] = []
        
        # Current timeout
        self.current_timeout = base_timeout
    
    def record_call(self, duration: float, success: bool) -> None:
        """
        Record a call result.
        
        Args:
            duration: Call duration in seconds
            success: Whether call succeeded
        """
        # Add to history
        self.history.append((duration, success))
        
        # Trim history
        if len(self.history) > self.history_size:
            self.history = self.history[-self.history_size:]
            
        # Recalculate timeout
        self._update_timeout()
    
    def _update_timeout(self) -> None:
        """Update timeout based on call history."""
        if not self.history:
            return
            
        # Get successful calls
        successful_calls = [(d, s) for d, s in self.history if s]
        
        if not successful_calls:
            # No successful calls, increase timeout
            self.current_timeout = min(self.current_timeout * 1.2, self.max_timeout)
            return
            
        # Calculate p95 of successful call durations
        durations = sorted([d for d, _ in successful_calls])
        p95_index = int(len(durations) * 0.95)
        p95_duration = durations[min(p95_index, len(durations) - 1)]
        
        # Failed calls proportion
        failed_proportion = 1 - (len(successful_calls) / len(self.history))
        
        # Calculate new timeout
        # Start with p95 and add margin based on failure rate
        margin_factor = 1.2 + (failed_proportion * 0.8)  # 1.2x to 2.0x
        new_timeout = p95_duration * margin_factor
        
        # Ensure timeout is within bounds
        self.current_timeout = max(self.min_timeout, min(new_timeout, self.max_timeout))
    
    def get_timeout(self) -> float:
        """Get current timeout value."""
        return self.current_timeout
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        successful_calls = [(d, s) for d, s in self.history if s]
        failed_calls = [(d, s) for d, s in self.history if not s]
        
        avg_duration = (
            sum(d for d, _ in successful_calls) / len(successful_calls) 
            if successful_calls else 0
        )
        
        return {
            "current_timeout": self.current_timeout,
            "base_timeout": self.base_timeout,
            "min_timeout": self.min_timeout,
            "max_timeout": self.max_timeout,
            "history_size": len(self.history),
            "success_rate": len(successful_calls) / len(self.history) if self.history else 0,
            "average_duration": avg_duration,
            "timeout_multiplier": self.current_timeout / (avg_duration if avg_duration > 0 else self.base_timeout)
        }