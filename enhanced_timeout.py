"""
Enhanced timeout and resilience manager for Vinkeljernet.

Provides adaptive timeout management, progressive fallback, and
resilience patterns for API calls with statistics.
"""

import asyncio
import functools
import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

# Typing variables
T = TypeVar('T')
AsyncFunc = Callable[..., Any]

# Configure logging
logger = logging.getLogger("vinkeljernet.enhanced_timeout")

class TimeoutStrategy(Enum):
    """Strategies for timeout handling."""
    FIXED = "fixed"  # Use a fixed timeout
    ADAPTIVE = "adaptive"  # Adjust timeout based on recent performance
    PROGRESSIVE = "progressive"  # Start low and increase if needed


class FallbackStrategy(Enum):
    """Strategies for handling timeouts and failures."""
    NONE = "none"  # No fallback, just fail
    RETRY = "retry"  # Retry the request
    CACHED = "cached"  # Use cached data if available
    DEGRADED = "degraded"  # Use a degraded/simplified response
    SYNTHETIC = "synthetic"  # Generate a synthetic response


@dataclass
class TimeoutStats:
    """Statistics for timeout management."""
    total_calls: int = 0
    successful_calls: int = 0
    timed_out_calls: int = 0
    avg_response_time: float = 0
    p95_response_time: float = 0
    recent_timeouts: List[datetime] = None
    recent_response_times: List[float] = None
    fallback_used_count: Dict[FallbackStrategy, int] = None
    
    def __post_init__(self):
        """Initialize mutable default fields."""
        if self.recent_timeouts is None:
            self.recent_timeouts = []
        if self.recent_response_times is None:
            self.recent_response_times = []
        if self.fallback_used_count is None:
            self.fallback_used_count = {strategy: 0 for strategy in FallbackStrategy}
    
    def record_success(self, response_time: float) -> None:
        """Record a successful call."""
        self.total_calls += 1
        self.successful_calls += 1
        
        # Track response time
        self.recent_response_times.append(response_time)
        if len(self.recent_response_times) > 100:
            self.recent_response_times.pop(0)
            
        # Update average
        if self.recent_response_times:
            self.avg_response_time = sum(self.recent_response_times) / len(self.recent_response_times)
            
            # Update p95 (95th percentile)
            sorted_times = sorted(self.recent_response_times)
            p95_index = int(0.95 * len(sorted_times))
            self.p95_response_time = sorted_times[p95_index]
    
    def record_timeout(self) -> None:
        """Record a timeout."""
        self.total_calls += 1
        self.timed_out_calls += 1
        self.recent_timeouts.append(datetime.now())
        
        # Only keep last 20 timeouts
        if len(self.recent_timeouts) > 20:
            self.recent_timeouts.pop(0)
    
    def record_fallback_used(self, strategy: FallbackStrategy) -> None:
        """Record that a fallback strategy was used."""
        self.fallback_used_count[strategy] += 1
    
    @property
    def timeout_rate(self) -> float:
        """Calculate the timeout rate."""
        if self.total_calls == 0:
            return 0
        return (self.timed_out_calls / self.total_calls) * 100
    
    @property
    def success_rate(self) -> float:
        """Calculate the success rate."""
        if self.total_calls == 0:
            return 100
        return (self.successful_calls / self.total_calls) * 100
    
    @property
    def recent_timeout_rate(self) -> float:
        """Calculate the recent timeout rate (last 1 hour)."""
        recent_time = datetime.now() - timedelta(hours=1)
        recent_timeouts = [t for t in self.recent_timeouts if t > recent_time]
        
        if self.total_calls == 0:
            return 0
        
        # Estimate total calls in last hour
        recent_calls_estimate = max(10, int(self.total_calls * 0.1))  # at least 10 or 10% of total
        
        return (len(recent_timeouts) / recent_calls_estimate) * 100


# Global registry of timeout stats by function/service
timeout_stats: Dict[str, TimeoutStats] = {}


def get_adaptive_timeout(func_name: str, base_timeout: float) -> float:
    """
    Calculate an adaptive timeout based on recent performance.
    
    Args:
        func_name: Function or service name
        base_timeout: Base timeout value
        
    Returns:
        Adapted timeout value
    """
    if func_name not in timeout_stats:
        return base_timeout
        
    stats = timeout_stats[func_name]
    
    # If no response times yet, return base timeout
    if not stats.recent_response_times:
        return base_timeout
    
    # Calculate adaptive timeout based on recent p95 response time
    p95_time = stats.p95_response_time
    
    # Add margin (50% more than p95)
    adaptive_timeout = p95_time * 1.5
    
    # Ensure timeout is reasonable (not too short or too long)
    min_timeout = base_timeout * 0.5  # Not less than half the base
    max_timeout = base_timeout * 2.0  # Not more than double the base
    
    return max(min_timeout, min(adaptive_timeout, max_timeout))


async def with_progressive_timeout(
    coroutine,
    timeout: float,
    fallback_value: Any = None,
    retry_count: int = 1,
    func_name: str = "unknown"
) -> Any:
    """
    Execute a coroutine with progressively increasing timeouts and fallbacks.
    
    Args:
        coroutine: Coroutine to execute
        timeout: Initial timeout in seconds
        fallback_value: Value to return if all attempts fail
        retry_count: Number of retry attempts
        func_name: Function name for stats tracking
        
    Returns:
        Result of coroutine or fallback value
    """
    # Initialize stats if needed
    if func_name not in timeout_stats:
        timeout_stats[func_name] = TimeoutStats()
    
    stats = timeout_stats[func_name]
    
    # Initial timeout
    current_timeout = timeout
    
    for attempt in range(retry_count + 1):  # +1 for initial attempt
        try:
            start_time = time.time()
            
            if attempt > 0:
                # Increase timeout for retries (add 50% each time)
                current_timeout *= 1.5
                logger.info(f"Retry {attempt} for {func_name} with timeout {current_timeout:.2f}s")
            
            # Execute with timeout
            result = await asyncio.wait_for(coroutine, timeout=current_timeout)
            
            # Record success
            end_time = time.time()
            response_time = end_time - start_time
            stats.record_success(response_time)
            
            return result
            
        except asyncio.TimeoutError:
            stats.record_timeout()
            
            # If this was the last attempt, use fallback
            if attempt == retry_count:
                logger.warning(f"All {retry_count+1} attempts timed out for {func_name}")
                stats.record_fallback_used(FallbackStrategy.DEGRADED)
                return fallback_value
            
            # Otherwise continue to next attempt with increased timeout
            logger.warning(f"Timeout occurred for {func_name} (attempt {attempt+1}/{retry_count+1})")
            
        except Exception as e:
            logger.error(f"Error in {func_name}: {e}")
            stats.record_fallback_used(FallbackStrategy.DEGRADED)
            return fallback_value


def adaptive_timeout(
    base_timeout: float = 30.0,
    retry_count: int = 1,
    fallback_value: Any = None,
    strategy: TimeoutStrategy = TimeoutStrategy.ADAPTIVE
) -> Callable:
    """
    Decorator for functions with adaptive timeout handling.
    
    Args:
        base_timeout: Base timeout in seconds
        retry_count: Number of retry attempts
        fallback_value: Value to return if all attempts fail
        strategy: Timeout strategy to use
        
    Returns:
        Decorated function
    """
    def decorator(func: AsyncFunc) -> AsyncFunc:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            func_name = func.__qualname__
            
            # Determine actual timeout based on strategy
            if strategy == TimeoutStrategy.ADAPTIVE:
                timeout = get_adaptive_timeout(func_name, base_timeout)
            elif strategy == TimeoutStrategy.PROGRESSIVE:
                # Start with a shorter timeout that will increase on retries
                timeout = base_timeout * 0.7
            else:
                # Fixed timeout
                timeout = base_timeout
            
            return await with_progressive_timeout(
                func(*args, **kwargs),
                timeout=timeout,
                fallback_value=fallback_value,
                retry_count=retry_count,
                func_name=func_name
            )
            
        return wrapper
    
    return decorator


def get_timeout_stats() -> Dict[str, Dict[str, Any]]:
    """
    Get timeout statistics for all tracked functions.
    
    Returns:
        Dict mapping function names to stat dictionaries
    """
    result = {}
    
    for func_name, stats in timeout_stats.items():
        result[func_name] = {
            "total_calls": stats.total_calls,
            "success_rate": f"{stats.success_rate:.1f}%",
            "timeout_rate": f"{stats.timeout_rate:.1f}%",
            "recent_timeout_rate": f"{stats.recent_timeout_rate:.1f}%",
            "avg_response_time": f"{stats.avg_response_time:.2f}s",
            "p95_response_time": f"{stats.p95_response_time:.2f}s",
            "fallbacks_used": {
                strategy.name: count
                for strategy, count in stats.fallback_used_count.items()
                if count > 0
            }
        }
    
    return result


class parallel_tasks:
    """Context manager for executing multiple tasks in parallel with resilience."""
    
    def __init__(self, base_timeout: float = 30.0, collect_results: bool = True):
        """
        Initialize parallel tasks context.
        
        Args:
            base_timeout: Base timeout for all tasks
            collect_results: Whether to collect and return results
        """
        self.base_timeout = base_timeout
        self.collect_results = collect_results
        self.tasks = []
        self.task_names = []
        self.fallbacks = {}
        self.timeouts = {}
        self.results = {}
    
    def add_task(
        self, 
        coro, 
        name: str, 
        timeout: Optional[float] = None, 
        fallback: Any = None
    ) -> None:
        """
        Add a task to be executed in parallel.
        
        Args:
            coro: Coroutine to execute
            name: Task name for identification
            timeout: Custom timeout (uses base_timeout if None)
            fallback: Fallback value if task fails
        """
        self.tasks.append(coro)
        self.task_names.append(name)
        self.fallbacks[name] = fallback
        self.timeouts[name] = timeout or self.base_timeout
    
    async def __aenter__(self):
        """Enter context manager."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and execute all tasks."""
        if not self.tasks:
            return
        
        # Create Tasks
        await_tasks = []
        for i, task_coro in enumerate(self.tasks):
            name = self.task_names[i]
            timeout = self.timeouts[name]
            fallback = self.fallbacks[name]
            
            # Wrap in timeout handler
            wrapped_task = with_progressive_timeout(
                task_coro,
                timeout=timeout,
                fallback_value=fallback,
                retry_count=1,
                func_name=name
            )
            
            await_tasks.append(asyncio.create_task(wrapped_task))
        
        # Wait for all tasks to complete
        if self.collect_results:
            results = await asyncio.gather(*await_tasks, return_exceptions=True)
            
            # Store results by task name
            for i, result in enumerate(results):
                name = self.task_names[i]
                self.results[name] = result
        else:
            # Just execute tasks without collecting results
            await asyncio.gather(*await_tasks, return_exceptions=True)
    
    def get_results(self) -> Dict[str, Any]:
        """Get results of completed tasks by name."""
        return self.results