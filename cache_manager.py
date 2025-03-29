"""
Cache manager module for Vinkeljernet project.

This module provides functionality to cache API responses to disk,
with configurable time-to-live (TTL) and cache invalidation options.
"""

import os
import json
import time
import hashlib
import functools
import asyncio
import inspect
from typing import Any, Callable, Dict, Optional, Union, TypeVar
from pathlib import Path

# Type variables for function annotations
T = TypeVar('T')
AsyncFunc = Callable[..., Any]
SyncFunc = Callable[..., Any]

# Cache configuration
DEFAULT_CACHE_DIR = Path(os.path.expanduser("~/.vinkeljernet/cache"))
DEFAULT_TTL = 3600  # 1 hour in seconds
CACHE_VERSION = 1  # Increment this when cache format changes

# Ensure cache directory exists
DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """
    Generate a unique cache key based on function name and arguments.
    
    Args:
        func_name: Name of the function being cached
        args: Positional arguments to the function
        kwargs: Keyword arguments to the function
        
    Returns:
        str: A hash string that uniquely identifies this function call
    """
    # Convert args and kwargs to a string representation
    args_str = str(args) if args else ""
    kwargs_str = str(sorted(kwargs.items())) if kwargs else ""
    
    # Create a hash of the function name and arguments
    key_data = f"{func_name}:{args_str}:{kwargs_str}".encode('utf-8')
    return hashlib.md5(key_data).hexdigest()


def save_to_cache(cache_key: str, data: Any, ttl: int, cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
    """
    Save data to the cache file.
    
    Args:
        cache_key: Unique key for this cache entry
        data: Data to be cached (must be JSON serializable)
        ttl: Time-to-live in seconds
        cache_dir: Directory to store cache files
    """
    cache_file = cache_dir / f"{cache_key}.json"
    
    cache_data = {
        "timestamp": time.time(),
        "expires": time.time() + ttl,
        "version": CACHE_VERSION,
        "data": data
    }
    
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False)
    except Exception as e:
        print(f"Warning: Could not write to cache: {e}")


def load_from_cache(cache_key: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> Optional[Any]:
    """
    Load data from the cache if it exists and has not expired.
    
    Args:
        cache_key: Unique key for this cache entry
        cache_dir: Directory where cache files are stored
        
    Returns:
        Optional[Any]: Cached data if found and valid, None otherwise
    """
    cache_file = cache_dir / f"{cache_key}.json"
    
    if not cache_file.exists():
        return None
        
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
            
        # Check if cache entry has expired or version mismatch
        current_time = time.time()
        if cache_data["expires"] < current_time:
            # Cache expired, delete the file
            cache_file.unlink(missing_ok=True)
            return None
            
        if cache_data.get("version") != CACHE_VERSION:
            # Version mismatch, delete the file
            cache_file.unlink(missing_ok=True)
            return None
            
        return cache_data["data"]
    except Exception as e:
        print(f"Warning: Could not read from cache: {e}")
        return None


def clear_cache(cache_dir: Path = DEFAULT_CACHE_DIR) -> int:
    """
    Clear all cached data.
    
    Args:
        cache_dir: Directory where cache files are stored
        
    Returns:
        int: Number of cache files deleted
    """
    count = 0
    for cache_file in cache_dir.glob("*.json"):
        try:
            cache_file.unlink()
            count += 1
        except Exception:
            pass
    return count


def purge_expired_cache(cache_dir: Path = DEFAULT_CACHE_DIR) -> int:
    """
    Remove all expired cache entries.
    
    Args:
        cache_dir: Directory where cache files are stored
        
    Returns:
        int: Number of expired cache files deleted
    """
    current_time = time.time()
    count = 0
    
    for cache_file in cache_dir.glob("*.json"):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            if cache_data.get("expires", 0) < current_time:
                cache_file.unlink()
                count += 1
        except Exception:
            # If we can't read the file, delete it
            try:
                cache_file.unlink()
                count += 1
            except Exception:
                pass
                
    return count


def cached_api(ttl: int = DEFAULT_TTL, 
               cache_dir: Path = DEFAULT_CACHE_DIR) -> Callable:
    """
    Decorator to cache API responses.
    
    Args:
        ttl: Time-to-live in seconds (default: 1 hour)
        cache_dir: Directory to store cache files
        
    Returns:
        Callable: Decorated function with caching
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, bypass_cache=False, **kwargs):
            # Extract bypass_cache parameter and remove it from kwargs if present
            if "bypass_cache" in kwargs:
                bypass_cache = kwargs.pop("bypass_cache")
                
            # Generate cache key
            cache_key = get_cache_key(func.__name__, args, kwargs)
            
            # Check if we should use the cache
            if not bypass_cache:
                cached_result = load_from_cache(cache_key, cache_dir)
                if cached_result is not None:
                    print(f"Cache hit for {func.__name__}")
                    return cached_result
            
            # Cache miss or bypass requested, call the function
            result = await func(*args, **kwargs)
            
            # Save result to cache if it's not None
            if result is not None:
                save_to_cache(cache_key, result, ttl, cache_dir)
                
            return result
            
        @functools.wraps(func)
        def sync_wrapper(*args, bypass_cache=False, **kwargs):
            # Extract bypass_cache parameter and remove it from kwargs if present
            if "bypass_cache" in kwargs:
                bypass_cache = kwargs.pop("bypass_cache")
                
            # Generate cache key
            cache_key = get_cache_key(func.__name__, args, kwargs)
            
            # Check if we should use the cache
            if not bypass_cache:
                cached_result = load_from_cache(cache_key, cache_dir)
                if cached_result is not None:
                    print(f"Cache hit for {func.__name__}")
                    return cached_result
            
            # Cache miss or bypass requested, call the function
            result = func(*args, **kwargs)
            
            # Save result to cache if it's not None
            if result is not None:
                save_to_cache(cache_key, result, ttl, cache_dir)
                
            return result
        
        # Return appropriate wrapper based on whether the decorated function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator