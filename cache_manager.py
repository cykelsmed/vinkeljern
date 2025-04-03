"""
Cache manager module for Vinkeljernet project.

This module provides functionality to cache API responses with configurable
time-to-live (TTL), compression, and improved memory efficiency.
"""

import os
import json
import time
import hashlib
import functools
import asyncio
import inspect
import logging
import lzma
import pickle
import shutil
from typing import Any, Callable, Dict, Optional, Union, TypeVar, List, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta

# Type variables for function annotations
T = TypeVar('T')
AsyncFunc = Callable[..., Any]
SyncFunc = Callable[..., Any]

# Configure logging
logger = logging.getLogger("vinkeljernet.cache")

# Cache configuration
DEFAULT_CACHE_DIR = Path(os.path.expanduser("~/.vinkeljernet/cache"))
DEFAULT_TTL = 3600  # 1 hour in seconds
CACHE_VERSION = 2  # Increment this when cache format changes
MEMORY_CACHE_SIZE = 100  # Maximum items in memory cache
COMPRESSION_THRESHOLD = 10240  # Bytes (10KB) - compress if larger than this
MAX_CACHE_SIZE_MB = 500  # Maximum cache size in MB

# Ensure cache directory exists
DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# In-memory LRU cache
memory_cache: Dict[str, Tuple[float, Any]] = {}
memory_cache_keys: List[str] = []

@dataclass
class CacheStats:
    """Statistics for cache usage."""
    hits: int = 0
    misses: int = 0
    disk_hits: int = 0
    memory_hits: int = 0
    write_count: int = 0
    compression_count: int = 0
    compressed_bytes_saved: int = 0
    last_cleanup: Optional[datetime] = None
    
# Global stats object
stats = CacheStats()

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
    
    # Sort kwargs for consistent ordering
    kwargs_str = str(sorted(kwargs.items())) if kwargs else ""
    
    # Create a hash of the function name and arguments
    key_data = f"{func_name}:{args_str}:{kwargs_str}".encode('utf-8')
    return hashlib.md5(key_data).hexdigest()

def _is_json_serializable(data: Any) -> bool:
    """Check if data can be serialized to JSON."""
    try:
        json.dumps(data)
        return True
    except (TypeError, OverflowError):
        return False

def _compress_data(data: Any) -> Tuple[bytes, bool]:
    """
    Compress data if it's large enough to benefit from compression.
    
    Returns:
        Tuple of (compressed_data, is_compressed)
    """
    # First pickle the data, then check size
    pickled_data = pickle.dumps(data)
    
    if len(pickled_data) > COMPRESSION_THRESHOLD:
        # Compress the pickled data
        compressed_data = lzma.compress(pickled_data)
        
        # Update stats
        stats.compression_count += 1
        stats.compressed_bytes_saved += len(pickled_data) - len(compressed_data)
        
        return compressed_data, True
    
    return pickled_data, False

def _decompress_data(data: bytes, is_compressed: bool) -> Any:
    """
    Decompress data if it was compressed.
    """
    if is_compressed:
        decompressed_data = lzma.decompress(data)
        return pickle.loads(decompressed_data)
    
    return pickle.loads(data)

def save_to_cache(cache_key: str, data: Any, ttl: int, cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
    """
    Save data to the cache (both in-memory and on disk).
    
    Args:
        cache_key: Unique key for this cache entry
        data: Data to be cached
        ttl: Time-to-live in seconds
        cache_dir: Directory to store cache files
    """
    # Update stats
    stats.write_count += 1
    
    # Save to memory cache
    memory_cache[cache_key] = (time.time() + ttl, data)
    
    # Add to keys list and maintain LRU order
    if cache_key in memory_cache_keys:
        memory_cache_keys.remove(cache_key)
    memory_cache_keys.append(cache_key)
    
    # Trim memory cache if needed
    if len(memory_cache_keys) > MEMORY_CACHE_SIZE:
        oldest_key = memory_cache_keys.pop(0)
        if oldest_key in memory_cache:
            del memory_cache[oldest_key]
    
    # Process for disk storage
    try:
        # Determine if data needs special handling
        compressed_data, is_compressed = _compress_data(data)
        
        # Create cache entry
        cache_entry = {
            "timestamp": time.time(),
            "expires": time.time() + ttl,
            "version": CACHE_VERSION,
            "compressed": is_compressed,
            # Store binary data in serialized form
            "data_binary": True
        }
        
        cache_file = cache_dir / f"{cache_key}.cache"
        
        # Write cache entry metadata
        with open(cache_file, 'wb') as f:
            # First write the metadata as JSON
            metadata = json.dumps(cache_entry).encode('utf-8')
            f.write(len(metadata).to_bytes(4, byteorder='little'))  # Store metadata length
            f.write(metadata)
            # Then write the binary data
            f.write(compressed_data)
            
    except Exception as e:
        logger.warning(f"Could not write to disk cache: {e}")
        
    # Periodically check cache size (every 20 writes)
    if stats.write_count % 20 == 0:
        asyncio.create_task(_check_cache_size(cache_dir))

async def _check_cache_size(cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
    """
    Asynchronously check and manage cache size.
    
    This runs in the background to avoid blocking the main thread.
    """
    try:
        total_size = 0
        cache_files = []
        
        # Get list of cache files with their size and modification time
        for cache_file in cache_dir.glob("*.cache"):
            try:
                stats = cache_file.stat()
                total_size += stats.st_size
                cache_files.append((cache_file, stats.st_mtime, stats.st_size))
            except (FileNotFoundError, PermissionError):
                continue
                
        # Convert to MB
        total_size_mb = total_size / (1024 * 1024)
        
        if total_size_mb > MAX_CACHE_SIZE_MB:
            # Sort cache files by modification time (oldest first)
            cache_files.sort(key=lambda x: x[1])
            
            # Delete oldest files until we're under the limit
            size_to_remove = total_size - (MAX_CACHE_SIZE_MB * 0.9 * 1024 * 1024)  # Target 90% of max
            removed_size = 0
            
            for file_path, mtime, size in cache_files:
                try:
                    file_path.unlink()
                    removed_size += size
                    if removed_size > size_to_remove:
                        break
                except (FileNotFoundError, PermissionError):
                    continue
                    
            logger.info(f"Cache cleanup: removed {removed_size/(1024*1024):.2f}MB of data")
            stats.last_cleanup = datetime.now()
            
    except Exception as e:
        logger.error(f"Error during cache size check: {e}")

def load_from_cache(cache_key: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> Optional[Any]:
    """
    Load data from the cache (checks memory first, then disk).
    
    Args:
        cache_key: Unique key for this cache entry
        cache_dir: Directory where cache files are stored
        
    Returns:
        Optional[Any]: Cached data if found and valid, None otherwise
    """
    current_time = time.time()
    
    # Check memory cache first
    if cache_key in memory_cache:
        expires, data = memory_cache[cache_key]
        
        if expires > current_time:
            # Memory cache hit - update LRU order
            if cache_key in memory_cache_keys:
                memory_cache_keys.remove(cache_key)
            memory_cache_keys.append(cache_key)
            
            # Update stats
            stats.hits += 1
            stats.memory_hits += 1
            
            return data
        else:
            # Expired - remove from memory cache
            del memory_cache[cache_key]
            if cache_key in memory_cache_keys:
                memory_cache_keys.remove(cache_key)
    
    # Memory cache miss, check disk cache
    cache_file = cache_dir / f"{cache_key}.cache"
    
    if not cache_file.exists():
        stats.misses += 1
        return None
        
    try:
        with open(cache_file, 'rb') as f:
            # Read metadata length
            metadata_length_bytes = f.read(4)
            if not metadata_length_bytes:
                raise ValueError("Invalid cache file format")
                
            metadata_length = int.from_bytes(metadata_length_bytes, byteorder='little')
            
            # Read metadata
            metadata_bytes = f.read(metadata_length)
            if not metadata_bytes:
                raise ValueError("Invalid cache file format")
                
            cache_entry = json.loads(metadata_bytes.decode('utf-8'))
            
            # Check expiration and version
            if cache_entry["expires"] < current_time:
                # Cache expired, delete the file
                cache_file.unlink(missing_ok=True)
                stats.misses += 1
                return None
                
            if cache_entry.get("version") != CACHE_VERSION:
                # Version mismatch, delete the file
                cache_file.unlink(missing_ok=True)
                stats.misses += 1
                return None
            
            # Read binary data
            compressed_data = f.read()
            
            # Decompress and load data
            data = _decompress_data(compressed_data, cache_entry.get("compressed", False))
            
            # Store in memory cache for faster future access
            memory_cache[cache_key] = (cache_entry["expires"], data)
            if cache_key in memory_cache_keys:
                memory_cache_keys.remove(cache_key)
            memory_cache_keys.append(cache_key)
            
            # Trim memory cache if needed
            if len(memory_cache_keys) > MEMORY_CACHE_SIZE:
                oldest_key = memory_cache_keys.pop(0)
                if oldest_key in memory_cache:
                    del memory_cache[oldest_key]
            
            # Update stats
            stats.hits += 1
            stats.disk_hits += 1
            
            return data
    except Exception as e:
        logger.warning(f"Could not read from cache: {e}")
        try:
            # Delete corrupted cache file
            cache_file.unlink(missing_ok=True)
        except:
            pass
        
        stats.misses += 1
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
    
    # Clear memory cache
    memory_cache.clear()
    memory_cache_keys.clear()
    
    # Clear disk cache
    for cache_file in cache_dir.glob("*.cache"):
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
    
    # Clear expired memory cache entries
    expired_keys = []
    for key, (expires, _) in memory_cache.items():
        if expires < current_time:
            expired_keys.append(key)
    
    for key in expired_keys:
        del memory_cache[key]
        if key in memory_cache_keys:
            memory_cache_keys.remove(key)
    
    # Clear expired disk cache entries
    for cache_file in cache_dir.glob("*.cache"):
        try:
            with open(cache_file, 'rb') as f:
                # Read metadata length
                metadata_length_bytes = f.read(4)
                if not metadata_length_bytes:
                    cache_file.unlink()
                    count += 1
                    continue
                    
                metadata_length = int.from_bytes(metadata_length_bytes, byteorder='little')
                
                # Read metadata
                metadata_bytes = f.read(metadata_length)
                if not metadata_bytes:
                    cache_file.unlink()
                    count += 1
                    continue
                    
                cache_entry = json.loads(metadata_bytes.decode('utf-8'))
                
                if cache_entry.get("expires", 0) < current_time:
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

def get_cache_stats() -> Dict[str, Any]:
    """
    Get statistics about the cache usage.
    
    Returns:
        Dict with cache statistics
    """
    # Calculate hit rate
    total_requests = stats.hits + stats.misses
    hit_rate = (stats.hits / total_requests) * 100 if total_requests > 0 else 0
    
    # Calculate memory hit percentage of total hits
    memory_hit_rate = (stats.memory_hits / stats.hits) * 100 if stats.hits > 0 else 0
    
    # Get disk cache size
    disk_cache_size = 0
    try:
        for cache_file in DEFAULT_CACHE_DIR.glob("*.cache"):
            disk_cache_size += cache_file.stat().st_size
    except:
        pass
    
    # Convert to MB
    disk_cache_size_mb = disk_cache_size / (1024 * 1024)
    
    # Format compression savings
    compression_saved_mb = stats.compressed_bytes_saved / (1024 * 1024)
    
    return {
        "hit_rate": f"{hit_rate:.1f}%",
        "hits": stats.hits,
        "misses": stats.misses,
        "memory_hits": stats.memory_hits,
        "disk_hits": stats.disk_hits,
        "memory_hit_rate": f"{memory_hit_rate:.1f}%",
        "memory_cache_size": len(memory_cache),
        "memory_cache_limit": MEMORY_CACHE_SIZE,
        "disk_cache_size": f"{disk_cache_size_mb:.2f} MB",
        "max_disk_cache_size": f"{MAX_CACHE_SIZE_MB} MB",
        "cache_writes": stats.write_count,
        "compression_count": stats.compression_count,
        "compression_saved": f"{compression_saved_mb:.2f} MB",
        "last_cleanup": stats.last_cleanup.isoformat() if stats.last_cleanup else None
    }

def optimize_cache(cache_dir: Path = DEFAULT_CACHE_DIR) -> Dict[str, Any]:
    """
    Optimize the cache by recompressing files and removing orphaned data.
    
    Returns:
        Dict with optimization statistics
    """
    files_processed = 0
    files_recompressed = 0
    files_removed = 0
    bytes_saved = 0
    
    # Process all cache files
    for cache_file in cache_dir.glob("*.cache"):
        try:
            files_processed += 1
            
            with open(cache_file, 'rb') as f:
                # Read metadata length
                metadata_length_bytes = f.read(4)
                if not metadata_length_bytes:
                    cache_file.unlink()
                    files_removed += 1
                    continue
                    
                metadata_length = int.from_bytes(metadata_length_bytes, byteorder='little')
                
                # Read metadata
                metadata_bytes = f.read(metadata_length)
                if not metadata_bytes:
                    cache_file.unlink()
                    files_removed += 1
                    continue
                    
                cache_entry = json.loads(metadata_bytes.decode('utf-8'))
                
                # Check expiration and version
                current_time = time.time()
                if cache_entry.get("expires", 0) < current_time:
                    cache_file.unlink()
                    files_removed += 1
                    continue
                    
                # Skip already optimized/compressed files with current version
                if (cache_entry.get("version") == CACHE_VERSION and 
                    cache_entry.get("compressed", False) and
                    cache_entry.get("data_binary", False)):
                    continue
                
                # Read old data
                old_data = f.read()
                
                # Handle old format data
                if not cache_entry.get("data_binary", False):
                    # Old format stored JSON data directly
                    try:
                        if cache_entry.get("compressed", False):
                            # Old compressed format
                            data = _decompress_data(old_data, True)
                        else:
                            # Old JSON format
                            data = json.loads(old_data.decode('utf-8'))
                    except:
                        # If we can't load the data, delete the file
                        cache_file.unlink()
                        files_removed += 1
                        continue
                else:
                    # New format with binary data
                    try:
                        data = _decompress_data(old_data, cache_entry.get("compressed", False))
                    except:
                        # If we can't load the data, delete the file
                        cache_file.unlink()
                        files_removed += 1
                        continue
                
                # Reprocess the data with current format and compression
                old_size = cache_file.stat().st_size
                compressed_data, is_compressed = _compress_data(data)
                
                # Create new cache entry
                new_cache_entry = {
                    "timestamp": cache_entry["timestamp"],
                    "expires": cache_entry["expires"],
                    "version": CACHE_VERSION,
                    "compressed": is_compressed,
                    "data_binary": True
                }
                
                # Write new optimized cache file
                with open(str(cache_file) + ".tmp", 'wb') as f_new:
                    metadata = json.dumps(new_cache_entry).encode('utf-8')
                    f_new.write(len(metadata).to_bytes(4, byteorder='little'))
                    f_new.write(metadata)
                    f_new.write(compressed_data)
                
                # Replace old file with new one
                os.replace(str(cache_file) + ".tmp", str(cache_file))
                
                # Calculate bytes saved
                new_size = cache_file.stat().st_size
                size_diff = old_size - new_size
                
                if size_diff > 0:
                    bytes_saved += size_diff
                    files_recompressed += 1
                    
        except Exception as e:
            logger.warning(f"Error optimizing cache file {cache_file}: {e}")
            try:
                # Clean up temp file if it exists
                if os.path.exists(str(cache_file) + ".tmp"):
                    os.unlink(str(cache_file) + ".tmp")
            except:
                pass
    
    return {
        "files_processed": files_processed,
        "files_recompressed": files_recompressed,
        "files_removed": files_removed,
        "bytes_saved": bytes_saved,
        "mb_saved": bytes_saved / (1024 * 1024)
    }

def cached_api(ttl: int = DEFAULT_TTL,
               cache_dir: Path = DEFAULT_CACHE_DIR,
               key_prefix: str = "") -> Callable:
    """
    Decorator to cache API responses with improved efficiency.
    
    Args:
        ttl: Time-to-live in seconds (default: 1 hour)
        cache_dir: Directory to store cache files
        key_prefix: Optional prefix for cache keys (for namespacing)
        
    Returns:
        Callable: Decorated function with caching
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, bypass_cache=False, **kwargs):
            # Extract bypass_cache parameter and remove it from kwargs if present
            if "bypass_cache" in kwargs:
                bypass_cache = kwargs.pop("bypass_cache")
            
            # Remove any 'proxies' parameter from kwargs
            if "proxies" in kwargs:
                kwargs.pop("proxies")
                
            # Generate cache key (with optional prefix)
            cache_key = f"{key_prefix}_{get_cache_key(func.__name__, args, kwargs)}" if key_prefix else get_cache_key(func.__name__, args, kwargs)
            
            # Check if we should use the cache
            if not bypass_cache:
                cached_result = load_from_cache(cache_key, cache_dir)
                if cached_result is not None:
                    logger.info(f"Cache hit for {func.__name__}")
                    return cached_result
            
            # Cache miss or bypass requested, call the function
            logger.info(f"Cache miss for {func.__name__}, calling function")
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
            
            # Remove any 'proxies' parameter from kwargs
            if "proxies" in kwargs:
                kwargs.pop("proxies")
                
            # Generate cache key (with optional prefix)
            cache_key = f"{key_prefix}_{get_cache_key(func.__name__, args, kwargs)}" if key_prefix else get_cache_key(func.__name__, args, kwargs)
            
            # Check if we should use the cache
            if not bypass_cache:
                cached_result = load_from_cache(cache_key, cache_dir)
                if cached_result is not None:
                    logger.info(f"Cache hit for {func.__name__}")
                    return cached_result
            
            # Cache miss or bypass requested, call the function
            logger.info(f"Cache miss for {func.__name__}, calling function")
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