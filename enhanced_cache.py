"""
Enhanced cache manager for Vinkeljernet.

Provides improved caching mechanisms with partial result caching,
aggressive memory caching, and intelligent TTL-management.
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
from typing import Any, Callable, Dict, Optional, Union, TypeVar, List, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import zlib

# Import original cache manager for compatibility
from cache_manager import (
    DEFAULT_CACHE_DIR,
    DEFAULT_TTL,
    CACHE_VERSION,
    MEMORY_CACHE_SIZE,
    COMPRESSION_THRESHOLD,
    MAX_CACHE_SIZE_MB,
    get_cache_stats,
    optimize_cache
)

# Add these imports at the top of the file, after the existing imports
from collections import defaultdict
import re
from difflib import SequenceMatcher
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
try:
    stopwords.words('danish')
except:
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')

# Type variables for function annotations
T = TypeVar('T')
AsyncFunc = Callable[..., Any]
SyncFunc = Callable[..., Any]

# Configure logging
logger = logging.getLogger("vinkeljernet.enhanced_cache")

# Enhanced in-memory cache with tiered structure
# Level 1: Fast access, short TTL (10 minutes)
L1_CACHE: Dict[str, Tuple[float, Any]] = {}
L1_CACHE_KEYS: List[str] = []
L1_MAX_SIZE = 100
L1_TTL = 600  # 10 minutes

# Level 2: Medium access, medium TTL (1 hour)
L2_CACHE: Dict[str, Tuple[float, Any]] = {}
L2_CACHE_KEYS: List[str] = []
L2_MAX_SIZE = 200
L2_TTL = 3600  # 1 hour

# Persistent path for partial results
PARTIAL_RESULTS_DIR = DEFAULT_CACHE_DIR / "partial"
PARTIAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Access tracking for smart cache promotion/demotion
cache_access_counts: Dict[str, int] = {}
last_cache_cleanup = time.time()
CLEANUP_INTERVAL = 3600  # 1 hour

# Similarity threshold for related topics (0-1)
SIMILARITY_THRESHOLD = 0.75
# Maximum number of similar topics to track
MAX_SIMILAR_TOPICS = 20

# Track related topics for intelligent caching
_topic_similarity_cache = defaultdict(list)  # topic -> [(similar_topic, similarity_score)]

@dataclass
class EnhancedCacheStats:
    """Enhanced statistics for cache usage."""
    hits: int = 0
    misses: int = 0
    partial_hits: int = 0  # New: hits from partial results
    l1_hits: int = 0       # Hits from L1 cache
    l2_hits: int = 0       # Hits from L2 cache
    disk_hits: int = 0     # Hits from disk cache
    saved_api_calls: int = 0
    saved_processing_time: float = 0
    compression_ratio: float = 0
    items_in_l1: int = 0
    items_in_l2: int = 0
    items_on_disk: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        if total == 0:
            return 0
        return (self.hits / total) * 100
    
    @property
    def l1_hit_rate(self) -> float:
        """Calculate L1 cache hit rate."""
        if self.hits == 0:
            return 0
        return (self.l1_hits / self.hits) * 100
    
    @property
    def l2_hit_rate(self) -> float:
        """Calculate L2 cache hit rate."""
        if self.hits == 0:
            return 0
        return (self.l2_hits / self.hits) * 100
        
    @property
    def disk_hit_rate(self) -> float:
        """Calculate disk cache hit rate."""
        if self.hits == 0:
            return 0
        return (self.disk_hits / self.hits) * 100
    
    @property
    def partial_hit_rate(self) -> float:
        """Calculate partial result hit rate."""
        total = self.hits + self.misses
        if total == 0:
            return 0
        return (self.partial_hits / total) * 100


# Global stats instance
enhanced_stats = EnhancedCacheStats()


def make_cache_key(func_name: str, args: Tuple, kwargs: Dict[str, Any], namespace: str = "") -> str:
    """
    Create a consistent cache key from function arguments with namespace support.
    
    Args:
        func_name: Name of the function
        args: Positional arguments
        kwargs: Keyword arguments
        namespace: Optional namespace to separate keys
        
    Returns:
        Cache key string (MD5 hash)
    """
    # Ignore bypass_cache kwargs and any callback functions to ensure consistent keys
    kwargs_copy = {
        k: v for k, v in kwargs.items() 
        if k != 'bypass_cache' and not callable(v) and not k.endswith('_callback')
    }
    
    # Convert args and kwargs to a stable string representation
    key_data = f"{namespace}:{func_name}:{str(args)}:{str(sorted(kwargs_copy.items()))}"
    
    # Create a hash for the cache key (more collision-resistant than simple MD5)
    cache_key = hashlib.blake2b(key_data.encode(), digest_size=16).hexdigest()
    
    return cache_key


async def _check_and_cleanup_cache() -> None:
    """
    Periodically check and clean up cache based on access patterns.
    
    This function optimizes cache by:
    1. Removing expired entries
    2. Promoting frequently accessed items to higher cache tiers
    3. Demoting rarely accessed items to lower cache tiers
    4. Ensuring cache size limits are respected
    """
    global last_cache_cleanup, L1_CACHE, L2_CACHE, L1_CACHE_KEYS, L2_CACHE_KEYS
    
    # Only perform cleanup periodically
    current_time = time.time()
    if current_time - last_cache_cleanup < CLEANUP_INTERVAL:
        return
        
    last_cache_cleanup = current_time
    
    # Step 1: Remove expired entries from L1 cache
    expired_l1 = []
    for key in L1_CACHE_KEYS:
        if key in L1_CACHE:
            timestamp, _ = L1_CACHE[key]
            if current_time - timestamp > L1_TTL:
                expired_l1.append(key)
    
    # Remove expired keys from L1
    for key in expired_l1:
        if key in L1_CACHE:
            del L1_CACHE[key]
        if key in L1_CACHE_KEYS:
            L1_CACHE_KEYS.remove(key)
    
    # Step 2: Remove expired entries from L2 cache
    expired_l2 = []
    for key in L2_CACHE_KEYS:
        if key in L2_CACHE:
            timestamp, _ = L2_CACHE[key]
            if current_time - timestamp > L2_TTL:
                expired_l2.append(key)
    
    # Remove expired keys from L2
    for key in expired_l2:
        if key in L2_CACHE:
            del L2_CACHE[key]
        if key in L2_CACHE_KEYS:
            L2_CACHE_KEYS.remove(key)
    
    # Step 3: Promote frequently accessed items from L2 to L1
    # Only if L1 has space or we can make space
    for key in list(L2_CACHE_KEYS):
        if key in cache_access_counts and cache_access_counts[key] > 3:
            if key in L2_CACHE:
                # Only promote if key is not expired
                timestamp, value = L2_CACHE[key]
                if current_time - timestamp <= L2_TTL:
                    # Promote to L1 cache
                    L1_CACHE[key] = (current_time, value)  # Update timestamp
                    L1_CACHE_KEYS.append(key)
                    
                    # Remove from L2
                    del L2_CACHE[key]
                    L2_CACHE_KEYS.remove(key)
                    
                    # Reset access count after promotion
                    cache_access_counts[key] = 0
    
    # Step 4: Ensure L1 cache size limits are respected
    if len(L1_CACHE_KEYS) > L1_MAX_SIZE:
        # Sort keys by access frequency (descending) and timestamp (ascending)
        sorted_keys = sorted(
            L1_CACHE_KEYS,
            key=lambda k: (-cache_access_counts.get(k, 0), L1_CACHE.get(k, (0, None))[0])
        )
        
        # Keep only the most recently accessed items within the size limit
        keys_to_keep = sorted_keys[:L1_MAX_SIZE]
        keys_to_demote = sorted_keys[L1_MAX_SIZE:]
        
        # Demote excess items to L2 cache
        for key in keys_to_demote:
            if key in L1_CACHE:
                # Move to L2 cache
                L2_CACHE[key] = L1_CACHE[key]
                L2_CACHE_KEYS.append(key)
                
                # Remove from L1
                del L1_CACHE[key]
                
        # Update L1 keys list
        L1_CACHE_KEYS = keys_to_keep
    
    # Step 5: Ensure L2 cache size limits are respected
    if len(L2_CACHE_KEYS) > L2_MAX_SIZE:
        # Sort keys by access frequency (descending) and timestamp (ascending)
        sorted_keys = sorted(
            L2_CACHE_KEYS,
            key=lambda k: (-cache_access_counts.get(k, 0), L2_CACHE.get(k, (0, None))[0])
        )
        
        # Keep only the most frequently accessed items within the size limit
        keys_to_keep = sorted_keys[:L2_MAX_SIZE]
        keys_to_remove = sorted_keys[L2_MAX_SIZE:]
        
        # Remove excess items (they'll stay on disk if they were persisted)
        for key in keys_to_remove:
            if key in L2_CACHE:
                del L2_CACHE[key]
                
        # Update L2 keys list
        L2_CACHE_KEYS = keys_to_keep
    
    # Update cache statistics
    enhanced_stats.items_in_l1 = len(L1_CACHE)
    enhanced_stats.items_in_l2 = len(L2_CACHE)
    
    logger.debug(f"Cache cleanup completed. L1: {len(L1_CACHE)}, L2: {len(L2_CACHE)}")


def register_cache_access(key: str) -> None:
    """
    Register a cache access to track usage patterns.
    
    Args:
        key: The cache key that was accessed
    """
    global cache_access_counts
    cache_access_counts[key] = cache_access_counts.get(key, 0) + 1


def save_partial_result(cache_key: str, field: str, value: Any, ttl: int = 86400) -> None:
    """
    Save a partial result to the partial results cache.
    
    Args:
        cache_key: The main cache key
        field: The field/attribute name in the result
        value: The value to save
        ttl: Time-to-live in seconds (default: 24 hours)
    """
    # Create a unique key for the partial result
    partial_key = f"{cache_key}:{field}"
    
    # Create the file path
    file_path = PARTIAL_RESULTS_DIR / f"{partial_key}.pkl"
    
    try:
        # Serialize and compress data with timestamp and TTL
        data = {
            "value": value,
            "timestamp": time.time(),
            "ttl": ttl,
            "field": field,
            "parent_key": cache_key
        }
        
        # Calculate data size for compression decision
        pickled_data = pickle.dumps(data)
        
        # Compress if data is large enough
        if len(pickled_data) > COMPRESSION_THRESHOLD:
            compressed_data = zlib.compress(pickled_data)
            with open(file_path, 'wb') as f:
                f.write(compressed_data)
        else:
            with open(file_path, 'wb') as f:
                f.write(pickled_data)
                
        logger.debug(f"Saved partial result for {field} to {file_path}")
    except Exception as e:
        logger.error(f"Error saving partial result: {e}")


def get_partial_result(cache_key: str, field: str) -> Tuple[Optional[Any], bool]:
    """
    Get a partial result from the partial results cache.
    
    Args:
        cache_key: The main cache key
        field: The field/attribute name in the result
        
    Returns:
        Tuple containing:
        - The partial result value or None if not found/expired
        - Boolean indicating whether the result was found
    """
    # Create a unique key for the partial result
    partial_key = f"{cache_key}:{field}"
    
    # Create the file path
    file_path = PARTIAL_RESULTS_DIR / f"{partial_key}.pkl"
    
    # Check if the file exists
    if not file_path.exists():
        return None, False
    
    try:
        # Read the file
        with open(file_path, 'rb') as f:
            content = f.read()
        
        # Try to decompress if compressed
        try:
            decompressed_data = zlib.decompress(content)
            data = pickle.loads(decompressed_data)
        except zlib.error:
            # Not compressed
            data = pickle.loads(content)
        
        # Check if expired
        if time.time() - data["timestamp"] > data["ttl"]:
            # Remove expired file
            file_path.unlink(missing_ok=True)
            return None, False
        
        # Register cache access
        register_cache_access(partial_key)
        
        # Update stats
        enhanced_stats.partial_hits += 1
        
        return data["value"], True
    except Exception as e:
        logger.error(f"Error reading partial result: {e}")
        return None, False


def preprocess_topic(topic: str) -> str:
    """
    Preprocess a topic string for similarity comparison.
    
    Args:
        topic: The topic string to preprocess
        
    Returns:
        Preprocessed string
    """
    # Convert to lowercase
    text = topic.lower()
    
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Get Danish stopwords or use English as fallback
    try:
        stop_words = set(stopwords.words('danish'))
    except:
        stop_words = set(stopwords.words('english'))
    
    # Remove stopwords
    tokens = word_tokenize(text)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    
    return ' '.join(filtered_tokens)

def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two strings using SequenceMatcher.
    
    Args:
        text1: First string
        text2: Second string
        
    Returns:
        Similarity score between 0 and 1
    """
    # Preprocess texts
    proc_text1 = preprocess_topic(text1)
    proc_text2 = preprocess_topic(text2)
    
    # Calculate similarity
    matcher = SequenceMatcher(None, proc_text1, proc_text2)
    return matcher.ratio()

def register_topic_for_similarity(topic: str, cache_key: str) -> None:
    """
    Register a topic for similarity matching.
    
    Args:
        topic: The topic string
        cache_key: The associated cache key
    """
    # Compare to existing topics
    for existing_topic, existing_keys in list(_topic_similarity_cache.items()):
        similarity = calculate_similarity(topic, existing_topic)
        
        # If similar, add to each other's lists
        if similarity >= SIMILARITY_THRESHOLD:
            # Add to existing topic's list
            _topic_similarity_cache[existing_topic].append((topic, cache_key, similarity))
            # Sort and limit list
            _topic_similarity_cache[existing_topic] = sorted(
                _topic_similarity_cache[existing_topic], 
                key=lambda x: x[2], 
                reverse=True
            )[:MAX_SIMILAR_TOPICS]
            
            # Add existing topic to new topic's list
            for _, existing_key, _ in existing_keys:
                if existing_key not in [k for _, k, _ in _topic_similarity_cache[topic]]:
                    _topic_similarity_cache[topic].append((existing_topic, existing_key, similarity))
            
    # Ensure topic has an entry even if no similar topics found
    if topic not in _topic_similarity_cache:
        _topic_similarity_cache[topic] = []

def find_similar_topic_cache_keys(topic: str) -> List[Tuple[str, str, float]]:
    """
    Find cache keys for similar topics.
    
    Args:
        topic: The topic to find similar topics for
        
    Returns:
        List of tuples (similar_topic, cache_key, similarity_score)
    """
    # Start with exact matches
    if topic in _topic_similarity_cache:
        return _topic_similarity_cache[topic]
    
    similar_topics = []
    # Calculate similarity with all existing topics
    for existing_topic, similar_info in _topic_similarity_cache.items():
        similarity = calculate_similarity(topic, existing_topic)
        if similarity >= SIMILARITY_THRESHOLD:
            # Get cache keys for this similar topic
            for sim_topic, cache_key, sim_score in similar_info:
                similar_topics.append((sim_topic, cache_key, sim_score * similarity))  # Adjust by current similarity
    
    # Return sorted by similarity
    return sorted(similar_topics, key=lambda x: x[2], reverse=True)[:MAX_SIMILAR_TOPICS]

def get_similar_topic_cache(topic: str) -> Optional[Dict[str, Any]]:
    """
    Get cached data for similar topics.
    
    Args:
        topic: The topic to find similar cached data for
        
    Returns:
        Dictionary mapping similar topics to their cached data
    """
    similar_topics = find_similar_topic_cache_keys(topic)
    if not similar_topics:
        return None
        
    result = {}
    from cache_manager import load_from_cache
    
    for sim_topic, cache_key, similarity in similar_topics:
        cached_data = load_from_cache(cache_key)
        if cached_data:
            result[sim_topic] = {
                "data": cached_data,
                "similarity": similarity,
                "cache_key": cache_key
            }
    
    return result if result else None

def enhanced_cached_api(ttl: int = DEFAULT_TTL, namespace: str = "api", check_similar: bool = True) -> Callable:
    """
    Enhanced decorator for caching API results with multi-tiered caching and similarity matching.
    
    Args:
        ttl: Time-to-live in seconds
        namespace: Cache namespace
        check_similar: If True, check for similar topics in cache
        
    Returns:
        Decorator function
    """
    def decorator(func: AsyncFunc) -> AsyncFunc:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Skip cache if explicitly requested
            bypass_cache = kwargs.get('bypass_cache', False)
            
            # Create cache key
            func_name = func.__qualname__
            cache_key = make_cache_key(func_name, args, kwargs, namespace)
            
            # Check if this is a topic-related function (for similarity caching)
            is_topic_func = 'topic' in kwargs or (args and isinstance(args[0], str))
            topic = kwargs.get('topic', args[0] if args else None) if is_topic_func else None
            
            # Check for bypass_cache
            if not bypass_cache:
                # Check periodically if cache needs cleanup
                await _check_and_cleanup_cache()
                
                # Check L1 cache (fastest)
                if cache_key in L1_CACHE:
                    timestamp, value = L1_CACHE[cache_key]
                    # Check if not expired
                    if time.time() - timestamp <= L1_TTL:
                        register_cache_access(cache_key)
                        enhanced_stats.hits += 1
                        enhanced_stats.l1_hits += 1
                        logger.debug(f"L1 cache hit: {func_name}")
                        return value
                
                # Check L2 cache
                if cache_key in L2_CACHE:
                    timestamp, value = L2_CACHE[cache_key]
                    # Check if not expired
                    if time.time() - timestamp <= L2_TTL:
                        register_cache_access(cache_key)
                        # Promote to L1 cache if it has room
                        if len(L1_CACHE) < L1_MAX_SIZE:
                            L1_CACHE[cache_key] = (time.time(), value)
                            L1_CACHE_KEYS.append(cache_key)
                            
                        enhanced_stats.hits += 1
                        enhanced_stats.l2_hits += 1
                        logger.debug(f"L2 cache hit: {func_name}")
                        return value
                
                # Check disk cache (original implementation)
                from cache_manager import load_from_cache
                result = load_from_cache(cache_key)
                if result is not None:
                    register_cache_access(cache_key)
                    
                    # Store in L2 cache for faster access next time
                    if len(L2_CACHE) < L2_MAX_SIZE:
                        L2_CACHE[cache_key] = (time.time(), result)
                        L2_CACHE_KEYS.append(cache_key)
                    
                    enhanced_stats.hits += 1
                    enhanced_stats.disk_hits += 1
                    logger.debug(f"Disk cache hit: {func_name}")
                    return result
                
                # NEW: Check for similar topics if enabled and topic is provided
                if check_similar and topic and is_topic_func:
                    similar_results = get_similar_topic_cache(topic)
                    if similar_results:
                        # Use the result from the most similar topic
                        most_similar = max(similar_results.items(), key=lambda x: x[1]["similarity"])
                        sim_topic, data = most_similar
                        sim_result = data["data"]
                        sim_score = data["similarity"]
                        
                        logger.info(f"Using cache from similar topic '{sim_topic}' (similarity: {sim_score:.2f}) for '{topic}'")
                        
                        # Cache this lookup for future use with adjusted TTL
                        adjusted_ttl = int(ttl * sim_score)  # Shorter TTL for less similar topics
                        from cache_manager import save_to_cache
                        save_to_cache(cache_key, sim_result, adjusted_ttl)
                        
                        # Store in L1 cache for immediate reuse
                        L1_CACHE[cache_key] = (time.time(), sim_result)
                        L1_CACHE_KEYS.append(cache_key)
                        
                        enhanced_stats.hits += 1
                        enhanced_stats.partial_hits += 1  # Count as partial hit since it's from a similar topic
                        
                        return sim_result
            
            # Execute the function and save result to cache
            start_time = time.time()
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Save result if not None and not in bypass mode
            if result is not None and not bypass_cache:
                # Save to cache
                from cache_manager import save_to_cache
                save_to_cache(cache_key, result, ttl)
                
                # Also save to in-memory cache for faster access
                L1_CACHE[cache_key] = (time.time(), result)
                L1_CACHE_KEYS.append(cache_key)
                
                # If result is a dictionary, save partial results
                if isinstance(result, dict):
                    for field, value in result.items():
                        # Only save non-empty values that are not too complex
                        if value and not isinstance(value, (dict, list)):
                            save_partial_result(cache_key, field, value, ttl)
                
                # Register topic for similarity matching if relevant
                if topic and is_topic_func:
                    register_topic_for_similarity(topic, cache_key)
                
                # Update metrics
                enhanced_stats.saved_processing_time += execution_time
                enhanced_stats.misses += 1
                
                # Keep memory cache from growing too large
                if len(L1_CACHE_KEYS) > L1_MAX_SIZE:
                    oldest_key = L1_CACHE_KEYS[0]
                    L1_CACHE_KEYS.pop(0)
                    if oldest_key in L1_CACHE:
                        # Move to L2 cache instead of discarding
                        L2_CACHE[oldest_key] = L1_CACHE[oldest_key]
                        L2_CACHE_KEYS.append(oldest_key)
                        del L1_CACHE[oldest_key]
            
            return result
        
        return wrapper
    
    return decorator


def get_enhanced_cache_stats() -> EnhancedCacheStats:
    """
    Get enhanced cache usage statistics.
    
    Returns:
        EnhancedCacheStats object
    """
    # Update disk items count
    if PARTIAL_RESULTS_DIR.exists():
        enhanced_stats.items_on_disk = len(list(PARTIAL_RESULTS_DIR.glob("*.pkl")))
    
    return enhanced_stats


async def clear_partial_cache() -> int:
    """
    Clear all partial result cache files.
    
    Returns:
        Number of files removed
    """
    if not PARTIAL_RESULTS_DIR.exists():
        return 0
        
    count = 0
    for file_path in PARTIAL_RESULTS_DIR.glob("*.pkl"):
        try:
            file_path.unlink()
            count += 1
        except Exception as e:
            logger.error(f"Error removing partial cache file {file_path}: {e}")
    
    return count


async def optimize_enhanced_cache(max_age_days: int = 7, silent: bool = False) -> Dict[str, Any]:
    """
    Optimize the enhanced cache for better performance.
    
    Args:
        max_age_days: Maximum age in days for cached items
        silent: Whether to suppress log messages
        
    Returns:
        Dict with optimization statistics
    """
    # Start with base optimization
    from cache_manager import optimize_cache as base_optimize
    base_stats = await base_optimize(max_age_days, silent)
    
    # Clear expired partial results
    if not silent:
        logger.info("Optimizing partial results cache...")
    
    current_time = time.time()
    max_age_seconds = max_age_days * 86400
    expired_count = 0
    kept_count = 0
    
    if PARTIAL_RESULTS_DIR.exists():
        for file_path in PARTIAL_RESULTS_DIR.glob("*.pkl"):
            try:
                # Check file age
                file_age = current_time - file_path.stat().st_mtime
                
                if file_age > max_age_seconds:
                    # Remove old file
                    file_path.unlink()
                    expired_count += 1
                else:
                    kept_count += 1
            except Exception as e:
                logger.error(f"Error during partial cache optimization: {e}")
    
    # Clear memory caches older than max age
    l1_cleared = 0
    for key in list(L1_CACHE_KEYS):
        if key in L1_CACHE:
            timestamp, _ = L1_CACHE[key]
            if current_time - timestamp > max_age_seconds:
                del L1_CACHE[key]
                L1_CACHE_KEYS.remove(key)
                l1_cleared += 1
    
    l2_cleared = 0
    for key in list(L2_CACHE_KEYS):
        if key in L2_CACHE:
            timestamp, _ = L2_CACHE[key]
            if current_time - timestamp > max_age_seconds:
                del L2_CACHE[key]
                L2_CACHE_KEYS.remove(key)
                l2_cleared += 1
    
    # Combine stats
    stats = {
        "base_optimization": base_stats,
        "partial_cache": {
            "expired_removed": expired_count,
            "kept": kept_count
        },
        "memory_cache": {
            "l1_cleared": l1_cleared,
            "l2_cleared": l2_cleared,
            "l1_remaining": len(L1_CACHE),
            "l2_remaining": len(L2_CACHE)
        }
    }
    
    if not silent:
        logger.info(f"Enhanced cache optimization complete. "
                   f"Removed {expired_count} expired partial results, "
                   f"{l1_cleared} L1 items, and {l2_cleared} L2 items.")
    
    return stats