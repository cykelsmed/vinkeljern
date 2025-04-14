"""
Enhanced caching module for Vinkeljernet project.

This module extends the basic cache_manager with intelligent topic-profile 
caching, partial result storage, and smarter TTL strategies.
"""

import os
import json
import time
import hashlib
import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta

# Import base caching functionality
from cache_manager import (
    save_to_cache,
    load_from_cache,
    get_cache_key,
    DEFAULT_CACHE_DIR,
    DEFAULT_TTL
)

# Configure logging
logger = logging.getLogger("vinkeljernet.enhanced_cache")

# Enhanced cache configuration
RESULTS_DIR = Path(os.path.expanduser("~/.vinkeljernet/results"))
PARTIAL_RESULTS_DIR = Path(os.path.expanduser("~/.vinkeljernet/partial_results"))
REQUEST_HISTORY_DIR = Path(os.path.expanduser("~/.vinkeljernet/history"))

# Ensure directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PARTIAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REQUEST_HISTORY_DIR.mkdir(parents=True, exist_ok=True)

# Cache TTL strategies (in seconds)
TTL_STRATEGIES = {
    "popular_topic": 172800,  # 48 hours for popular topics
    "standard": 86400,        # 24 hours for standard requests
    "fast_changing": 3600,    # 1 hour for fast-changing topics
    "expensive_request": 259200,  # 3 days for computationally expensive requests
    "test": 60               # 1 minute for testing
}

class PartialResult:
    """Store and manage partial results during multi-step API processes."""
    
    def __init__(self, request_id: str, total_steps: int = 5):
        self.request_id = request_id
        self.total_steps = total_steps
        self.completed_steps = 0
        self.results = {}
        self.errors = []
        self.start_time = time.time()
        self.last_update = self.start_time
        self.is_complete = False
    
    def add_result(self, step_name: str, result: Any) -> None:
        """Add a step result."""
        self.results[step_name] = result
        self.completed_steps += 1
        self.last_update = time.time()
        
        # Auto-save after each step
        self.save()
    
    def add_error(self, step_name: str, error: str) -> None:
        """Add an error for a step."""
        self.errors.append({"step": step_name, "error": error, "time": time.time()})
        self.last_update = time.time()
        
        # Auto-save after recording error
        self.save()
    
    def mark_complete(self) -> None:
        """Mark the process as complete."""
        self.is_complete = True
        self.last_update = time.time()
        self.save()
    
    def get_progress(self) -> float:
        """Get progress as a percentage."""
        if self.total_steps <= 0:
            return 0.0
        return min(100.0, (self.completed_steps / self.total_steps) * 100)
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "request_id": self.request_id,
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
            "results": self.results,
            "errors": self.errors,
            "start_time": self.start_time,
            "last_update": self.last_update,
            "is_complete": self.is_complete,
            "progress": self.get_progress(),
            "elapsed_seconds": self.get_elapsed_time()
        }
    
    def save(self) -> None:
        """Save the partial result to disk."""
        partial_file = PARTIAL_RESULTS_DIR / f"{self.request_id}.json"
        
        try:
            with open(partial_file, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Could not save partial result: {e}")
    
    @classmethod
    def load(cls, request_id: str) -> Optional['PartialResult']:
        """Load a partial result from disk."""
        partial_file = PARTIAL_RESULTS_DIR / f"{request_id}.json"
        
        if not partial_file.exists():
            return None
        
        try:
            with open(partial_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            result = cls(request_id, data.get("total_steps", 5))
            result.completed_steps = data.get("completed_steps", 0)
            result.results = data.get("results", {})
            result.errors = data.get("errors", [])
            result.start_time = data.get("start_time", time.time())
            result.last_update = data.get("last_update", time.time())
            result.is_complete = data.get("is_complete", False)
            
            return result
        except Exception as e:
            logger.warning(f"Could not load partial result: {e}")
            return None

def generate_topic_profile_key(topic: str, profile_id: str) -> str:
    """
    Generate a cache key for a topic-profile combination.
    
    Args:
        topic: The news topic
        profile_id: Identifier for the editorial profile
        
    Returns:
        str: A unique key for this topic-profile combination
    """
    # Normalize inputs to avoid case sensitivity issues
    normalized_topic = topic.lower().strip()
    normalized_profile = profile_id.lower().strip()
    
    # Create a hash
    key_data = f"topic:{normalized_topic}:profile:{normalized_profile}".encode('utf-8')
    return hashlib.md5(key_data).hexdigest()

def determine_ttl_strategy(topic: str, complexity: int = 1) -> int:
    """
    Determine appropriate cache TTL based on topic and complexity.
    
    Args:
        topic: The news topic
        complexity: Request complexity (1-5)
        
    Returns:
        int: TTL in seconds
    """
    # List of keywords indicating fast-changing topics
    fast_changing_keywords = [
        'breaking', 'live', 'update', 'udvikling', 'seneste', 
        'valg', 'krig', 'ulykke', 'katastrofe'
    ]
    
    # Check if any fast-changing keywords are in the topic
    if any(keyword in topic.lower() for keyword in fast_changing_keywords):
        return TTL_STRATEGIES["fast_changing"]
    
    # For very complex requests, use longer TTL
    if complexity >= 4:
        return TTL_STRATEGIES["expensive_request"]
    
    # Default to standard TTL
    return TTL_STRATEGIES["standard"]

def record_request(
    topic: str, 
    profile_id: str, 
    was_cached: bool, 
    execution_time: float,
    status: str = "success"
) -> None:
    """
    Record request information for analytics.
    
    Args:
        topic: News topic
        profile_id: Profile identifier
        was_cached: Whether result was served from cache
        execution_time: Time taken in seconds
        status: Request status
    """
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    
    # Daily log file for requests
    log_file = REQUEST_HISTORY_DIR / f"requests_{date_str}.jsonl"
    
    record = {
        "timestamp": now.isoformat(),
        "topic": topic,
        "profile_id": profile_id,
        "was_cached": was_cached,
        "execution_time": execution_time,
        "status": status
    }
    
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.warning(f"Could not record request: {e}")

def save_final_result(
    topic: str,
    profile_id: str,
    result: Any,
    execution_time: float
) -> str:
    """
    Save final result to a persistent file storage.
    
    Args:
        topic: News topic
        profile_id: Profile identifier
        result: Result data to save
        execution_time: Time taken in seconds
        
    Returns:
        str: Result ID
    """
    # Generate a unique ID for this result
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_id = f"{profile_id}_{timestamp}_{hashlib.md5(topic.encode('utf-8')).hexdigest()[:8]}"
    
    # Save the result
    result_file = RESULTS_DIR / f"{result_id}.json"
    
    metadata = {
        "id": result_id,
        "topic": topic,
        "profile_id": profile_id,
        "timestamp": datetime.now().isoformat(),
        "execution_time": execution_time
    }
    
    try:
        # Combine metadata and result
        full_data = {
            "metadata": metadata,
            "result": result
        }
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(full_data, f, ensure_ascii=False, indent=2)
            
        return result_id
    except Exception as e:
        logger.warning(f"Could not save result: {e}")
        return ""

def load_result(result_id: str) -> Optional[Dict[str, Any]]:
    """
    Load a saved result by ID.
    
    Args:
        result_id: Result identifier
        
    Returns:
        Optional[Dict]: The saved result or None if not found
    """
    result_file = RESULTS_DIR / f"{result_id}.json"
    
    if not result_file.exists():
        return None
    
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load result {result_id}: {e}")
        return None

def get_recent_results(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get a list of recent results with metadata.
    
    Args:
        limit: Maximum number of results to return
        
    Returns:
        List[Dict]: Recent results
    """
    results = []
    
    try:
        # Get all result files sorted by modification time (newest first)
        result_files = sorted(
            RESULTS_DIR.glob("*.json"), 
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        # Load metadata from each file
        for i, file_path in enumerate(result_files):
            if i >= limit:
                break
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    results.append(data.get("metadata", {}))
            except Exception as e:
                logger.warning(f"Could not read result file {file_path}: {e}")
    except Exception as e:
        logger.error(f"Error listing recent results: {e}")
    
    return results

def get_cached_topics(profile_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get a list of topics that have cached results.
    
    Args:
        profile_id: Optional profile ID to filter by
        
    Returns:
        List[Dict]: Topics with cache information
    """
    topics = []
    
    try:
        # Get all result files
        result_files = RESULTS_DIR.glob("*.json")
        
        # Extract topic information from each file
        for file_path in result_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    metadata = data.get("metadata", {})
                    
                    # Filter by profile if specified
                    if profile_id and metadata.get("profile_id") != profile_id:
                        continue
                    
                    topic_info = {
                        "topic": metadata.get("topic", "Unknown"),
                        "profile_id": metadata.get("profile_id", "Unknown"),
                        "timestamp": metadata.get("timestamp"),
                        "result_id": metadata.get("id")
                    }
                    topics.append(topic_info)
            except Exception as e:
                logger.warning(f"Could not read result file {file_path}: {e}")
    except Exception as e:
        logger.error(f"Error listing cached topics: {e}")
    
    return topics

def cleanup_old_partial_results(max_age_hours: int = 24) -> int:
    """
    Clean up partial results older than specified age.
    
    Args:
        max_age_hours: Maximum age in hours to keep
        
    Returns:
        int: Number of files removed
    """
    count = 0
    threshold = time.time() - (max_age_hours * 3600)
    
    try:
        for file_path in PARTIAL_RESULTS_DIR.glob("*.json"):
            try:
                # Check last modification time
                if file_path.stat().st_mtime < threshold:
                    file_path.unlink()
                    count += 1
            except Exception:
                pass
    except Exception as e:
        logger.error(f"Error cleaning up partial results: {e}")
    
    return count

def intelligent_cache_key(func_name: str, topic: str, profile_id: str, **kwargs) -> str:
    """
    Generate an intelligent cache key for vinkel-related functions.
    
    Args:
        func_name: Name of the function
        topic: News topic
        profile_id: Profile identifier
        **kwargs: Other parameters
        
    Returns:
        str: Cache key
    """
    # Start with the function name
    key_components = [func_name]
    
    # Add normalized topic and profile
    key_components.append(topic.lower().strip())
    key_components.append(profile_id.lower().strip())
    
    # Add select kwargs that influence the result
    important_keys = [
        "detailed", "include_expert_sources", "include_knowledge_distillate", 
        "complexity", "tone", "max_angles"
    ]
    
    for k in important_keys:
        if k in kwargs:
            key_components.append(f"{k}:{kwargs[k]}")
    
    # Create a string and hash it
    key_data = ":".join(str(c) for c in key_components)
    return hashlib.md5(key_data.encode('utf-8')).hexdigest()

async def cache_topic_profile_result(
    func_name: str,
    topic: str,
    profile_id: str,
    result: Any,
    complexity: int = 1,
    **kwargs
) -> None:
    """
    Cache result with intelligent topic-profile key.
    
    Args:
        func_name: Name of the function being cached
        topic: News topic
        profile_id: Profile identifier
        result: Result to cache
        complexity: Request complexity (1-5)
        **kwargs: Other parameters that affect caching
    """
    # Generate cache key
    cache_key = intelligent_cache_key(func_name, topic, profile_id, **kwargs)
    
    # Determine TTL based on topic and complexity
    ttl = determine_ttl_strategy(topic, complexity)
    
    # Save to both memory and disk cache
    save_to_cache(cache_key, result, ttl)
    
    # Save as a final result file as well
    execution_time = kwargs.get("execution_time", 0)
    await asyncio.to_thread(save_final_result, topic, profile_id, result, execution_time)

async def load_topic_profile_result(
    func_name: str,
    topic: str,
    profile_id: str,
    **kwargs
) -> Optional[Any]:
    """
    Load cached result with intelligent topic-profile key.
    
    Args:
        func_name: Name of the function being cached
        topic: News topic
        profile_id: Profile identifier
        **kwargs: Other parameters that affect caching
        
    Returns:
        Optional[Any]: Cached result or None
    """
    # Generate cache key
    cache_key = intelligent_cache_key(func_name, topic, profile_id, **kwargs)
    
    # Try to load from cache
    return load_from_cache(cache_key)