"""
Parallel API processing for Vinkeljernet.

This module provides functionality for making parallel API requests
to improve performance when multiple calls are needed.
"""

import asyncio
import logging
from typing import List, Dict, Any, Callable, Optional, TypeVar, Tuple
from functools import partial
import time

# Configure logging
logger = logging.getLogger("vinkeljernet.parallel_api")

# Type variable for generic functions
T = TypeVar('T')

class BatchProcessor:
    """
    Handles batch processing of API requests to improve performance.
    """
    
    def __init__(self, 
                 max_concurrent: int = 5,
                 batch_interval: float = 0.1,
                 max_batch_size: int = 10):
        """
        Initialize the batch processor.
        
        Args:
            max_concurrent: Maximum number of concurrent API calls
            batch_interval: Time interval between batches in seconds
            max_batch_size: Maximum number of requests in a single batch
        """
        self.max_concurrent = max_concurrent
        self.batch_interval = batch_interval
        self.max_batch_size = max_batch_size
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.queue = asyncio.Queue()
        self.batch_running = False
        self.batch_task = None
        self.results: Dict[str, Any] = {}
        self.current_batch_id = 0
        self.stats = {
            "total_requests": 0,
            "batched_requests": 0,
            "direct_requests": 0,
            "errors": 0,
            "avg_response_time": 0,
            "total_response_time": 0
        }
    
    async def _execute_with_semaphore(self, 
                                      func: Callable, 
                                      *args, 
                                      **kwargs) -> Any:
        """
        Execute a function with semaphore-based concurrency control.
        
        Args:
            func: The async function to execute
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            The result of the function call
        """
        async with self.semaphore:
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                self.stats["total_response_time"] += time.time() - start_time
                self.stats["total_requests"] += 1
                if self.stats["total_requests"] > 0:
                    self.stats["avg_response_time"] = (
                        self.stats["total_response_time"] / self.stats["total_requests"]
                    )
                return result
            except Exception as e:
                self.stats["errors"] += 1
                logger.error(f"Error in API call: {e}")
                raise
    
    async def execute(self, 
                      func: Callable, 
                      *args, 
                      batch: bool = True, 
                      batch_key: Optional[str] = None, 
                      **kwargs) -> Any:
        """
        Execute an async function, optionally as part of a batch.
        
        Args:
            func: The async function to execute
            *args: Positional arguments to pass to the function
            batch: Whether to batch this request if possible
            batch_key: Custom key to identify this request in batch results
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            The result of the function call
        """
        # If batching is disabled, execute directly
        if not batch:
            self.stats["direct_requests"] += 1
            return await self._execute_with_semaphore(func, *args, **kwargs)
        
        # Create a unique identifier for this request if not provided
        if batch_key is None:
            batch_key = f"batch_request_{id(func)}_{id(args)}_{id(kwargs)}_{time.time()}"
        
        # Put the request in the queue
        await self.queue.put((batch_key, func, args, kwargs))
        self.stats["batched_requests"] += 1
        
        # Start the batch processing if not already running
        if not self.batch_running:
            self.batch_running = True
            self.batch_task = asyncio.create_task(self._process_batch())
        
        # Wait for the result
        while batch_key not in self.results:
            await asyncio.sleep(0.01)
        
        # Get and remove the result
        result = self.results.pop(batch_key)
        
        # If the result is an exception, raise it
        if isinstance(result, Exception):
            raise result
        
        return result
    
    async def _process_batch(self) -> None:
        """
        Process requests from the queue in batches.
        """
        try:
            while True:
                # Wait for items to be in the queue
                if self.queue.empty():
                    await asyncio.sleep(0.05)
                    if self.queue.empty():
                        self.batch_running = False
                        break
                
                # Create a batch
                batch: List[Tuple[str, Callable, tuple, dict]] = []
                self.current_batch_id += 1
                batch_id = self.current_batch_id
                
                # Get items from the queue, up to max_batch_size
                for _ in range(self.max_batch_size):
                    if self.queue.empty():
                        break
                    
                    item = await self.queue.get()
                    batch.append(item)
                
                logger.debug(f"Processing batch {batch_id} with {len(batch)} requests")
                
                # Process the batch
                tasks = []
                for batch_key, func, args, kwargs in batch:
                    task = asyncio.create_task(
                        self._process_batch_item(batch_key, func, args, kwargs)
                    )
                    tasks.append(task)
                
                # Wait for all tasks in the batch to complete
                for task in tasks:
                    await task
                
                # Wait a bit before processing the next batch
                await asyncio.sleep(self.batch_interval)
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            self.batch_running = False
    
    async def _process_batch_item(self, 
                                  batch_key: str, 
                                  func: Callable, 
                                  args: tuple, 
                                  kwargs: dict) -> None:
        """
        Process a single item from a batch.
        
        Args:
            batch_key: The key to store the result under
            func: The async function to execute
            args: Positional arguments to pass to the function
            kwargs: Keyword arguments to pass to the function
        """
        try:
            result = await self._execute_with_semaphore(func, *args, **kwargs)
            self.results[batch_key] = result
        except Exception as e:
            self.results[batch_key] = e
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the batch processor.
        
        Returns:
            Dictionary with statistics
        """
        return {
            **self.stats,
            "queue_size": self.queue.qsize() if not self.queue.empty() else 0,
            "is_batch_running": self.batch_running
        }
    
    async def wait_for_completion(self, timeout: float = 30.0) -> None:
        """
        Wait for all queued requests to complete.
        
        Args:
            timeout: Maximum time to wait in seconds
        """
        start_time = time.time()
        
        while not self.queue.empty() or self.batch_running:
            if time.time() - start_time > timeout:
                logger.warning(f"Timed out waiting for batch completion after {timeout} seconds")
                break
            
            await asyncio.sleep(0.1)
        
        if self.batch_task and not self.batch_task.done():
            self.batch_task.cancel()
            
        self.batch_running = False

# Global instance
batch_processor = BatchProcessor()

async def execute_in_parallel(
    func: Callable[[Any], Any],
    items: List[Any],
    max_concurrency: int = 5,
    process_func: Callable[[Any, Any], Any] = None,
    progress_callback: Callable[[float], None] = None
) -> List[Any]:
    """
    Execute a function in parallel for multiple items.
    
    Args:
        func: The async function to execute
        items: List of items to process
        max_concurrency: Maximum number of concurrent executions
        process_func: Optional function to process results
        progress_callback: Optional callback for progress updates
        
    Returns:
        List of results in the same order as items
    """
    if not items:
        return []
    
    # Prepare semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def process_item(item, index):
        async with semaphore:
            try:
                result = await func(item)
                if process_func:
                    result = process_func(item, result)
                
                # Update progress if callback provided
                if progress_callback:
                    progress = (index + 1) / len(items) * 100
                    await progress_callback(progress)
                    
                return result
            except Exception as e:
                logger.error(f"Error processing item {index}: {e}")
                return None
    
    # Create tasks for all items
    tasks = [
        asyncio.create_task(process_item(item, i))
        for i, item in enumerate(items)
    ]
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks)
    
    return results

async def parallel_api_calls(
    api_func: Callable,
    param_sets: List[Dict[str, Any]],
    max_concurrency: int = 3
) -> List[Any]:
    """
    Execute multiple API calls in parallel with controlled concurrency.
    
    Args:
        api_func: The API function to call
        param_sets: List of parameter dictionaries for each call
        max_concurrency: Maximum number of concurrent API calls
        
    Returns:
        List of API call results
    """
    semaphore = asyncio.Semaphore(max_concurrency)
    results = []
    
    async def call_api_with_params(params):
        async with semaphore:
            start_time = time.time()
            try:
                result = await api_func(**params)
                logger.debug(f"API call completed in {time.time() - start_time:.2f}s")
                return result
            except Exception as e:
                logger.error(f"API call failed: {e}")
                return None
    
    tasks = [call_api_with_params(params) for params in param_sets]
    results = await asyncio.gather(*tasks)
    
    return results

# Helper function for partial topic information 
async def fetch_multiple_topics(
    fetch_topic_func: Callable, 
    topics: List[str],
    max_concurrency: int = 3,
    bypass_cache: bool = False,
    progress_callback: Optional[Callable] = None
) -> Dict[str, str]:
    """
    Fetch information for multiple topics in parallel.
    
    Args:
        fetch_topic_func: Function to fetch topic information
        topics: List of topics to fetch information for
        max_concurrency: Maximum number of concurrent API calls
        bypass_cache: Whether to bypass the cache
        progress_callback: Optional callback for progress updates
        
    Returns:
        Dictionary mapping topics to their information
    """
    # Define parameters for each call
    param_sets = [
        {"topic": topic, "bypass_cache": bypass_cache}
        for topic in topics
    ]
    
    # Track progress
    completed = 0
    total = len(topics)
    
    async def fetch_with_progress(params):
        nonlocal completed
        result = await fetch_topic_func(**params)
        completed += 1
        if progress_callback:
            await progress_callback(completed / total * 100)
        return (params["topic"], result)
    
    # Execute API calls in parallel
    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = []
    
    for params in param_sets:
        task = asyncio.create_task(
            async_with_semaphore(semaphore, fetch_with_progress, params)
        )
        tasks.append(task)
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks)
    
    # Build result dictionary
    return {topic: info for topic, info in results if info is not None}

async def async_with_semaphore(semaphore, func, *args, **kwargs):
    """Helper function to execute with a semaphore."""
    async with semaphore:
        return await func(*args, **kwargs)
"""