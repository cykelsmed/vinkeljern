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
import concurrent.futures
import sys

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

class AsyncAPIClient:
    """
    Optimeret asynkron API-klient med connection pooling, 
    ratelimiting og adaptive timeout.
    """
    
    def __init__(
        self, 
        base_url: str, 
        api_key: str = None,
        max_concurrent_requests: int = 5,
        timeout: float = 60.0,
        retry_attempts: int = 3,
        retry_delay: float = 2.0,
        headers: Dict[str, str] = None
    ):
        """
        Initialiserer API-klienten.
        
        Args:
            base_url: Base URL for API-kald
            api_key: API-nøgle til autentificering
            max_concurrent_requests: Maksimalt antal samtidige requests
            timeout: Timeout i sekunder for API-kald
            retry_attempts: Antal forsøg ved fejl
            retry_delay: Tid i sekunder mellem forsøg (fordobles for hver retry)
            headers: Standard headers til alle requests
        """
        self.base_url = base_url
        self.api_key = api_key
        self.session = None
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.default_headers = headers or {}
        self.logger = logger
        
        # Metrikker til monitorering
        self.metrics = {
            "requests_total": 0,
            "requests_success": 0,
            "requests_failed": 0,
            "total_time": 0.0,
            "avg_response_time": 0.0,
            "rate_limited_count": 0,
            "retry_count": 0,
            "timeout_count": 0
        }
    
    async def initialize(self):
        """Initialiserer aiohttp session med connection pooling"""
        if self.session is None:
            import aiohttp
            
            # Opret connector med god TCP pooling konfiguration
            connector = aiohttp.TCPConnector(
                limit=20,  # Maksimalt antal forbindelser (per host)
                ttl_dns_cache=300,  # Cacher DNS opslag i 5 minutter
                use_dns_cache=True,
                ssl=False  # Slå SSL-verifikation fra hvis nødvendigt
            )
            
            # Opret session med timeout
            timeout = aiohttp.ClientTimeout(
                total=self.timeout,
                sock_connect=30.0,  # Socket connection timeout
                sock_read=self.timeout  # Socket read timeout
            )
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers=self._get_default_headers()
            )
            
            self.logger.info(f"Initialized AsyncAPIClient for {self.base_url}")
    
    def _get_default_headers(self) -> Dict[str, str]:
        """Genererer default headers til alle requests"""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "VinkeljernetClient/1.0"
        }
        
        # Tilføj API-nøgle hvis tilgængelig
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            
        # Tilføj kundetilpassede headers
        headers.update(self.default_headers)
        
        return headers
    
    async def close(self):
        """Lukker sessionen korrekt"""
        if self.session:
            await self.session.close()
            self.session = None
            self.logger.debug("Closed API client session")
    
    async def request(
        self, 
        method: str, 
        endpoint: str, 
        payload: Dict[str, Any] = None, 
        headers: Dict[str, str] = None,
        query_params: Dict[str, str] = None,
        timeout_override: float = None,
        retry_attempts_override: int = None
    ) -> Dict[str, Any]:
        """
        Sender en API-forespørgsel med ratelimiting, retries og metrikmåling.
        
        Args:
            method: HTTP-metode (GET, POST, osv.)
            endpoint: API-endepunkt (tilføjes til base_url)
            payload: JSON-payload til request body
            headers: Ekstra headers til denne request
            query_params: URL query parametre
            timeout_override: Specifik timeout for denne request
            retry_attempts_override: Specifikt antal forsøg for denne request
            
        Returns:
            Dict[str, Any]: API-svar som dict
        """
        import aiohttp
        
        # Initialisér session hvis ikke allerede gjort
        if not self.session:
            await self.initialize()
            
        # Kombiner endpoint med base URL
        full_url = f"{self.base_url}{endpoint}"
        
        # Kombiner standard headers med request-specifikke headers
        request_headers = self._get_default_headers()
        if headers:
            request_headers.update(headers)
            
        # Brug request-specifikke værdier eller defaults
        timeout = timeout_override or self.timeout
        retry_attempts = retry_attempts_override or self.retry_attempts
        
        # Brug semaforen til at begrænse samtidige forespørgsler
        async with self.semaphore:
            start_time = time.time()
            self.metrics["requests_total"] += 1
            
            for attempt in range(retry_attempts):
                try:
                    # Opret timeout context for denne request
                    request_timeout = aiohttp.ClientTimeout(total=timeout)
                    
                    async with self.session.request(
                        method=method,
                        url=full_url,
                        json=payload,
                        headers=request_headers,
                        params=query_params,
                        timeout=request_timeout
                    ) as response:
                        # Håndter ratebegrænsning
                        if response.status == 429:
                            self.metrics["rate_limited_count"] += 1
                            retry_after = int(response.headers.get("Retry-After", self.retry_delay * 2**attempt))
                            self.logger.warning(f"Rate limited, waiting {retry_after} seconds")
                            await asyncio.sleep(retry_after)
                            continue
                            
                        # Håndter serverfejl (500+)
                        if response.status >= 500:
                            if attempt < retry_attempts - 1:
                                delay = self.retry_delay * 2**attempt  # Eksponentiel backoff
                                self.logger.warning(f"Server error {response.status}, retry {attempt+1} after {delay}s")
                                self.metrics["retry_count"] += 1
                                await asyncio.sleep(delay)
                                continue
                                
                        # Håndter client errors (400-499, men ikke 429)
                        if 400 <= response.status < 500 and response.status != 429:
                            error_text = await response.text()
                            self.metrics["requests_failed"] += 1
                            self.logger.error(f"Client error {response.status}: {error_text}")
                            
                            # Forsøg at parse fejlmeddelelse fra JSON hvis muligt
                            try:
                                error_json = await response.json()
                                if isinstance(error_json, dict) and "error" in error_json:
                                    error_detail = error_json["error"]
                                    raise Exception(f"API error: {response.status} - {error_detail}")
                            except:
                                pass
                                
                            raise Exception(f"API error: {response.status} - {error_text}")
                            
                        # Håndter succesfulde svar (200-299)
                        if 200 <= response.status < 300:
                            result = await response.json()
                            self.metrics["requests_success"] += 1
                            elapsed = time.time() - start_time
                            self.metrics["total_time"] += elapsed
                            self.metrics["avg_response_time"] = (
                                self.metrics["total_time"] / self.metrics["requests_success"]
                            )
                            self.logger.debug(f"API call to {endpoint} successful in {elapsed:.2f}s")
                            return result
                            
                        # Andre uventede statuskoder
                        error_text = await response.text()
                        self.metrics["requests_failed"] += 1
                        self.logger.error(f"Unexpected API response: {response.status} - {error_text}")
                        raise Exception(f"Unexpected API status: {response.status}")
                        
                except asyncio.TimeoutError:
                    self.metrics["timeout_count"] += 1
                    if attempt < retry_attempts - 1:
                        delay = self.retry_delay * 2**attempt
                        self.logger.warning(f"Request timeout, retry {attempt+1} after {delay}s")
                        await asyncio.sleep(delay)
                    else:
                        self.metrics["requests_failed"] += 1
                        elapsed = time.time() - start_time
                        self.logger.error(f"Request timed out after {elapsed:.2f}s and {retry_attempts} attempts")
                        raise Exception(f"API request timed out after {retry_attempts} attempts")
                except Exception as e:
                    if attempt < retry_attempts - 1:
                        delay = self.retry_delay * 2**attempt
                        self.logger.warning(f"Request error: {str(e)}, retry {attempt+1} after {delay}s")
                        self.metrics["retry_count"] += 1
                        await asyncio.sleep(delay)
                    else:
                        self.metrics["requests_failed"] += 1
                        self.logger.error(f"Request failed after {retry_attempts} attempts: {str(e)}")
                        raise
            
            # Hvis vi når hertil, er alle forsøg fejlet
            self.metrics["requests_failed"] += 1
            raise Exception(f"All {retry_attempts} request attempts failed")

    async def batch_requests(
        self,
        requests: List[Dict[str, Any]],
        max_concurrent: int = None
    ) -> List[Dict[str, Any]]:
        """
        Udfører flere API-forespørgsler parallelt, med respekt for ratelimits.
        
        Args:
            requests: Liste af forespørgselsobjekter med følgende felter:
                - method: HTTP-metode
                - endpoint: API-endepunkt
                - payload: JSON-payload
                - headers: Ekstra headers
                - query_params: URL-parametre
            max_concurrent: Maks antal samtidige forespørgsler
            
        Returns:
            Liste af API-svar i samme rækkefølge som forespørgslerne
        """
        if not requests:
            return []
            
        if max_concurrent:
            # Skab en midlertidig semafor for denne batch
            batch_semaphore = asyncio.Semaphore(max_concurrent)
        else:
            # Brug den globale semafor
            batch_semaphore = self.semaphore
            
        async def call_api_with_params(req):
            async with batch_semaphore:
                return await self.request(
                    method=req.get("method", "POST"),
                    endpoint=req.get("endpoint", ""),
                    payload=req.get("payload"),
                    headers=req.get("headers"),
                    query_params=req.get("query_params"),
                    timeout_override=req.get("timeout"),
                    retry_attempts_override=req.get("retry_attempts")
                )
        
        # Start alle forespørgsler og vent på resultater
        tasks = [call_api_with_params(req) for req in requests]
        return await asyncio.gather(*tasks, return_exceptions=True)
        
    def get_metrics(self) -> Dict[str, Union[int, float]]:
        """Returnerer metrikker om API-forbrug og ydelse"""
        return self.metrics.copy()

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

async def fetch_expert_sources_for_angles(
    angles: List[Dict[str, Any]], 
    topic: str,
    generate_expert_sources_func: Callable,
    bypass_cache: bool = False,
    max_workers: int = 3,
    max_angles: int = 3,
    progress_callback: Optional[Callable[[int], Any]] = None
) -> List[Dict[str, Any]]:
    """
    Parallelt henter ekspertkilder for flere vinkler ved hjælp af concurrent.futures.
    
    Args:
        angles: Liste af vinkelobjekter
        topic: Hovedemnet
        generate_expert_sources_func: Funktion til at generere ekspertkilder
        bypass_cache: Om cache skal ignoreres
        max_workers: Maksimalt antal samtidige workers
        max_angles: Maksimalt antal vinkler at hente kilder til
        progress_callback: Callback-funktion til fremskridtsrapportering
        
    Returns:
        Liste af vinkler med tilføjede ekspertkilder
    """
    if not angles:
        return []
    
    # Tag en kopi af vinklerne så vi ikke ændrer originalen
    enriched_angles = angles.copy()
    
    # Begrænset til top-N vinkler
    angles_to_process = angles[:max_angles] if len(angles) > max_angles else angles
    
    logger.info(f"Fetching expert sources for {len(angles_to_process)} angles with {max_workers} workers")
    
    # Forbered loop til at køre async kode
    loop = asyncio.get_event_loop()
    
    def run_async_fetch(angle_index: int, angle: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """Kører async kode i en thread til at hente ekspertkilder"""
        headline = angle.get('overskrift', f"Vinkel om {topic}")
        description = angle.get('beskrivelse', "")
        
        # Opret en ny event loop til denne thread
        asyncio.set_event_loop(asyncio.new_event_loop())
        
        try:
            # Kør den asynkrone funktion og få resultatet
            expert_sources = asyncio.run(generate_expert_sources_func(
                topic=topic,
                angle_headline=headline,
                angle_description=description,
                bypass_cache=bypass_cache
            ))
            
            # Rapporter fremskridt hvis callback er givet
            if progress_callback and callable(progress_callback):
                asyncio.run(progress_callback((angle_index + 1) * 100 // len(angles_to_process)))
                
            return angle_index, expert_sources
        except Exception as e:
            logger.error(f"Error fetching expert sources for angle {angle_index}: {e}")
            return angle_index, None
    
    try:
        # Opret ThreadPoolExecutor med det angivne antal workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Send hver vinkel til en thread og få en future
            future_to_angle = {
                executor.submit(run_async_fetch, i, angle): (i, angle) 
                for i, angle in enumerate(angles_to_process)
            }
            
            # Behandl resultaterne, efterhånden som de bliver færdige
            for future in concurrent.futures.as_completed(future_to_angle):
                try:
                    angle_index, expert_sources = future.result()
                    
                    # Hvis vi fik ekspertkilder, tilføj dem til vinklen
                    if expert_sources is not None and angle_index < len(enriched_angles):
                        enriched_angles[angle_index]['ekspertKilder'] = expert_sources
                        enriched_angles[angle_index]['harEkspertKilder'] = True
                    elif angle_index < len(enriched_angles):
                        # Tilføj en tom struktur hvis der ikke var nogen kilder
                        enriched_angles[angle_index]['ekspertKilder'] = {
                            "experts": [],
                            "institutions": [],
                            "data_sources": [],
                            "error": "Kunne ikke generere ekspertkilder"
                        }
                        enriched_angles[angle_index]['harEkspertKilder'] = False
                except Exception as e:
                    logger.error(f"Error processing expert sources result: {e}")
    
    except Exception as e:
        logger.error(f"Error in fetch_expert_sources_for_angles: {e}")
    
    return enriched_angles

async def batch_generate_angles(
    topic: str,
    profile: Any, # RedaktionelDNA
    generate_angles_func: Callable,
    batch_count: int = 3,
    angles_per_batch: int = 4,
    bypass_cache: bool = False,
    progress_callback: Optional[Callable[[int, str], Any]] = None
) -> List[Dict[str, Any]]:
    """
    Genererer vinkler i flere parallelle batches med forskellige fokusområder
    for bedre ydeevne, robusthed og variation.
    
    Args:
        topic: Emne at generere vinkler for
        profile: Redaktionel DNA-profil
        generate_angles_func: Funktion til at generere vinkler
        batch_count: Antal parallelle batches
        angles_per_batch: Antal vinkler at generere per batch
        bypass_cache: Om cache skal ignoreres
        progress_callback: Callback-funktion for statusopdateringer
        
    Returns:
        Kombineret liste af genererede vinkler fra alle batches
    """
    import copy
    
    # Opdel profilen i forskellige fokusområder for hver batch
    batches = []
    
    # Hent alle fokusområder fra profilen (håndter forskellige attributnavne)
    all_focus_areas = []
    if hasattr(profile, 'fokusOmrader') and profile.fokusOmrader:
        all_focus_areas = profile.fokusOmrader
    elif hasattr(profile, 'fokus_omrader') and profile.fokus_omrader:
        all_focus_areas = profile.fokus_omrader
    
    if not all_focus_areas:
        all_focus_areas = ["Generelt"]
    
    # Hent nyhedskriterier for variationer
    news_criteria = {}
    if hasattr(profile, 'nyhedsprioritering') and profile.nyhedsprioritering:
        news_criteria = profile.nyhedsprioritering
    
    # Opret variationer af prompt til hver batch
    for i in range(batch_count):
        # Cirkulær fordeling af fokusområder for variation
        focus_start = i % max(1, len(all_focus_areas))
        selected_focus = all_focus_areas[focus_start:] + all_focus_areas[:focus_start]
        selected_focus = selected_focus[:max(2, len(selected_focus) // batch_count)]
        
        # Vælg forskellige nyhedskriterier for hver batch for variation
        selected_criteria = {}
        if news_criteria:
            # Vælg prioriterede kriterier baseret på batch-nummer
            sorted_criteria = sorted(news_criteria.items(), key=lambda x: x[1], reverse=True)
            offset = i % len(sorted_criteria)
            for j in range(min(3, len(sorted_criteria))):
                idx = (offset + j) % len(sorted_criteria)
                selected_criteria[sorted_criteria[idx][0]] = sorted_criteria[idx][1]
        
        # Opret en batch med varieret fokus
        batch_profile = copy.deepcopy(profile)
        
        # Sæt fokus og nyhedskriterier for denne batch
        if hasattr(batch_profile, 'fokusOmrader'):
            batch_profile.fokusOmrader = selected_focus
        if hasattr(batch_profile, 'fokus_omrader'):
            batch_profile.fokus_omrader = selected_focus
            
        # Gem batch info
        batches.append({
            "profile": batch_profile,
            "focus": ", ".join(selected_focus),
            "criteria": selected_criteria,
            "max_angles": angles_per_batch,
            "batch_id": i
        })
    
    # Rapporter start på batch processing
    if progress_callback:
        await progress_callback(5, f"Starter generering af vinkler i {batch_count} batches")
    
    # Funktion til at generere en enkelt batch
    async def generate_batch(batch_info):
        batch_id = batch_info["batch_id"]
        batch_profile = batch_info["profile"]
        focus = batch_info["focus"]
        max_angles = batch_info["max_angles"]
        
        try:
            # Generer vinkler for denne batch
            logger.info(f"Generating batch {batch_id} with focus: {focus}")
            
            angles = await generate_angles_func(
                topic=topic,
                profile=batch_profile,
                bypass_cache=bypass_cache
            )
            
            # Tag de bedste vinkler fra denne batch
            result = angles[:max_angles] if angles and len(angles) > max_angles else angles
            
            # Marker hver vinkel med batch info for debugging
            if result:
                for angle in result:
                    if isinstance(angle, dict):
                        angle['_batch_id'] = batch_id
                        angle['_batch_focus'] = focus
            
            logger.info(f"Batch {batch_id} complete, generated {len(result) if result else 0} angles")
            return result or []
        except Exception as e:
            logger.error(f"Error generating batch {batch_id}: {str(e)}")
            return []
    
    # Eksekver alle batches parallelt
    batch_tasks = []
    for batch in batches:
        task = asyncio.create_task(generate_batch(batch))
        batch_tasks.append(task)
    
    # Vent på at batch-tasks bliver færdige
    all_angles = []
    for i, task in enumerate(asyncio.as_completed(batch_tasks)):
        try:
            angles = await task
            all_angles.extend(angles)
            
            # Opdater fremskridt
            if progress_callback:
                progress = 10 + ((i + 1) * 60) // len(batches)
                await progress_callback(progress, f"Batch {i+1}/{len(batches)} færdig")
                
        except Exception as e:
            logger.error(f"Error waiting for batch task: {str(e)}")
    
    # Fjern eventuelle duplikater baseret på overskrift
    if progress_callback:
        await progress_callback(70, "Fjerner duplikate vinkler")
        
    unique_angles = {}
    for angle in all_angles:
        if isinstance(angle, dict) and "overskrift" in angle:
            headline = angle["overskrift"].lower()
            # Behold den første forekomst af hver overskrift (eller den med højere score)
            if headline not in unique_angles:
                unique_angles[headline] = angle
            # Alternativt, behold den med mest detaljeret beskrivelse
            elif len(angle.get("beskrivelse", "")) > len(unique_angles[headline].get("beskrivelse", "")):
                unique_angles[headline] = angle
    
    # Konverter tilbage til liste
    result_angles = list(unique_angles.values())
    
    # Sortér og rangér vinklerne, hvis det er nødvendigt
    try:
        if progress_callback:
            await progress_callback(80, "Sorterer og rangerer vinkler")
            
        # Importér rangerings-funktion hvis tilgængelig
        try:
            from angle_processor import filter_and_rank_angles
            result_angles = filter_and_rank_angles(result_angles, profile, min(15, len(result_angles)))
        except (ImportError, ModuleNotFoundError):
            # Fallback til simpel sorting efter beskrivelseslængde
            result_angles = sorted(
                result_angles,
                key=lambda x: len(x.get("beskrivelse", "")),
                reverse=True
            )
    except Exception as e:
        logger.error(f"Error sorting angles: {str(e)}")
    
    if progress_callback:
        await progress_callback(90, f"Færdig - genererede {len(result_angles)} unikke vinkler")
        
    return result_angles

class ProgressiveResultProcessor:
    """
    Håndterer progressiv visning af resultater ved at sende delvise resultater
    til en callback-funktion efterhånden som de bliver tilgængelige.
    """
    
    def __init__(
        self,
        result_callback: Callable[[Dict[str, Any]], Awaitable[None]],
        progress_callback: Optional[Callable[[int, str], Awaitable[None]]] = None,
        result_id: str = None
    ):
        """
        Initialiserer resultatprocessoren.
        
        Args:
            result_callback: Asynkron callback der kaldes med delvise resultater
            progress_callback: Asynkron callback for fremskridtsrapportering
            result_id: Unik ID til at identificere denne resultatsession
        """
        import uuid
        self.result_callback = result_callback
        self.progress_callback = progress_callback
        self.result_id = result_id or str(uuid.uuid4())
        self.completed = False
        self.angles = []
        self.last_update_time = time.time()
        self.update_interval = 0.5  # Sekunder mellem opdateringer
        self.start_time = time.time()
        self.has_error = False
        self.metadata = {}
        
    async def update_progress(self, percent: int, message: str):
        """Opdaterer fremskridt og sender til callback hvis defineret"""
        if self.progress_callback and callable(self.progress_callback):
            await self.progress_callback(percent, message)
            
        # Log progress
        logger.debug(f"Progress {percent}%: {message}")
    
    async def add_angle(self, angle: Dict[str, Any], send_update: bool = False):
        """
        Tilføjer en ny vinkel og sender opdatering hvis intervaltiden er nået
        eller send_update er True.
        """
        if not isinstance(angle, dict):
            logger.warning(f"Skipping non-dict angle: {type(angle)}")
            return
            
        self.angles.append(angle)
        
        # Kontroller om vi bør sende en opdatering
        current_time = time.time()
        if send_update or (current_time - self.last_update_time >= self.update_interval):
            await self.send_update()
            self.last_update_time = current_time
    
    async def add_angles(self, angles: List[Dict[str, Any]]):
        """Tilføjer flere vinkler på én gang og sender opdatering"""
        if not angles:
            return
            
        valid_angles = [a for a in angles if isinstance(a, dict)]
        if not valid_angles:
            return
            
        self.angles.extend(valid_angles)
        await self.send_update()
        self.last_update_time = time.time()
    
    async def send_update(self):
        """Sender den aktuelle tilstand til result_callback"""
        if not self.result_callback or not callable(self.result_callback):
            return
            
        result_package = {
            "result_id": self.result_id,
            "angles": self.angles.copy(),
            "completed": self.completed,
            "has_error": self.has_error,
            "timestamp": time.time(),
            "angle_count": len(self.angles),
            "elapsed_time": time.time() - self.start_time,
            "metadata": self.metadata
        }
        
        await self.result_callback(result_package)
    
    async def complete(self, metadata: Optional[Dict[str, Any]] = None):
        """Markerer resultaterne som færdige og sender den sidste opdatering"""
        self.completed = True
        
        # Tilføj metadata til den endelige pakke
        if metadata:
            self.metadata.update(metadata)
            
        self.metadata["final_angle_count"] = len(self.angles)
        self.metadata["total_time"] = time.time() - self.start_time
        
        # Send endelig opdatering
        await self.send_update()
            
        if self.progress_callback and callable(self.progress_callback):
            await self.progress_callback(100, "Færdig")
            
    async def set_error(self, error_message: str):
        """Markerer resultaterne som fejlede"""
        self.has_error = True
        self.metadata["error"] = error_message
        
        error_package = {
            "result_id": self.result_id,
            "angles": self.angles,
            "completed": False,
            "has_error": True,
            "error": error_message,
            "timestamp": time.time(),
            "elapsed_time": time.time() - self.start_time,
            "metadata": self.metadata
        }
        
        if self.result_callback and callable(self.result_callback):
            await self.result_callback(error_package)
            
        if self.progress_callback and callable(self.progress_callback):
            await self.progress_callback(-1, error_message)


async def progressive_angle_generation(
    topic: str,
    profile: Any,  # RedaktionelDNA
    result_callback: Callable[[Dict[str, Any]], Awaitable[None]],
    fetch_topic_func: Callable,
    generate_angles_func: Callable,
    generate_sources_func: Optional[Callable] = None,
    progress_callback: Optional[Callable[[int, str], Awaitable[None]]] = None,
    bypass_cache: bool = False,
    batch_count: int = 3,
    angles_per_batch: int = 4
) -> str:
    """
    Udfører vinkelgenerering med progressiv opdatering af resultater.
    
    Args:
        topic: Emnet at generere vinkler for
        profile: Redaktionel profil
        result_callback: Callback til at modtage delvise resultater
        fetch_topic_func: Funktion til at hente emneinfo
        generate_angles_func: Funktion til at generere vinkler
        generate_sources_func: Funktion til at generere ekspertkilder (valgfri)
        progress_callback: Callback til fremskridtsrapportering
        bypass_cache: Om cache skal ignoreres
        batch_count: Antal parallelle batches til vinkelgenerering
        angles_per_batch: Antal vinkler per batch
        
    Returns:
        Resultat-ID som kan bruges til at matche resultater fra callback
    """
    # Opret resultatprocessor
    processor = ProgressiveResultProcessor(
        result_callback=result_callback,
        progress_callback=progress_callback
    )
    
    # Send indledende status
    await processor.update_progress(5, f"Starter vinkelgenerering for '{topic}'...")
    
    try:
        # 1. Hent baggrundsinformation om emnet
        await processor.update_progress(10, "Henter baggrundsinformation...")
        
        topic_info = await fetch_topic_func(topic, bypass_cache=bypass_cache)
        
        if not topic_info or not isinstance(topic_info, str):
            topic_info = f"Emnet handler om {topic}. Ingen yderligere baggrundsinformation tilgængelig."
        
        processor.metadata["topic_info_length"] = len(topic_info)
        processor.metadata["topic"] = topic
        
        # 2. Generer vinkler i batches for bedre variation
        await processor.update_progress(20, "Genererer vinkler i batches...")
        
        # Funktion til at generere en enkelt batch
        async def generate_batch(batch_id: int, batch_profile: Any):
            try:
                # Opret variation i profilen for denne batch
                import copy
                modified_profile = copy.deepcopy(batch_profile)
                
                # Hent alle fokusområder (håndter forskellige attributnavne)
                all_focus_areas = []
                if hasattr(modified_profile, 'fokusOmrader') and modified_profile.fokusOmrader:
                    all_focus_areas = modified_profile.fokusOmrader
                elif hasattr(modified_profile, 'fokus_omrader') and modified_profile.fokus_omrader:
                    all_focus_areas = modified_profile.fokus_omrader
                
                if all_focus_areas:
                    # Rotér fokusområder baseret på batch-ID
                    offset = batch_id % max(1, len(all_focus_areas))
                    selected_focus = all_focus_areas[offset:] + all_focus_areas[:offset]
                    
                    # Begrænset til 2-3 fokusområder
                    selected_focus = selected_focus[:min(3, len(selected_focus))]
                    
                    # Opdater profilen
                    if hasattr(modified_profile, 'fokusOmrader'):
                        modified_profile.fokusOmrader = selected_focus
                    elif hasattr(modified_profile, 'fokus_omrader'):
                        modified_profile.fokus_omrader = selected_focus
                
                # Generer vinkler
                angles = await generate_angles_func(
                    topic=topic,
                    profile=modified_profile,
                    bypass_cache=bypass_cache
                )
                
                # Tag de bedste vinkler fra denne batch
                result = angles[:angles_per_batch] if angles and len(angles) > angles_per_batch else angles
                
                if result:
                    # Tilføj batch-info til hver vinkel
                    for angle in result:
                        if isinstance(angle, dict):
                            angle['_batch_id'] = batch_id
                    
                    # Tilføj progressivt til resultater
                    for angle in result:
                        await processor.add_angle(angle)
                
                return result or []
            except Exception as e:
                logger.error(f"Error generating batch {batch_id}: {str(e)}")
                return []
        
        # Start alle batches parallelt
        batch_tasks = [
            asyncio.create_task(generate_batch(i, profile))
            for i in range(batch_count)
        ]
        
        # Behandl resultaterne, efterhånden som de kommer ind
        for i, task in enumerate(asyncio.as_completed(batch_tasks)):
            try:
                # Vi behøver ikke resultatet her, da vinkler allerede er tilføjet via add_angle
                await task
                # Opdater fremskridt
                progress = 25 + ((i + 1) * 30) // batch_count
                await processor.update_progress(progress, f"Batch {i+1}/{batch_count} modtaget")
            except Exception as e:
                logger.error(f"Error waiting for batch {i}: {str(e)}")
        
        # Berig vinkler med kildeforslag
        if generate_sources_func and callable(generate_sources_func):
            await processor.update_progress(60, "Finder ekspertkilder...")
            
            # Begrænset til top 3 vinkler for at spare API-kald
            top_angles = processor.angles[:3] if len(processor.angles) >= 3 else processor.angles
            enriched_angles = await fetch_expert_sources_for_angles(
                angles=top_angles,
                topic=topic,
                generate_expert_sources_func=generate_sources_func,
                bypass_cache=bypass_cache,
                max_workers=len(top_angles)
            )
            
            # Opdater de originale vinkler med ekspertkilder
            for i, angle in enumerate(enriched_angles):
                if i < len(processor.angles):
                    if 'ekspertKilder' in angle:
                        processor.angles[i]['ekspertKilder'] = angle['ekspertKilder']
                        processor.angles[i]['harEkspertKilder'] = angle['harEkspertKilder']
                        
                        # Send update efter hver berigelse
                        await processor.send_update()
        
        # Sorter og ranger vinkler
        await processor.update_progress(80, "Sorterer og prioriterer vinkler...")
        
        try:
            from angle_processor import filter_and_rank_angles
            ranked_angles = filter_and_rank_angles(processor.angles, profile, 10)
            
            # Opdater angles med de rangerede resultater
            processor.angles = ranked_angles
            
            # Send en opdatering med de rangerede vinkler
            await processor.send_update()
        except (ImportError, ModuleNotFoundError):
            # Fallback til simpel sorting hvis angle_processor ikke er tilgængelig
            processor.angles = sorted(
                processor.angles,
                key=lambda x: len(x.get("beskrivelse", "")),
                reverse=True
            )
        except Exception as e:
            logger.error(f"Error ranking angles: {str(e)}")
        
        # Opdater og færdiggør
        await processor.update_progress(90, "Finaliserer resultater...")
        
        # Marker processen som afsluttet
        await processor.complete({
            "processing_time": time.time() - processor.start_time,
            "total_angles_generated": len(processor.angles),
            "topic_info_summary": topic_info[:100] + "..." if len(topic_info) > 100 else topic_info
        })
        
        return processor.result_id
        
    except Exception as e:
        logger.error(f"Error in progressive_angle_generation: {str(e)}")
        await processor.set_error(f"Fejl under vinkelgenerering: {str(e)}")
        return processor.result_id