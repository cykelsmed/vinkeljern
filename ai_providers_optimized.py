"""
AI providers module for Vinkeljernet project - Optimized Version

This module provides a unified abstraction layer for different AI providers
like OpenAI (GPT), Anthropic (Claude), and Perplexity. It handles authentication,
request formatting, error handling, and response parsing for each provider.

The optimized version includes:
1. True parallel request handling
2. Efficient model selection based on task complexity
3. Enhanced caching with LRU and memory/disk strategies 
4. Improved error handling with fallbacks
5. Performance monitoring and circuit breakers
"""

import os
import json
import time
import logging
import ssl
import certifi
import requests
import asyncio
import aiohttp
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Type, Union, Callable, Tuple
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

# Import configuration
from config import OPENAI_API_KEY, ANTHROPIC_API_KEY, PERPLEXITY_API_KEY
from retry_manager import (
    retry_with_circuit_breaker, 
    CircuitOpenError, 
    MaxRetriesExceededError,
    get_circuit_stats
)
from cache_manager import cached_api
from error_handling import (
    APIKeyMissingError,
    APIConnectionError,
    APIResponseError,
    SSLVerificationError,
    log_info, 
    log_warning, 
    log_error
)

# Configure module logger
logger = logging.getLogger("vinkeljernet.ai_providers_optimized")

# Maximum concurrent requests 
MAX_CONCURRENT_REQUESTS = 10

# Connection pools
_thread_pool = None
_session_pool = {}

# Performance metrics
@dataclass
class ProviderMetrics:
    """Track performance metrics for providers."""
    requests: int = 0
    successes: int = 0
    failures: int = 0
    total_latency: float = 0.0
    max_latency: float = 0.0
    min_latency: float = float('inf')
    cache_hits: int = 0
    start_time: float = field(default_factory=time.time)
    
    def record_request(self, success: bool, latency: float, cache_hit: bool = False):
        """Record a request and its metrics."""
        self.requests += 1
        
        if success:
            self.successes += 1
        else:
            self.failures += 1
            
        if cache_hit:
            self.cache_hits += 1
        else:
            self.total_latency += latency
            self.max_latency = max(self.max_latency, latency)
            self.min_latency = min(self.min_latency, latency) if latency > 0 else self.min_latency
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current metrics as dictionary."""
        uptime = time.time() - self.start_time
        total_api_calls = self.requests - self.cache_hits
        
        avg_latency = self.total_latency / max(1, total_api_calls) if total_api_calls > 0 else 0
        success_rate = (self.successes / max(1, self.requests)) * 100 if self.requests > 0 else 0
        cache_hit_rate = (self.cache_hits / max(1, self.requests)) * 100 if self.requests > 0 else 0
        
        return {
            "uptime_seconds": uptime,
            "uptime_formatted": f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m {int(uptime % 60)}s",
            "total_requests": self.requests,
            "successful_requests": self.successes,
            "failed_requests": self.failures,
            "success_rate": f"{success_rate:.1f}%", 
            "average_latency_ms": int(avg_latency * 1000),
            "min_latency_ms": int(self.min_latency * 1000) if self.min_latency != float('inf') else 0,
            "max_latency_ms": int(self.max_latency * 1000),
            "cache_hits": self.cache_hits,
            "cache_hit_rate": f"{cache_hit_rate:.1f}%"
        }

# Global metrics
metrics = {
    "anthropic": ProviderMetrics(),
    "openai": ProviderMetrics(),
    "perplexity": ProviderMetrics()
}

def get_thread_pool(max_workers=MAX_CONCURRENT_REQUESTS):
    """Get or create the global thread pool executor."""
    global _thread_pool
    if _thread_pool is None:
        _thread_pool = ThreadPoolExecutor(max_workers=max_workers)
    return _thread_pool

async def get_aiohttp_session(key="default", timeout=60):
    """Get or create an aiohttp session from the pool."""
    global _session_pool
    
    if key not in _session_pool or _session_pool[key].closed:
        # Configure timeout
        timeout_config = aiohttp.ClientTimeout(total=timeout)
        
        # Create connector with SSL context
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        connector = aiohttp.TCPConnector(ssl=ssl_context, limit=MAX_CONCURRENT_REQUESTS)
        
        _session_pool[key] = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout_config,
            raise_for_status=False
        )
        
    return _session_pool[key]

async def close_all_sessions():
    """Close all aiohttp sessions in the pool."""
    for key, session in list(_session_pool.items()):
        if not session.closed:
            await session.close()
    
    _session_pool.clear()

class ProviderType(Enum):
    """Enum for supported AI providers."""
    OPENAI = auto()
    ANTHROPIC = auto()
    PERPLEXITY = auto()

class ModelTier(Enum):
    """Capability tiers for AI models."""
    FAST = auto()    # Fastest models for simple tasks
    STANDARD = auto() # Good balance of speed/capability 
    ADVANCED = auto() # Most capable models for complex tasks

@dataclass
class ProviderConfig:
    """Configuration for AI provider."""
    api_key: str
    api_url: str = ""
    model: str = ""
    max_tokens: int = 1000
    temperature: float = 0.7
    timeout: int = 60
    dev_mode: bool = False
    tier: ModelTier = ModelTier.STANDARD
    extra_options: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.api_key:
            raise APIKeyMissingError("API key is required")
        
        # If API URL is provided, validate it's a proper URL
        if self.api_url:
            try:
                parsed = urlparse(self.api_url)
                if not all([parsed.scheme, parsed.netloc]):
                    raise ValueError(f"Invalid API URL: {self.api_url}")
            except Exception as e:
                raise ValueError(f"Invalid API URL: {self.api_url} - {str(e)}")

@dataclass
class ProviderResponse:
    """Standardized response from AI providers."""
    content: str
    usage: Dict[str, Any] = field(default_factory=dict)
    model: str = ""
    raw_response: Any = None
    created_at: float = field(default_factory=time.time)
    latency: float = 0.0
    
    @property
    def token_usage(self) -> int:
        """Get total token usage if available."""
        return self.usage.get('total_tokens', 0)

class BaseProvider(ABC):
    """Abstract base class for AI providers."""
    
    def __init__(self, config: ProviderConfig):
        """
        Initialize the provider.
        
        Args:
            config: Provider configuration.
        """
        self.config = config
        self.validate_config()
        self._setup_logging()
        self.metrics = None
        self._lock = threading.RLock()
    
    def validate_config(self) -> None:
        """Validate provider-specific configuration."""
        if not self.config.api_key:
            raise APIKeyMissingError(f"API key is required for {self.__class__.__name__}")
    
    def _setup_logging(self) -> None:
        """Set up logging for this provider."""
        self.logger = logging.getLogger(f"vinkeljernet.ai_providers.{self.__class__.__name__}")
    
    def _get_metrics(self) -> ProviderMetrics:
        """Get the metrics object for this provider."""
        provider_name = self.__class__.__name__.lower().replace('provider', '')
        
        if provider_name.startswith('anthropic'):
            return metrics["anthropic"]
        elif provider_name.startswith('openai'):
            return metrics["openai"]
        elif provider_name.startswith('perplexity'):
            return metrics["perplexity"]
        else:
            # Default to a new metrics object if provider not recognized
            if self.metrics is None:
                self.metrics = ProviderMetrics()
            return self.metrics
    
    def get_model_for_tier(self, tier: ModelTier) -> str:
        """
        Get the appropriate model name for the requested tier.
        
        Args:
            tier: The capability tier requested
            
        Returns:
            str: Model name appropriate for the provider and tier
        """
        # Default implementation returns the configured model
        return self.config.model
    
    @abstractmethod
    async def generate_async(self, prompt: str, **kwargs) -> ProviderResponse:
        """
        Generate a response asynchronously.
        
        Args:
            prompt: The prompt to send to the AI model.
            **kwargs: Additional provider-specific options.
            
        Returns:
            ProviderResponse: The generated response.
        """
        pass
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> ProviderResponse:
        """
        Generate a response synchronously.
        
        Args:
            prompt: The prompt to send to the AI model.
            **kwargs: Additional provider-specific options.
            
        Returns:
            ProviderResponse: The generated response.
        """
        pass

    async def create_secure_api_session(self) -> aiohttp.ClientSession:
        """
        Create a secure aiohttp session with proper SSL handling.
        
        Returns:
            aiohttp.ClientSession: A configured session.
        """
        # Configure SSL context
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        
        if self.config.dev_mode:
            # Warning: This disables SSL verification and should only be used in development
            import warnings
            warnings.warn(
                "⚠️ WARNING: SSL verification disabled! This is insecure and should ONLY be used in development.",
                category=Warning
            )
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
        
        # Configure timeout
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        
        # Create connector with SSL context
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        
        # Create session
        return aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )

class AnthropicProvider(BaseProvider):
    """Provider implementation for Anthropic Claude API."""
    
    def __init__(self, config: Optional[ProviderConfig] = None):
        """
        Initialize the Anthropic provider.
        
        Args:
            config: Provider configuration. If None, uses default config from environment.
        """
        # Default configuration
        if config is None:
            config = ProviderConfig(
                api_key=ANTHROPIC_API_KEY,
                api_url="https://api.anthropic.com/v1/messages",
                model="claude-3-opus-20240229",
                max_tokens=1000,
                temperature=0.7
            )
        
        super().__init__(config)
    
    def validate_config(self) -> None:
        """Validate Anthropic-specific configuration."""
        super().validate_config()
        
        if not self.config.model:
            self.config.model = "claude-3-opus-20240229"
        
        valid_models = {
            "claude-3-opus-20240229", 
            "claude-3-sonnet-20240229", 
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0",
        }
        
        if self.config.model not in valid_models:
            log_warning(f"Unrecognized Anthropic model: {self.config.model}. " 
                        f"Recognized models are: {', '.join(valid_models)}")

    def get_model_for_tier(self, tier: ModelTier) -> str:
        """Get the appropriate Claude model for the requested tier."""
        if tier == ModelTier.FAST:
            return "claude-3-haiku-20240307"
        elif tier == ModelTier.STANDARD:
            return "claude-3-sonnet-20240229"
        elif tier == ModelTier.ADVANCED:
            return "claude-3-opus-20240229"
        else:
            return self.config.model  # Default

    @cached_api(ttl=3600)
    @retry_with_circuit_breaker(
        max_retries=3,
        initial_backoff=1.0,
        backoff_factor=2.0,
        exceptions=[requests.RequestException, ConnectionError, TimeoutError],
        circuit_name="anthropic_api"
    )
    def generate(self, prompt: str, **kwargs) -> ProviderResponse:
        """
        Generate a response synchronously using the Anthropic Claude API.
        
        Args:
            prompt: The prompt to send to Claude.
            **kwargs: Additional options to override default configuration.
            
        Returns:
            ProviderResponse: The generated response.
        """
        # Process special parameters
        bypass_cache = kwargs.pop('bypass_cache', False)
        tier = kwargs.pop('tier', None)
        
        # Merge config with any overrides from kwargs
        max_tokens = kwargs.get('max_tokens', self.config.max_tokens)
        temperature = kwargs.get('temperature', self.config.temperature)
        
        # Use tier-based model selection if requested
        if tier is not None:
            model = self.get_model_for_tier(tier)
        else:
            model = kwargs.get('model', self.config.model)
            
        system_prompt = kwargs.get('system_prompt', 
                              "Du er en erfaren journalist med ekspertise i at udvikle kreative og relevante nyhedsvinkler.")
        
        # Log masked API key
        masked_key = self.config.api_key[:5] + "..." + self.config.api_key[-3:] if len(self.config.api_key) > 8 else "INVALID"
        log_info(f"Using Anthropic API key (masked): {masked_key}")
        
        start_time = time.time()
        cache_hit = False
        success = False
        
        try:
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.config.api_key,
                "anthropic-version": "2023-06-01"
            }
            
            payload = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system_prompt,
                "messages": [{"role": "user", "content": prompt}],
            }
            
            # Make the API call
            response = requests.post(
                self.config.api_url,
                headers=headers,
                json=payload,
                timeout=self.config.timeout
            )
            
            # Check for errors
            if response.status_code != 200:
                error_text = response.text
                try:
                    error_json = json.loads(error_text)
                    error_message = error_json.get('error', {}).get('message', error_text)
                except json.JSONDecodeError:
                    error_message = error_text
                
                log_error(f"Anthropic API Error: Status {response.status_code}: {error_message}")
                raise APIResponseError(f"Anthropic API Error: {error_message}", 
                                      status_code=response.status_code, 
                                      response_body=error_text)
            
            # Parse and return the response
            response_data = response.json()
            content = response_data['content'][0]['text']
            
            # Extract usage if available
            usage = {}
            if 'usage' in response_data:
                usage = response_data['usage']
            
            latency = time.time() - start_time
            success = True
            
            result = ProviderResponse(
                content=content,
                usage=usage,
                model=model,
                raw_response=response_data,
                latency=latency
            )
            
            # Record metrics
            self._get_metrics().record_request(
                success=success,
                latency=latency,
                cache_hit=cache_hit
            )
            
            return result
            
        except requests.exceptions.SSLError as e:
            log_error(f"SSL Error: {e}")
            raise SSLVerificationError(f"SSL Error when connecting to Anthropic API: {e}")
        
        except requests.exceptions.ConnectionError as e:
            log_error(f"Connection Error: {e}")
            raise APIConnectionError(f"Failed to connect to Anthropic API: {e}")
        
        except requests.exceptions.Timeout as e:
            log_error(f"Timeout Error: {e}")
            raise APIConnectionError(f"Anthropic API request timed out: {e}")
        
        except requests.exceptions.RequestException as e:
            log_error(f"Request Error: {e}")
            raise APIConnectionError(f"Anthropic API request failed: {e}")
        
        except Exception as e:
            log_error(f"Unexpected error with Anthropic API: {e}")
            raise
        finally:
            # Record metrics if not already done
            if not success:
                latency = time.time() - start_time
                self._get_metrics().record_request(
                    success=False,
                    latency=latency,
                    cache_hit=cache_hit
                )

    @cached_api(ttl=3600)
    @retry_with_circuit_breaker(
        max_retries=3,
        initial_backoff=1.0,
        backoff_factor=2.0,
        exceptions=[aiohttp.ClientError, asyncio.TimeoutError, ConnectionError],
        circuit_name="anthropic_api_async"
    )
    async def generate_async(self, prompt: str, **kwargs) -> ProviderResponse:
        """
        Generate a response asynchronously using the Anthropic Claude API.
        
        Args:
            prompt: The prompt to send to Claude.
            **kwargs: Additional options to override default configuration.
            
        Returns:
            ProviderResponse: The generated response.
        """
        # Process special parameters
        bypass_cache = kwargs.pop('bypass_cache', False)
        tier = kwargs.pop('tier', None)
        
        # Merge config with any overrides from kwargs
        max_tokens = kwargs.get('max_tokens', self.config.max_tokens)
        temperature = kwargs.get('temperature', self.config.temperature)
        
        # Use tier-based model selection if requested
        if tier is not None:
            model = self.get_model_for_tier(tier)
        else:
            model = kwargs.get('model', self.config.model)
            
        system_prompt = kwargs.get('system_prompt', 
                              "Du er en erfaren journalist med ekspertise i at udvikle kreative og relevante nyhedsvinkler.")
        
        # Check if streaming is requested
        use_streaming = kwargs.get('streaming', False)
        
        # Log masked API key
        masked_key = self.config.api_key[:5] + "..." + self.config.api_key[-3:] if len(self.config.api_key) > 8 else "INVALID"
        log_info(f"Using Anthropic API key (masked): {masked_key}")
        
        start_time = time.time()
        cache_hit = False
        success = False
        
        try:
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.config.api_key,
                "anthropic-version": "2023-06-01"
            }
            
            payload = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system_prompt,
                "messages": [{"role": "user", "content": prompt}],
            }
            
            if use_streaming:
                # Add streaming parameters
                payload["stream"] = True
                headers["anthropic-beta"] = "claude-3-opus-20240229-streaming-v1"
                
                # Use streaming implementation
                return await self._generate_streaming(headers, payload)
            
            # Create a secure session
            session_key = f"anthropic_{model}"
            session = await get_aiohttp_session(key=session_key, timeout=self.config.timeout)
            
            # Make the API call
            async with session.post(
                self.config.api_url,
                headers=headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    try:
                        error_json = json.loads(error_text)
                        error_message = error_json.get('error', {}).get('message', error_text)
                    except json.JSONDecodeError:
                        error_message = error_text
                    
                    log_error(f"Anthropic API Error: Status {response.status}: {error_message}")
                    raise APIResponseError(f"Anthropic API Error: {error_message}", 
                                          status_code=response.status, 
                                          response_body=error_text)
                
                # Parse and return the response
                response_data = await response.json()
                content = response_data['content'][0]['text']
                
                # Extract usage if available
                usage = {}
                if 'usage' in response_data:
                    usage = response_data['usage']
                
                latency = time.time() - start_time
                success = True
                
                result = ProviderResponse(
                    content=content,
                    usage=usage,
                    model=model,
                    raw_response=response_data,
                    latency=latency
                )
                
                # Record metrics
                self._get_metrics().record_request(
                    success=success,
                    latency=latency,
                    cache_hit=cache_hit
                )
                
                return result
                
        except aiohttp.ClientSSLError as e:
            log_error(f"SSL Error: {e}")
            raise SSLVerificationError(f"SSL Error when connecting to Anthropic API: {e}")
        
        except aiohttp.ClientConnectorError as e:
            log_error(f"Connection Error: {e}")
            raise APIConnectionError(f"Failed to connect to Anthropic API: {e}")
        
        except asyncio.TimeoutError as e:
            log_error(f"Timeout Error: {e}")
            raise APIConnectionError(f"Anthropic API request timed out: {e}")
        
        except aiohttp.ClientError as e:
            log_error(f"Request Error: {e}")
            raise APIConnectionError(f"Anthropic API request failed: {e}")
        
        except Exception as e:
            log_error(f"Unexpected error with Anthropic API: {e}")
            raise
        finally:
            # Record metrics if not already done
            if not success:
                latency = time.time() - start_time
                self._get_metrics().record_request(
                    success=False,
                    latency=latency,
                    cache_hit=cache_hit
                )
    
    async def _generate_streaming(self, headers: Dict[str, str], payload: Dict[str, Any]) -> ProviderResponse:
        """
        Generate a response using streaming mode.
        
        Args:
            headers: Request headers
            payload: Request payload
            
        Returns:
            ProviderResponse: The assembled response
        """
        start_time = time.time()
        session_key = f"anthropic_stream_{payload.get('model', 'default')}"
        session = await get_aiohttp_session(key=session_key, timeout=self.config.timeout)
        
        try:
            async with session.post(
                self.config.api_url,
                headers=headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    try:
                        error_json = json.loads(error_text)
                        error_message = error_json.get('error', {}).get('message', error_text)
                    except json.JSONDecodeError:
                        error_message = error_text
                    
                    log_error(f"Anthropic Streaming API Error: Status {response.status}: {error_message}")
                    raise APIResponseError(f"Anthropic API Error: {error_message}", 
                                          status_code=response.status, 
                                          response_body=error_text)
                
                # Process the streaming response
                full_content = ""
                raw_response = None
                
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if not line or line == "data: [DONE]":
                        continue
                        
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])  # Remove "data: " prefix
                            
                            # Extract content from delta
                            if 'delta' in data and 'text' in data['delta']:
                                chunk = data['delta']['text']
                                full_content += chunk
                            
                            # Store the last chunk as raw_response
                            # (not ideal but gives us something)
                            raw_response = data
                            
                        except json.JSONDecodeError:
                            log_warning(f"Failed to parse streaming response line: {line}")
                
                latency = time.time() - start_time
                
                # Create a response with the assembled content
                result = ProviderResponse(
                    content=full_content,
                    usage={},  # Usage info not available in streaming mode
                    model=payload.get('model', ''),
                    raw_response=raw_response,
                    latency=latency
                )
                
                # Record metrics
                self._get_metrics().record_request(
                    success=True,
                    latency=latency,
                    cache_hit=False
                )
                
                return result
                
        except Exception as e:
            latency = time.time() - start_time
            log_error(f"Error in streaming response: {e}")
            
            # Record metrics
            self._get_metrics().record_request(
                success=False,
                latency=latency,
                cache_hit=False
            )
            
            raise


class OpenAIProvider(BaseProvider):
    """Provider implementation for OpenAI API."""
    
    def __init__(self, config: Optional[ProviderConfig] = None):
        """
        Initialize the OpenAI provider.
        
        Args:
            config: Provider configuration. If None, uses default config from environment.
        """
        # Default configuration
        if config is None:
            config = ProviderConfig(
                api_key=OPENAI_API_KEY,
                model="gpt-4",
                max_tokens=1000,
                temperature=0.7
            )
        
        super().__init__(config)
        
        # Import OpenAI here to avoid import errors if not installed
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.config.api_key)
        except ImportError:
            log_error("OpenAI package not installed. Please install it with: pip install openai")
            raise
    
    def validate_config(self) -> None:
        """Validate OpenAI-specific configuration."""
        super().validate_config()
        
        if not self.config.model:
            self.config.model = "gpt-4"
        
        valid_models = {
            "gpt-4", 
            "gpt-4-turbo", 
            "gpt-4-vision-preview",
            "gpt-4-0125-preview",
            "gpt-4-1106-preview",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0125",
        }
        
        if self.config.model not in valid_models:
            log_warning(f"Unrecognized OpenAI model: {self.config.model}. " 
                        f"Recognized models are: {', '.join(valid_models)}")

    def get_model_for_tier(self, tier: ModelTier) -> str:
        """Get the appropriate OpenAI model for the requested tier."""
        if tier == ModelTier.FAST:
            return "gpt-3.5-turbo-0125"
        elif tier == ModelTier.STANDARD:
            return "gpt-4-0125-preview"
        elif tier == ModelTier.ADVANCED:
            return "gpt-4-turbo"
        else:
            return self.config.model  # Default

    @cached_api(ttl=3600)
    @retry_with_circuit_breaker(
        max_retries=3,
        initial_backoff=1.0,
        backoff_factor=2.0,
        exceptions=[ConnectionError, TimeoutError, Exception],
        circuit_name="openai_api"
    )
    def generate(self, prompt: str, **kwargs) -> ProviderResponse:
        """
        Generate a response synchronously using the OpenAI API.
        
        Args:
            prompt: The prompt to send to the model.
            **kwargs: Additional options to override default configuration.
            
        Returns:
            ProviderResponse: The generated response.
        """
        # Import error types
        try:
            from openai.types.error import APIError, APIConnectionError, RateLimitError
        except ImportError:
            try:
                from openai.error import APIError, APIConnectionError, RateLimitError
            except ImportError:
                class APIError(Exception): pass
                class APIConnectionError(Exception): pass
                class RateLimitError(Exception): pass
        
        # Process special parameters
        bypass_cache = kwargs.pop('bypass_cache', False)
        tier = kwargs.pop('tier', None)
        
        # Merge config with any overrides from kwargs
        max_tokens = kwargs.get('max_tokens', self.config.max_tokens)
        temperature = kwargs.get('temperature', self.config.temperature)
        
        # Use tier-based model selection if requested
        if tier is not None:
            model = self.get_model_for_tier(tier)
        else:
            model = kwargs.get('model', self.config.model)
            
        system_prompt = kwargs.get('system_prompt', 
                                  "Du er en erfaren journalist med ekspertise i at udvikle kreative og relevante nyhedsvinkler.")
        
        # Log masked API key
        masked_key = self.config.api_key[:5] + "..." + self.config.api_key[-3:] if len(self.config.api_key) > 8 else "INVALID"
        log_info(f"Using OpenAI API key (masked): {masked_key}")
        
        start_time = time.time()
        cache_hit = False
        success = False
        
        try:
            # Call the OpenAI API
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Extract the response content
            content = response.choices[0].message.content
            
            # Extract usage information
            usage = {}
            if hasattr(response, 'usage'):
                usage = {
                    'prompt_tokens': getattr(response.usage, 'prompt_tokens', 0),
                    'completion_tokens': getattr(response.usage, 'completion_tokens', 0),
                    'total_tokens': getattr(response.usage, 'total_tokens', 0)
                }
            
            latency = time.time() - start_time
            success = True
            
            result = ProviderResponse(
                content=content,
                usage=usage,
                model=model,
                raw_response=response,
                latency=latency
            )
            
            # Record metrics
            self._get_metrics().record_request(
                success=success,
                latency=latency,
                cache_hit=cache_hit
            )
            
            return result
            
        except (APIError, APIConnectionError, RateLimitError) as e:
            log_error(f"OpenAI API Error: {e}")
            raise APIResponseError(f"OpenAI API Error: {e}")
        
        except Exception as e:
            log_error(f"Unexpected error with OpenAI API: {e}")
            raise
        finally:
            # Record metrics if not already done
            if not success:
                latency = time.time() - start_time
                self._get_metrics().record_request(
                    success=False,
                    latency=latency,
                    cache_hit=cache_hit
                )

    @cached_api(ttl=3600)
    @retry_with_circuit_breaker(
        max_retries=3,
        initial_backoff=1.0,
        backoff_factor=2.0,
        exceptions=[ConnectionError, TimeoutError, Exception],
        circuit_name="openai_api_async"
    )
    async def generate_async(self, prompt: str, **kwargs) -> ProviderResponse:
        """
        Generate a response asynchronously using the OpenAI API.
        
        This implementation uses the optimized approach of running async
        in a thread pool to avoid blocking the event loop while still
        providing asynchronous behavior.
        
        Args:
            prompt: The prompt to send to the model.
            **kwargs: Additional options to override default configuration.
            
        Returns:
            ProviderResponse: The generated response.
        """
        # Get thread pool
        loop = asyncio.get_event_loop()
        executor = get_thread_pool()
        
        # Run the synchronous method in a thread pool
        return await loop.run_in_executor(
            executor, lambda: self.generate(prompt, **kwargs)
        )


class PerplexityProvider(BaseProvider):
    """Provider implementation for Perplexity API."""
    
    def __init__(self, config: Optional[ProviderConfig] = None):
        """
        Initialize the Perplexity provider.
        
        Args:
            config: Provider configuration. If None, uses default config from environment.
        """
        # Default configuration
        if config is None:
            config = ProviderConfig(
                api_key=PERPLEXITY_API_KEY,
                api_url="https://api.perplexity.ai/chat/completions",
                model="sonar",
                max_tokens=1000,
                temperature=0.2
            )
        
        super().__init__(config)
    
    def validate_config(self) -> None:
        """Validate Perplexity-specific configuration."""
        super().validate_config()
        
        if not self.config.model:
            self.config.model = "sonar"
        
        if not self.config.api_url:
            self.config.api_url = "https://api.perplexity.ai/chat/completions"
        
        # Perplexity currently supports these models
        valid_models = {"sonar", "mixtral-8x7b", "llama-3-70b", "llama-3-8b", "codellama-70b"}
        
        if self.config.model not in valid_models:
            log_warning(f"Unrecognized Perplexity model: {self.config.model}. " 
                        f"Recognized models are: {', '.join(valid_models)}")

    def get_model_for_tier(self, tier: ModelTier) -> str:
        """Get the appropriate Perplexity model for the requested tier."""
        if tier == ModelTier.FAST:
            return "llama-3-8b"
        elif tier == ModelTier.STANDARD:
            return "sonar"
        elif tier == ModelTier.ADVANCED:
            return "llama-3-70b"
        else:
            return self.config.model  # Default

    @cached_api(ttl=3600)
    @retry_with_circuit_breaker(
        max_retries=3,
        initial_backoff=1.0,
        backoff_factor=2.0,
        exceptions=[requests.RequestException, ConnectionError, TimeoutError],
        circuit_name="perplexity_api"
    )
    def generate(self, prompt: str, **kwargs) -> ProviderResponse:
        """
        Generate a response synchronously using the Perplexity API.
        
        Args:
            prompt: The prompt to send to the model.
            **kwargs: Additional options to override default configuration.
            
        Returns:
            ProviderResponse: The generated response.
        """
        # Process special parameters
        bypass_cache = kwargs.pop('bypass_cache', False)
        tier = kwargs.pop('tier', None)
        
        # Merge config with any overrides from kwargs
        max_tokens = kwargs.get('max_tokens', self.config.max_tokens)
        temperature = kwargs.get('temperature', self.config.temperature)
        
        # Use tier-based model selection if requested
        if tier is not None:
            model = self.get_model_for_tier(tier)
        else:
            model = kwargs.get('model', self.config.model)
            
        system_prompt = kwargs.get('system_prompt', 
                                  "Du er en professionel journalist, der skal give koncis og faktabaseret information om et aktuelt nyhedsemne.")
        
        # Log masked API key
        masked_key = self.config.api_key[:5] + "..." + self.config.api_key[-3:] if len(self.config.api_key) > 8 else "INVALID"
        log_info(f"Using Perplexity API key (masked): {masked_key}")
        
        start_time = time.time()
        cache_hit = False
        success = False
        
        try:
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "return_images": False,
                "return_related_questions": False
            }
            
            # Make the API call
            response = requests.post(
                self.config.api_url,
                headers=headers,
                json=payload,
                timeout=self.config.timeout
            )
            
            # Check for errors
            if response.status_code != 200:
                error_text = response.text
                try:
                    error_json = json.loads(error_text)
                    error_message = error_json.get('error', {}).get('message', error_text)
                except json.JSONDecodeError:
                    error_message = error_text
                
                log_error(f"Perplexity API Error: Status {response.status_code}: {error_message}")
                raise APIResponseError(f"Perplexity API Error: {error_message}", 
                                      status_code=response.status_code, 
                                      response_body=error_text)
            
            # Parse and return the response
            data = response.json()
            
            try:
                content = data['choices'][0]['message']['content']
                
                # Extract usage information
                usage = {}
                if 'usage' in data:
                    usage = data['usage']
                
                latency = time.time() - start_time
                success = True
                
                result = ProviderResponse(
                    content=content,
                    usage=usage,
                    model=model,
                    raw_response=data,
                    latency=latency
                )
                
                # Record metrics
                self._get_metrics().record_request(
                    success=success,
                    latency=latency,
                    cache_hit=cache_hit
                )
                
                return result
                
            except (KeyError, IndexError) as e:
                log_error(f"Unexpected response format from Perplexity API: {e}")
                raise APIResponseError(f"Unexpected response format from Perplexity API: {e}")
            
        except requests.exceptions.SSLError as e:
            log_error(f"SSL Error: {e}")
            raise SSLVerificationError(f"SSL Error when connecting to Perplexity API: {e}")
        
        except requests.exceptions.ConnectionError as e:
            log_error(f"Connection Error: {e}")
            raise APIConnectionError(f"Failed to connect to Perplexity API: {e}")
        
        except requests.exceptions.Timeout as e:
            log_error(f"Timeout Error: {e}")
            raise APIConnectionError(f"Perplexity API request timed out: {e}")
        
        except requests.exceptions.RequestException as e:
            log_error(f"Request Error: {e}")
            raise APIConnectionError(f"Perplexity API request failed: {e}")
        
        except Exception as e:
            log_error(f"Unexpected error with Perplexity API: {e}")
            raise
        finally:
            # Record metrics if not already done
            if not success:
                latency = time.time() - start_time
                self._get_metrics().record_request(
                    success=False,
                    latency=latency,
                    cache_hit=cache_hit
                )

    @cached_api(ttl=3600)
    @retry_with_circuit_breaker(
        max_retries=3,
        initial_backoff=1.0,
        backoff_factor=2.0,
        exceptions=[aiohttp.ClientError, asyncio.TimeoutError, ConnectionError],
        circuit_name="perplexity_api_async"
    )
    async def generate_async(self, prompt: str, **kwargs) -> ProviderResponse:
        """
        Generate a response asynchronously using the Perplexity API.
        
        Args:
            prompt: The prompt to send to the model.
            **kwargs: Additional options to override default configuration.
            
        Returns:
            ProviderResponse: The generated response.
        """
        # Process special parameters
        bypass_cache = kwargs.pop('bypass_cache', False)
        tier = kwargs.pop('tier', None)
        
        # Merge config with any overrides from kwargs
        max_tokens = kwargs.get('max_tokens', self.config.max_tokens)
        temperature = kwargs.get('temperature', self.config.temperature)
        
        # Use tier-based model selection if requested
        if tier is not None:
            model = self.get_model_for_tier(tier)
        else:
            model = kwargs.get('model', self.config.model)
            
        system_prompt = kwargs.get('system_prompt', 
                                  "Du er en professionel journalist, der skal give koncis og faktabaseret information om et aktuelt nyhedsemne.")
        
        # Log masked API key
        masked_key = self.config.api_key[:5] + "..." + self.config.api_key[-3:] if len(self.config.api_key) > 8 else "INVALID"
        log_info(f"Using Perplexity API key (masked): {masked_key}")
        
        start_time = time.time()
        cache_hit = False
        success = False
        
        try:
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "return_images": False,
                "return_related_questions": False
            }
            
            # Create a secure session
            session_key = f"perplexity_{model}"
            session = await get_aiohttp_session(key=session_key, timeout=self.config.timeout)
            
            # Make the API call
            async with session.post(
                self.config.api_url,
                headers=headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    try:
                        error_json = json.loads(error_text)
                        error_message = error_json.get('error', {}).get('message', error_text)
                    except json.JSONDecodeError:
                        error_message = error_text
                    
                    log_error(f"Perplexity API Error: Status {response.status}: {error_message}")
                    raise APIResponseError(f"Perplexity API Error: {error_message}", 
                                          status_code=response.status, 
                                          response_body=error_text)
                
                # Parse and return the response
                data = await response.json()
                
                try:
                    content = data['choices'][0]['message']['content']
                    
                    # Extract usage information
                    usage = {}
                    if 'usage' in data:
                        usage = data['usage']
                    
                    latency = time.time() - start_time
                    success = True
                    
                    result = ProviderResponse(
                        content=content,
                        usage=usage,
                        model=model,
                        raw_response=data,
                        latency=latency
                    )
                    
                    # Record metrics
                    self._get_metrics().record_request(
                        success=success,
                        latency=latency,
                        cache_hit=cache_hit
                    )
                    
                    return result
                    
                except (KeyError, IndexError) as e:
                    log_error(f"Unexpected response format from Perplexity API: {e}")
                    raise APIResponseError(f"Unexpected response format from Perplexity API: {e}")
                
        except aiohttp.ClientSSLError as e:
            log_error(f"SSL Error: {e}")
            raise SSLVerificationError(f"SSL Error when connecting to Perplexity API: {e}")
        
        except aiohttp.ClientConnectorError as e:
            log_error(f"Connection Error: {e}")
            raise APIConnectionError(f"Failed to connect to Perplexity API: {e}")
        
        except asyncio.TimeoutError as e:
            log_error(f"Timeout Error: {e}")
            raise APIConnectionError(f"Perplexity API request timed out: {e}")
        
        except aiohttp.ClientError as e:
            log_error(f"Request Error: {e}")
            raise APIConnectionError(f"Perplexity API request failed: {e}")
        
        except Exception as e:
            log_error(f"Unexpected error with Perplexity API: {e}")
            raise
        finally:
            # Record metrics if not already done
            if not success:
                latency = time.time() - start_time
                self._get_metrics().record_request(
                    success=False,
                    latency=latency,
                    cache_hit=cache_hit
                )


class ProviderFactory:
    """Factory for creating AI providers."""
    
    _instances = {}
    
    @staticmethod
    def create_provider(provider_type: ProviderType, config: Optional[ProviderConfig] = None) -> BaseProvider:
        """
        Create a provider instance based on type.
        
        Args:
            provider_type: Type of provider to create.
            config: Optional provider configuration.
            
        Returns:
            BaseProvider: The created provider instance.
            
        Raises:
            ValueError: If the provider type is not supported.
        """
        if provider_type == ProviderType.OPENAI:
            if ProviderType.OPENAI not in ProviderFactory._instances:
                ProviderFactory._instances[ProviderType.OPENAI] = OpenAIProvider(config)
            return ProviderFactory._instances[ProviderType.OPENAI]
        elif provider_type == ProviderType.ANTHROPIC:
            if ProviderType.ANTHROPIC not in ProviderFactory._instances:
                ProviderFactory._instances[ProviderType.ANTHROPIC] = AnthropicProvider(config)
            return ProviderFactory._instances[ProviderType.ANTHROPIC]
        elif provider_type == ProviderType.PERPLEXITY:
            if ProviderType.PERPLEXITY not in ProviderFactory._instances:
                ProviderFactory._instances[ProviderType.PERPLEXITY] = PerplexityProvider(config)
            return ProviderFactory._instances[ProviderType.PERPLEXITY]
        else:
            raise ValueError(f"Unsupported provider type: {provider_type}")
    
    @staticmethod
    def get_provider_by_name(name: str, config: Optional[ProviderConfig] = None) -> BaseProvider:
        """
        Get a provider instance by its name.
        
        Args:
            name: Name of the provider (case-insensitive)
            config: Optional provider configuration
            
        Returns:
            BaseProvider: The provider instance
            
        Raises:
            ValueError: If the provider name is not recognized
        """
        name_lower = name.lower()
        
        if 'openai' in name_lower or 'gpt' in name_lower:
            return ProviderFactory.create_provider(ProviderType.OPENAI, config)
        elif 'anthropic' in name_lower or 'claude' in name_lower:
            return ProviderFactory.create_provider(ProviderType.ANTHROPIC, config)
        elif 'perplexity' in name_lower or 'pplx' in name_lower:
            return ProviderFactory.create_provider(ProviderType.PERPLEXITY, config)
        else:
            raise ValueError(f"Unrecognized provider name: {name}")


# Optimized async multi-provider functions

async def generate_with_providers(
    prompt: str,
    providers: List[Union[str, BaseProvider, ProviderType]],
    tier: ModelTier = ModelTier.STANDARD,
    system_prompt: str = "",
    max_tokens: int = 1000,
    temperature: float = 0.7,
    timeout: int = 60,
    compare_outputs: bool = False
) -> Union[ProviderResponse, List[ProviderResponse]]:
    """
    Generate responses using multiple providers in parallel.
    
    Args:
        prompt: The prompt to send to the models
        providers: List of providers (can be names, types, or instances)
        tier: Capability tier to use for all providers
        system_prompt: System prompt to use
        max_tokens: Maximum tokens to generate
        temperature: Temperature for generation
        timeout: Timeout for all requests
        compare_outputs: If True, return all responses; if False, return the first successful one
        
    Returns:
        Union[ProviderResponse, List[ProviderResponse]]: Generated response(s)
    """
    provider_instances = []
    
    # Convert all providers to instances
    for p in providers:
        if isinstance(p, str):
            provider_instances.append(ProviderFactory.get_provider_by_name(p))
        elif isinstance(p, ProviderType):
            provider_instances.append(ProviderFactory.create_provider(p))
        elif isinstance(p, BaseProvider):
            provider_instances.append(p)
        else:
            raise ValueError(f"Unsupported provider type: {type(p)}")
    
    # Create tasks for all providers
    tasks = []
    for provider in provider_instances:
        task = asyncio.create_task(provider.generate_async(
            prompt=prompt,
            tier=tier,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature
        ))
        tasks.append(task)
    
    # Wait for the first successful result or all results
    if compare_outputs:
        # Wait for all results, gathering successful ones
        all_results = []
        all_exceptions = []
        
        for task in asyncio.as_completed(tasks, timeout=timeout):
            try:
                result = await task
                all_results.append(result)
            except Exception as e:
                all_exceptions.append(e)
        
        if not all_results:
            # All tasks failed
            if all_exceptions:
                raise all_exceptions[0]
            else:
                raise TimeoutError("All providers timed out or failed")
                
        return all_results
    else:
        # Return first successful result
        exceptions = []
        
        for task in asyncio.as_completed(tasks, timeout=timeout):
            try:
                return await task
            except Exception as e:
                exceptions.append(e)
                
        # If we got here, all tasks failed
        if exceptions:
            raise exceptions[0]
        else:
            raise TimeoutError("All providers timed out")

async def fetch_information_async(
    topic: str,
    provider: Union[str, BaseProvider, ProviderType] = "perplexity",
    tier: ModelTier = ModelTier.FAST,
    detailed: bool = False,
    timeout: int = 30
) -> Optional[str]:
    """
    Fetch information about a topic using an AI provider.
    
    Args:
        topic: The topic to fetch information about
        provider: Provider to use (defaults to Perplexity)
        tier: Capability tier to use
        detailed: If True, get more detailed information
        timeout: Timeout in seconds
        
    Returns:
        Optional[str]: The information retrieved or None if failed
    """
    # Convert provider to instance if needed
    if isinstance(provider, str):
        provider_instance = ProviderFactory.get_provider_by_name(provider)
    elif isinstance(provider, ProviderType):
        provider_instance = ProviderFactory.create_provider(provider)
    elif isinstance(provider, BaseProvider):
        provider_instance = provider
    else:
        raise ValueError(f"Unsupported provider type: {type(provider)}")
    
    # Construct an efficient prompt based on detail level
    if detailed:
        prompt = f"""
        Giv en udførlig og velstruktureret analyse af '{topic}' med følgende sektioner:
        
        # OVERSIGT
        # BAGGRUND
        # AKTUEL STATUS
        # NØGLETAL
        # PERSPEKTIVER
        # RELEVANS FOR DANMARK
        # FREMTIDSUDSIGTER
        
        Inkluder konkrete fakta, tal og datoer hvor relevant. Hold dig til fakta og vær objektiv.
        """
        max_tokens = 1500
    else:
        prompt = f"""
        Giv en kort og præcis oversigt over emnet '{topic}' med fokus på:
        
        # HVAD (Hvad handler emnet om - 2-3 sætninger)
        # HVORFOR (Hvorfor er det aktuelt lige nu - 2-3 sætninger)
        # HVEM (Hvem er involveret - kort liste)
        # NØGLEFAKTA (2-3 vigtige fakta eller tal)
        
        Hold svaret kortfattet og faktuelt med maksimalt 400 ord.
        """
        max_tokens = 750
    
    system_prompt = "Du er en præcis og faktabaseret research-assistent med ekspertise i at give velstrukturerede, objektive informationer. Vær faktabaseret og undgå overflødige beskrivelser."
    
    try:
        # Start time measurement
        start_time = time.time()
        
        response = await provider_instance.generate_async(
            prompt=prompt,
            system_prompt=system_prompt,
            tier=tier,
            max_tokens=max_tokens,
            temperature=0.2,
            timeout=timeout
        )
        
        logger.info(f"Information retrieved in {time.time() - start_time:.2f} seconds")
        return response.content
    except Exception as e:
        logger.error(f"Error fetching information: {e}")
        return None

# Advanced parallel processing function
async def process_generation_request_parallel(
    topic: str,
    profile: Any,  # Should be RedaktionelDNA type from models.py
    bypass_cache: bool = False,
    progress_callback = None,
    timeout: int = 45,
    use_fast_models: bool = False
) -> List[Dict[str, Any]]:
    """
    Process an angle generation request with optimized parallel API calls.
    
    Args:
        topic: News topic to generate angles for
        profile: Editorial DNA profile to use
        bypass_cache: If True, ignore cached results
        progress_callback: Optional callback function for progress updates
        timeout: Overall timeout for the entire operation
        use_fast_models: If True, use faster but less capable models for preliminary steps
        
    Returns:
        List[Dict]: Generated angles
    """
    # Update progress
    if progress_callback:
        await progress_callback(10)
    
    # Configure tier based on speed preference
    info_tier = ModelTier.FAST if use_fast_models else ModelTier.STANDARD
    angle_tier = ModelTier.STANDARD  # Always use standard or better for angles
    source_tier = ModelTier.FAST     # Always use fast model for sources
    
    # Convert profile into strings for prompt construction
    try:
        principper = "\n".join([f"- {p}" for p in profile.kerneprincipper])
        nyhedskriterier = "\n".join([f"- {k}: {v}" for k, v in profile.nyhedsprioritering.items()])
        fokusomrader = "\n".join([f"- {f}" for f in profile.fokusOmrader])
        nogo_omrader = "\n".join([f"- {n}" for n in profile.noGoOmrader]) if profile.noGoOmrader else "Ingen"
    except Exception as e:
        logger.error(f"Error processing profile: {e}")
        principper = ""
        nyhedskriterier = ""
        fokusomrader = ""
        nogo_omrader = "Ingen"
    
    # Launch all initial tasks in parallel
    info_task = asyncio.create_task(
        fetch_information_async(
            topic, 
            provider="perplexity", 
            tier=info_tier, 
            detailed=not use_fast_models
        )
    )
    
    # Update progress
    if progress_callback:
        await progress_callback(25)
    
    # Wait for topic info - needed for angle generation
    try:
        # We need to wait for this since it's required for the angle prompt
        topic_info = await asyncio.wait_for(info_task, timeout=timeout)
        
        if not topic_info:
            topic_info = f"Emnet handler om {topic}. Ingen yderligere baggrundsinformation tilgængelig."
    except asyncio.TimeoutError:
        logger.warning(f"Topic info request timed out after {timeout} seconds")
        topic_info = f"Emnet handler om {topic}. Ingen yderligere baggrundsinformation tilgængelig."
    except Exception as e:
        logger.error(f"Error fetching topic info: {e}")
        topic_info = f"Emnet handler om {topic}. Ingen yderligere baggrundsinformation tilgængelig."
    
    # Update progress
    if progress_callback:
        await progress_callback(40)
    
    # Create the angle generation prompt
    from prompt_engineering import construct_angle_prompt
    
    prompt = construct_angle_prompt(
        topic,
        topic_info,
        principper,
        profile.tone_og_stil,
        fokusomrader,
        nyhedskriterier,
        nogo_omrader
    )
    
    # Launch source suggestions task in parallel with angle generation
    source_task = asyncio.create_task(
        fetch_information_async(
            f"Relevante kilder om {topic}", 
            provider="anthropic", 
            tier=source_tier, 
            detailed=False
        )
    )
    
    # Generate angles with Claude API
    try:
        # Get provider
        provider = ProviderFactory.create_provider(ProviderType.ANTHROPIC)
        
        # Use appropriate tier based on speed preference  
        if use_fast_models:
            model = provider.get_model_for_tier(ModelTier.STANDARD)
        else:
            model = provider.get_model_for_tier(ModelTier.ADVANCED)
            
        response = await provider.generate_async(
            prompt=prompt,
            model=model,
            max_tokens=2500,
            temperature=0.7,
            streaming=True
        )
        
        # Update progress
        if progress_callback:
            await progress_callback(70)
        
        # Parse angles from response
        from prompt_engineering import parse_angles_from_response
        angles = parse_angles_from_response(response.content)
        
        if not angles:
            logger.error("No angles parsed from response")
            return []
        
        # Add background info to each angle
        perplexity_extract = topic_info[:800] + ("..." if len(topic_info) > 800 else "")
        for angle in angles:
            if isinstance(angle, dict):
                angle['perplexityInfo'] = perplexity_extract
        
        # Wait for source suggestions to complete with timeout
        try:
            source_text = await asyncio.wait_for(source_task, timeout=15)  # Short timeout for sources
            if source_text:
                for angle in angles:
                    if isinstance(angle, dict):
                        angle['kildeForslagInfo'] = source_text
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning(f"Source suggestions timed out or failed: {e}")
            # Continue without source suggestions if they fail
        
        # Update progress
        if progress_callback:
            await progress_callback(85)
        
        # Filter and rank angles
        from angle_processor import filter_and_rank_angles
        ranked_angles = filter_and_rank_angles(angles, profile, 5)
        
        # Final progress update
        if progress_callback:
            await progress_callback(100)
        
        return ranked_angles
    
    except Exception as e:
        logger.error(f"Error generating angles: {e}")
        raise

# Convenience functions for backward compatibility

def get_openai_provider() -> OpenAIProvider:
    """
    Get an OpenAI provider instance with default configuration.
    
    Returns:
        OpenAIProvider: The OpenAI provider instance.
    """
    return ProviderFactory.create_provider(ProviderType.OPENAI)

def get_anthropic_provider() -> AnthropicProvider:
    """
    Get an Anthropic provider instance with default configuration.
    
    Returns:
        AnthropicProvider: The Anthropic provider instance.
    """
    return ProviderFactory.create_provider(ProviderType.ANTHROPIC)

def get_perplexity_provider() -> PerplexityProvider:
    """
    Get a Perplexity provider instance with default configuration.
    
    Returns:
        PerplexityProvider: The Perplexity provider instance.
    """
    return ProviderFactory.create_provider(ProviderType.PERPLEXITY)

async def fetch_topic_information_async(topic: str, dev_mode: bool = False, bypass_cache: bool = False, progress_callback=None) -> Optional[str]:
    """
    Fetch information about a topic using the Perplexity API asynchronously.
    
    This is a backward-compatible replacement for the function in api_clients.py.
    
    Args:
        topic: The news topic to fetch information about
        dev_mode: If True, disables SSL verification (development only!)
        bypass_cache: If True, ignore cached results
        progress_callback: Optional callback function to report progress (0-100)
        
    Returns:
        Optional[str]: The information retrieved or None if failed
    """
    try:
        # Update progress if callback provided
        if progress_callback:
            await progress_callback(25)
        
        # Create provider with dev_mode setting
        provider_config = ProviderConfig(
            api_key=PERPLEXITY_API_KEY,
            model="sonar",
            temperature=0.2,
            max_tokens=1000,
            dev_mode=dev_mode
        )
        
        # Use our optimized implementation
        result = await fetch_information_async(
            topic,
            provider=PerplexityProvider(provider_config),
            tier=ModelTier.STANDARD
        )
        
        # Final progress update
        if progress_callback:
            await progress_callback(100)
        
        return result
    except Exception as e:
        log_error(f"Error fetching topic information: {e}")
        # Final progress update on error
        if progress_callback:
            await progress_callback(100)
        return None

def generate_angles_with_provider(emne: str, topic_info: str, profile: Any, bypass_cache: bool = False) -> List[Dict[str, Any]]:
    """
    Generate news angles for the given topic and profile using Claude.
    
    This is a backward-compatible replacement for the function in api_clients.py.
    
    Args:
        emne: The news topic.
        topic_info: Background information on the topic.
        profile: The editorial DNA profile.
        bypass_cache: if True, bypass local cache.
        
    Returns:
        A list of angle dictionaries.
        
    Raises:
        ValueError: If angles cannot be generated
    """
    from prompt_engineering import construct_angle_prompt, parse_angles_from_response
    
    try:
        # Convert profile into strings for prompt construction
        principper = "\n".join([f"- {p}" for p in profile.kerneprincipper])
        nyhedskriterier = "\n".join([f"- {k}: {v}" for k, v in profile.nyhedsprioritering.items()])
        fokusomrader = "\n".join([f"- {f}" for f in profile.fokusOmrader])
        nogo_omrader = "\n".join([f"- {n}" for n in profile.noGoOmrader]) if profile.noGoOmrader else "Ingen"
        
        # Create the prompt
        prompt = construct_angle_prompt(
            emne,
            topic_info,
            principper,
            profile.tone_og_stil,
            fokusomrader,
            nyhedskriterier,
            nogo_omrader
        )
        
        # Use the provider to generate angles
        provider = get_anthropic_provider()
        response = provider.generate(
            prompt=prompt,
            max_tokens=2500,
            temperature=0.7,
            bypass_cache=bypass_cache
        )
        
        # Parse the angles from the response
        angles = parse_angles_from_response(response.content)
        
        # Log success
        log_info(f"Generated {len(angles)} angles successfully")
        
        # Make sure we return a list
        if not isinstance(angles, list):
            if isinstance(angles, dict):
                # Single angle in dict format
                angles = [angles]
            else:
                log_error(f"Unexpected format: {type(angles)}")
                raise ValueError(f"Unexpected format: {type(angles)}. Expected a list or dict.")
        
        # Add perplexity information to each angle if available
        if topic_info and isinstance(topic_info, str):
            try:
                # Extract first 1000 chars to keep it concise
                perplexity_extract = topic_info[:1000] + ("..." if len(topic_info) > 1000 else "")
                for angle in angles:
                    if isinstance(angle, dict):
                        angle['perplexityInfo'] = perplexity_extract
            except Exception as e:
                # Log but don't fail if we can't add perplexity info
                log_warning(f"Could not add perplexity information: {e}")
        
        return angles
        
    except Exception as e:
        log_error(f"Unexpected error generating angles: {e}")
        raise ValueError(
            f"Unexpected error generating angles: {e}. "
            "Please contact support if the problem persists."
        )

def get_provider_performance_metrics():
    """Get performance metrics for all providers."""
    return {
        "anthropic": metrics["anthropic"].get_stats(),
        "openai": metrics["openai"].get_stats(),
        "perplexity": metrics["perplexity"].get_stats(),
        "circuits": get_circuit_stats()
    }

# Initialize and cleanup functions
async def initialize_providers():
    """Initialize all providers for optimal performance."""
    # Pre-create instances
    get_anthropic_provider()
    get_openai_provider()
    get_perplexity_provider()
    
    # Pre-warm sessions
    await get_aiohttp_session("anthropic", 60)
    await get_aiohttp_session("perplexity", 30)
    
    # Initialize thread pool
    get_thread_pool()
    
    logger.info("AI providers initialized and ready")

async def shutdown_providers():
    """Properly shut down all provider connections."""
    # Close all sessions
    await close_all_sessions()
    
    # Shut down thread pool
    if _thread_pool is not None:
        _thread_pool.shutdown(wait=True)
        
    logger.info("AI providers shut down successfully")