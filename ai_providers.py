"""
AI providers module for Vinkeljernet project.

This module provides a unified abstraction layer for different AI providers
like OpenAI (GPT), Anthropic (Claude), and Perplexity. It handles authentication,
request formatting, error handling, and response parsing for each provider.

It offers both synchronous and asynchronous interfaces, with built-in retry logic,
circuit breaker pattern to prevent excessive requests during outages, and caching
to reduce API costs and improve performance.
"""

import os
import json
import time
import logging
import ssl
import certifi
import requests
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Type, Union, Callable, Tuple
from urllib.parse import urlparse

# Try to import aiohttp, install if not available
try:
    import aiohttp
except ImportError:
    print("Error: aiohttp package not found. Installing now...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "aiohttp==3.9.1"])
    import aiohttp

# Import configuration
from config import OPENAI_API_KEY, ANTHROPIC_API_KEY, PERPLEXITY_API_KEY
from retry_manager import (
    retry_with_circuit_breaker, 
    CircuitOpenError, 
    MaxRetriesExceededError
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
logger = logging.getLogger("vinkeljernet.ai_providers")


class ProviderType(Enum):
    """Enum for supported AI providers."""
    OPENAI = auto()
    ANTHROPIC = auto()
    PERPLEXITY = auto()


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
    
    def validate_config(self) -> None:
        """Validate provider-specific configuration."""
        if not self.config.api_key:
            raise APIKeyMissingError(f"API key is required for {self.__class__.__name__}")
    
    def _setup_logging(self) -> None:
        """Set up logging for this provider."""
        self.logger = logging.getLogger(f"vinkeljernet.ai_providers.{self.__class__.__name__}")

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
        # Merge config with any overrides from kwargs
        max_tokens = kwargs.get('max_tokens', self.config.max_tokens)
        temperature = kwargs.get('temperature', self.config.temperature)
        model = kwargs.get('model', self.config.model)
        system_prompt = kwargs.get('system_prompt', 
                                  "Du er en erfaren journalist med ekspertise i at udvikle kreative og relevante nyhedsvinkler.")
        
        # Log masked API key
        masked_key = self.config.api_key[:5] + "..." + self.config.api_key[-3:] if len(self.config.api_key) > 8 else "INVALID"
        log_info(f"Using Anthropic API key (masked): {masked_key}")
        
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
            
            return ProviderResponse(
                content=content,
                usage=usage,
                model=model,
                raw_response=response_data
            )
            
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
        # Merge config with any overrides from kwargs
        max_tokens = kwargs.get('max_tokens', self.config.max_tokens)
        temperature = kwargs.get('temperature', self.config.temperature)
        model = kwargs.get('model', self.config.model)
        system_prompt = kwargs.get('system_prompt', 
                                  "Du er en erfaren journalist med ekspertise i at udvikle kreative og relevante nyhedsvinkler.")
        
        # Log masked API key
        masked_key = self.config.api_key[:5] + "..." + self.config.api_key[-3:] if len(self.config.api_key) > 8 else "INVALID"
        log_info(f"Using Anthropic API key (masked): {masked_key}")
        
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
            
            # Create a secure session
            async with await self.create_secure_api_session() as session:
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
                    
                    return ProviderResponse(
                        content=content,
                        usage=usage,
                        model=model,
                        raw_response=response_data
                    )
                    
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
        
        # Merge config with any overrides from kwargs
        max_tokens = kwargs.get('max_tokens', self.config.max_tokens)
        temperature = kwargs.get('temperature', self.config.temperature)
        model = kwargs.get('model', self.config.model)
        system_prompt = kwargs.get('system_prompt', 
                                  "Du er en erfaren journalist med ekspertise i at udvikle kreative og relevante nyhedsvinkler.")
        
        # Log masked API key
        masked_key = self.config.api_key[:5] + "..." + self.config.api_key[-3:] if len(self.config.api_key) > 8 else "INVALID"
        log_info(f"Using OpenAI API key (masked): {masked_key}")
        
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
            
            return ProviderResponse(
                content=content,
                usage=usage,
                model=model,
                raw_response=response
            )
            
        except (APIError, APIConnectionError, RateLimitError) as e:
            log_error(f"OpenAI API Error: {e}")
            raise APIResponseError(f"OpenAI API Error: {e}")
        
        except Exception as e:
            log_error(f"Unexpected error with OpenAI API: {e}")
            raise

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
        
        NOTE: The official OpenAI Python client doesn't support async natively,
        so this implementation runs the synchronous client in a thread pool.
        
        Args:
            prompt: The prompt to send to the model.
            **kwargs: Additional options to override default configuration.
            
        Returns:
            ProviderResponse: The generated response.
        """
        # Run the synchronous method in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.generate(prompt, **kwargs)
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
        # Merge config with any overrides from kwargs
        max_tokens = kwargs.get('max_tokens', self.config.max_tokens)
        temperature = kwargs.get('temperature', self.config.temperature)
        model = kwargs.get('model', self.config.model)
        system_prompt = kwargs.get('system_prompt', 
                                  "Du er en professionel journalist, der skal give koncis og faktabaseret information om et aktuelt nyhedsemne.")
        
        # Log masked API key
        masked_key = self.config.api_key[:5] + "..." + self.config.api_key[-3:] if len(self.config.api_key) > 8 else "INVALID"
        log_info(f"Using Perplexity API key (masked): {masked_key}")
        
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
                
                return ProviderResponse(
                    content=content,
                    usage=usage,
                    model=model,
                    raw_response=data
                )
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
        # Merge config with any overrides from kwargs
        max_tokens = kwargs.get('max_tokens', self.config.max_tokens)
        temperature = kwargs.get('temperature', self.config.temperature)
        model = kwargs.get('model', self.config.model)
        system_prompt = kwargs.get('system_prompt', 
                                  "Du er en professionel journalist, der skal give koncis og faktabaseret information om et aktuelt nyhedsemne.")
        
        # Log masked API key
        masked_key = self.config.api_key[:5] + "..." + self.config.api_key[-3:] if len(self.config.api_key) > 8 else "INVALID"
        log_info(f"Using Perplexity API key (masked): {masked_key}")
        
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
            async with await self.create_secure_api_session() as session:
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
                        
                        return ProviderResponse(
                            content=content,
                            usage=usage,
                            model=model,
                            raw_response=data
                        )
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


class ProviderFactory:
    """Factory for creating AI providers."""
    
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
            return OpenAIProvider(config)
        elif provider_type == ProviderType.ANTHROPIC:
            return AnthropicProvider(config)
        elif provider_type == ProviderType.PERPLEXITY:
            return PerplexityProvider(config)
        else:
            raise ValueError(f"Unsupported provider type: {provider_type}")


# Convenience functions for backward compatibility

def get_openai_provider() -> OpenAIProvider:
    """
    Get an OpenAI provider instance with default configuration.
    
    Returns:
        OpenAIProvider: The OpenAI provider instance.
    """
    return OpenAIProvider()

def get_anthropic_provider() -> AnthropicProvider:
    """
    Get an Anthropic provider instance with default configuration.
    
    Returns:
        AnthropicProvider: The Anthropic provider instance.
    """
    return AnthropicProvider()

def get_perplexity_provider() -> PerplexityProvider:
    """
    Get a Perplexity provider instance with default configuration.
    
    Returns:
        PerplexityProvider: The Perplexity provider instance.
    """
    return PerplexityProvider()

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
        provider = PerplexityProvider(provider_config)
        
        # Update progress before making the request
        if progress_callback:
            await progress_callback(40)
        
        # Construct prompt
        prompt = f"Giv mig en grundig, men velstruktureret oversigt over den aktuelle situation vedrørende følgende nyhedsemne: {topic}. Inkluder relevante fakta, baggrund, kontekst og eventuelle nylige udviklinger. Vær præcis og faktabaseret."
        
        # Make the request with the provider
        response = await provider.generate_async(
            prompt=prompt,
            max_tokens=1000,
            temperature=0.2,
            bypass_cache=bypass_cache
        )
        
        # Update progress after receiving response
        if progress_callback:
            await progress_callback(75)
        
        # Final progress update
        if progress_callback:
            await progress_callback(100)
        
        return response.content
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