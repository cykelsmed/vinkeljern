# AI Providers for Vinkeljernet

This document explains how to use the AI provider abstraction layer in Vinkeljernet.

## Overview

The `ai_providers.py` module provides a unified interface for working with different AI providers:

- Anthropic (Claude)
- OpenAI (GPT)
- Perplexity

Key features:
- Common interface for all providers
- Both synchronous and asynchronous API
- Automatic retries with exponential backoff
- Circuit breaker pattern to prevent excessive API calls during outages
- Caching of responses to reduce costs
- Comprehensive error handling
- Standardized response format

## Basic Usage

### Quick Start

```python
from ai_providers import get_anthropic_provider

# Create a provider with default settings
provider = get_anthropic_provider()

# Generate a response
response = provider.generate(
    prompt="Explain quantum computing in simple terms.",
    max_tokens=500
)

# Use the response
print(response.content)
```

### Using the Factory

```python
from ai_providers import ProviderType, ProviderFactory, ProviderConfig

# Create a provider with custom settings
config = ProviderConfig(
    api_key="your_api_key",
    model="claude-3-opus-20240229",
    max_tokens=1000,
    temperature=0.7
)

provider = ProviderFactory.create_provider(
    provider_type=ProviderType.ANTHROPIC,
    config=config
)

# Generate a response
response = provider.generate(
    prompt="Write a short story about a robot learning to paint.",
    temperature=0.8  # Override the config
)
```

### Asynchronous API

```python
import asyncio
from ai_providers import get_perplexity_provider

async def fetch_data():
    provider = get_perplexity_provider()
    
    response = await provider.generate_async(
        prompt="Summarize the current situation regarding climate change.",
        max_tokens=500
    )
    
    return response.content

# Run the async function
result = asyncio.run(fetch_data())
```

## Provider Types

### Anthropic (Claude)

Claude is ideal for:
- Complex reasoning
- Long contexts
- Creative writing
- Instruction following

```python
from ai_providers import get_anthropic_provider

provider = get_anthropic_provider()

# Available models:
# - claude-3-opus-20240229 (default, most capable)
# - claude-3-sonnet-20240229 (balanced)
# - claude-3-haiku-20240307 (fast)
# - claude-2.1
# - claude-2.0

response = provider.generate(
    prompt="Write a news article about...",
    model="claude-3-haiku-20240307",  # Use a faster model
    max_tokens=1000,
    temperature=0.7
)
```

### Perplexity

Perplexity is ideal for:
- Knowledge-intensive tasks
- Fact retrieval
- Summarization
- Research assistance

```python
from ai_providers import get_perplexity_provider

provider = get_perplexity_provider()

# Available models:
# - sonar (default)
# - mixtral-8x7b
# - llama-3-70b
# - llama-3-8b
# - codellama-70b

response = provider.generate(
    prompt="Explain the current state of...",
    model="sonar",
    max_tokens=500,
    temperature=0.2  # Lower temperature for more factual responses
)
```

### OpenAI (GPT)

OpenAI GPT is ideal for:
- General purpose tasks
- Coding assistance
- Structured data generation
- Fine-tuned models

```python
from ai_providers import get_openai_provider

provider = get_openai_provider()

# Available models:
# - gpt-4 (default)
# - gpt-4-turbo
# - gpt-4-vision-preview
# - gpt-4-0125-preview
# - gpt-3.5-turbo
# - gpt-3.5-turbo-0125

response = provider.generate(
    prompt="Generate JSON data for...",
    model="gpt-4",
    max_tokens=500,
    temperature=0.3
)
```

## Response Format

All providers return a standardized `ProviderResponse` object:

```python
@dataclass
class ProviderResponse:
    content: str                      # The generated text
    usage: Dict[str, Any]             # Token usage information
    model: str                        # The model used
    raw_response: Any                 # The original response from the API
    created_at: float                 # Timestamp of when the response was created
    
    @property
    def token_usage(self) -> int:
        # Helper to get the total token usage
        return self.usage.get('total_tokens', 0)
```

## Backward Compatibility

For backward compatibility with the rest of the codebase, utility functions are provided:

```python
# Instead of import from api_clients, use:
from ai_providers import fetch_topic_information_async, generate_angles_with_provider

# These provide the same functionality as their api_clients.py counterparts
topic_info = await fetch_topic_information_async(
    topic="Some topic", 
    dev_mode=False
)

angles = generate_angles_with_provider(
    emne="Some topic",
    topic_info=topic_info,
    profile=profile_obj
)
```

## Configuration

Each provider accepts a `ProviderConfig` object:

```python
@dataclass
class ProviderConfig:
    api_key: str                      # Required API key
    api_url: str = ""                 # API endpoint URL
    model: str = ""                   # Model to use (provider-specific)
    max_tokens: int = 1000            # Max tokens to generate
    temperature: float = 0.7          # Temperature for generation
    timeout: int = 60                 # Timeout in seconds
    dev_mode: bool = False            # If True, disables SSL verification (dev only!)
    extra_options: Dict[str, Any] = field(default_factory=dict)  # Additional options
```

## Testing

Run the test script to verify your setup:

```bash
python test_ai_providers.py
```

This will test each provider and their functionality.

## Error Handling

The providers use a consistent error hierarchy:

- `APIKeyMissingError`: API key is missing or invalid
- `APIConnectionError`: Cannot connect to the API
- `APIResponseError`: API returned an error
- `SSLVerificationError`: SSL verification failed

Example:

```python
from ai_providers import get_anthropic_provider
from error_handling import APIResponseError, APIConnectionError

try:
    provider = get_anthropic_provider()
    response = provider.generate("Generate something")
    print(response.content)
except APIConnectionError as e:
    print(f"Connection error: {e}")
except APIResponseError as e:
    print(f"API error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```