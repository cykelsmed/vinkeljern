# Vinkeljernet Configuration Management

This document explains how to use the new centralized configuration management system in Vinkeljernet.

## Overview

The `config_manager.py` module provides a centralized configuration system with the following features:

- Configuration loading from multiple sources (YAML files, environment variables, defaults)
- Environment-specific configuration (development, testing, production)
- Validation using Pydantic models
- Secure handling of sensitive values like API keys
- Hierarchical configuration structure
- Convenient access to configuration values

## Configuration Sources

The configuration is loaded from the following sources, in order of priority:

1. **Environment Variables**: Take highest precedence
2. **Configuration Files**: Environment-specific YAML files
3. **Default Values**: Built into the configuration models

## Directory Structure

```
vinkeljernet/
├── config/
│   ├── config.development.yaml  # Development environment config
│   ├── config.testing.yaml      # Testing environment config
│   ├── config.production.yaml   # Production environment config
│   └── ...                      # Editorial profile files
├── config_manager.py            # Configuration management module
└── ...
```

## Configuration Files

The configuration files are YAML files that can contain any of the configuration settings. A basic example:

```yaml
# Development environment configuration
env: development
debug: true

# API configurations
api:
  openai:
    model: "gpt-4"
    max_tokens: 1000
    temperature: 0.7
  
  anthropic:
    api_url: "https://api.anthropic.com/v1/messages"
    model: "claude-3-opus-20240229"
    max_tokens: 1000
    temperature: 0.7

  # ... more settings
```

## Environment Variables

Environment variables can be used to override configuration values. They should be prefixed with `VINKELJERNET_` and use underscores to represent nested values.

For example:

```
VINKELJERNET_ENV=production
VINKELJERNET_DEBUG=false
VINKELJERNET_API_OPENAI_API_KEY=your-api-key-here
VINKELJERNET_API_ANTHROPIC_API_KEY=your-api-key-here
VINKELJERNET_CACHE_ENABLED=true
```

## Using the Configuration

### Basic Usage

```python
from config_manager import get_config

# Get the configuration
config = get_config()

# Access configuration values
debug_mode = config.debug
openai_model = config.api.openai.model
cache_enabled = config.cache.enabled
```

### Environment-Specific Configuration

You can load a specific environment's configuration:

```python
from config_manager import get_config, Environment

# Load production configuration
config = get_config(env=Environment.PRODUCTION)

# Or using a string
config = get_config(env="production")
```

### Convenience Functions

For common values, convenience functions are provided:

```python
from config_manager import get_openai_api_key, get_anthropic_api_key, get_perplexity_api_key

openai_key = get_openai_api_key()
anthropic_key = get_anthropic_api_key()
perplexity_key = get_perplexity_api_key()
```

### Backward Compatibility

For backward compatibility, the module exports the API keys as constants:

```python
from config_manager import OPENAI_API_KEY, ANTHROPIC_API_KEY, PERPLEXITY_API_KEY
```

## Configuration Structure

The configuration is organized into the following sections:

### Core Settings

- `env`: Environment (development, testing, production)
- `debug`: Debug mode flag
- `version`: Application version

### API Configuration

```python
config.api.openai.api_key  # OpenAI API key (SecretStr)
config.api.openai.model    # OpenAI model name
# ... more OpenAI settings

config.api.anthropic.api_key  # Anthropic API key (SecretStr)
config.api.anthropic.api_url  # Anthropic API URL
config.api.anthropic.model    # Anthropic model name
# ... more Anthropic settings

config.api.perplexity.api_key  # Perplexity API key (SecretStr)
config.api.perplexity.api_url  # Perplexity API URL
config.api.perplexity.model    # Perplexity model name
# ... more Perplexity settings
```

### Cache Configuration

```python
config.cache.enabled      # Whether caching is enabled
config.cache.ttl          # Cache TTL in seconds
config.cache.directory    # Cache directory
config.cache.max_size_mb  # Maximum cache size in MB
```

### Logging Configuration

```python
config.logging.level    # Logging level
config.logging.file     # Log file path
config.logging.format   # Log format
config.logging.console  # Whether to log to console
```

### Application Configuration

```python
config.app.interface             # Interface type (cli, web, both)
config.app.default_profile       # Default profile name
config.app.profile_directory     # Directory containing profiles
config.app.results_directory     # Directory for results
config.app.default_output_format # Default output format
config.app.num_angles            # Number of angles to generate
```

### Extra Settings

Extra settings can be accessed through the `extra` field:

```python
config.extra.get("some_setting")  # Get an extra setting
```

## Security

Sensitive values like API keys are handled securely:

- API keys are stored as `SecretStr` objects in the configuration
- They don't appear in string representations or logs
- Access to the actual value requires explicitly using `.get_secret_value()`
- The module recommends using environment variables or .env files for sensitive values

## Default Configuration Files

The module can create default configuration files for development, testing, and production environments if they don't exist:

```python
from config_manager import create_default_config_files

create_default_config_files()
```

## Initialization

The module initializes the configuration and logging automatically when imported, but you can explicitly initialize it:

```python
from config_manager import initialize_config

config = initialize_config(env="production")
```

## Environment Selection

The environment is determined in the following order:

1. Explicitly passed to `get_config()`
2. `VINKELJERNET_ENV` environment variable
3. Default to `development`