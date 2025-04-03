"""
Centralized configuration management for Vinkeljernet.

This module provides a centralized configuration management system that:
1. Loads configuration from YAML files, environment variables, and defaults
2. Validates configuration values using Pydantic
3. Supports different environments (development, testing, production)
4. Provides a clean interface to access configuration values
5. Handles sensitive values like API keys securely
"""

import os
import yaml
import logging
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Type, TypeVar
from pydantic import BaseModel, Field, validator, SecretStr
from functools import lru_cache
from dotenv import load_dotenv

# Set up logger
logger = logging.getLogger("vinkeljernet.config")

# Type variable for configuration models
ConfigT = TypeVar('ConfigT', bound='BaseConfig')


class Environment(str, Enum):
    """Environment in which the application is running."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Log level for the application."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class APIProviderConfig(BaseModel):
    """Configuration for an API provider."""
    api_key: SecretStr
    api_url: Optional[str] = None
    model: Optional[str] = None
    max_tokens: int = 1000
    temperature: float = 0.7
    timeout: int = 60
    extra_options: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        """Configuration for the Pydantic model."""
        extra = "allow"


class APIConfig(BaseModel):
    """Configuration for all API providers."""
    openai: APIProviderConfig
    anthropic: APIProviderConfig
    perplexity: APIProviderConfig


class CacheConfig(BaseModel):
    """Configuration for caching."""
    enabled: bool = True
    ttl: int = 3600  # Default TTL in seconds (1 hour)
    directory: str = "~/.vinkeljernet/cache"
    max_size_mb: int = 100  # Maximum cache size in MB
    
    @validator('directory')
    def expand_user_dir(cls, v: str) -> str:
        """Expand user directory in path."""
        return os.path.expanduser(v)


class LoggingConfig(BaseModel):
    """Configuration for logging."""
    level: LogLevel = LogLevel.INFO
    file: Optional[str] = "vinkeljernet.log"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    console: bool = True


class AppConfig(BaseModel):
    """Application-specific configuration."""
    interface: str = "cli"  # cli, web, or both
    default_profile: Optional[str] = None
    profile_directory: str = "config"
    results_directory: str = "results"
    default_output_format: str = "markdown"  # json, markdown, html
    num_angles: int = 5  # Default number of angles to generate
    
    @validator('profile_directory', 'results_directory')
    def validate_directory(cls, v: str) -> str:
        """Validate that directory exists or create it."""
        directory = Path(v)
        if not directory.exists():
            try:
                directory.mkdir(parents=True)
                logger.info(f"Created directory: {directory}")
            except (PermissionError, FileNotFoundError) as e:
                logger.warning(f"Could not create directory {directory}: {e}. Using default.")
                # Fall back to a safe default in the current directory
                if "profile" in v:
                    fallback = Path("config")
                else:
                    fallback = Path("results")
                
                if not fallback.exists():
                    fallback.mkdir(exist_ok=True)
                
                logger.info(f"Using fallback directory: {fallback}")
                return str(fallback)
        return str(directory)


class BaseConfig(BaseModel):
    """Base configuration for Vinkeljernet."""
    # Core settings
    env: Environment = Environment.DEVELOPMENT
    debug: bool = Field(default=False, description="Debug mode flag")
    version: str = "0.1.0"
    
    # Component configurations
    api: APIConfig
    cache: CacheConfig = CacheConfig()
    logging: LoggingConfig = LoggingConfig()
    app: AppConfig = AppConfig()
    
    # Extra settings
    extra: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        """Configuration for the Pydantic model."""
        extra = "allow"
        validate_assignment = True


def load_yaml_config(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        file_path: Path to the YAML configuration file
        
    Returns:
        Dict containing configuration values
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        yaml.YAMLError: If the file is not valid YAML
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    try:
        with open(path, 'r', encoding='utf-8') as file:
            config_data = yaml.safe_load(file)
        return config_data or {}
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML config file {path}: {e}")
        raise


def load_env_vars(prefix: str = "VINKELJERNET_") -> Dict[str, Any]:
    """
    Load configuration from environment variables.
    
    This function loads environment variables with the given prefix
    and converts them to a nested dictionary structure.
    
    For example:
        VINKELJERNET_API_OPENAI_API_KEY=abc123
        
    Will be converted to:
        {"api": {"openai": {"api_key": "abc123"}}}
    
    Special case:
        Environment variables with underscores in their names like
        VINKELJERNET_APP_NUM_ANGLES will be handled correctly.
        
    Args:
        prefix: Prefix for environment variables to load
        
    Returns:
        Dict containing configuration values from environment variables
    """
    # Load .env file if it exists
    load_dotenv()
    
    result = {}
    
    # Special case handling for known fields with underscores
    underscore_mappings = {
        "APP_NUM_ANGLES": ["app", "num_angles"],
        "API_OPENAI_API_KEY": ["api", "openai", "api_key"],
        "API_ANTHROPIC_API_KEY": ["api", "anthropic", "api_key"],
        "API_PERPLEXITY_API_KEY": ["api", "perplexity", "api_key"],
    }
    
    # Find all environment variables with the given prefix
    for key, value in os.environ.items():
        if key.startswith(prefix):
            # Strip prefix
            config_key = key[len(prefix):]
            
            # Check if this is a special case
            parts = None
            for special_key, path in underscore_mappings.items():
                if config_key == special_key:
                    parts = path
                    break
            
            # If not a special case, split by underscore
            if parts is None:
                parts = config_key.lower().split('_')
            
            # Build nested dictionary
            current = result
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Get the last part (key to set)
            last_part = parts[-1]
            
            # Try to parse value as boolean, integer, or float
            # Convert boolean strings
            if value.lower() in ("true", "yes", "1", "on"):
                current[last_part] = True
            elif value.lower() in ("false", "no", "0", "off"):
                current[last_part] = False
            # Convert numeric strings
            elif value.isdigit():
                current[last_part] = int(value)
            elif value.replace(".", "", 1).isdigit() and value.count(".") == 1:
                current[last_part] = float(value)
            # Leave as string
            else:
                current[last_part] = value
    
    return result


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    This function recursively merges the override dictionary into the base dictionary.
    If a key exists in both dictionaries and both values are dictionaries,
    the values are merged recursively. Otherwise, the value from the override
    dictionary is used.
    
    Args:
        base: Base dictionary
        override: Dictionary with values to override
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        if (
            key in result and 
            isinstance(result[key], dict) and 
            isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def create_api_config(config_data: Dict[str, Any]) -> APIConfig:
    """
    Create API configuration from configuration data.
    
    This function extracts API configuration from the configuration data,
    with special handling for API keys from environment variables.
    
    Args:
        config_data: Configuration data
        
    Returns:
        APIConfig object
    """
    # Default API configuration
    api_defaults = {
        "openai": {
            "api_key": os.getenv("OPENAI_API_KEY", ""),
            "model": "gpt-4"
        },
        "anthropic": {
            "api_key": os.getenv("ANTHROPIC_API_KEY", ""),
            "api_url": "https://api.anthropic.com/v1/messages",
            "model": "claude-3-opus-20240229"
        },
        "perplexity": {
            "api_key": os.getenv("PERPLEXITY_API_KEY", ""),
            "api_url": "https://api.perplexity.ai/chat/completions",
            "model": "sonar"
        }
    }
    
    # Extract API configuration from config_data or use defaults
    api_config = config_data.get("api", {})
    
    # Merge with defaults
    api_config = deep_merge(api_defaults, api_config)
    
    # Create APIConfig object
    return APIConfig(
        openai=APIProviderConfig(**api_config["openai"]),
        anthropic=APIProviderConfig(**api_config["anthropic"]),
        perplexity=APIProviderConfig(**api_config["perplexity"])
    )


@lru_cache(maxsize=1)
def get_config(
    env: Optional[str] = None,
    config_file: Optional[str] = None
) -> BaseConfig:
    """
    Get the application configuration.
    
    This function loads the configuration from:
    1. Default values
    2. Configuration file (if provided)
    3. Environment variables
    
    Args:
        env: Environment (development, testing, production) or None to use VINKELJERNET_ENV
        config_file: Path to configuration file or None to use default
        
    Returns:
        BaseConfig object
    """
    # Determine environment
    if env is None:
        env = os.getenv("VINKELJERNET_ENV", Environment.DEVELOPMENT)
    
    # Determine config file
    if config_file is None:
        config_dir = Path(os.getenv("VINKELJERNET_CONFIG_DIR", "config"))
        config_file = config_dir / f"config.{env.lower()}.yaml"
    
    # Start with an empty configuration
    config_data = {}
    
    # Load from file if it exists
    try:
        file_config = load_yaml_config(config_file)
        config_data = deep_merge(config_data, file_config)
    except FileNotFoundError:
        logger.warning(f"Configuration file not found: {config_file}")
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML config file: {e}")
    
    # Load from environment variables
    env_config = load_env_vars()
    logger.debug(f"Environment config: {env_config}")
    config_data = deep_merge(config_data, env_config)
    
    # Create API configuration
    api_config = create_api_config(config_data)
    
    # Create final configuration
    config_data["api"] = api_config.dict()
    
    # Create BaseConfig object
    return BaseConfig(**config_data)


def setup_logging(config: BaseConfig) -> None:
    """
    Set up logging based on configuration.
    
    Args:
        config: Application configuration
    """
    log_config = config.logging
    log_level = getattr(logging, log_config.level)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add file handler if configured
    if log_config.file:
        file_handler = logging.FileHandler(log_config.file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(log_config.format))
        root_logger.addHandler(file_handler)
    
    # Add console handler if configured
    if log_config.console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter(log_config.format))
        root_logger.addHandler(console_handler)


def initialize_config(
    env: Optional[str] = None,
    config_file: Optional[str] = None
) -> BaseConfig:
    """
    Initialize the application configuration.
    
    This function loads the configuration and sets up logging.
    
    Args:
        env: Environment (development, testing, production) or None to use VINKELJERNET_ENV
        config_file: Path to configuration file or None to use default
        
    Returns:
        BaseConfig object
    """
    config = get_config(env, config_file)
    setup_logging(config)
    return config


# Convenience functions for accessing configuration values

def get_openai_api_key() -> str:
    """Get the OpenAI API key."""
    config = get_config()
    return config.api.openai.api_key.get_secret_value()


def get_anthropic_api_key() -> str:
    """Get the Anthropic API key."""
    config = get_config()
    return config.api.anthropic.api_key.get_secret_value()


def get_perplexity_api_key() -> str:
    """Get the Perplexity API key."""
    config = get_config()
    return config.api.perplexity.api_key.get_secret_value()


# Create default configuration files

def create_default_config_files() -> None:
    """
    Create default configuration files.
    
    This function creates default configuration files for development,
    testing, and production environments if they don't exist.
    """
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    # Default configuration
    default_config = {
        "env": "development",
        "debug": True,
        "api": {
            "openai": {
                "model": "gpt-4",
                "max_tokens": 1000,
                "temperature": 0.7
            },
            "anthropic": {
                "api_url": "https://api.anthropic.com/v1/messages",
                "model": "claude-3-opus-20240229",
                "max_tokens": 1000,
                "temperature": 0.7
            },
            "perplexity": {
                "api_url": "https://api.perplexity.ai/chat/completions",
                "model": "sonar", 
                "max_tokens": 1000,
                "temperature": 0.2
            }
        },
        "cache": {
            "enabled": True,
            "ttl": 3600,
            "directory": "~/.vinkeljernet/cache",
            "max_size_mb": 100
        },
        "logging": {
            "level": "INFO",
            "file": "vinkeljernet.log",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "console": True
        },
        "app": {
            "interface": "cli",
            "default_profile": "dr_profil",
            "profile_directory": "config",
            "results_directory": "results",
            "default_output_format": "markdown",
            "num_angles": 5
        }
    }
    
    # Environment-specific configurations
    env_configs = {
        "development": {
            "env": "development",
            "debug": True,
            "logging": {
                "level": "DEBUG"
            }
        },
        "testing": {
            "env": "testing",
            "debug": False,
            "logging": {
                "level": "INFO"
            },
            "cache": {
                "enabled": False
            }
        },
        "production": {
            "env": "production",
            "debug": False,
            "logging": {
                "level": "WARNING"
            }
        }
    }
    
    # Create configuration files
    for env, env_config in env_configs.items():
        config_file = config_dir / f"config.{env}.yaml"
        if not config_file.exists():
            # Merge with default configuration
            merged_config = deep_merge(default_config, env_config)
            
            # Write to file
            with open(config_file, 'w', encoding='utf-8') as file:
                yaml.dump(merged_config, file, default_flow_style=False, allow_unicode=True, sort_keys=False)
            
            logger.info(f"Created default configuration file: {config_file}")


# For backward compatibility with existing code
OPENAI_API_KEY = get_openai_api_key()
ANTHROPIC_API_KEY = get_anthropic_api_key()
PERPLEXITY_API_KEY = get_perplexity_api_key()

# Create an instance for immediate use
config = get_config()

if __name__ == "__main__":
    # Create default configuration files
    create_default_config_files()
    
    # Print current configuration
    print(f"Current environment: {config.env}")
    print(f"Debug mode: {config.debug}")
    print(f"OpenAI model: {config.api.openai.model}")
    print(f"Anthropic model: {config.api.anthropic.model}")
    print(f"Perplexity model: {config.api.perplexity.model}")
    print(f"Cache enabled: {config.cache.enabled}")
    print(f"Logging level: {config.logging.level}")