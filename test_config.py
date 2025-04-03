#!/usr/bin/env python3
"""
Test script for the Vinkeljernet configuration system.

This script tests the configuration system by loading and displaying
configuration values from different environments and sources.
"""

import os
import sys
from pathlib import Path
from pprint import pprint

# Make sure we can import modules from the project
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config_manager import (
    get_config,
    Environment,
    get_openai_api_key,
    get_anthropic_api_key,
    get_perplexity_api_key,
    create_default_config_files
)


def print_section(title):
    """Print a section title."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)


def test_get_config():
    """Test the get_config function."""
    print_section("Testing get_config()")
    
    # Test default configuration
    config = get_config()
    print(f"Environment: {config.env}")
    print(f"Debug mode: {config.debug}")
    print(f"OpenAI model: {config.api.openai.model}")
    print(f"Anthropic model: {config.api.anthropic.model}")
    print(f"Perplexity model: {config.api.perplexity.model}")
    print(f"Cache enabled: {config.cache.enabled}")
    print(f"Logging level: {config.logging.level}")


def test_environment_specific_configs():
    """Test loading configuration for different environments."""
    print_section("Testing Environment-Specific Configurations")
    
    for env in [Environment.DEVELOPMENT, Environment.TESTING, Environment.PRODUCTION]:
        print(f"\nEnvironment: {env}")
        config = get_config(env=env)
        print(f"Debug mode: {config.debug}")
        print(f"OpenAI model: {config.api.openai.model}")
        print(f"Cache enabled: {config.cache.enabled}")
        print(f"Logging level: {config.logging.level}")


def test_api_keys():
    """Test retrieving API keys."""
    print_section("Testing API Keys")
    
    # Test convenience functions
    print("Using convenience functions:")
    print(f"OpenAI API key: {'*' * len(get_openai_api_key()) if get_openai_api_key() else 'Not set'}")
    print(f"Anthropic API key: {'*' * len(get_anthropic_api_key()) if get_anthropic_api_key() else 'Not set'}")
    print(f"Perplexity API key: {'*' * len(get_perplexity_api_key()) if get_perplexity_api_key() else 'Not set'}")
    
    # Test direct access (SecretStr)
    print("\nUsing direct access (SecretStr):")
    config = get_config()
    print(f"OpenAI API key: {config.api.openai.api_key}")  # Will print SecretStr('**********') for security
    print(f"Anthropic API key: {config.api.anthropic.api_key}")
    print(f"Perplexity API key: {config.api.perplexity.api_key}")


def test_environment_variable_override():
    """Test overriding configuration with environment variables."""
    print_section("Testing Environment Variable Override")
    
    # Save original environment variable if it exists
    original_env = os.environ.get("VINKELJERNET_ENV")
    original_debug = os.environ.get("VINKELJERNET_DEBUG")
    
    try:
        # Override with environment variables
        os.environ["VINKELJERNET_ENV"] = "testing"
        os.environ["VINKELJERNET_DEBUG"] = "true"
        
        # Force reload configuration
        get_config.cache_clear()
        
        # Get configuration
        config = get_config()
        print(f"Environment (should be testing): {config.env}")
        print(f"Debug mode (should be True): {config.debug}")
        
        # Change environment variable
        os.environ["VINKELJERNET_DEBUG"] = "false"
        
        # Force reload configuration
        get_config.cache_clear()
        
        # Get configuration again
        config = get_config()
        print(f"Debug mode (should be False): {config.debug}")
        
    finally:
        # Restore original environment variables
        if original_env is not None:
            os.environ["VINKELJERNET_ENV"] = original_env
        else:
            os.environ.pop("VINKELJERNET_ENV", None)
            
        if original_debug is not None:
            os.environ["VINKELJERNET_DEBUG"] = original_debug
        else:
            os.environ.pop("VINKELJERNET_DEBUG", None)
        
        # Force reload configuration
        get_config.cache_clear()


def test_create_default_configs():
    """Test creating default configuration files."""
    print_section("Testing create_default_config_files()")
    
    # Create default configuration files
    create_default_config_files()
    
    # Check if files were created
    config_dir = Path("config")
    dev_config = config_dir / "config.development.yaml"
    test_config = config_dir / "config.testing.yaml"
    prod_config = config_dir / "config.production.yaml"
    
    print(f"Development config exists: {dev_config.exists()}")
    print(f"Testing config exists: {test_config.exists()}")
    print(f"Production config exists: {prod_config.exists()}")


def main():
    """Run all tests."""
    print_section("Vinkeljernet Configuration System Test")
    
    test_get_config()
    test_environment_specific_configs()
    test_api_keys()
    test_environment_variable_override()
    test_create_default_configs()


if __name__ == "__main__":
    main()