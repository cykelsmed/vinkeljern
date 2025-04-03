#!/usr/bin/env python3
"""
Example usage of the Vinkeljernet configuration system.

This script demonstrates how to use the new configuration system
in a typical Vinkeljernet file.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any

# Make sure we can import modules from the project
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import from the new configuration system
from config_manager import (
    get_config, 
    Environment,
    get_openai_api_key,
    get_anthropic_api_key,
    get_perplexity_api_key
)

# Import other Vinkeljernet components
from config_loader import load_and_validate_profile


def create_api_client(provider: str) -> Dict[str, Any]:
    """
    Create an API client configuration for the specified provider.
    
    Args:
        provider: The provider name (openai, anthropic, perplexity)
        
    Returns:
        A dictionary with API client configuration
    """
    # Get the configuration
    config = get_config()
    
    # Get provider configuration
    provider_config = getattr(config.api, provider)
    
    # Create client configuration
    client_config = {
        "api_key": provider_config.api_key.get_secret_value(),
        "model": provider_config.model,
        "max_tokens": provider_config.max_tokens,
        "temperature": provider_config.temperature,
        "timeout": provider_config.timeout
    }
    
    # Add API URL if available
    if provider_config.api_url:
        client_config["api_url"] = provider_config.api_url
    
    # Add any extra options
    for key, value in provider_config.extra_options.items():
        client_config[key] = value
    
    return client_config


def get_default_profile() -> str:
    """
    Get the default profile from configuration.
    
    Returns:
        The path to the default profile
    """
    config = get_config()
    profile_name = config.app.default_profile or "dr_profil"
    profile_dir = config.app.profile_directory
    
    # Make sure profile has .yaml extension
    if not profile_name.endswith(".yaml"):
        profile_name = f"{profile_name}.yaml"
    
    # Return full path
    return str(Path(profile_dir) / profile_name)


def main() -> None:
    """Run the example."""
    # Get configuration
    config = get_config()
    
    print(f"Running in {config.env} environment")
    print(f"Debug mode: {config.debug}")
    print(f"Logging level: {config.logging.level}")
    
    # Load default profile
    profile_path = get_default_profile()
    print(f"Loading profile from: {profile_path}")
    
    try:
        profile = load_and_validate_profile(profile_path)
        print(f"Loaded profile: {profile.navn}")
    except Exception as e:
        print(f"Error loading profile: {e}")
    
    # Create API clients
    for provider in ["openai", "anthropic", "perplexity"]:
        client_config = create_api_client(provider)
        print(f"\n{provider.title()} API Client Configuration:")
        
        # Print non-sensitive values only
        for key, value in client_config.items():
            if key != "api_key":
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {'*' * 10}")  # Mask API key
    
    # Use convenience functions for backward compatibility
    openai_key = get_openai_api_key()
    anthropic_key = get_anthropic_api_key()
    perplexity_key = get_perplexity_api_key()
    
    print("\nAPI Keys (masked):")
    print(f"  OpenAI: {'*' * 10}")
    print(f"  Anthropic: {'*' * 10}")
    print(f"  Perplexity: {'*' * 10}")
    
    # Use application settings
    print("\nApplication Settings:")
    print(f"  Interface: {config.app.interface}")
    print(f"  Number of angles: {config.app.num_angles}")
    print(f"  Output format: {config.app.default_output_format}")
    
    # Force configuration reload to pick up any environment variables
    get_config.cache_clear()
    fresh_config = get_config()
    
    # Show any overrides
    if fresh_config.app.num_angles != 5:  # Default is 5
        print(f"  Overridden number of angles: {fresh_config.app.num_angles}")


if __name__ == "__main__":
    main()