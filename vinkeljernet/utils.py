"""
Utility functions for the Vinkeljernet application.

This module provides utility functions for file operations, logging,
and other common tasks.
"""

import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

from config_manager import get_config

# Configure logger
logger = logging.getLogger("vinkeljernet.utils")


def setup_logging(debug: bool = False) -> None:
    """
    Set up logging for the application.
    
    Args:
        debug: Whether to enable debug mode
    """
    config = get_config()
    log_level = logging.DEBUG if debug else getattr(logging, config.logging.level)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Configure file handler
    if config.logging.file:
        file_handler = logging.FileHandler(config.logging.file)
        file_formatter = logging.Formatter(config.logging.format)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Configure console handler if enabled
    if config.logging.console:
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(config.logging.format)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)


def get_available_profiles(profile_dir: Optional[str] = None) -> List[str]:
    """
    Get a list of available profile paths.
    
    Args:
        profile_dir: Directory containing profile files,
                    or None to use the default
                    
    Returns:
        List of profile paths
    """
    config = get_config()
    directory = Path(profile_dir or config.app.profile_directory)
    
    if not directory.exists() or not directory.is_dir():
        return []
    
    profiles = list(directory.glob("*.yaml"))
    return [str(p) for p in profiles]


def get_profile_names() -> List[str]:
    """
    Get a list of available profile names.
    
    Returns:
        List of profile names (without path and extension)
    """
    profiles = get_available_profiles()
    return [Path(p).stem for p in profiles]


def get_default_profile() -> str:
    """
    Get the default profile path.
    
    Returns:
        Path to the default profile
    """
    config = get_config()
    default_profile = config.app.default_profile or "dr_profil"
    
    # Add .yaml extension if needed
    if not default_profile.endswith(".yaml"):
        default_profile += ".yaml"
    
    # Get profile directory
    profile_dir = config.app.profile_directory
    
    # Return full path
    return str(Path(profile_dir) / default_profile)


def clear_api_cache() -> int:
    """
    Clear the API cache.
    
    Returns:
        Number of cache files removed
    """
    try:
        from cache_manager import clear_cache
        return clear_cache()
    except ImportError:
        logger.warning("Could not import cache_manager.clear_cache")
        return 0


def reset_circuit_breakers() -> None:
    """
    Reset all circuit breakers to closed state.
    """
    try:
        from retry_manager import reset_circuit
        reset_circuit("perplexity_api")
        reset_circuit("openai_api")
        reset_circuit("anthropic_api")
        logger.info("All circuit breakers reset")
    except ImportError:
        logger.warning("Could not import retry_manager.reset_circuit")


def get_circuit_stats() -> Dict[str, Any]:
    """
    Get statistics for all circuit breakers.
    
    Returns:
        Dictionary with circuit breaker statistics
    """
    try:
        from retry_manager import get_circuit_stats
        return get_circuit_stats()
    except ImportError:
        logger.warning("Could not import retry_manager.get_circuit_stats")
        return {}