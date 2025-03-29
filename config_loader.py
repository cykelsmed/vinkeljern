"""
Configuration loader module for Vinkeljernet project.

This module provides functionality to load and validate YAML profile configurations
using the RedaktionelDNA Pydantic model.
"""

import yaml
from pathlib import Path
from pydantic import ValidationError
from models import RedaktionelDNA


def load_and_validate_profile(profile_path: str | Path) -> RedaktionelDNA:
    """
    Load and validate a YAML profile against the RedaktionelDNA model.

    Args:
        profile_path: Path to the YAML profile file (string or Path object)

    Returns:
        RedaktionelDNA: Validated profile as a RedaktionelDNA object

    Raises:
        FileNotFoundError: If the profile file doesn't exist
        yaml.YAMLError: If there's an error parsing the YAML file
        ValueError: If the profile data fails validation against the RedaktionelDNA model
    """
    # Convert to Path object if string is provided
    path = Path(profile_path)

    # Check if file exists
    if not path.exists():
        raise FileNotFoundError(f"Profile file not found: {path}")
    
    # Read and parse YAML file
    try:
        with open(path, 'r', encoding='utf-8') as file:
            profile_data = yaml.safe_load(file)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        raise
    
    # Validate against RedaktionelDNA model
    try:
        profile = RedaktionelDNA(**profile_data)
        return profile
    except ValidationError as e:
        error_message = "Profile validation failed:\n"
        for error in e.errors():
            location = " -> ".join(str(loc) for loc in error["loc"])
            error_message += f"- {location}: {error['msg']}\n"
        print(error_message)
        raise ValueError(error_message) from e