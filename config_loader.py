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


def validate_all_profiles(config_dir: str = "config") -> None:
    """
    Validate all YAML profiles in the given config directory.

    This function loads each YAML file in the config directory, validates it 
    using the RedaktionelDNA model, and specifically checks that the 
    'tone_og_stil' field is a simple string and not a nested dictionary.
    
    Examples:
    ---------
    Correct YAML formatting for tone_og_stil:
    
      tone_og_stil: "En klar, præcis og engagerende tone."
    
    Incorrect YAML formatting for tone_og_stil:
    
      tone_og_stil:
        type: "engagerende"
        description: "En klar, præcis og engagerende tone."
    
    Raises:
        ValueError: If 'tone_og_stil' is not a simple string.
    """
    config_path = Path(config_dir)
    yaml_files = list(config_path.glob("*.yaml")) + list(config_path.glob("*.yml"))
    
    for file in yaml_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            
            # Validate using the Pydantic model
            profile = RedaktionelDNA(**data)
            
            # Check that tone_og_stil is a simple string (not a dict)
            if not isinstance(profile.tone_og_stil, str):
                raise ValueError(f"In file '{file}', field 'tone_og_stil' must be a simple string, found {type(profile.tone_og_stil).__name__}")
            
            print(f"[VALID] Profile '{profile.navn}' in {file} is valid.")
        except Exception as e:
            print(f"[ERROR] Validating profile in {file}: {e}")


# Example usage:
if __name__ == "__main__":
    validate_all_profiles("config")