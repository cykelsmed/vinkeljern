"""
Configuration loader module for Vinkeljernet project.

This module provides functionality to load and validate YAML profile configurations
using the RedaktionelDNA Pydantic model and secure YAML processing.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Set, Optional, Union

from pydantic import ValidationError
from models import RedaktionelDNA
from secure_yaml import secure_load_yaml, secure_load_all_yaml_files, YAMLSecurityError

# Configure logging
logger = logging.getLogger("vinkeljernet.config_loader")

# Define profile field validation constraints
REQUIRED_PROFILE_FIELDS = {
    'navn', 'beskrivelse', 'kerneprincipper', 'tone_og_stil', 
    'nyhedsprioritering', 'fokusOmrader'
}

OPTIONAL_PROFILE_FIELDS = {'noGoOmrader'}
ALL_PROFILE_FIELDS = REQUIRED_PROFILE_FIELDS | OPTIONAL_PROFILE_FIELDS

def load_and_validate_profile(profile_path: Union[str, Path]) -> RedaktionelDNA:
    """
    Load and validate a YAML profile against the RedaktionelDNA model.

    Args:
        profile_path: Path to the YAML profile file (string or Path object)

    Returns:
        RedaktionelDNA: Validated profile as a RedaktionelDNA object

    Raises:
        FileNotFoundError: If the profile file doesn't exist
        YAMLSecurityError: If the profile contains dangerous YAML constructs
        ValueError: If the profile data fails validation against the RedaktionelDNA model
    """
    try:
        # Load YAML securely
        profile_data = secure_load_yaml(profile_path)
        
        # Check for required fields (pre-validation)
        missing_fields = REQUIRED_PROFILE_FIELDS - set(profile_data.keys())
        if missing_fields:
            raise ValueError(f"Missing required fields in profile: {', '.join(missing_fields)}")
        
        # Check for unexpected fields 
        unexpected_fields = set(profile_data.keys()) - ALL_PROFILE_FIELDS
        if unexpected_fields:
            logger.warning(f"Unexpected fields in profile {profile_path}: {', '.join(unexpected_fields)}")
            # Remove unexpected fields for safety
            for field in unexpected_fields:
                del profile_data[field]
        
        # Validate tone_og_stil is a string, not a dict (common error)
        if not isinstance(profile_data.get('tone_og_stil', ''), str):
            raise ValueError(
                "Field 'tone_og_stil' must be a simple string, not a dictionary or list."
            )
        
        # Validate nyhedsprioritering values are integers
        if 'nyhedsprioritering' in profile_data:
            for key, value in profile_data['nyhedsprioritering'].items():
                if not isinstance(value, int):
                    raise ValueError(
                        f"Value for nyhedsprioritering.{key} must be an integer, got {type(value).__name__}"
                    )
        
        # Validate against RedaktionelDNA model
        try:
            profile = RedaktionelDNA(**profile_data)
            logger.info(f"Successfully loaded and validated profile: {profile.navn}")
            return profile
        except ValidationError as e:
            error_message = "Profile validation failed:\n"
            for error in e.errors():
                location = " -> ".join(str(loc) for loc in error["loc"])
                error_message += f"- {location}: {error['msg']}\n"
            logger.error(f"Validation error for profile {profile_path}: {error_message}")
            raise ValueError(error_message) from e
            
    except YAMLSecurityError as e:
        logger.error(f"Security error in profile {profile_path}: {e}")
        raise ValueError(f"Security issue detected in profile: {e}") from e
    except Exception as e:
        logger.error(f"Error loading profile {profile_path}: {e}")
        raise

def get_available_profiles(config_dir: str = "config") -> List[Dict[str, Any]]:
    """
    Get a list of all available profiles with metadata.
    
    Args:
        config_dir: Directory to search for profiles
        
    Returns:
        List of dictionaries containing profile metadata
    """
    profiles = []
    config_path = Path(config_dir)
    
    try:
        # Find all YAML files in the config directory
        yaml_files = list(config_path.glob("*.yaml")) + list(config_path.glob("*.yml"))
        
        for file_path in yaml_files:
            try:
                # Try to load just the basic profile metadata
                profile_data = secure_load_yaml(file_path)
                
                # Create a simplified profile entry with just the metadata
                profile_info = {
                    'filename': file_path.name,
                    'path': str(file_path),
                    'navn': profile_data.get('navn', 'Unnamed Profile'),
                    'beskrivelse': profile_data.get('beskrivelse', 'No description')
                }
                
                profiles.append(profile_info)
            except Exception as e:
                logger.warning(f"Error loading profile {file_path}: {e}")
                # Skip this profile but continue with others
                continue
    except Exception as e:
        logger.error(f"Error scanning for profiles in {config_dir}: {e}")
    
    return profiles

def validate_all_profiles(config_dir: str = "config") -> Dict[str, List[str]]:
    """
    Validate all YAML profiles in the given config directory.

    Args:
        config_dir: Directory to search for profiles
        
    Returns:
        Dictionary with lists of valid and invalid profile paths
    """
    valid_profiles = []
    invalid_profiles = []
    
    config_path = Path(config_dir)
    yaml_files = list(config_path.glob("*.yaml")) + list(config_path.glob("*.yml"))
    
    for file_path in yaml_files:
        try:
            profile = load_and_validate_profile(file_path)
            valid_profiles.append(str(file_path))
            logger.info(f"Valid profile: {profile.navn} in {file_path}")
        except Exception as e:
            invalid_profiles.append(str(file_path))
            logger.warning(f"Invalid profile in {file_path}: {e}")
    
    return {
        'valid': valid_profiles,
        'invalid': invalid_profiles
    }

def create_profile_backup(profile_path: Union[str, Path], backup_dir: Optional[str] = None) -> str:
    """
    Create a backup of a profile file before modifying it.
    
    Args:
        profile_path: Path to the profile file
        backup_dir: Directory to store backups (defaults to 'config/backups')
        
    Returns:
        Path to the backup file
    """
    path = Path(profile_path)
    
    # Default backup directory
    if backup_dir is None:
        backup_dir = path.parent / 'backups'
    else:
        backup_dir = Path(backup_dir)
    
    # Create backup directory if it doesn't exist
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamp for backup filename
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create backup filename
    backup_filename = f"{path.stem}_{timestamp}{path.suffix}"
    backup_path = backup_dir / backup_filename
    
    # Copy file to backup
    import shutil
    shutil.copy2(path, backup_path)
    
    logger.info(f"Created backup of {path.name} at {backup_path}")
    return str(backup_path)

# Test/example code
if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(level=logging.INFO)
    
    # Validate all profiles
    results = validate_all_profiles("config")
    print(f"Valid profiles: {len(results['valid'])}")
    print(f"Invalid profiles: {len(results['invalid'])}")