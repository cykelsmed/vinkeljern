"""
Secure YAML processing module for Vinkeljernet.

This module provides functions for securely loading and validating YAML files,
with additional protection against common YAML-based vulnerabilities.
"""

import yaml
import re
import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Set, Optional, Union

# Configure logging
logger = logging.getLogger("vinkeljernet.secure_yaml")

# Maximum file size for YAML files (100KB)
MAX_YAML_SIZE = 100 * 1024

# Set of dangerous YAML tags that should be blocked
DANGEROUS_YAML_TAGS = {
    '!!python/object',
    '!!python/name',
    '!!python/module',
    '!!python/object/apply',
    '!!python/object/new',
}

class YAMLSecurityError(Exception):
    """Exception raised for YAML security violations."""
    pass

def check_yaml_safety(content: str) -> bool:
    """
    Check if YAML content contains potentially dangerous patterns.
    
    Args:
        content: The YAML content to check
        
    Returns:
        True if safe, False if potentially dangerous
        
    Raises:
        YAMLSecurityError: If dangerous patterns are found
    """
    # Check for dangerous tags
    for tag in DANGEROUS_YAML_TAGS:
        if tag in content:
            raise YAMLSecurityError(f"Dangerous YAML tag found: {tag}")
    
    # Check for any tag directive (custom tags)
    if re.search(r'!![a-z]+/[a-z]+', content):
        raise YAMLSecurityError("Custom YAML tag directive found")
        
    # Check for suspicious include directives
    if re.search(r'include\s*:\s*[\'"]?[/~]', content):
        raise YAMLSecurityError("Suspicious include directive found")
        
    return True

def secure_load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Securely load a YAML file with additional security checks.
    
    Args:
        file_path: Path to the YAML file
        
    Returns:
        Dict containing the YAML data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        PermissionError: If the file can't be read
        YAMLSecurityError: If security checks fail
        yaml.YAMLError: If there's an error parsing the YAML
    """
    # Ensure we have a Path object
    path = Path(file_path)
    
    # Security check: File must exist
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
        
    # Security check: Check file permissions (Unix-like systems)
    if hasattr(os, 'access') and hasattr(os, 'R_OK'):
        if not os.access(path, os.R_OK):
            raise PermissionError(f"No permission to read file: {path}")
    
    # Security check: Max file size
    file_size = path.stat().st_size
    if file_size > MAX_YAML_SIZE:
        raise YAMLSecurityError(
            f"YAML file too large: {file_size} bytes (max: {MAX_YAML_SIZE})"
        )
    
    # Read the file content
    try:
        with open(path, 'r', encoding='utf-8') as file:
            content = file.read()
    except Exception as e:
        logger.error(f"Error reading YAML file {path}: {e}")
        raise
    
    # Security check: Check for dangerous patterns
    check_yaml_safety(content)
    
    # Parse the YAML using the safe loader
    try:
        data = yaml.safe_load(content)
        return data
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {path}: {e}")
        raise

def secure_load_yaml_string(content: str) -> Dict[str, Any]:
    """
    Securely load YAML content from a string.
    
    Args:
        content: YAML content as a string
        
    Returns:
        Dict containing the YAML data
        
    Raises:
        YAMLSecurityError: If security checks fail
        yaml.YAMLError: If there's an error parsing the YAML
    """
    # Security check: Max content size
    if len(content) > MAX_YAML_SIZE:
        raise YAMLSecurityError(
            f"YAML content too large: {len(content)} bytes (max: {MAX_YAML_SIZE})"
        )
    
    # Security check: Check for dangerous patterns
    check_yaml_safety(content)
    
    # Parse the YAML using the safe loader
    try:
        data = yaml.safe_load(content)
        return data
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML content: {e}")
        raise

def secure_load_all_yaml_files(directory: Union[str, Path], pattern: str = "*.yaml") -> List[Dict[str, Any]]:
    """
    Securely load all YAML files in a directory.
    
    Args:
        directory: Directory to scan for YAML files
        pattern: Glob pattern to match files
        
    Returns:
        List of dictionaries containing YAML data
        
    Raises:
        FileNotFoundError: If the directory doesn't exist
    """
    # Ensure we have a Path object
    path = Path(directory)
    
    # Security check: Directory must exist
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Directory not found: {path}")
    
    # Find all matching files and load each one securely
    results = []
    for yaml_file in path.glob(pattern):
        try:
            yaml_data = secure_load_yaml(yaml_file)
            # Add the filename for reference
            yaml_data['_filename'] = yaml_file.name
            results.append(yaml_data)
        except Exception as e:
            logger.warning(f"Error loading YAML file {yaml_file}: {e}")
            # Continue with other files
            continue
            
    return results

def secure_dump_yaml(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """
    Securely dump data to a YAML file.
    
    Args:
        data: Data to write to the file
        file_path: Path to the output file
        
    Raises:
        PermissionError: If the file can't be written
    """
    # Ensure we have a Path object
    path = Path(file_path)
    
    # Create parent directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write the data to the file using the safe dumper
    try:
        with open(path, 'w', encoding='utf-8') as file:
            yaml.safe_dump(data, file, default_flow_style=False)
    except Exception as e:
        logger.error(f"Error writing YAML file {path}: {e}")
        raise

def validate_yaml_structure(data: Dict[str, Any], 
                          required_fields: Set[str], 
                          optional_fields: Optional[Set[str]] = None) -> bool:
    """
    Validate that a YAML structure contains required fields and only allowed fields.
    
    Args:
        data: The YAML data to validate
        required_fields: Set of field names that must be present
        optional_fields: Set of field names that are allowed but not required
        
    Returns:
        True if valid, False otherwise
        
    Raises:
        ValueError: If validation fails
    """
    # Check for required fields
    missing_fields = required_fields - set(data.keys())
    if missing_fields:
        raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
    
    # Check for unexpected fields if optional_fields is provided
    if optional_fields is not None:
        allowed_fields = required_fields | optional_fields
        unexpected_fields = set(data.keys()) - allowed_fields
        if unexpected_fields:
            raise ValueError(f"Unexpected fields: {', '.join(unexpected_fields)}")
    
    return True