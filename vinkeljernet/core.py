"""
Core functionality for the Vinkeljernet application.

This module contains the core functionality for generating angles,
independent of the user interface.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple

from rich import print as rprint
from rich.console import Console

from vinkeljernet.ui_utils import (
    display_profile_info,
    create_progress_spinner,
    display_angles_table,
    display_angles_panels
)

from config_manager import get_config
from models import RedaktionelDNA
from config_loader import load_and_validate_profile
from api_clients import fetch_topic_information, generate_angles
from angle_processor import filter_and_rank_angles

# Configure logger
logger = logging.getLogger("vinkeljernet.core")

# Get configuration
config = get_config()


async def process_generation_request(
    topic: str,
    profile_path: str,
    format_type: str = "json",
    output_path: Optional[str] = None,
    dev_mode: bool = False,
    bypass_cache: bool = False,
    debug: bool = False,
    progress_callback: Optional[Callable[[int], None]] = None
) -> Tuple[List[Dict[str, Any]], RedaktionelDNA, str]:
    """
    Process a request to generate angles.
    
    Args:
        topic: The news topic to generate angles for
        profile_path: Path to the profile YAML file
        format_type: Output format type (json, markdown, html)
        output_path: Path to save output (optional)
        dev_mode: Whether to run in development mode
        bypass_cache: Whether to bypass cache
        debug: Whether to enable debug mode
        progress_callback: Optional callback for progress updates
        
    Returns:
        Tuple containing:
        - List of ranked angles
        - RedaktionelDNA profile
        - Background information
    
    Raises:
        FileNotFoundError: If profile file doesn't exist
        ValueError: If validation fails or no angles can be generated
    """
    # Load and validate profile
    profile = load_and_validate_profile(Path(profile_path))
    logger.info(f"Loaded profile: {profile.navn}")
    
    # Get information about the topic
    logger.info(f"Fetching information for topic: {topic}")
    topic_info = await fetch_topic_information(
        topic, 
        dev_mode=dev_mode, 
        bypass_cache=bypass_cache,
        progress_callback=progress_callback
    )
    
    if topic_info:
        logger.info("Background information retrieved successfully")
    else:
        logger.warning("Could not retrieve detailed background information")
        # Set a minimal fallback for topic_info
        topic_info = f"Emnet handler om {topic}. Ingen yderligere baggrundsinformation tilgængelig."
    
    # Generate angles
    logger.info("Generating angles...")
    angles = generate_angles(topic, topic_info, profile, bypass_cache=bypass_cache)
    
    if not angles:
        error_msg = "No angles could be generated."
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info(f"Generated {len(angles)} raw angles")
    
    # Filter and rank angles
    logger.info("Filtering and ranking angles...")
    ranked_angles = filter_and_rank_angles(angles, profile, config.app.num_angles)
    
    if not ranked_angles:
        error_msg = "No angles left after filtering."
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info(f"Ranked and filtered to {len(ranked_angles)} angles")
    
    # Save to output file if specified
    if output_path:
        save_output(ranked_angles, profile, topic, format_type, output_path)
    
    return ranked_angles, profile, topic_info


def safe_process_angles(angles: List[Dict[str, Any]], profile: RedaktionelDNA, num_angles: int = 5) -> List[Dict[str, Any]]:
    """
    Process angles with robust error handling.

    Tries to filter and rank angles using filter_and_rank_angles. If an AttributeError
    is raised (for example, a missing attribute), prints a friendly error message and
    falls back to a simple sort based on the number of news criteria in each angle.
    
    Args:
        angles: List of angle dictionaries.
        profile: Editorial DNA profile.
        num_angles: Number of angles to return.
        
    Returns:
        A list of angles.
    """
    try:
        return filter_and_rank_angles(angles, profile, num_angles)
    except AttributeError as e:
        logger.error(f"AttributeError during filtering: {e}")
        print(f"AttributeError during filtering: {e}")
        print("Falling back to simple sort of angles based on the count of news criteria.")
        sorted_angles = sorted(angles, key=lambda x: len(x.get("nyhedskriterier", [])), reverse=True)
        return sorted_angles[:num_angles]
    except Exception as e:
        logger.error(f"Unexpected error during filtering: {e}")
        print(f"Unexpected error during filtering: {e}")
        print("Falling back to unfiltered angle list.")
        return angles[:num_angles]


def save_output(
    angles: List[Dict[str, Any]],
    profile: RedaktionelDNA,
    topic: str,
    format_type: str,
    output_path: str
) -> None:
    """
    Save the generated angles to a file.
    
    Args:
        angles: The angles to save
        profile: The editorial DNA profile
        topic: The news topic
        format_type: The output format (json, markdown, html)
        output_path: The path to save the file to
    """
    try:
        from formatters import format_angles
        
        # Extract profile name
        profile_name = Path(profile.navn).stem if hasattr(profile, 'navn') else "unknown"
        
        format_angles(
            angles, 
            format_type=format_type,
            profile_name=profile_name,
            topic=topic,
            output_path=output_path
        )
        
        logger.info(f"Results saved to {output_path} ({format_type} format)")
    except ImportError:
        # Fallback to JSON if formatter module not available
        with open(output_path, 'w', encoding='utf-8') as outfile:
            json.dump(angles, outfile, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to {output_path} (JSON format, formatter not available)")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        raise