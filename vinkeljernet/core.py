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
    display_angles_panels,
    ProcessStage  # Import ProcessStage enum here
)

from config_manager import get_config
from models import RedaktionelDNA
from config_loader import load_and_validate_profile
from api_clients_wrapper import (
    fetch_topic_information, 
    generate_angles, 
    generate_expert_source_suggestions,
    generate_knowledge_distillate,
    process_generation_request as api_process_generation
)
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
    progress_callback: Optional[Callable[[int], Any]] = None,
    progress_stages: Optional[Dict[str, Callable]] = None
) -> Tuple[List[Dict], RedaktionelDNA, str]:
    """
    Process a generation request with all necessary steps.
    
    Args:
        topic: News topic to generate angles for
        profile_path: Path to the profile YAML file
        format_type: Output format (json, markdown, html)
        output_path: Optional path to save output
        dev_mode: If True, disables SSL verification (for development only)
        bypass_cache: If True, bypass cache
        debug: Enable debug output
        progress_callback: Optional progress callback function
        progress_stages: Dict of stage name to setter function
        
    Returns:
        Tuple containing angles list, profile object, and topic info
    """
    # Load the editorial DNA profile
    profile = None
    topic_info = None
    knowledge_distillate = None
    progress_stages = progress_stages or {}
    
    try:
        # Load profile first - this doesn't require any API calls
        profile = load_and_validate_profile(profile_path)
        # Use profile.navn instead of profile.name (Danish for 'name')
        logger.info(f"Loaded profile: {profile.navn}")
        
        # Set stage to fetching background info if callback provided
        if "FETCHING_INFO" in progress_stages:
            progress_stages["FETCHING_INFO"](ProcessStage.FETCHING_INFO, 
                                           f"Henter baggrundsinformation om '{topic}'...")
        
        # Get background information about the topic
        logger.info(f"Fetching information for topic: {topic}")
        try:
            from api_clients_wrapper import fetch_topic_information
            topic_info = await fetch_topic_information(
                topic=topic,
                dev_mode=dev_mode,
                bypass_cache=bypass_cache,
                progress_callback=progress_callback,
                detailed=True
            )
            
            if not topic_info:
                logger.warning("Could not retrieve detailed background information")
                topic_info = f"Emnet handler om {topic}. Ingen yderligere baggrundsinformation tilgængelig."
                
        except Exception as e:
            logger.error(f"Error fetching topic information: {str(e)}")
            topic_info = f"Emnet handler om {topic}. Ingen yderligere baggrundsinformation tilgængelig."

        # Set stage to generating knowledge distillate if callback provided
        if "GENERATING_KNOWLEDGE" in progress_stages:
            progress_stages["GENERATING_KNOWLEDGE"](
                ProcessStage.GENERATING_KNOWLEDGE,
                "Genererer videndistillat fra baggrundsinformation..."
            )
            
        # Generate knowledge distillate from background info
        try:
            logger.info("Generating knowledge distillate from background information")
            from api_clients_wrapper import generate_knowledge_distillate
            knowledge_distillate = await generate_knowledge_distillate(
                topic_info=topic_info,
                topic=topic,
                bypass_cache=bypass_cache,
                progress_callback=progress_callback
            )
            logger.info("Knowledge distillate generated successfully")
        except Exception as e:
            logger.error(f"Error generating knowledge distillate: {str(e)}")
            knowledge_distillate = None
            
        # Set stage to generating angles if callback provided
        if "GENERATING_ANGLES" in progress_stages:
            progress_stages["GENERATING_ANGLES"](ProcessStage.GENERATING_ANGLES, 
                                              f"Genererer vinkler for '{topic}' med {profile.navn} profilen...")
                                              
        # Generate angles from the topic and profile
        logger.info("Generating angles...")
        
        # Use optimized process_generation_request directly if available
        raw_angles = []
        try:
            from api_clients_wrapper import process_generation_request
            # Try to use the optimized version first
            raw_angles = await process_generation_request(
                topic=topic,
                profile=profile,
                bypass_cache=bypass_cache,
                progress_callback=progress_callback,
                include_expert_sources=True,
                include_knowledge_distillate=True
            )
            
        except (ImportError, AttributeError, NotImplementedError) as e:
            logger.warning(f"Could not use optimized process_generation_request: {str(e)}")
            try:
                # Fall back to standard generate_angles if optimized version fails
                from api_clients_wrapper import generate_angles
                raw_angles = await generate_angles(
                    emne=topic,
                    topic_info=topic_info,
                    profile=profile,
                    bypass_cache=bypass_cache
                )
            except Exception as e:
                logger.error(f"Error in generate_angles: {str(e)}")
                # Return a minimal result with error information
                raw_angles = [{
                    "overskrift": f"Fejl under vinkelgenerering: {str(e)}",
                    "beskrivelse": "Der opstod en fejl under generering af vinkler. Se logs for detaljer.",
                    "nyhedskriterier": ["aktualitet"],
                    "error": str(e)
                }]
        
        logger.info(f"Generated {len(raw_angles)} raw angles")
        
        # Set stage to filtering and ranking angles if callback provided
        if "FILTERING_ANGLES" in progress_stages:
            progress_stages["FILTERING_ANGLES"](ProcessStage.FILTERING_ANGLES, 
                                             "Filtrerer og rangerer vinkler...")
                                             
        # Filter and rank the angles
        logger.info("Filtering and ranking angles...")
        from angle_processor import filter_and_rank_angles
        ranked_angles = filter_and_rank_angles(raw_angles, profile, 5)
        logger.info(f"Ranked and filtered to {len(ranked_angles)} angles")
        
        # Set stage to generating source suggestions if callback provided
        if "GENERATING_SOURCES" in progress_stages:
            progress_stages["GENERATING_SOURCES"](ProcessStage.GENERATING_SOURCES, 
                                               "Finder eksperter og kilder...")
                                               
        # Generate source suggestions
        logger.info("Generating general source suggestions...")
        try:
            from api_clients_wrapper import fetch_source_suggestions
            source_text = await fetch_source_suggestions(topic, bypass_cache=bypass_cache)
            if source_text:
                for angle in ranked_angles:
                    if isinstance(angle, dict):
                        angle['kildeForslagInfo'] = source_text
        except Exception as e:
            logger.error(f"Error fetching source suggestions: {str(e)}")
            # Continue without source suggestions
            
        # Set stage to generating expert sources if callback provided
        if "GENERATING_EXPERT_SOURCES" in progress_stages:
            progress_stages["GENERATING_EXPERT_SOURCES"](ProcessStage.GENERATING_EXPERT_SOURCES, 
                                                     "Finder ekspertkilder til vinkler...")
                                                     
        # Generate expert sources for each angle
        logger.info("Generating expert sources for each angle...")
        for i, angle in enumerate(ranked_angles):
            if not isinstance(angle, dict):
                continue
                
            # Only process top 3 angles to limit API calls
            if i >= 3:
                break
                
            try:
                headline = angle.get('overskrift', f"Vinkel om {topic}")
                description = angle.get('beskrivelse', "")
                
                from api_clients_wrapper import generate_expert_source_suggestions
                expert_sources = await generate_expert_source_suggestions(
                    topic=topic,
                    angle_headline=headline,
                    angle_description=description,
                    bypass_cache=bypass_cache
                )
                
                if expert_sources:
                    angle['ekspertKilder'] = expert_sources
                    angle['harEkspertKilder'] = True
                    
            except Exception as e:
                logger.warning(f"Failed to generate expert sources for angle: {headline}")
                # Continue without expert sources for this angle
        
        # Save to file if requested
        if output_path:
            from formatters import save_results
            save_results(ranked_angles, profile, topic_info, output_path, format_type)
            
        return ranked_angles, profile, topic_info
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in process_generation_request: {str(e)}", exc_info=True)
        raise


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