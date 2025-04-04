"""
Optimized core functionality for the Vinkeljernet application.

This module contains the core functionality for generating angles,
with optimizations for performance, concurrency, and resilience.
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple, Set, Union
from datetime import datetime

from rich import print as rprint
from rich.console import Console

# Import custom utilities
from vinkeljernet.ui_utils import (
    display_profile_info,
    create_progress_spinner,
    display_angles_table,
    display_angles_panels,
    ProcessStage
)

# Import configuration and models
from config_manager import get_config
from models import RedaktionelDNA
from config_loader import load_and_validate_profile

# Import API clients
from api_clients_optimized import (
    fetch_topic_information, 
    generate_angles, 
    generate_expert_source_suggestions,
    generate_knowledge_distillate,
    process_generation_request as api_process_generation,
    get_performance_metrics,
    initialize_api_client,
    shutdown_api_client
)
from angle_processor import filter_and_rank_angles
from cache_manager import optimize_cache, get_cache_stats

# Configure logger
logger = logging.getLogger("vinkeljernet.optimized_core")

# Get configuration
config = get_config()

# Global performance metrics
performance_data = {
    "topic_info_time": 0,
    "angles_generation_time": 0,
    "knowledge_distillate_time": 0,
    "expert_sources_time": 0,
    "total_time": 0,
    "cache_hits": 0,
    "api_calls": 0
}

class ParallelStageExecutor:
    """Manages parallel execution of generation stages with timeouts and fallbacks."""
    
    def __init__(self, timeout_multiplier: float = 1.0):
        """
        Initialize the parallel stage executor.
        
        Args:
            timeout_multiplier: Multiplier for default timeouts (1.0 = normal timeouts)
        """
        self.timeout_multiplier = timeout_multiplier
        self.stage_timeouts = {
            "topic_info": 30 * timeout_multiplier,  # 30 seconds
            "angles": 60 * timeout_multiplier,      # 60 seconds
            "knowledge_distillate": 45 * timeout_multiplier,  # 45 seconds
            "expert_sources": 25 * timeout_multiplier,  # 25 seconds per expert
            "sources": 20 * timeout_multiplier      # 20 seconds
        }
        self.tasks = {}
        self._start_times = {}
        self._completed_stages = set()
        self._failed_stages = set()
        
    async def execute_with_timeout(
        self, 
        stage_name: str,
        coroutine,
        timeout: Optional[float] = None,
        fallback_value: Any = None
    ) -> Any:
        """
        Execute a coroutine with timeout and return result or fallback.
        
        Args:
            stage_name: Name of the execution stage
            coroutine: Coroutine to execute
            timeout: Timeout in seconds (uses default for stage if None)
            fallback_value: Value to return if execution times out or fails
            
        Returns:
            Result of coroutine execution or fallback value
        """
        if timeout is None:
            timeout = self.stage_timeouts.get(stage_name, 30.0)
            
        # Record start time
        self._start_times[stage_name] = time.time()
        
        try:
            # Create and run the task with timeout
            result = await asyncio.wait_for(coroutine, timeout=timeout)
            execution_time = time.time() - self._start_times[stage_name]
            logger.info(f"Stage '{stage_name}' completed in {execution_time:.2f}s")
            
            # Record stage completion
            self._completed_stages.add(stage_name)
            
            # Update performance metrics
            if stage_name in performance_data:
                performance_data[stage_name + "_time"] = execution_time
                
            return result
        except asyncio.TimeoutError:
            execution_time = time.time() - self._start_times[stage_name]
            logger.warning(f"Stage '{stage_name}' timed out after {execution_time:.2f}s")
            self._failed_stages.add(stage_name)
            return fallback_value
        except Exception as e:
            execution_time = time.time() - self._start_times[stage_name]
            logger.error(f"Stage '{stage_name}' failed after {execution_time:.2f}s: {e}")
            self._failed_stages.add(stage_name)
            return fallback_value
    
    def get_completed_stages(self) -> Set[str]:
        """Get set of completed stage names."""
        return self._completed_stages
    
    def get_failed_stages(self) -> Set[str]:
        """Get set of failed stage names."""
        return self._failed_stages
    
    def get_execution_time(self, stage_name: str) -> float:
        """Get execution time for a specific stage."""
        if stage_name in self._start_times:
            end_time = time.time()
            if stage_name in self._completed_stages or stage_name in self._failed_stages:
                # For completed or failed stages, we already calculated the time
                if stage_name + "_time" in performance_data:
                    return performance_data[stage_name + "_time"]
                return 0
            else:
                # For running stages, calculate current duration
                return end_time - self._start_times[stage_name]
        return 0


async def optimized_process_generation(
    topic: str,
    profile_path: str,
    format_type: str = "json",
    output_path: Optional[str] = None,
    dev_mode: bool = False,
    bypass_cache: bool = False,
    debug: bool = False,
    progress_callback: Optional[Callable[[int], None]] = None,
    progress_stages: Optional[Dict[str, Callable]] = None,
    optimize_performance: bool = True,
    timeout_multiplier: float = 1.0,
    enable_progressive_display: bool = True
) -> Tuple[List[Dict[str, Any]], RedaktionelDNA, str]:
    """
    Optimized process to generate angles with parallel execution and resilience.
    
    Args:
        topic: The news topic to generate angles for
        profile_path: Path to the profile YAML file
        format_type: Output format type (json, markdown, html)
        output_path: Path to save output (optional)
        dev_mode: Whether to run in development mode
        bypass_cache: Whether to bypass cache
        debug: Whether to enable debug mode
        progress_callback: Optional callback for progress updates
        progress_stages: Optional dict of stage name to stage callback function
        optimize_performance: Whether to apply additional performance optimizations
        timeout_multiplier: Multiplier for timeouts (higher = more time allowed)
        enable_progressive_display: Whether to display results progressively
        
    Returns:
        Tuple containing:
        - List of ranked angles
        - RedaktionelDNA profile
        - Background information
    """
    total_start_time = time.time()
    
    try:
        # Initialize API client and optimize cache if needed
        if optimize_performance:
            await initialize_api_client()
            await optimize_cache(silent=True)
        
        # Create parallel stage executor
        executor = ParallelStageExecutor(timeout_multiplier)
        
        # Load and validate profile
        try:
            profile = load_and_validate_profile(Path(profile_path))
            logger.info(f"Loaded profile: {profile.navn}")
        except Exception as e:
            logger.error(f"Failed to load profile: {e}")
            raise ValueError(f"Could not load profile: {e}")
            
        # Notify stage: Fetching background information
        if progress_stages and "FETCHING_INFO" in progress_stages:
            try:
                progress_stages["FETCHING_INFO"](
                    ProcessStage.FETCHING_INFO, 
                    f"Indhenter baggrundsinformation om '{topic}'..."
                )
            except (ImportError, KeyError) as e:
                logger.warning(f"Error in progress stage callback: {e}")
                
        if progress_callback:
            await progress_callback(5)
        
        # 1. Start topic information task
        logger.info(f"Fetching information for topic: {topic}")
        topic_info_task = asyncio.create_task(
            fetch_topic_information(
                topic, 
                dev_mode=dev_mode, 
                bypass_cache=bypass_cache,
                progress_callback=progress_callback
            )
        )
        
        # Wait for topic info with timeout
        topic_info = await executor.execute_with_timeout(
            "topic_info",
            topic_info_task, 
            fallback_value=f"Emnet handler om {topic}. Ingen yderligere baggrundsinformation tilgængelig."
        )
        
        if progress_callback:
            await progress_callback(20)
        
        # Convert profile into strings for prompt construction
        principper = "\n".join([f"- {p}" for p in profile.kerneprincipper])
        nyhedskriterier = "\n".join([f"- {k}: {v}" for k, v in profile.nyhedsprioritering.items()])
        fokusomrader = "\n".join([f"- {f}" for f in profile.fokusOmrader])
        nogo_omrader = "\n".join([f"- {n}" for n in profile.noGoOmrader]) if profile.noGoOmrader else "Ingen"
        
        # 2. Start knowledge distillate task (can run concurrently with angles generation)
        if progress_stages and "GENERATING_KNOWLEDGE" in progress_stages:
            try:
                progress_stages["GENERATING_KNOWLEDGE"](
                    ProcessStage.GENERATING_KNOWLEDGE, 
                    f"Genererer videndistillat om '{topic}'..."
                )
            except (ImportError, KeyError) as e:
                logger.warning(f"Error in progress stage callback: {e}")
                
        knowledge_distillate_task = asyncio.create_task(
            generate_knowledge_distillate(
                topic_info=topic_info,
                topic=topic,
                bypass_cache=bypass_cache
            )
        )
        
        # 3. Start angles generation
        if progress_stages and "GENERATING_ANGLES" in progress_stages:
            try:
                progress_stages["GENERATING_ANGLES"](
                    ProcessStage.GENERATING_ANGLES, 
                    f"Genererer vinkler for '{topic}' med {profile.navn} profilen..."
                )
            except (ImportError, KeyError) as e:
                logger.warning(f"Error in progress stage callback: {e}")
                
        # Get angles (no need for timeout here since this is our main function)
        from prompt_engineering import construct_angle_prompt
        
        prompt = construct_angle_prompt(
            topic,
            topic_info,
            principper,
            profile.tone_og_stil,
            fokusomrader,
            nyhedskriterier,
            nogo_omrader
        )
        
        # 4. Start source suggestions task (parallel with angles)
        source_task = asyncio.create_task(
            fetch_source_suggestions(topic, bypass_cache=bypass_cache)
        )
        
        if progress_callback:
            await progress_callback(40)
            
        # Wait for angles
        angles = await generate_angles(topic, topic_info, profile, bypass_cache=bypass_cache)
        
        if not angles:
            error_msg = "No angles could be generated."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Generated {len(angles)} raw angles")
        
        if progress_callback:
            await progress_callback(60)
        
        # Filter and rank angles
        if progress_stages and "FILTERING_ANGLES" in progress_stages:
            try:
                progress_stages["FILTERING_ANGLES"](
                    ProcessStage.FILTERING_ANGLES, 
                    f"Filtrerer og rangerer de genererede vinkler..."
                )
            except (ImportError, KeyError) as e:
                logger.warning(f"Error in progress stage callback: {e}")
                
        ranked_angles = filter_and_rank_angles(angles, profile, config.app.num_angles)
        
        if not ranked_angles:
            error_msg = "No angles left after filtering."
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        logger.info(f"Ranked and filtered to {len(ranked_angles)} angles")
        
        # Add perplexity extract to each angle
        perplexity_extract = topic_info[:1000] + ("..." if len(topic_info) > 1000 else "")
        for angle in ranked_angles:
            if isinstance(angle, dict):
                angle['perplexityInfo'] = perplexity_extract
        
        # Progressive display of initial results
        if enable_progressive_display:
            # Here we could display preliminary results before all data is ready
            pass
            
        # Wait for source suggestions
        if progress_callback:
            await progress_callback(70)
            
        source_result = await executor.execute_with_timeout(
            "sources",
            source_task,
            fallback_value=None
        )
        
        if source_result:
            for angle in ranked_angles:
                if isinstance(angle, dict):
                    angle['kildeForslagInfo'] = source_result
        
        # Wait for knowledge distillate
        knowledge_distillate = await executor.execute_with_timeout(
            "knowledge_distillate",
            knowledge_distillate_task,
            fallback_value=None
        )
        
        if knowledge_distillate:
            for angle in ranked_angles:
                if isinstance(angle, dict):
                    angle['videnDistillat'] = knowledge_distillate
                    angle['harVidenDistillat'] = True
        
        # Get expert sources in parallel for top angles
        if progress_stages and "GENERATING_EXPERT_SOURCES" in progress_stages:
            try:
                progress_stages["GENERATING_EXPERT_SOURCES"](
                    ProcessStage.GENERATING_EXPERT_SOURCES,
                    f"Finder ekspertkilder til specifikke vinkler..."
                )
            except (ImportError, KeyError) as e:
                logger.warning(f"Error in progress stage callback: {e}")
                
        expert_tasks = []
        
        # Start tasks for up to 3 top angles
        for i, angle in enumerate(ranked_angles[:3]):
            if not isinstance(angle, dict):
                continue
                
            headline = angle.get('overskrift', f"Vinkel om {topic}")
            description = angle.get('beskrivelse', "")
            
            task = asyncio.create_task(
                generate_expert_source_suggestions(
                    topic=topic,
                    angle_headline=headline,
                    angle_description=description,
                    bypass_cache=bypass_cache
                )
            )
            expert_tasks.append((i, angle, task))
        
        if progress_callback:
            await progress_callback(80)
        
        # Wait for expert sources with aggressive timeouts
        for i, angle, task in expert_tasks:
            expert_result = await executor.execute_with_timeout(
                "expert_sources",
                task,
                # Only wait 10-15 seconds per expert source to keep overall latency low
                timeout=executor.stage_timeouts["expert_sources"],
                fallback_value={"experts": [], "institutions": [], "data_sources": []}
            )
            
            # Add expert sources to angle
            if expert_result:
                angle['ekspertKilder'] = expert_result
                angle['harEkspertKilder'] = True
            else:
                angle['ekspertKilder'] = {"experts": [], "institutions": [], "data_sources": []}
                angle['harEkspertKilder'] = False
        
        if progress_callback:
            await progress_callback(95)
            
        # Final result processing
        for angle in ranked_angles:
            if isinstance(angle, dict):
                # Ensure all angles have the metadata fields
                angle['harVidenDistillat'] = 'videnDistillat' in angle
                angle['harEkspertKilder'] = 'ekspertKilder' in angle
                
                # Ensure fields exist to prevent UI errors
                if 'videnDistillat' not in angle:
                    angle['videnDistillat'] = {
                        "key_statistics": [], 
                        "key_claims": [], 
                        "perspectives": [], 
                        "important_dates": []
                    }
                if 'ekspertKilder' not in angle:
                    angle['ekspertKilder'] = {
                        "experts": [], 
                        "institutions": [], 
                        "data_sources": []
                    }
        
        # Save to output file if specified
        if output_path:
            save_output(ranked_angles, profile, topic, format_type, output_path)
            
        # Record total performance time
        total_time = time.time() - total_start_time
        performance_data["total_time"] = total_time
        logger.info(f"Total generation time: {total_time:.2f} seconds")
        
        if progress_callback:
            await progress_callback(100)
            
        # Clean up and return results
        return ranked_angles, profile, topic_info
            
    except Exception as e:
        total_time = time.time() - total_start_time
        logger.error(f"Error in optimized process generation (after {total_time:.2f}s): {e}")
        
        # Even in case of error, try to record performance data
        performance_data["total_time"] = total_time
        
        # Re-raise the exception
        raise
        
    finally:
        # Always clean up resources
        if optimize_performance:
            try:
                await shutdown_api_client()
            except Exception as e:
                logger.warning(f"Error during API client shutdown: {e}")


def save_output(
    angles: List[Dict[str, Any]],
    profile: RedaktionelDNA,
    topic: str,
    format_type: str,
    output_path: str
) -> None:
    """
    Save the generated angles to a file with optimized encoding.
    
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


def get_performance_report() -> Dict[str, Any]:
    """
    Get a comprehensive performance report of the last generation run.
    
    Returns:
        Dict with performance metrics
    """
    # Get API-level metrics
    api_metrics = get_performance_metrics()
    cache_metrics = get_cache_stats()
    
    # Combine with our metrics
    report = {
        "process_times": {
            "total_execution_time": performance_data["total_time"],
            "topic_info_fetch_time": performance_data["topic_info_time"],
            "angles_generation_time": performance_data["angles_generation_time"],
            "knowledge_distillate_time": performance_data["knowledge_distillate_time"],
            "expert_sources_time": performance_data["expert_sources_time"],
        },
        "api_metrics": api_metrics,
        "cache_metrics": cache_metrics,
        "timestamp": datetime.now().isoformat()
    }
    
    return report


async def fetch_source_suggestions(topic: str, bypass_cache: bool = False) -> Optional[str]:
    """
    Fetch source suggestions for a topic using API client.
    
    This function is a wrapper to maintain backward compatibility.
    
    Args:
        topic: The topic to fetch sources for
        bypass_cache: Whether to bypass cache
        
    Returns:
        Source suggestions or None if failed
    """
    from api_clients_optimized import fetch_source_suggestions as api_fetch_source_suggestions
    
    return await api_fetch_source_suggestions(topic, bypass_cache=bypass_cache)