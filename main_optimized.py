"""
Optimized main entry point for the Vinkeljernet application.

This module provides a high-performance implementation of the main Vinkeljernet
application with parallel processing, caching, and resilience features.
"""
import argparse
import asyncio
import json
import sys
import yaml
import logging
import os
import glob
import time
from argparse import Namespace
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime

from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.live import Live
from rich.layout import Layout

# Import prompt_toolkit for interactive CLI
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter, FuzzyCompleter
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.shortcuts import clear, message_dialog, button_dialog
from prompt_toolkit.application import run_in_terminal

# Import models
from models import RedaktionelDNA, ExpertSource, KnowledgeDistillate

# Configure root logger
logging.basicConfig(
    filename='vinkeljernet.log',
    filemode='w',  # 'w' to overwrite, 'a' to append
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Module logger
logger = logging.getLogger("vinkeljernet.optimized_main")
logger.info("Starting optimized Vinkeljernet")

# Import configuration components
try:
    from config import OPENAI_API_KEY, PERPLEXITY_API_KEY, ANTHROPIC_API_KEY
    # Verify API keys are present with more useful error messages
    missing_keys = []
    if not ANTHROPIC_API_KEY:
        missing_keys.append("ANTHROPIC_API_KEY")
    if not PERPLEXITY_API_KEY:
        missing_keys.append("PERPLEXITY_API_KEY")
        
    if missing_keys:
        rprint("[bold yellow]Advarsel:[/bold yellow] Følgende API-nøgler mangler i konfigurationen:")
        for key in missing_keys:
            rprint(f"  - {key}")
        rprint("[yellow]Nogle funktioner vil måske ikke virke korrekt.[/yellow]")
except Exception as e:
    rprint(f"[bold red]Fejl ved indlæsning af API nøgler:[/bold red] {e}")
    rprint("[yellow]Opret en .env fil med API nøglerne eller konfigurer dem i config.py[/yellow]")
    sys.exit(1)

# Import core components with optimized path
from config_loader import load_and_validate_profile
from optimized_core import (
    optimized_process_generation,
    get_performance_report,
    ParallelStageExecutor
)
from angle_processor import filter_and_rank_angles
from cache_manager import optimize_cache, get_cache_stats

# Import modular components for CLI UI
from vinkeljernet.ui_utils import (
    display_profile_info,
    display_angles_panels,
    display_angles_table,
    display_knowledge_distillate,
    display_expert_sources,
    ProcessStage
)

# Initialize console
console = Console()

def parse_arguments() -> Namespace:
    """
    Parse command-line arguments with enhanced options for performance optimization.
    
    Returns:
        Namespace with parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Vinkeljernet - Optimeret værktøj til generering af journalistiske vinkler"
    )
    
    # Main arguments
    parser.add_argument(
        'topic', 
        nargs='?',
        help='Nyhedsemne at generere vinkler for'
    )
    
    # Profile selection options
    profile_group = parser.add_argument_group('Profil')
    profile_group.add_argument(
        '--profile', '-p', 
        help='Sti til en profil-YAML-fil der definerer redaktionel DNA'
    )
    
    # Output options
    output_group = parser.add_argument_group('Output')
    output_group.add_argument(
        '--format', '-f',
        choices=['json', 'markdown', 'md', 'html', 'text', 'txt'],
        default='text',
        help='Output format (default: text)'
    )
    output_group.add_argument(
        '--output', '-o',
        help='Sti til output-fil'
    )
    
    # Mode options
    mode_group = parser.add_argument_group('Kørselstilstand')
    mode_group.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Kør i interaktiv tilstand'
    )
    mode_group.add_argument(
        '--list-profiles', '-l',
        action='store_true',
        help='List alle tilgængelige profiler'
    )
    mode_group.add_argument(
        '--detailed', '-d',
        action='store_true',
        help='Vis detaljeret output'
    )
    
    # Performance options (new)
    perf_group = parser.add_argument_group('Ydeevne')
    perf_group.add_argument(
        '--optimize',
        action='store_true',
        help='Anvend ydeevne-optimeringer (anbefales)'
    )
    perf_group.add_argument(
        '--timeout-multiplier',
        type=float,
        default=1.0,
        help='Multiplier for timeouts (1.0 = normal, 2.0 = dobbelt tid)'
    )
    perf_group.add_argument(
        '--disable-progress',
        action='store_true',
        help='Deaktiver progressiv visning af resultater'
    )
    perf_group.add_argument(
        '--stats',
        action='store_true',
        help='Vis ydeevne-statistik efter kørsel'
    )
    perf_group.add_argument(
        '--max-concurrent',
        type=int,
        default=5,
        help='Maksimalt antal samtidige API-kald'
    )
    perf_group.add_argument(
        '--optimize-cache',
        action='store_true',
        help='Optimer cache før kørsel'
    )
    
    # Advanced options
    adv_group = parser.add_argument_group('Avanceret')
    adv_group.add_argument(
        '--dev-mode',
        action='store_true',
        help='Kør i udvikler-tilstand'
    )
    adv_group.add_argument(
        '--bypass-cache',
        action='store_true',
        help='Ignorer cache og hent friske data'
    )
    adv_group.add_argument(
        '--debug',
        action='store_true',
        help='Aktiver debug-udskrift'
    )
    adv_group.add_argument(
        '--skip-expert-sources',
        action='store_true',
        help='Skip generation af ekspertkilder (hurtigere)'
    )
    adv_group.add_argument(
        '--skip-knowledge-distillate',
        action='store_true',
        help='Skip generation af videndistillat (hurtigere)'
    )
    
    return parser.parse_args()


def list_available_profiles(profile_dir: str = "config") -> None:
    """
    List all available profiles with enhanced descriptions.
    
    Args:
        profile_dir: Directory containing profile YAML files
    """
    profile_files = glob.glob(f"{profile_dir}/*_profil.yaml")
    
    if not profile_files:
        rprint("[bold yellow]Ingen profiler fundet i katalog:[/bold yellow]", profile_dir)
        return
        
    table = Table(title="Tilgængelige Profiler")
    table.add_column("Profil", style="cyan")
    table.add_column("Beskrivelse", style="green")
    table.add_column("Fokusområder", style="yellow")
    
    for profile_path in sorted(profile_files):
        try:
            profile = load_and_validate_profile(Path(profile_path))
            
            # Extract file name without extension for cleaner display
            file_name = os.path.basename(profile_path).replace("_profil.yaml", "")
            description = getattr(profile, 'beskrivelse', 'Ingen beskrivelse')
            
            # Format focus areas as comma-separated list
            focus_areas = getattr(profile, 'fokusOmrader', [])
            focus_str = ", ".join(focus_areas[:3])
            if len(focus_areas) > 3:
                focus_str += f" (+{len(focus_areas)-3} mere)"
                
            table.add_row(file_name, description, focus_str)
        except Exception as e:
            table.add_row(
                os.path.basename(profile_path),
                f"[red]Fejl ved indlæsning: {e}[/red]",
                ""
            )
    
    console.print(table)


def get_default_profile_path(profile_dir: str = "config") -> Optional[str]:
    """
    Get the default profile path based on naming convention.
    
    Args:
        profile_dir: Directory containing profile YAML files
        
    Returns:
        Default profile path or None if not found
    """
    # First try to find a 'default' profile
    default_patterns = [
        f"{profile_dir}/default_profil.yaml",
        f"{profile_dir}/standard_profil.yaml"
    ]
    
    for pattern in default_patterns:
        if os.path.exists(pattern):
            return pattern
    
    # If no default profile, return the first profile found
    profile_files = glob.glob(f"{profile_dir}/*_profil.yaml")
    if profile_files:
        return profile_files[0]
        
    return None


class OptimizedProgressTracker:
    """Enhanced progress tracker with responsive UI and more detailed stages."""
    
    def __init__(self, topic: str, detailed: bool = False):
        """
        Initialize the progress tracker.
        
        Args:
            topic: The topic being processed
            detailed: Whether to show detailed progress
        """
        self.topic = topic
        self.detailed = detailed
        self.current_stage = None
        self.stage_weights = {
            ProcessStage.FETCHING_INFO: 20,
            ProcessStage.GENERATING_KNOWLEDGE: 15,
            ProcessStage.GENERATING_ANGLES: 30,
            ProcessStage.FILTERING_ANGLES: 5,
            ProcessStage.GENERATING_SOURCES: 10,
            ProcessStage.GENERATING_EXPERT_SOURCES: 20,
        }
        self.stage_progress = {stage: 0 for stage in ProcessStage}
        self.stage_start_times = {}
        self.stage_messages = {
            ProcessStage.FETCHING_INFO: f"Indhenter baggrundsinformation om '{topic}'...",
            ProcessStage.GENERATING_KNOWLEDGE: "Genererer videndistillat...",
            ProcessStage.GENERATING_ANGLES: "Genererer vinkler...",
            ProcessStage.FILTERING_ANGLES: "Filtrerer og prioriterer vinkler...",
            ProcessStage.GENERATING_SOURCES: "Finder relevante kilder...",
            ProcessStage.GENERATING_EXPERT_SOURCES: "Finder ekspertkilder til vinkler...",
        }
        self.additional_message = ""
        
        # Initialize progress display
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        )
        self.task_id = self.progress.add_task("Initialiserer...", total=100)
    
    async def progress_callback(self, percent: int) -> None:
        """
        Update progress within the current stage.
        
        Args:
            percent: Progress percentage within the current stage (0-100)
        """
        if self.current_stage:
            # Update stage-specific progress
            self.stage_progress[self.current_stage] = percent
            
            # Calculate overall progress based on stage weights
            overall_progress = self._calculate_overall_progress()
            
            # Update progress bar
            self.progress.update(self.task_id, completed=overall_progress)
    
    def stage_callback(self, stage: ProcessStage, message: str) -> None:
        """
        Called when the processing stage changes.
        
        Args:
            stage: The new processing stage
            message: Message describing the current activity
        """
        # Record stage start time
        self.stage_start_times[stage] = time.time()
        
        # Update current stage
        self.current_stage = stage
        
        # Reset progress for this stage
        self.stage_progress[stage] = 0
        
        # Update task description
        self.progress.update(
            self.task_id, 
            description=f"[cyan]{message}[/cyan]"
        )
        
        # If detailed, print a message about the stage change
        if self.detailed:
            stage_name = stage.name if hasattr(stage, 'name') else str(stage)
            self.additional_message = f"Stage: [bold]{stage_name}[/bold] - {message}"
    
    def _calculate_overall_progress(self) -> float:
        """
        Calculate overall progress as weighted average of stage progress.
        
        Returns:
            Overall progress (0-100)
        """
        total_weight = sum(self.stage_weights.values())
        weighted_progress = 0
        
        for stage, weight in self.stage_weights.items():
            weighted_progress += (self.stage_progress.get(stage, 0) * weight / 100)
            
        return (weighted_progress / total_weight) * 100
    
    def __enter__(self):
        """Start the progress display."""
        self.progress.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the progress display."""
        self.progress.stop()
        
        # If detailed, print timing information
        if self.detailed and self.stage_start_times:
            console.print("\n[bold]Stage timing:[/bold]")
            now = time.time()
            
            timing_table = Table(show_header=True)
            timing_table.add_column("Stage", style="cyan")
            timing_table.add_column("Duration", style="green")
            
            for stage, start_time in self.stage_start_times.items():
                # Calculate end time (either next stage start time or now)
                end_time = now
                for next_stage, next_start in self.stage_start_times.items():
                    if next_start > start_time and next_start < end_time:
                        end_time = next_start
                
                # Calculate duration
                duration = end_time - start_time
                stage_name = stage.name if hasattr(stage, 'name') else str(stage)
                
                timing_table.add_row(
                    stage_name,
                    f"{duration:.2f} seconds"
                )
            
            console.print(timing_table)


async def generate_angles_async(args: Namespace) -> None:
    """
    Generate angles for a given topic with optimized async processing.
    
    Args:
        args: Command-line arguments
    """
    # Validate and prepare parameters
    topic = args.topic
    if not topic:
        console.print("[bold red]Fejl:[/bold red] Intet emne angivet.")
        return
        
    # Determine profile path
    profile_path = args.profile
    if not profile_path:
        profile_path = get_default_profile_path()
        if not profile_path:
            console.print("[bold red]Fejl:[/bold red] Ingen profil angivet, og ingen standardprofil fundet.")
            return
    
    # Validate profile path
    if not os.path.exists(profile_path):
        console.print(f"[bold red]Fejl:[/bold red] Profil '{profile_path}' findes ikke.")
        return
        
    # Prepare format type
    format_type = args.format
    if format_type == 'md':
        format_type = 'markdown'
    elif format_type in ['txt', 'text']:
        format_type = 'text'
        
    # Optimize cache if requested
    if args.optimize_cache:
        console.print("[cyan]Optimerer cache...[/cyan]")
        await optimize_cache()
    
    # Initialize progress tracking
    with OptimizedProgressTracker(topic, detailed=args.detailed) as progress_tracker:
        try:
            # Process angle generation with optimized core
            features = {
                "include_expert_sources": not args.skip_expert_sources,
                "include_knowledge_distillate": not args.skip_knowledge_distillate
            }
            
            # Define progress callback functions
            progress_cb = progress_tracker.progress_callback
            stage_callbacks = {
                "FETCHING_INFO": progress_tracker.stage_callback,
                "GENERATING_KNOWLEDGE": progress_tracker.stage_callback,
                "GENERATING_ANGLES": progress_tracker.stage_callback,
                "FILTERING_ANGLES": progress_tracker.stage_callback,
                "GENERATING_SOURCES": progress_tracker.stage_callback,
                "GENERATING_EXPERT_SOURCES": progress_tracker.stage_callback
            }
            
            # Generate angles with optimized core
            start_time = time.time()
            
            angles, profile, topic_info = await optimized_process_generation(
                topic=topic,
                profile_path=profile_path,
                format_type=format_type,
                output_path=args.output,
                dev_mode=args.dev_mode,
                bypass_cache=args.bypass_cache,
                debug=args.debug,
                progress_callback=progress_cb,
                progress_stages=stage_callbacks,
                optimize_performance=args.optimize,
                timeout_multiplier=args.timeout_multiplier,
                enable_progressive_display=not args.disable_progress
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Tæl succesfulde vinkler og fejl
            successful_angles = [angle for angle in angles if not angle.get('error')]
            error_angles_count = len(angles) - len(successful_angles)

            # Display results based on format
            if format_type == 'text':
                display_angles_panels(angles, verbose=args.detailed, detailed_sources=args.detailed)
                if successful_angles:
                    console.print(f"\n[green]✓[/green] Genereret {len(successful_angles)} vinkler succesfuldt på [bold]{processing_time:.2f}[/bold] sekunder.")
                else:
                    console.print(f"\n[bold red]Ingen vinkler blev genereret succesfuldt.[/bold red]")
                if error_angles_count > 0:
                    console.print(f"[yellow]![/yellow] {error_angles_count} vinkel(er) kunne ikke genereres pga. fejl.")
            elif format_type == 'markdown':
                # Display as markdown
                from formatters import format_angles_markdown
                markdown_output = format_angles_markdown(angles, profile_name=profile.navn, topic=topic)
                console.print(Markdown(markdown_output))
            else:
                # Just show summary for other formats
                if successful_angles:
                    console.print(f"[green]✓[/green] Genereret {len(successful_angles)} vinkler i {format_type} format på {processing_time:.2f} sekunder.")
                else:
                    console.print(f"[bold red]Ingen vinkler blev genereret succesfuldt i {format_type} format.[/bold red]")
                if error_angles_count > 0:
                    console.print(f"[yellow]![/yellow] {error_angles_count} vinkel(er) i {format_type} format kunne ikke genereres pga. fejl.")
                
                if args.output:
                    console.print(f"[green]Resultat gemt i: [bold]{args.output}[/bold][/green]")
            
            # Show performance stats if requested
            if args.stats:
                performance_report = get_performance_report()
                
                stats_table = Table(title="Ydeevnestatistik")
                stats_table.add_column("Metrik", style="cyan")
                stats_table.add_column("Værdi", style="green")
                
                # Add process times
                for key, value in performance_report["process_times"].items():
                    if isinstance(value, (int, float)):
                        formatted_value = f"{value:.2f} sekunder" if key != "cache_hits" else str(value)
                        stats_table.add_row(key.replace("_", " ").title(), formatted_value)
                
                # Add API stats if available
                if "api" in performance_report["api_metrics"]:
                    api_stats = performance_report["api_metrics"]["api"]
                    stats_table.add_row("", "")  # Blank row for separation
                    stats_table.add_row("[bold]API Statistik[/bold]", "")
                    stats_table.add_row("Succesrate", api_stats.get("success_rate", "N/A"))
                    stats_table.add_row("Cache Hit Rate", api_stats.get("cache_hit_rate", "N/A"))
                    stats_table.add_row("Gns. Svartid", f"{api_stats.get('average_latency_ms', 0)/1000:.2f} sekunder")
                
                console.print(stats_table)
                
        except Exception as e:
            console.print(f"[bold red]Fejl under generering af vinkler:[/bold red] {e}")
            if args.debug:
                console.print_exception()


def main():
    """Main entry point for the optimized application."""
    args = parse_arguments()
    
    # Handle list profiles request
    if args.list_profiles:
        list_available_profiles()
        return
        
    # Handle interactive mode
    if args.interactive:
        console.print("[yellow]Interaktiv tilstand er ikke understøttet i den optimerede version endnu.[/yellow]")
        return
        
    # Handle regular mode
    if args.topic:
        try:
            # Run the async generation function
            asyncio.run(generate_angles_async(args))
        except KeyboardInterrupt:
            console.print("\n[yellow]Afbrudt af bruger.[/yellow]")
        except Exception as e:
            console.print(f"\n[bold red]En uventet fejl opstod:[/bold red] {e}")
            if args.debug:
                console.print_exception()
    else:
        console.print("[bold red]Fejl:[/bold red] Intet emne angivet. Brug --help for at se hjælp.")


if __name__ == "__main__":
    main()