"""
Standard CLI mode for the Vinkeljernet application.

This module handles the standard (non-interactive) CLI mode for the application.
"""

import asyncio
import logging
import sys
from argparse import Namespace
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable

from rich import print as rprint
from rich.console import Console
from rich.table import Table

from vinkeljernet.core import process_generation_request
from vinkeljernet.ui_utils import (
    display_welcome_panel,
    display_profile_info,
    create_progress_spinner,
    display_angles_panels
)
from vinkeljernet.utils import clear_api_cache, reset_circuit_breakers, get_circuit_stats

# Configure logger
logger = logging.getLogger("vinkeljernet.standard_mode")

# Create console
console = Console()


async def run_standard_mode(args: Namespace) -> None:
    """
    Run the application in standard CLI mode.
    
    Args:
        args: Command-line arguments
    """
    # Print start message with nice formatting
    display_welcome_panel()
    
    # Display received arguments
    console.print("[bold blue]Running with the following parameters:[/bold blue]")
    console.print(f"  [green]News topic:[/green] {args.emne}")
    console.print(f"  [green]Profile file:[/green] {args.profil}")
    console.print(f"  [green]Output file:[/green] {args.output if args.output else 'None (display in terminal)'}")
    
    if args.dev_mode:
        console.print("  [yellow]⚠️ Development mode enabled (insecure SSL)[/yellow]")
    
    # Handle cache and circuit breaker options
    if args.clear_cache:
        num_cleared = clear_api_cache()
        console.print(f"[yellow]Cache cleared: {num_cleared} files deleted[/yellow]")
    
    if args.reset_circuits:
        reset_circuit_breakers()
        console.print("[yellow]All circuit breakers reset[/yellow]")
    
    if args.show_circuits:
        display_circuit_stats()
    
    # Load and validate profile with progress spinner
    with create_progress_spinner("Loading editorial profile...") as progress:
        task = progress.add_task("Loading", total=None)
        
        # Create a placeholder for background information update
        async def progress_callback(percent: int):
            progress.update(task, completed=percent)
    
    # Generate angles and show results
    try:
        ranked_angles, profile, topic_info = await process_generation_request(
            topic=args.emne,
            profile_path=args.profil,
            format_type=args.format,
            output_path=args.output,
            dev_mode=args.dev_mode,
            bypass_cache=args.bypass_cache,
            debug=args.debug,
            progress_callback=progress_callback
        )
        
        # Display profile summary
        display_profile_info(profile)
        
        # Present results with nice formatting
        display_angles_panels(ranked_angles)
        
        if args.output:
            console.print(f"\n[green]✓[/green] Results saved to {args.output} ({args.format} format)")
            
    except FileNotFoundError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        console.print("[yellow]Tip:[/yellow] Check the file path and try again")
        sys.exit(1)
    except ValueError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


def display_circuit_stats() -> None:
    """Display statistics for circuit breakers."""
    stats = get_circuit_stats()
    
    circuit_table = Table(show_header=True, header_style="bold blue")
    circuit_table.add_column("API", style="dim")
    circuit_table.add_column("State")
    circuit_table.add_column("Successful calls")
    circuit_table.add_column("Failed calls")
    circuit_table.add_column("Consecutive failures")
    circuit_table.add_column("Retry count")
    circuit_table.add_column("Last failure")
    
    for name, data in stats.items():
        state_style = "green" if data["state"] == "closed" else "red" if data["state"] == "open" else "yellow"
        circuit_table.add_row(
            name,
            f"[{state_style}]{data['state']}[/{state_style}]",
            str(data["success_count"]),
            str(data["failure_count"]),
            str(data["consecutive_failures"]),
            str(data["total_retries"]),
            data["last_failure"] or "None"
        )
    
    console.print("\n[bold blue]Circuit Breaker Status:[/bold blue]")
    console.print(circuit_table)