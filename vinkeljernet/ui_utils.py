"""
UI utilities for the Vinkeljernet application.

This module provides common UI utilities for the Vinkeljernet application,
including progress bars, tables, and panels.
"""

import logging
from typing import Dict, List, Any, Optional, Callable

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.markdown import Markdown
from rich.syntax import Syntax

from models import RedaktionelDNA

# Configure logger
logger = logging.getLogger("vinkeljernet.ui_utils")

# Create console for rich output
console = Console()


def create_progress_spinner(
    message: str,
    transient: bool = True
) -> Progress:
    """
    Create a progress spinner with a message.
    
    Args:
        message: The message to display
        transient: Whether the spinner should be transient
        
    Returns:
        Progress: The created progress
    """
    return Progress(
        SpinnerColumn(),
        TextColumn(f"[bold blue]{message}"),
        console=console,
        transient=transient
    )


def display_profile_info(profile: RedaktionelDNA) -> None:
    """
    Display information about a profile in a table.
    
    Args:
        profile: The editorial DNA profile
    """
    console.print("[green]✓[/green] Profile loaded and validated")
    
    table = Table(show_header=True, header_style="bold blue")
    table.add_column("Profile element", style="dim")
    table.add_column("Value")
    
    # Display core principles
    principles = "\n".join([f"- {p}" for p in profile.kerneprincipper])
    table.add_row("Core principles", principles)
    table.add_row("Tone and style", profile.tone_og_stil)
    table.add_row("Number of news criteria", str(len(profile.nyhedsprioritering)))
    table.add_row("Number of focus areas", str(len(profile.fokusOmrader)))
    table.add_row("Number of no-go areas", str(len(profile.noGoOmrader)))

    console.print(table)


def display_angles_table(angles: List[Dict[str, Any]]) -> None:
    """
    Display a table of generated angles.
    
    Args:
        angles: List of angle dictionaries
    """
    table = Table(show_header=True, header_style="bold blue")
    table.add_column("#", style="dim")
    table.add_column("Headline", style="bold")
    table.add_column("Description")
    table.add_column("News criteria")
    
    for i, angle in enumerate(angles, 1):
        headline = angle.get('overskrift', 'No headline')
        description = angle.get('beskrivelse', 'No description')
        criteria = ", ".join(angle.get('nyhedskriterier', []))
        
        table.add_row(
            str(i),
            headline,
            description[:80] + ("..." if len(description) > 80 else ""),
            criteria
        )
    
    console.print(table)


def display_angles_panels(angles: List[Dict[str, Any]]) -> None:
    """
    Display detailed panels for each angle.
    
    Args:
        angles: List of angle dictionaries
    """
    console.print("\n[bold blue]🎯 Recommended angles:[/bold blue]")
    
    for i, angle in enumerate(angles, 1):
        # Handle potentially missing keys with .get()
        headline = angle.get('overskrift', 'No headline')
        description = angle.get('beskrivelse', 'No description')
        rationale = angle.get('begrundelse', 'No rationale')
        criteria = angle.get('nyhedskriterier', [])
        questions = angle.get('startSpørgsmål', [])
        score = angle.get('kriterieScore', 'N/A')
        
        # Create panel for each angle
        panel_content = [
            f"[bold white]{headline}[/bold white]",
            f"\n{description}",
            f"\n[dim blue]Rationale:[/dim blue] [dim]{rationale}[/dim]",
            f"\n[dim blue]News criteria:[/dim blue] [dim]{', '.join(criteria)}[/dim]"
        ]
        
        # Add start questions if available
        if questions:
            panel_content.append(f"\n[dim blue]Initial questions:[/dim blue]")
            for q in questions:
                panel_content.append(f"[dim]• {q}[/dim]")
        
        if score != 'N/A':
            panel_content.append(f"\n[dim blue]Score:[/dim blue] [dim]{score}[/dim]")
        
        console.print(Panel(
            "\n".join(panel_content),
            title=f"[bold green]Angle {i}[/bold green]",
            border_style="green",
            expand=False
        ))


def display_welcome_panel(title: str = "Vinkeljernet") -> None:
    """
    Display a welcome panel.
    
    Args:
        title: The title for the welcome panel
    """
    console.print(Panel.fit(
        f"[bold blue]{title}[/bold blue] - Journalistic angle generator",
        border_style="blue"
    ))


def display_error(message: str, title: str = "Error") -> None:
    """
    Display an error message.
    
    Args:
        message: The error message
        title: The error title
    """
    console.print(Panel(
        f"[bold red]{message}[/bold red]",
        title=f"[bold red]{title}[/bold red]",
        border_style="red"
    ))


def display_warning(message: str, title: str = "Warning") -> None:
    """
    Display a warning message.
    
    Args:
        message: The warning message
        title: The warning title
    """
    console.print(Panel(
        f"[bold yellow]{message}[/bold yellow]",
        title=f"[bold yellow]{title}[/bold yellow]",
        border_style="yellow"
    ))


def display_success(message: str, title: str = "Success") -> None:
    """
    Display a success message.
    
    Args:
        message: The success message
        title: The success title
    """
    console.print(Panel(
        f"[bold green]{message}[/bold green]",
        title=f"[bold green]{title}[/bold green]",
        border_style="green"
    ))


def display_info(message: str, title: str = "Information") -> None:
    """
    Display an information message.
    
    Args:
        message: The information message
        title: The information title
    """
    console.print(Panel(
        f"[bold blue]{message}[/bold blue]",
        title=f"[bold blue]{title}[/bold blue]",
        border_style="blue"
    ))


def display_help_text(help_text: str) -> None:
    """
    Display help text as markdown.
    
    Args:
        help_text: The help text in markdown format
    """
    md = Markdown(help_text)
    console.print(md)