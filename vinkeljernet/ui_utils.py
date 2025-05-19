"""
UI utilities for the Vinkeljernet application.

This module provides common UI utilities for the Vinkeljernet application,
including progress bars, tables, and panels.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Callable, Tuple
from enum import Enum
from datetime import datetime, timedelta

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.live import Live
from rich.layout import Layout
from rich import box

from models import RedaktionelDNA

# Configure logger
logger = logging.getLogger("vinkeljernet.ui_utils")

# Create console for rich output
console = Console()


class ProcessStage(Enum):
    """Enum for the various processing stages in the application."""
    INITIALIZING = "Initialiserer"
    LOADING_PROFILE = "Indlæser profil"
    FETCHING_INFO = "Henter baggrundsinformation"
    GENERATING_KNOWLEDGE = "Genererer videndistillat"
    GENERATING_ANGLES = "Genererer vinkler"
    FILTERING_ANGLES = "Filtrerer og rangerer vinkler"
    GENERATING_SOURCES = "Finder eksperter og kilder"
    GENERATING_EXPERT_SOURCES = "Finder ekspertkilder til vinkler"
    FINALIZING = "Færdiggør"
    COMPLETE = "Færdig"


class ProgressTracker:
    """
    Track progress across multiple stages of the angle generation process.
    
    This class provides a way to track and display progress across the different
    stages of the angle generation pipeline, with estimates of time remaining.
    """
    def __init__(self, total_stages: int = 8):
        self.stages = {
            ProcessStage.INITIALIZING: {"weight": 5, "progress": 0, "start_time": None, "end_time": None, "message": ""},
            ProcessStage.LOADING_PROFILE: {"weight": 5, "progress": 0, "start_time": None, "end_time": None, "message": ""},
            ProcessStage.FETCHING_INFO: {"weight": 15, "progress": 0, "start_time": None, "end_time": None, "message": ""},
            ProcessStage.GENERATING_KNOWLEDGE: {"weight": 15, "progress": 0, "start_time": None, "end_time": None, "message": ""},
            ProcessStage.GENERATING_ANGLES: {"weight": 20, "progress": 0, "start_time": None, "end_time": None, "message": ""},
            ProcessStage.FILTERING_ANGLES: {"weight": 10, "progress": 0, "start_time": None, "end_time": None, "message": ""},
            ProcessStage.GENERATING_SOURCES: {"weight": 10, "progress": 0, "start_time": None, "end_time": None, "message": ""},
            ProcessStage.GENERATING_EXPERT_SOURCES: {"weight": 15, "progress": 0, "start_time": None, "end_time": None, "message": ""},
            ProcessStage.FINALIZING: {"weight": 5, "progress": 0, "start_time": None, "end_time": None, "message": ""},
            ProcessStage.COMPLETE: {"weight": 0, "progress": 100, "start_time": None, "end_time": None, "message": "Færdig"}
        }
        self.current_stage = ProcessStage.INITIALIZING
        self.start_time = time.time()
        self.completed = False
        self._update_listeners = []
        
        # Initialize the first stage
        self.stages[self.current_stage]["start_time"] = self.start_time
        
    def set_stage(self, stage: ProcessStage, message: str = ""):
        """Set the current processing stage."""
        # Mark the previous stage as complete if we're moving to a new stage
        if stage != self.current_stage:
            # End the previous stage
            now = time.time()
            self.stages[self.current_stage]["progress"] = 100
            self.stages[self.current_stage]["end_time"] = now
            
            # Start the new stage
            self.current_stage = stage
            self.stages[stage]["start_time"] = now
            self.stages[stage]["message"] = message
            
            # If this is the final stage, mark it as complete
            if stage == ProcessStage.COMPLETE:
                self.completed = True
                self.stages[stage]["progress"] = 100
                self.stages[stage]["end_time"] = now
                
            self._notify_listeners()
                
    def update_progress(self, progress: int, message: str = ""):
        """Update the progress of the current stage."""
        if 0 <= progress <= 100:
            self.stages[self.current_stage]["progress"] = progress
            if message:
                self.stages[self.current_stage]["message"] = message
            self._notify_listeners()
            
    def get_overall_progress(self) -> int:
        """Calculate overall progress across all stages."""
        total_weight = sum(stage["weight"] for stage in self.stages.values())
        weighted_progress = sum(
            (stage["progress"] / 100) * stage["weight"] 
            for stage in self.stages.values()
        )
        return int((weighted_progress / total_weight) * 100)
    
    def get_estimated_time_remaining(self) -> Tuple[float, str]:
        """
        Estimate time remaining based on progress so far.
        
        Returns:
            Tuple containing:
            - Estimated seconds remaining (float)
            - Formatted time string (str)
        """
        if self.completed:
            return 0, "0s"
            
        # Calculate elapsed time
        elapsed = time.time() - self.start_time
        
        # Calculate overall progress
        progress = self.get_overall_progress()
        
        # Avoid division by zero
        if progress <= 0:
            return 999, "Beregner..."
            
        # Estimate total time based on progress so far
        estimated_total = elapsed * 100 / progress
        
        # Calculate remaining time
        remaining_seconds = estimated_total - elapsed
        
        # Format the remaining time
        if remaining_seconds < 60:
            formatted = f"{int(remaining_seconds)}s"
        elif remaining_seconds < 3600:
            minutes = int(remaining_seconds / 60)
            seconds = int(remaining_seconds % 60)
            formatted = f"{minutes}m {seconds}s"
        else:
            hours = int(remaining_seconds / 3600)
            minutes = int((remaining_seconds % 3600) / 60)
            formatted = f"{hours}h {minutes}m"
            
        return remaining_seconds, formatted
    
    def get_current_message(self) -> str:
        """Get the message for the current stage."""
        return self.stages[self.current_stage].get("message", "")
    
    def get_stage_info(self) -> Dict:
        """Get information about all stages for display."""
        stage_info = {}
        for stage, info in self.stages.items():
            # Skip the COMPLETE stage for display
            if stage == ProcessStage.COMPLETE:
                continue
                
            # Calculate elapsed time for this stage if it has started
            elapsed = ""
            if info["start_time"]:
                if info["end_time"]:
                    # Stage is complete
                    elapsed = format_seconds(info["end_time"] - info["start_time"])
                else:
                    # Stage is in progress
                    elapsed = format_seconds(time.time() - info["start_time"])
                
            stage_info[stage] = {
                "name": stage.value,
                "progress": info["progress"],
                "elapsed": elapsed,
                "status": self._get_stage_status(stage),
                "message": info.get("message", "")
            }
        return stage_info
        
    def _get_stage_status(self, stage: ProcessStage) -> str:
        """Get the status of a stage (waiting, in progress, complete)."""
        if stage == self.current_stage:
            return "in_progress"
        elif self.stages[stage]["end_time"]:
            return "complete"
        else:
            return "waiting"
            
    def add_update_listener(self, callback: Callable):
        """Add a callback to be notified when progress updates."""
        self._update_listeners.append(callback)
        
    def _notify_listeners(self):
        """Notify all registered listeners of a progress update."""
        for callback in self._update_listeners:
            try:
                callback(self)
            except Exception as e:
                logger.error(f"Error in progress listener callback: {e}")


def format_seconds(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        seconds = int(seconds % 60)
        return f"{minutes}m {seconds}s"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h {minutes}m"


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


def create_detailed_progress_display() -> Tuple[Live, ProgressTracker]:
    """
    Create a detailed progress display with live updating.
    
    Returns:
        Tuple containing:
        - Live display object
        - ProgressTracker instance
    """
    # Create the tracker
    tracker = ProgressTracker()
    
    # Create the layout
    layout = Layout()
    
    # Split into main sections
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=3)
    )
    
    # Setup the stages display in the main area
    layout["main"].split_row(
        Layout(name="stages", ratio=3),
        Layout(name="details", ratio=2)
    )
    
    # Create the live display
    live = Live(layout, refresh_per_second=4, console=console)
    
    # Define the update function
    def update_display(tracker: ProgressTracker):
        # Update the header - overall progress
        progress_bar = f"[{'#' * (tracker.get_overall_progress() // 2):<50}]"
        overall_percent = tracker.get_overall_progress()
        _, remaining = tracker.get_estimated_time_remaining()
        
        layout["header"].update(
            Panel(
                f"Samlet fremskridt: {progress_bar} {overall_percent}% (Estimeret tid tilbage: {remaining})",
                title="[bold blue]Vinkeljernet[/bold blue]",
                border_style="blue"
            )
        )
        
        # Update stages table
        stages_table = Table(show_header=True, header_style="bold blue", box=box.ROUNDED)
        stages_table.add_column("Trin")
        stages_table.add_column("Status")
        stages_table.add_column("Fremskridt")
        stages_table.add_column("Tid")
        
        stage_info = tracker.get_stage_info()
        for stage, info in stage_info.items():
            name = info["name"]
            
            # Format status with appropriate color
            if info["status"] == "complete":
                status = "[green]✓ Færdig[/green]"
                style = "dim"
            elif info["status"] == "in_progress":
                status = "[bold yellow]⟳ I gang[/bold yellow]"
                style = "bold"
            else:
                status = "[dim]○ Venter[/dim]"
                style = "dim"
                
            # Format progress
            if info["progress"] == 100:
                progress = "[green]100%[/green]"
            elif info["progress"] > 0:
                progress = f"[yellow]{info['progress']}%[/yellow]"
            else:
                progress = "[dim]0%[/dim]"
                
            # Format time
            time_str = info["elapsed"] if info["elapsed"] else "-"
                
            stages_table.add_row(
                f"[{style}]{name}[/{style}]", 
                status,
                progress,
                time_str
            )
            
        layout["stages"].update(
            Panel(
                stages_table,
                title="[bold blue]Procestrin[/bold blue]",
                border_style="blue"
            )
        )
        
        # Update details - current stage details
        current_message = tracker.get_current_message()
        current_stage_name = tracker.current_stage.value
        current_progress = tracker.stages[tracker.current_stage]["progress"]
        
        # Create a mini progress bar for the current stage
        mini_bar = f"[{'#' * (current_progress // 2):<50}]"
        
        details_content = [
            f"[bold]Nuværende trin:[/bold] {current_stage_name}",
            f"[bold]Fremskridt:[/bold] {mini_bar} {current_progress}%",
            "",
            f"[dim]{current_message}[/dim]"
        ]
        
        layout["details"].update(
            Panel(
                "\n".join(details_content),
                title="[bold blue]Detaljer[/bold blue]",
                border_style="blue"
            )
        )
        
        # Update footer
        elapsed = format_seconds(time.time() - tracker.start_time)
        layout["footer"].update(
            Panel(
                f"Samlet tid: {elapsed} | Tryk Ctrl+C for at afbryde",
                border_style="blue"
            )
        )
    
    # Add the update function as a listener
    tracker.add_update_listener(update_display)
    
    # Initial update
    update_display(tracker)
    
    return live, tracker


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


def display_knowledge_distillate(knowledge_distillate: Dict[str, Any]) -> None:
    """
    Display knowledge distillate in a nicely formatted panel.
    
    Args:
        knowledge_distillate: Knowledge distillate dictionary
    """
    console.print("\n[bold blue]📊 Videndistillat:[/bold blue]")
    
    if not knowledge_distillate or not isinstance(knowledge_distillate, dict):
        console.print(Panel("Ingen videndistillat tilgængeligt", title="[bold blue]Videndistillat[/bold blue]", border_style="blue"))
        return
        
    distillate_content = []
    
    # Add main points if available
    if 'hovedpunkter' in knowledge_distillate and knowledge_distillate['hovedpunkter']:
        distillate_content.append("[bold cyan]Hovedpunkter:[/bold cyan]")
        for point in knowledge_distillate['hovedpunkter']:
            distillate_content.append(f"• {point}")
        distillate_content.append("")
    
    # Add key statistics if available
    if 'noegletal' in knowledge_distillate and knowledge_distillate['noegletal']:
        distillate_content.append("[bold cyan]Nøgletal:[/bold cyan]")
        for stat in knowledge_distillate['noegletal']:
            if isinstance(stat, dict):
                tal = stat.get('tal', 'N/A')
                beskrivelse = stat.get('beskrivelse', '')
                kilde = stat.get('kilde', '')
                source = f" [dim]({kilde})[/dim]" if kilde else ""
                distillate_content.append(f"• [bold]{tal}[/bold]: {beskrivelse}{source}")
        distillate_content.append("")
    
    # Add key claims if available
    if 'centralePaastand' in knowledge_distillate and knowledge_distillate['centralePaastand']:
        distillate_content.append("[bold cyan]Centrale påstande:[/bold cyan]")
        for claim in knowledge_distillate['centralePaastand']:
            if isinstance(claim, dict):
                paastand = claim.get('paastand', '')
                kilde = claim.get('kilde', '')
                source = f" [dim]({kilde})[/dim]" if kilde else ""
                distillate_content.append(f"• {paastand}{source}")
            elif isinstance(claim, str):
                distillate_content.append(f"• {claim}")
        distillate_content.append("")
    
    # Add different perspectives if available
    if 'vinkler' in knowledge_distillate and knowledge_distillate['vinkler']:
        distillate_content.append("[bold cyan]Perspektiver:[/bold cyan]")
        for perspective in knowledge_distillate['vinkler']:
            if isinstance(perspective, dict):
                vinkel = perspective.get('vinkel', '')
                aktor = perspective.get('aktør', '')
                actor_text = f" [dim]({aktor})[/dim]" if aktor else ""
                distillate_content.append(f"• {vinkel}{actor_text}")
            elif isinstance(perspective, str):
                distillate_content.append(f"• {perspective}")
        distillate_content.append("")
    
    # Add important dates if available
    if 'datoer' in knowledge_distillate and knowledge_distillate['datoer']:
        distillate_content.append("[bold cyan]Vigtige datoer:[/bold cyan]")
        for date_info in knowledge_distillate['datoer']:
            if isinstance(date_info, dict):
                dato = date_info.get('dato', '')
                begivenhed = date_info.get('begivenhed', '')
                betydning = date_info.get('betydning', '')
                importance = f" - {betydning}" if betydning else ""
                distillate_content.append(f"• [bold]{dato}[/bold]: {begivenhed}{importance}")
    
    # If no content could be extracted, show a message
    if not distillate_content:
        distillate_content.append("[dim]Kunne ikke vise videndistillat i korrekt format.[/dim]")
        # Add raw data as fallback
        distillate_content.append("\n[dim]Rå data:[/dim]")
        try:
            import json
            distillate_content.append(json.dumps(knowledge_distillate, indent=2, ensure_ascii=False)[:500])
        except Exception:
            distillate_content.append(str(knowledge_distillate)[:500])
    
    # Display the distillate panel
    console.print(Panel(
        "\n".join(distillate_content),
        title="[bold blue]Videndistillat[/bold blue]",
        border_style="blue",
        expand=False
    ))


def display_angles_panels(angles: List[Dict[str, Any]], verbose: bool = False, detailed_sources: bool = False) -> None:
    """
    Display detailed panels for each angle.
    
    Args:
        angles: List of angle dictionaries
        verbose: If True, show more detailed information about sources
        detailed_sources: If True, show all available fields for all sources
    """
    # First check if we have a knowledge distillate to display
    has_distillate = False
    for angle in angles:
        if angle.get('videnDistillat') and not has_distillate:
            has_distillate = True
            display_knowledge_distillate(angle.get('videnDistillat'))
            break
    
    console.print("\n[bold blue]🎯 Anbefalede vinkler:[/bold blue]")
    
    for i, angle in enumerate(angles, 1):
        # Handle potentially missing keys with .get()
        headline = angle.get('overskrift', 'Ingen overskrift')
        description = angle.get('beskrivelse', 'Ingen beskrivelse')
        rationale = angle.get('begrundelse', 'Ingen begrundelse')
        criteria = angle.get('nyhedskriterier', [])
        questions = angle.get('startSpørgsmål', [])
        score = angle.get('kriterieScore', None)
        
        # Get the expert and source suggestions
        experts = angle.get('ekspertForslag', [])
        sources = angle.get('kildeForslag', [])
        statistics = angle.get('statistikForslag', [])
        
        # Create panel for each angle
        panel_content = [
            f"[bold white]{headline}[/bold white]",
            f"\n{description}",
            f"\n[dim blue]Begrundelse:[/dim blue] [dim]{rationale}[/dim]",
            f"\n[dim blue]Nyhedskriterier:[/dim blue] [dim]{', '.join(criteria)}[/dim]"
        ]
        
        # Add start questions if available
        if questions:
            panel_content.append(f"\n[dim blue]Startspørgsmål:[/dim blue]")
            for q in questions:
                panel_content.append(f"[dim]• {q}[/dim]")
        
        # Vis kun score hvis det er en reel numerisk værdi
        if isinstance(score, (int, float)):
            panel_content.append(f"\n[dim blue]Score:[/dim blue] [dim]{score}[/dim]")
        
        # Check for new expert sources format first
        if angle.get('ekspertKilder'):
            ekspert_kilder = angle.get('ekspertKilder')
            # Understøt både engelsk og dansk format
            experts = ekspert_kilder.get('experts') or ekspert_kilder.get('eksperter')
            institutions = ekspert_kilder.get('institutions') or ekspert_kilder.get('institutioner')
            data_sources = ekspert_kilder.get('data_sources') or ekspert_kilder.get('datakilder')

            if experts:
                panel_content.append(f"\n[bold cyan]Ekspertkilder:[/bold cyan]")
                # Vis alle eksperter hvis detailed_sources er True
                expert_limit = len(experts) if detailed_sources else (5 if verbose else 3)
                for expert in experts[:expert_limit]:
                    name = expert.get('navn', '')
                    title = expert.get('titel', '')
                    org = expert.get('organisation', '')
                    expertise = expert.get('ekspertise', '')
                    kontakt = expert.get('kontakt', '')
                    relevans = expert.get('relevans', '')
                    
                    expert_line = f"[bold]• {name}[/bold]"
                    if title:
                        expert_line += f", {title}"
                    if org:
                        expert_line += f", {org}"
                    panel_content.append(expert_line)
                    
                    # Vis alle detaljer hvis detailed_sources er True
                    if detailed_sources:
                        if expertise:
                            panel_content.append(f"  [dim italic]Ekspertise:[/dim italic] [dim]{expertise}[/dim]")
                        if kontakt:
                            panel_content.append(f"  [dim italic]Kontakt:[/dim italic] [dim]{kontakt}[/dim]")
                        if relevans:
                            panel_content.append(f"  [dim italic]Relevans:[/dim italic] [dim]{relevans}[/dim]")
                    else:
                        # Vis kun ekspertise i normal mode
                        if expertise:
                            panel_content.append(f"  [dim]{expertise}[/dim]")
                
                if not detailed_sources and len(experts) > expert_limit:
                    panel_content.append(f"[dim]+ {len(experts) - expert_limit} flere eksperter...[/dim]")
            
            if institutions:
                panel_content.append(f"\n[bold cyan]Institutioner:[/bold cyan]")
                # Vis alle institutioner hvis detailed_sources er True
                inst_limit = len(institutions) if detailed_sources else (3 if verbose else 2)
                for inst in institutions[:inst_limit]:
                    navn = inst.get('navn', '')
                    type_str = inst.get('type', '')
                    relevans = inst.get('relevans', '')
                    kontaktperson = inst.get('kontaktperson', '')
                    kontakt = inst.get('kontakt', '')
                    
                    inst_line = f"[bold]• {navn}[/bold]"
                    if type_str:
                        inst_line += f" ({type_str})"
                    panel_content.append(inst_line)
                    
                    # Vis alle detaljer hvis detailed_sources er True
                    if detailed_sources:
                        if relevans:
                            panel_content.append(f"  [dim italic]Relevans:[/dim italic] [dim]{relevans}[/dim]")
                        if kontaktperson:
                            panel_content.append(f"  [dim italic]Kontaktperson:[/dim italic] [dim]{kontaktperson}[/dim]")
                        if kontakt:
                            panel_content.append(f"  [dim italic]Kontakt:[/dim italic] [dim]{kontakt}[/dim]")
                
                if not detailed_sources and len(institutions) > inst_limit:
                    panel_content.append(f"[dim]+ {len(institutions) - inst_limit} flere institutioner...[/dim]")
            
            if data_sources:
                panel_content.append(f"\n[bold cyan]Datakilder:[/bold cyan]")
                # Vis alle datakilder hvis detailed_sources er True
                ds_limit = len(data_sources) if detailed_sources else (3 if verbose else 2)
                for ds in data_sources[:ds_limit]:
                    titel = ds.get('titel', 'N/A')
                    udgiver = ds.get('udgiver', '')
                    beskrivelse = ds.get('beskrivelse', '')
                    link = ds.get('link', '') or ds.get('url', '') or ds.get('website', '')
                    opdateret = ds.get('senest_opdateret', '')
                    
                    ds_line = f"[bold]• {titel}[/bold]"
                    if udgiver:
                        ds_line += f" ({udgiver})"
                    panel_content.append(ds_line)
                    
                    # Vis alle detaljer hvis detailed_sources er True
                    if detailed_sources:
                        if beskrivelse:
                            panel_content.append(f"  [dim italic]Beskrivelse:[/dim italic] [dim]{beskrivelse}[/dim]")
                        if link:
                            panel_content.append(f"  [dim italic]Link:[/dim italic] [blue underline]{link}[/blue underline]")
                        if opdateret:
                            panel_content.append(f"  [dim italic]Senest opdateret:[/dim italic] [dim]{opdateret}[/dim]")
                    else:
                        # Vis kun beskrivelse og link i normal mode
                        if beskrivelse:
                            panel_content.append(f"  [dim]{beskrivelse}[/dim]")
                        if link:
                            panel_content.append(f"  [blue underline]{link}[/blue underline]")
                
                if not detailed_sources and len(data_sources) > ds_limit:
                    panel_content.append(f"[dim]+ {len(data_sources) - ds_limit} flere datakilder...[/dim]")
            
            if not (experts or institutions or data_sources):
                logger.warning(f"Ekspertkilder fundet men ingen kendte nøgler (ekspertKilder): {list(ekspert_kilder.keys())}")
        
        # Fall back to older expert and source formats if needed
        elif experts:
            panel_content.append(f"\n[bold blue]Relevante eksperter:[/bold blue]")
            for expert in experts:
                name = expert.get('navn', '')
                title = expert.get('titel', '')
                org = expert.get('organisation', '')
                expertise = expert.get('ekspertise', '')
                
                if name and (title or org):
                    expert_line = f"[bold]• {name}[/bold]"
                    if title:
                        expert_line += f", {title}"
                    if org:
                        expert_line += f", {org}"
                    panel_content.append(expert_line)
                    if expertise:
                        panel_content.append(f"  [dim]{expertise}[/dim]")
        
        # Add source suggestions (older format) if available
        if sources:
            panel_content.append(f"\n[bold blue]Relevante kilder:[/bold blue]")
            for source in sources:
                name = source.get('navn', '')
                type_str = source.get('type', '')
                desc = source.get('beskrivelse', '')
                link = source.get('link', '') or source.get('url', '') or source.get('website', '')
                if name:
                    source_line = f"[bold]• {name}[/bold]"
                    if type_str:
                        source_line += f" ({type_str})"
                    panel_content.append(source_line)
                    if desc:
                        panel_content.append(f"  [dim]{desc}[/dim]")
                    if link:
                        panel_content.append(f"    [blue underline]{link}[/blue underline]")
        
        # Fallback: vis tekstbaseret kildeforslag hvis ingen strukturerede kilder
        if not (sources or angle.get('ekspertKilder') or statistics) and angle.get('kildeForslagInfo'):
            panel_content.append(f"\n[bold yellow]FALLBACK: Generelle kilder (tekstformat):[/bold yellow]")
            panel_content.append(f"[dim italic]Note: Strukturerede ekspertkilder kunne ikke genereres. Viser generisk kildeliste i stedet.[/dim italic]")
            panel_content.append(f"[dim]{angle['kildeForslagInfo']}[/dim]")
        
        # Add statistics (older format) if available
        if statistics:
            panel_content.append(f"\n[bold blue]Relevante statistikker:[/bold blue]")
            for stat in statistics:
                desc = stat.get('beskrivelse', '')
                source = stat.get('kilde', '')
                
                if desc:
                    stat_line = f"[bold]• {desc}[/bold]"
                    if source:
                        stat_line += f" [dim]({source})[/dim]"
                    panel_content.append(stat_line)
        
        console.print(Panel(
            "\n".join(panel_content),
            title=f"[bold green]Vinkel {i}[/bold green]",
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