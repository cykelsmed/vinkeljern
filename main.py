"""
Main entry point for the Vinkeljernet application.

This module handles command-line arguments and orchestrates the application flow.
It supports both traditional CLI mode and an interactive mode using prompt_toolkit.
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
from typing import List, Dict, Any, Optional, Tuple

from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.markdown import Markdown
from rich.syntax import Syntax

# Import prompt_toolkit for interactive CLI
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter, FuzzyCompleter
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.shortcuts import clear, message_dialog, button_dialog
from prompt_toolkit.application import run_in_terminal

from models import RedaktionelDNA, KildeModel, KnowledgeDistillate  # <--- Updated imports

# Configure root logger
logging.basicConfig(
    filename='vinkeljernet.log',
    filemode='w',  # 'w' to overwrite, 'a' to append
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Test logging
logging.debug("Logging initialized")
logging.info("Vinkeljernet starting up")

# Import configuration components
try:
    from config import OPENAI_API_KEY, PERPLEXITY_API_KEY
    # Verify API keys are present
    if not OPENAI_API_KEY:
        rprint("[bold red]Fejl:[/bold red] OPENAI_API_KEY mangler i konfigurationen.")
        rprint("[yellow]Tip:[/yellow] Opret en .env fil med OPENAI_API_KEY=din_nøgle")
        sys.exit(1)
    if not PERPLEXITY_API_KEY:
        rprint("[bold yellow]Advarsel:[/bold yellow] PERPLEXITY_API_KEY mangler i konfigurationen.")
        rprint("[yellow]Applikationen vil fortsætte, men uden detaljeret baggrundsinformation.[/yellow]")
except Exception as e:
    rprint(f"[bold red]Fejl ved indlæsning af API nøgler:[/bold red] {e}")
    sys.exit(1)

from config_loader import load_and_validate_profile
from api_clients_wrapper import (
    fetch_topic_information, 
    generate_angles, 
    process_generation_request,
    get_performance_metrics,
    initialize_api_client,
    shutdown_api_client
)
from angle_processor import filter_and_rank_angles

# Initialize console
console = Console()

def parse_arguments() -> Namespace:
    """
    Parse command-line arguments for the Vinkeljernet application.
    
    Returns:
        Namespace: Object containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Vinkeljernet: Generate news angles based on editorial DNA profiles"
    )
    
    parser.add_argument(
        "--emne", "-e",
        type=str,
        required=False,  # Not required anymore because of interactive mode
        help="Det nyhedsemne, der skal genereres vinkler for."
    )
    
    parser.add_argument(
        "--profil", "-p",
        type=str,
        required=False,  # Not required anymore because of interactive mode
        help="Stien til YAML-filen med den redaktionelle DNA-profil."
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=False,
        default=None,
        help="Valgfri filsti til at gemme outputtet."
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Start i interaktiv CLI-tilstand"
    )
    
    parser.add_argument(
        "--dev-mode",
        action="store_true",
        help="Kør i udviklingstilstand (deaktiverer SSL-verifikation, usikkert!)"
    )
    
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Ryd cache før programmet kører"
    )
    
    parser.add_argument(
        "--bypass-cache",
        action="store_true",
        help="Ignorer cache og tving friske API kald"
    )
    
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "markdown", "html"],
        default="json",
        help="Format for output (when using --output)"
    )
    
    parser.add_argument(
        "--show-circuits",
        action="store_true",
        help="Vis status for circuit breakers"
    )

    parser.add_argument(
        "--reset-circuits",
        action="store_true",
        help="Nulstil alle circuit breakers til lukket tilstand"
    )
    
    parser.add_argument(
        "--performance",
        action="store_true",
        help="Vis ydeevne-statistik for API og cache"
    )
    
    parser.add_argument(
        "--optimize-cache",
        action="store_true",
        help="Optimer disk-cachen (fjern forældede og komprimér)"
    )
    
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Få mere detaljeret information om emnet"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Aktiver debug tilstand med ekstra output"
    )
    
    args = parser.parse_args()
    
    # Validate required arguments if not in interactive mode
    if not args.interactive and (not args.emne or not args.profil):
        if not args.emne:
            parser.error("Argument --emne/-e er påkrævet i ikke-interaktiv tilstand")
        if not args.profil:
            parser.error("Argument --profil/-p er påkrævet i ikke-interaktiv tilstand")
    
    return args


async def main_async() -> None:
    """
    Asynchronous main function that orchestrates the application flow.
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # Initialize API client for optimized performance
    try:
        await initialize_api_client()
    except Exception as e:
        logging.warning(f"Failed to initialize API client: {e}")
    
    # Print start message with nice formatting
    console.print(Panel.fit(
        "[bold blue]Vinkeljernet[/bold blue] - Journalistisk vinkelgenerator",
        border_style="blue"
    ))
    
    # Display received arguments
    console.print("[bold blue]Kører med følgende parametre:[/bold blue]")
    console.print(f"  [green]Nyhedsemne:[/green] {args.emne}")
    console.print(f"  [green]Profil fil:[/green] {args.profil}")
    console.print(f"  [green]Output fil:[/green] {args.output if args.output else 'Ingen (viser i terminalen)'}")
    if args.dev_mode:
        console.print("  [yellow]⚠️ Udvikler-tilstand aktiveret (usikker SSL)[/yellow]")
    if args.detailed:
        console.print("  [green]Detaljeret emne-information aktiveret[/green]")
    
    # Handle cache operations
    if args.clear_cache:
        from cache_manager import clear_cache
        num_cleared = clear_cache()
        console.print(f"[yellow]Cache ryddet: {num_cleared} filer slettet[/yellow]")
    
    if args.optimize_cache:
        from cache_manager import optimize_cache
        result = optimize_cache()
        console.print("[yellow]Cache optimeret[/yellow]")
        console.print(f"  Filer behandlet: {result['files_processed']}")
        console.print(f"  Filer komprimeret: {result['files_recompressed']}")
        console.print(f"  Forældede filer fjernet: {result['files_removed']}")
        console.print(f"  Plads sparet: {result['mb_saved']:.2f} MB")
    
    # Handle circuit breaker operations
    if args.reset_circuits:
        from retry_manager import reset_circuit
        reset_circuit("perplexity_api")
        reset_circuit("openai_api")
        reset_circuit("anthropic_api")
        console.print("[yellow]Alle circuit breakers nulstillet[/yellow]")

    # Show circuit breaker status if requested
    if args.show_circuits:
        from retry_manager import get_circuit_stats
        stats = get_circuit_stats()
        
        circuit_table = Table(show_header=True, header_style="bold blue")
        circuit_table.add_column("API", style="dim")
        circuit_table.add_column("Tilstand")
        circuit_table.add_column("Succesfulde kald")
        circuit_table.add_column("Fejlslagne kald")
        circuit_table.add_column("Konsekutive fejl")
        circuit_table.add_column("Antal gentag")
        circuit_table.add_column("Seneste fejl")
        
        for name, data in stats.items():
            state_style = "green" if data["state"] == "closed" else "red" if data["state"] == "open" else "yellow"
            circuit_table.add_row(
                name,
                f"[{state_style}]{data['state']}[/{state_style}]",
                str(data["success_count"]),
                str(data["failure_count"]),
                str(data["consecutive_failures"]),
                str(data["total_retries"]),
                data["last_failure"] or "Ingen"
            )
        
        console.print("\n[bold blue]Circuit Breaker Status:[/bold blue]")
        console.print(circuit_table)
    
    # Show performance metrics if requested
    if args.performance:
        metrics = get_performance_metrics()
        
        # Display API metrics
        api_metrics = metrics.get("api", {})
        console.print("\n[bold blue]API Ydeevne Statistik:[/bold blue]")
        
        api_table = Table(show_header=True, header_style="bold blue")
        api_table.add_column("Metrik", style="dim")
        api_table.add_column("Værdi")
        
        if "note" in api_metrics:
            # Simple metrics
            api_table.add_row("Besked", api_metrics["note"])
        else:
            # Advanced metrics
            api_table.add_row("Successrate", api_metrics.get("success_rate", "N/A"))
            api_table.add_row("Cache træfrate", api_metrics.get("cache_hit_rate", "N/A"))
            api_table.add_row("Total antal kald", str(api_metrics.get("total_requests", "N/A")))
            api_table.add_row("Succesfulde kald", str(api_metrics.get("successful_requests", "N/A")))
            api_table.add_row("Fejlede kald", str(api_metrics.get("failed_requests", "N/A")))
            api_table.add_row("Gennemsnitlig latens", f"{api_metrics.get('average_latency_ms', 'N/A')} ms")
            api_table.add_row("Driftstid", api_metrics.get("uptime_formatted", "N/A"))
        
        console.print(api_table)
        
        # Display cache metrics
        cache_metrics = metrics.get("cache", {})
        console.print("\n[bold blue]Cache Statistik:[/bold blue]")
        
        cache_table = Table(show_header=True, header_style="bold blue")
        cache_table.add_column("Metrik", style="dim")
        cache_table.add_column("Værdi")
        
        for key, value in cache_metrics.items():
            cache_table.add_row(key, str(value))
        
        console.print(cache_table)
    
    # Load and validate profile with progress spinner
    profile = None
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Indlæser redaktionel profil..."),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("Loading", total=None)
        try:
            profile_path = Path(args.profil)
            profile = load_and_validate_profile(profile_path)
            progress.update(task, completed=True)
        except FileNotFoundError:
            progress.update(task, completed=True)
            console.print(f"[bold red]Fejl:[/bold red] Profil filen '{args.profil}' blev ikke fundet")
            console.print("[yellow]Tip:[/yellow] Kontroller filstien og prøv igen")
            sys.exit(1)
        except yaml.YAMLError as e:
            progress.update(task, completed=True)
            console.print(f"[bold red]Fejl:[/bold red] Kunne ikke læse YAML profil: {e}")
            console.print("[yellow]Tip:[/yellow] Kontrollér at din YAML-fil er korrekt formateret")
            sys.exit(1)
        except (ValueError, TypeError) as e:
            progress.update(task, completed=True)
            console.print(f"[bold red]Fejl:[/bold red] Profilvalidering fejlede: {e}")
            console.print("[yellow]Tip:[/yellow] Kontrollér at din profil indeholder alle påkrævede felter")
            sys.exit(1)
    
    # Display profile summary
    console.print("[green]✓[/green] Profil indlæst og valideret")
    
    table = Table(show_header=True, header_style="bold blue")
    table.add_column("Profil-element", style="dim")
    table.add_column("Værdi")
    
    # Display kerneprincipper
    principles = "\n".join([f"- {p}" for p in profile.kerneprincipper])
    table.add_row("Kerneprincipper", principles)
    table.add_row("Tone og stil", profile.tone_og_stil)
    table.add_row("Antal nyhedskriterier", str(len(profile.nyhedsprioritering)))
    table.add_row("Antal fokusområder", str(len(profile.fokusOmrader)))
    table.add_row("Antal no-go områder", str(len(profile.noGoOmrader)))

    console.print(table)
    
    # Record start time for performance measurement
    start_time = time.time()
    
    # Process generation request with optimized client
    angles = None
    with Progress(
        SpinnerColumn(),
        BarColumn(),
        TextColumn("[bold blue]{task.description}[/bold blue] {task.percentage:>3.0f}%"),
        console=console,
        transient=True
    ) as progress:
        # Set up a callback to update progress
        async def progress_callback(percent: int):
            progress.update(task, completed=percent)
            
            # Update task description based on progress
            if percent < 20:
                progress.update(task, description="Henter information om emnet...")
            elif percent < 40:
                progress.update(task, description="Genererer videndistillat...")
            elif percent < 60:
                progress.update(task, description="Genererer vinkler...")
            elif percent < 80:
                progress.update(task, description="Finder ekspertkilder...")
            else:
                progress.update(task, description="Færdiggør resultater...")
        
        # Create progress task
        task = progress.add_task("Starter processen...", total=100)
        
        try:
            # Avoid name conflict: Import the API wrapper function with a different name
            from api_clients_wrapper import process_generation_request as api_process_request
            angles = await api_process_request(
                args.emne,
                profile,
                bypass_cache=args.bypass_cache,
                progress_callback=progress_callback,
                include_expert_sources=True,
                include_knowledge_distillate=True
            )
            
            # Ensure progress is complete
            progress.update(task, completed=100, description="Færdig!")
            
        except Exception as e:
            progress.update(task, completed=100, description="Fejlet!")
            console.print(f"\n[bold red]Fejl under vinkelgenerering:[/bold red] {e}")
            
            # Try fallback if the optimized client fails
            console.print("[yellow]Forsøger med backup-metode...[/yellow]")
            try:
                # First get topic info
                topic_info = await fetch_topic_information(
                    args.emne, 
                    dev_mode=args.dev_mode, 
                    bypass_cache=args.bypass_cache,
                    detailed=args.detailed
                )
                
                if not topic_info:
                    topic_info = f"Emnet handler om {args.emne}. Ingen yderligere baggrundsinformation tilgængelig."
                
                # Then generate angles with topic info
                angles = generate_angles(args.emne, topic_info, profile, bypass_cache=args.bypass_cache)
                
            except Exception as fallback_error:
                console.print(f"[bold red]Backup-metode fejlede også:[/bold red] {fallback_error}")
                sys.exit(1)
    
    # Calculate and display performance
    execution_time = time.time() - start_time
    console.print(f"[dim]Udførelsestid: {execution_time:.2f} sekunder[/dim]")
    
    # Check if we have any angles
    if not angles:
        console.print("[bold red]Ingen vinkler kunne genereres.[/bold red]")
        console.print("[yellow]Mulige årsager:[/yellow]")
        console.print("  - API-fejl ved forbindelse til AI-tjenester")
        console.print("  - Emnet er for specifikt eller ukendt")
        console.print("  - Profilen er for restriktiv")
        console.print("[yellow]Prøv et andet emne eller kontrollér API-nøglen.[/yellow]")
        sys.exit(1)
    
    console.print(f"[green]✓[/green] Genereret {len(angles)} vinkler")
    
    # Check if we have a knowledge distillate
    has_distillate = False
    for angle in angles:
        if angle.get('videnDistillat') and not has_distillate:
            has_distillate = True
            distillate = angle.get('videnDistillat')
            
            # Display knowledge distillate panel
            console.print("\n[bold blue]📊 Videndistillat:[/bold blue]")
            
            distillate_content = []
            
            # Add key statistics if available
            if 'noegletal' in distillate and distillate['noegletal']:
                distillate_content.append("[bold cyan]Nøgletal:[/bold cyan]")
                for stat in distillate['noegletal']:
                    source = f" [dim]({stat.get('kilde')})[/dim]" if stat.get('kilde') else ""
                    distillate_content.append(f"• [bold]{stat.get('tal')}[/bold]: {stat.get('beskrivelse')}{source}")
                distillate_content.append("")
            
            # Add key claims if available
            if 'centralePaastand' in distillate and distillate['centralePaastand']:
                distillate_content.append("[bold cyan]Centrale påstande:[/bold cyan]")
                for claim in distillate['centralePaastand']:
                    source = f" [dim]({claim.get('kilde')})[/dim]" if claim.get('kilde') else ""
                    distillate_content.append(f"• {claim.get('paastand')}{source}")
                distillate_content.append("")
            
            # Add different perspectives if available
            if 'vinkler' in distillate and distillate['vinkler']:
                distillate_content.append("[bold cyan]Perspektiver:[/bold cyan]")
                for perspective in distillate['vinkler']:
                    actor = f" [dim]({perspective.get('aktør')})[/dim]" if perspective.get('aktør') else ""
                    distillate_content.append(f"• {perspective.get('vinkel')}{actor}")
                distillate_content.append("")
            
            # Add important dates if available
            if 'datoer' in distillate and distillate['datoer']:
                distillate_content.append("[bold cyan]Vigtige datoer:[/bold cyan]")
                for date_info in distillate['datoer']:
                    importance = f" - {date_info.get('betydning')}" if date_info.get('betydning') else ""
                    distillate_content.append(f"• [bold]{date_info.get('dato')}[/bold]: {date_info.get('begivenhed')}{importance}")
            
            # Display the distillate panel
            console.print(Panel(
                "\n".join(distillate_content),
                title="[bold blue]Videndistillat[/bold blue]",
                border_style="blue",
                expand=False
            ))
            
            # Only show the distillate once
            break
    
    # Present results with nice formatting
    console.print("\n[bold blue]🎯 Anbefalede vinkler:[/bold blue]")
    for i, angle in enumerate(angles, 1):
        # Handle potentially missing keys with .get()
        headline = angle.get('overskrift', 'Ingen overskrift')
        description = angle.get('beskrivelse', 'Ingen beskrivelse')
        rationale = angle.get('begrundelse', 'Ingen begrundelse')
        criteria = angle.get('nyhedskriterier', [])
        questions = angle.get('startSpørgsmål', [])
        score = angle.get('kriterieScore', 'N/A')
        
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
        
        if score != 'N/A':
            panel_content.append(f"\n[dim blue]Score:[/dim blue] [dim]{score}[/dim]")
        
        # Add expert sources if available for this angle
        if angle.get('ekspertKilder'):
            panel_content.append(f"\n[bold cyan]Ekspertkilder:[/bold cyan]")
            
            # Add experts
            if 'experts' in angle['ekspertKilder'] and angle['ekspertKilder']['experts']:
                for expert in angle['ekspertKilder']['experts'][:3]:  # Limit to 3 experts to avoid overload
                    name = expert.get('navn', 'N/A')
                    title = expert.get('titel', 'N/A')
                    org = expert.get('organisation', 'N/A')
                    contact = f" [dim]({expert.get('kontakt')})[/dim]" if expert.get('kontakt') else ""
                    panel_content.append(f"• [bold]{name}[/bold]: {title}, {org}{contact}")
            
            # Add data sources (just a mention)
            if 'data_sources' in angle['ekspertKilder'] and angle['ekspertKilder']['data_sources']:
                data_count = len(angle['ekspertKilder']['data_sources'])
                panel_content.append(f"[dim]+ {data_count} datakilder tilgængelige[/dim]")
        
        console.print(Panel(
            "\n".join(panel_content),
            title=f"[bold green]Vinkel {i}[/bold green]",
            border_style="green",
            expand=False
        ))
    
    # Save to output file if specified
    if args.output:
        try:
            from formatters import format_angles
            
            # Extract profile name from the path
            profile_name = Path(args.profil).stem
            
            format_angles(
                angles, 
                format_type=args.format,
                profile_name=profile_name,
                topic=args.emne,
                output_path=args.output
            )
            
            console.print(f"\n[green]✓[/green] Resultater gemt i {args.output} ({args.format} format)")
        except ImportError:
            # Fallback to JSON if formatter module not available
            with open(args.output, 'w', encoding='utf-8') as outfile:
                json.dump(angles, outfile, ensure_ascii=False, indent=2)
            console.print(f"\n[green]✓[/green] Resultater gemt i {args.output} (JSON format)")
        except IOError as e:
            console.print(f"\n[bold red]Fejl ved skrivning til fil:[/bold red] {e}")
            console.print(f"[yellow]Tjek om stien eksisterer og om du har skriverettigheder.[/yellow]")
        except Exception as e:
            console.print(f"\n[bold red]Uventet fejl ved skrivning til fil:[/bold red] {e}")
            
    # Properly clean up API client on exit
    try:
        await shutdown_api_client()
    except Exception as e:
        logging.warning(f"Error shutting down API client: {e}")
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        print(f"[DEBUG] Loaded profile data: {profile.dict()}")
        print(f"[DEBUG] Topic information: {topic_info[:200]}...")


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
        print(f"AttributeError during filtering: {e}")
        print("Falling back to simple sort of angles based on the count of news criteria.")
        sorted_angles = sorted(angles, key=lambda x: len(x.get("nyhedskriterier", [])), reverse=True)
        return sorted_angles[:num_angles]
    except Exception as e:
        print(f"Unexpected error during filtering: {e}")
        print("Falling back to unfiltered angle list.")
        return angles[:num_angles]


# Define the interactive CLI functions
def get_available_profiles() -> List[str]:
    """Get list of available profile paths."""
    profile_dir = Path("config")
    if not profile_dir.exists() or not profile_dir.is_dir():
        return []
    
    profiles = list(profile_dir.glob("*.yaml"))
    return [str(p) for p in profiles]


def get_profile_names() -> List[str]:
    """Get list of available profile names (without path and extension)."""
    profiles = get_available_profiles()
    return [Path(p).stem for p in profiles]


def display_welcome_message() -> None:
    """Display a welcome message for the interactive CLI."""
    clear()
    console.print(Panel.fit(
        "[bold blue]Vinkeljernet[/bold blue] - Interaktiv journalistisk vinkelgenerator",
        border_style="blue"
    ))
    console.print(
        "\n[bold]Velkommen til Vinkeljernet CLI![/bold]\n\n"
        "Dette værktøj hjælper med at generere nyhedsvinkler baseret på redaktionelle DNA-profiler.\n"
        "Brug kommandoer for at udforske og generere vinkler. Skriv [bold]hjælp[/bold] for at se tilgængelige kommandoer.\n"
    )


def display_help() -> None:
    """Display help information for the interactive CLI."""
    help_text = """
    # Vinkeljernet Kommandoer
    
    ## Grundlæggende
    - `hjælp` - Vis denne hjælpetekst
    - `afslut` eller `quit` - Afslut programmet
    - `ryd` - Ryd skærmen
    
    ## Profiler
    - `profiler` - Vis tilgængelige redaktionelle profiler
    - `vis profil <navn>` - Vis detaljer om en specifik profil
    
    ## Generering
    - `generer <emne> <profil>` - Generer vinkler for et emne med en bestemt profil
       Eksempel: `generer klimaforandringer dr_profil`
    
    ## Konfiguration
    - `toggle debug` - Slå debug-tilstand til/fra
    - `toggle cache` - Slå caching til/fra
    - `ryd cache` - Ryd cache-filer
    
    ## System
    - `status` - Vis system-status inkl. circuit breakers
    """
    
    md = Markdown(help_text)
    console.print(md)


async def run_interactive_cli() -> None:
    """Run the interactive CLI interface."""
    # Setup the prompt session with history
    history_file = Path.home() / ".vinkeljernet_history"
    
    # Check if we're running in an environment that doesn't support prompt_toolkit
    try:
        session = PromptSession(history=FileHistory(str(history_file)))
    except Exception as e:
        # Fallback simple mode for non-interactive environments
        console.print(f"[yellow]Kunne ikke starte interaktiv tilstand: {e}[/yellow]")
        console.print("[yellow]Bruger simpel kommandolinje-tilstand i stedet.[/yellow]")
        return await run_simple_cli()
    
    # Set up the style
    style = Style.from_dict({
        'prompt': 'ansicyan bold',
        'command': 'ansigreen',
        'param': 'ansiyellow',
    })
    
    # Set up auto-completion
    commands = ['hjælp', 'afslut', 'quit', 'ryd', 'profiler', 
                'vis profil', 'generer', 'toggle debug', 
                'toggle cache', 'ryd cache', 'status']
    profile_names = get_profile_names()
    all_words = commands + profile_names + ['klimaforandringer', 'politik', 'økonomi', 'teknologi']
    word_completer = WordCompleter(all_words, ignore_case=True)
    
    # Cache some settings
    debug_mode = False
    bypass_cache = False
    
    # Main command loop
    display_welcome_message()
    
    while True:
        try:
            # Show the prompt with appropriate styling
            user_input = await session.prompt_async(
                HTML("<prompt>vinkeljernet></prompt> "),
                completer=FuzzyCompleter(word_completer),
                style=style
            )
            
            # Process the command
            cmd_parts = user_input.strip().split()
            if not cmd_parts:
                continue
                
            command = cmd_parts[0].lower()
            
            # Handle exit commands
            if command in ['afslut', 'quit', 'exit']:
                console.print("[yellow]Afslutter Vinkeljernet CLI. På gensyn![/yellow]")
                break
                
            # Handle help command
            elif command == 'hjælp':
                display_help()
                
            # Handle clear screen
            elif command == 'ryd':
                clear()
                display_welcome_message()
                
            # Handle show profiles
            elif command == 'profiler':
                profiles = get_profile_names()
                if not profiles:
                    console.print("[yellow]Ingen profiler fundet. Check config/ mappen.[/yellow]")
                else:
                    table = Table(title="Tilgængelige Profiler")
                    table.add_column("Profilnavn", style="cyan")
                    for profile in profiles:
                        table.add_row(profile)
                    console.print(table)
                    
            # Handle view profile
            elif command == 'vis' and len(cmd_parts) >= 3 and cmd_parts[1] == 'profil':
                profile_name = cmd_parts[2]
                profile_path = f"config/{profile_name}.yaml"
                if not profile_name.endswith('.yaml'):
                    profile_path = f"config/{profile_name}.yaml"
                
                try:
                    profile = load_and_validate_profile(Path(profile_path))
                    
                    # Display profile
                    console.print(f"[bold blue]Profil:[/bold blue] {profile_name}")
                    
                    table = Table(show_header=True, header_style="bold blue")
                    table.add_column("Profil-element", style="dim")
                    table.add_column("Værdi")
                    
                    principles = "\n".join([f"- {p}" for p in profile.kerneprincipper])
                    table.add_row("Kerneprincipper", principles)
                    table.add_row("Tone og stil", profile.tone_og_stil)
                    table.add_row("Antal nyhedskriterier", str(len(profile.nyhedsprioritering)))
                    table.add_row("Antal fokusområder", str(len(profile.fokusOmrader)))
                    table.add.row("Antal no-go områder", str(len(profile.noGoOmrader)))
                    
                    console.print(table)
                except FileNotFoundError:
                    console.print(f"[bold red]Fejl:[/bold red] Profil '{profile_name}' blev ikke fundet")
                except Exception as e:
                    console.print(f"[bold red]Fejl ved indlæsning af profil:[/bold red] {e}")
                    
            # Handle generate command
            elif command == 'generer' and len(cmd_parts) >= 3:
                topic = cmd_parts[1]
                profile_name = cmd_parts[2]
                
                # Check if this is a valid profile
                if not profile_name.endswith('.yaml'):
                    profile_path = f"config/{profile_name}.yaml"
                else:
                    profile_path = profile_name
                
                # Create an args-like object to pass to the main function
                class Args:
                    pass
                
                args = Args()
                args.emne = topic
                args.profil = profile_path
                args.output = None
                args.dev_mode = False
                args.clear_cache = False
                args.bypass_cache = bypass_cache
                args.format = "json"
                args.show_circuits = False
                args.reset_circuits = False
                args.debug = debug_mode
                
                try:
                    # Run the main function but capture its output for the interactive CLI
                    console.print(f"[bold blue]Genererer vinkler for emne '{topic}' med profil '{profile_name}'...[/bold blue]")
                    await run_in_terminal(lambda: console.print("Processing... (this may take a minute)"))
                    
                    # Call the main processing function
                    await process_generation_request(args)
                    
                except Exception as e:
                    console.print(f"[bold red]Fejl ved generering af vinkler:[/bold red] {e}")
            
            # Handle toggle debug command
            elif user_input.strip().lower() == 'toggle debug':
                debug_mode = not debug_mode
                console.print(f"[blue]Debug-tilstand er nu {'[green]aktiveret' if debug_mode else '[red]deaktiveret'}[/blue]")
                
            # Handle toggle cache command
            elif user_input.strip().lower() == 'toggle cache':
                bypass_cache = not bypass_cache
                console.print(f"[blue]Cache bypass er nu {'[green]aktiveret' if bypass_cache else '[red]deaktiveret'}[/blue]")
                
            # Handle clear cache command
            elif user_input.strip().lower() == 'ryd cache':
                from cache_manager import clear_cache
                num_cleared = clear_cache()
                console.print(f"[yellow]Cache ryddet: {num_cleared} filer slettet[/yellow]")
                
            # Handle status command
            elif command == 'status':
                from retry_manager import get_circuit_stats
                stats = get_circuit_stats()
                
                circuit_table = Table(show_header=True, header_style="bold blue")
                circuit_table.add_column("API", style="dim")
                circuit_table.add.column("Tilstand")
                circuit_table.add.column("Succesfulde kald")
                circuit_table.add.column("Fejlslagne kald")
                circuit_table.add.column("Konsekutive fejl")
                
                for name, data in stats.items():
                    state_style = "green" if data["state"] == "closed" else "red" if data["state"] == "open" else "yellow"
                    circuit_table.add.row(
                        name,
                        f"[{state_style}]{data['state']}[/{state_style}]",
                        str(data["success_count"]),
                        str(data["failure_count"]),
                        str(data["consecutive_failures"])
                    )
                
                console.print("\n[bold blue]Circuit Breaker Status:[/bold blue]")
                console.print(circuit_table)
                
                # Display cache status
                console.print("\n[bold blue]Cache Status:[/bold blue]")
                console.print(f"Cache bypass: {'[red]Aktiveret' if bypass_cache else '[green]Deaktiveret'}") 
                console.print(f"Debug mode: {'[green]Aktiveret' if debug_mode else '[red]Deaktiveret'}")
                
            # Unknown command
            else:
                console.print(f"[yellow]Ukendt kommando: {user_input}[/yellow]")
                console.print("Skriv [bold]hjælp[/bold] for at se tilgængelige kommandoer.")
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Operation afbrudt. Skriv 'afslut' for at lukke programmet.[/yellow]")
            continue
        except EOFError:
            console.print("\n[yellow]Afslutter Vinkeljernet CLI. På gensyn![/yellow]")
            break
        except Exception as e:
            console.print(f"[bold red]Fejl:[/bold red] {str(e)}")


async def process_generation_request(args) -> None:
    """Process a generation request with the given args using the optimized API client."""
    
    # Print start message with nice formatting
    console.print(Panel.fit(
        f"[bold blue]Genererer vinkler for: {args.emne}[/bold blue]",
        border_style="blue"
    ))
    
    # Initialize API client for optimized performance
    try:
        await initialize_api_client()
    except Exception as e:
        logging.warning(f"Failed to initialize API client: {e}")
    
    # Load and validate profile
    profile = None
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Indlæser redaktionel profil..."),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("Loading", total=None)
        try:
            profile_path = Path(args.profil)
            profile = load_and_validate_profile(profile_path)
            progress.update(task, completed=True)
        except FileNotFoundError:
            progress.update(task, completed=True)
            console.print(f"[bold red]Fejl:[/bold red] Profil filen '{args.profil}' blev ikke fundet")
            return
        except yaml.YAMLError as e:
            progress.update(task, completed=True)
            console.print(f"[bold red]Fejl:[/bold red] Kunne ikke læse YAML profil: {e}")
            return
        except (ValueError, TypeError) as e:
            progress.update(task, completed=True)
            console.print(f"[bold red]Fejl:[/bold red] Profilvalidering fejlede: {e}")
            return
    
    console.print("[green]✓[/green] Profil indlæst og valideret")
    
    # Record start time for performance measurement
    start_time = time.time()
    
    # Process generation request with optimized client
    angles = None
    with Progress(
        SpinnerColumn(),
        BarColumn(),
        TextColumn("[bold blue]{task.description}[/bold blue] {task.percentage:>3.0f}%"),
        console=console,
        transient=True
    ) as progress:
        # Set up a callback to update progress
        async def progress_callback(percent: int):
            progress.update(task, completed=percent)
            
            # Update task description based on progress
            if percent < 20:
                progress.update(task, description="Henter information om emnet...")
            elif percent < 40:
                progress.update(task, description="Genererer videndistillat...")
            elif percent < 60:
                progress.update(task, description="Genererer vinkler...")
            elif percent < 80:
                progress.update(task, description="Finder ekspertkilder...")
            else:
                progress.update(task, description="Færdiggør resultater...")
        
        # Create progress task
        task = progress.add_task("Starter processen...", total=100)
        
        try:
            # Avoid name conflict: Import the API wrapper function with a different name
            from api_clients_wrapper import process_generation_request as api_process_request
            angles = await api_process_request(
                args.emne,
                profile,
                bypass_cache=args.bypass_cache,
                progress_callback=progress_callback,
                include_expert_sources=True,
                include_knowledge_distillate=True
            )
            
            # Ensure progress is complete
            progress.update(task, completed=100, description="Færdig!")
            
        except Exception as e:
            progress.update(task, completed=100, description="Fejlet!")
            console.print(f"\n[bold red]Fejl under vinkelgenerering:[/bold red] {e}")
            
            # Try fallback if the optimized client fails
            console.print("[yellow]Forsøger med backup-metode...[/yellow]")
            try:
                # First get topic info
                topic_info = await fetch_topic_information(
                    args.emne, 
                    dev_mode=args.dev_mode, 
                    bypass_cache=args.bypass_cache
                )
                
                if not topic_info:
                    topic_info = f"Emnet handler om {args.emne}. Ingen yderligere baggrundsinformation tilgængelig."
                
                # Then generate angles with topic info
                angles = generate_angles(args.emne, topic_info, profile, bypass_cache=args.bypass_cache)
                
            except Exception as fallback_error:
                console.print(f"[bold red]Backup-metode fejlede også:[/bold red] {fallback_error}")
                return
    
    # Calculate and display performance
    execution_time = time.time() - start_time
    console.print(f"[dim]Udførelsestid: {execution_time:.2f} sekunder[/dim]")
    
    # Check if we have any angles
    if not angles:
        console.print("[bold red]Ingen vinkler kunne genereres.[/bold red]")
        console.print("[yellow]Mulige årsager:[/yellow]")
        console.print("  - API-fejl ved forbindelse til AI-tjenester")
        console.print("  - Emnet er for specifikt eller ukendt")
        console.print("  - Profilen er for restriktiv")
        return
    
    console.print(f"[green]✓[/green] Genereret {len(angles)} vinkler")
    
    # Check if we have a knowledge distillate
    has_distillate = False
    for angle in angles:
        if angle.get('videnDistillat') and not has_distillate:
            has_distillate = True
            distillate = angle.get('videnDistillat')
            
            # Display knowledge distillate panel
            console.print("\n[bold blue]📊 Videndistillat:[/bold blue]")
            
            distillate_content = []
            
            # Add key statistics if available
            if 'noegletal' in distillate and distillate['noegletal']:
                distillate_content.append("[bold cyan]Nøgletal:[/bold cyan]")
                for stat in distillate['noegletal']:
                    source = f" [dim]({stat.get('kilde')})[/dim]" if stat.get('kilde') else ""
                    distillate_content.append(f"• [bold]{stat.get('tal')}[/bold]: {stat.get('beskrivelse')}{source}")
                distillate_content.append("")
            
            # Add key claims if available
            if 'centralePaastand' in distillate and distillate['centralePaastand']:
                distillate_content.append("[bold cyan]Centrale påstande:[/bold cyan]")
                for claim in distillate['centralePaastand']:
                    source = f" [dim]({claim.get('kilde')})[/dim]" if claim.get('kilde') else ""
                    distillate_content.append(f"• {claim.get('paastand')}{source}")
                distillate_content.append("")
            
            # Add different perspectives if available
            if 'vinkler' in distillate and distillate['vinkler']:
                distillate_content.append("[bold cyan]Perspektiver:[/bold cyan]")
                for perspective in distillate['vinkler']:
                    actor = f" [dim]({perspective.get('aktør')})[/dim]" if perspective.get('aktør') else ""
                    distillate_content.append(f"• {perspective.get('vinkel')}{actor}")
                distillate_content.append("")
            
            # Add important dates if available
            if 'datoer' in distillate and distillate['datoer']:
                distillate_content.append("[bold cyan]Vigtige datoer:[/bold cyan]")
                for date_info in distillate['datoer']:
                    importance = f" - {date_info.get('betydning')}" if date_info.get('betydning') else ""
                    distillate_content.append(f"• [bold]{date_info.get('dato')}[/bold]: {date_info.get('begivenhed')}{importance}")
            
            # Display the distillate panel
            console.print(Panel(
                "\n".join(distillate_content),
                title="[bold blue]Videndistillat[/bold blue]",
                border_style="blue",
                expand=False
            ))
            
            # Only show the distillate once
            break
    
    # Present results with nice formatting
    console.print("\n[bold blue]🎯 Anbefalede vinkler:[/bold blue]")
    for i, angle in enumerate(angles, 1):
        # Handle potentially missing keys with .get()
        headline = angle.get('overskrift', 'Ingen overskrift')
        description = angle.get('beskrivelse', 'Ingen beskrivelse')
        rationale = angle.get('begrundelse', 'Ingen begrundelse')
        criteria = angle.get('nyhedskriterier', [])
        questions = angle.get('startSpørgsmål', [])
        score = angle.get('kriterieScore', 'N/A')
        
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
        
        if score != 'N/A':
            panel_content.append(f"\n[dim blue]Score:[/dim blue] [dim]{score}[/dim]")
        
        # Add expert sources if available for this angle
        if angle.get('ekspertKilder'):
            panel_content.append(f"\n[bold cyan]Ekspertkilder:[/bold cyan]")
            
            # Add experts
            if 'experts' in angle['ekspertKilder'] and angle['ekspertKilder']['experts']:
                for expert in angle['ekspertKilder']['experts'][:3]:  # Limit to 3 experts to avoid overload
                    name = expert.get('navn', 'N/A')
                    title = expert.get('titel', 'N/A')
                    org = expert.get('organisation', 'N/A')
                    contact = f" [dim]({expert.get('kontakt')})[/dim]" if expert.get('kontakt') else ""
                    panel_content.append(f"• [bold]{name}[/bold]: {title}, {org}{contact}")
            
            # Add data sources (just a mention)
            if 'data_sources' in angle['ekspertKilder'] and angle['ekspertKilder']['data_sources']:
                data_count = len(angle['ekspertKilder']['data_sources'])
                panel_content.append(f"[dim]+ {data_count} datakilder tilgængelige[/dim]")
        
        console.print(Panel(
            "\n".join(panel_content),
            title=f"[bold green]Vinkel {i}[/bold green]",
            border_style="green",
            expand=False
        ))
    
    # Clean up resources
    try:
        await shutdown_api_client()
    except Exception as e:
        logging.warning(f"Error shutting down API client: {e}")
    
    # Ask if user wants to save the output
    def ask_save_output():
        result = button_dialog(
            title="Gem resultat",
            text="Vil du gemme de genererede vinkler til en fil?",
            buttons=[
                ("Ja", True),
                ("Nej", False),
            ],
        ).run()
        
        if result:
            # Ask for filename
            path = session.prompt("Angiv filsti: ", default=f"{args.emne}_{Path(args.profil).stem}.md")
            
            try:
                from formatters import format_angles
                
                # Extract profile name from the path
                profile_name = Path(args.profil).stem
                
                # Determine format based on extension
                extension = Path(path).suffix.lower()
                if extension == '.json':
                    format_type = 'json'
                elif extension == '.md':
                    format_type = 'markdown'
                elif extension == '.html':
                    format_type = 'html'
                else:
                    format_type = 'markdown'  # Default
                
                format_angles(
                    ranked_angles, 
                    format_type=format_type,
                    profile_name=profile_name,
                    topic=args.emne,
                    output_path=path
                )
                
                console.print(f"\n[green]✓[/green] Resultater gemt i {path} ({format_type} format)")
            except Exception as e:
                console.print(f"\n[bold red]Fejl ved skrivning til fil:[/bold red] {e}")
    
    # This is commented out because it's causing issues with the interactive CLI
    # run_in_terminal(ask_save_output)


async def run_simple_cli() -> None:
    """Run a simplified CLI interface for environments where prompt_toolkit doesn't work."""
    display_welcome_message()
    display_help()
    
    # Get available profiles
    profiles = get_profile_names()
    console.print("\n[bold blue]Tilgængelige profiler:[/bold blue]")
    for profile in profiles:
        console.print(f"  • {profile}")
    
    # Main command loop with simple input
    while True:
        try:
            # Simple input prompt
            console.print("\n[bold cyan]vinkeljernet>[/bold cyan] ", end="")
            user_input = input()
            
            # Process the command
            cmd_parts = user_input.strip().split()
            if not cmd_parts:
                continue
                
            command = cmd_parts[0].lower()
            
            # Handle exit commands
            if command in ['afslut', 'quit', 'exit']:
                console.print("[yellow]Afslutter Vinkeljernet CLI. På gensyn![/yellow]")
                break
                
            # Handle help command
            elif command == 'hjælp':
                display_help()
                
            # Handle show profiles
            elif command == 'profiler':
                profiles = get_profile_names()
                if not profiles:
                    console.print("[yellow]Ingen profiler fundet. Check config/ mappen.[/yellow]")
                else:
                    table = Table(title="Tilgængelige Profiler")
                    table.add_column("Profilnavn", style="cyan")
                    for profile in profiles:
                        table.add_row(profile)
                    console.print(table)
            
            # Handle generate command with simple mode
            elif command == 'generer' and len(cmd_parts) >= 3:
                topic = cmd_parts[1]
                profile_name = cmd_parts[2]
                
                console.print(f"[bold]Genererer vinkler for '{topic}' med profil '{profile_name}'...[/bold]")
                
                # Construct a proper file path for the profile
                if not profile_name.endswith('.yaml'):
                    profile_path = f"config/{profile_name}.yaml"
                else:
                    profile_path = profile_name
                
                # Create a simpler processing object
                class Args:
                    pass
                
                args = Args()
                args.emne = topic
                args.profil = profile_path
                args.output = None
                args.dev_mode = False
                args.clear_cache = False
                args.bypass_cache = False
                args.format = "json"
                args.show_circuits = False
                args.reset_circuits = False
                args.debug = False
                
                try:
                    # Call the main processing function
                    await process_generation_request(args)
                except Exception as e:
                    console.print(f"[bold red]Fejl ved generering af vinkler:[/bold red] {e}")
            
            # Handle unknown commands
            else:
                console.print(f"[yellow]Ukendt kommando: {user_input}[/yellow]")
                console.print("Skriv [bold]hjælp[/bold] for at se tilgængelige kommandoer.")
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Operation afbrudt. Skriv 'afslut' for at lukke programmet.[/yellow]")
            continue
        except EOFError:
            console.print("\n[yellow]Afslutter Vinkeljernet CLI. På gensyn![/yellow]")
            break
        except Exception as e:
            console.print(f"[bold red]Fejl:[/bold red] {str(e)}")


def main() -> None:
    """
    Main function that orchestrates the application flow.
    """
    try:
        args = parse_arguments()
        
        # Check if we should run in interactive mode
        if args.interactive:
            asyncio.run(run_interactive_cli())
        else:
            # Traditional CLI mode
            asyncio.run(main_async())
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Program afbrudt af bruger.[/yellow]")
        # Do a final attempt to clean up resources
        try:
            asyncio.run(shutdown_api_client())
        except:
            pass
        sys.exit(0)
    except ValueError as e:
        # Håndter ValueError separat, da disse ofte er forventede fejl
        console.print(f"\n[bold red]Fejl:[/bold red] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]Uventet fejl:[/bold red] {e}")
        console.print("[yellow]Dette er sandsynligvis en bug i programmet. Indsend venligst en fejlrapport.[/yellow]")
        
        # Do a final attempt to clean up resources
        try:
            asyncio.run(shutdown_api_client())
        except:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()