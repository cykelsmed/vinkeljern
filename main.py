"""
Main entry point for the Vinkeljernet application.

This module handles command-line arguments and orchestrates the application flow.
"""
import argparse
import asyncio
import json
import sys
import yaml
import logging
from argparse import Namespace
from pathlib import Path
from typing import List, Dict, Any, Optional

from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from models import RedaktionelDNA  # <--- Added import here

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
from api_clients import fetch_topic_information, generate_angles
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
        required=True,
        help="Det nyhedsemne, der skal genereres vinkler for."
    )
    
    parser.add_argument(
        "--profil", "-p",
        type=str,
        required=True,
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
        "--debug",
        action="store_true",
        help="Aktiver debug tilstand med ekstra output"
    )
    
    return parser.parse_args()


async def main_async() -> None:
    """
    Asynchronous main function that orchestrates the application flow.
    """
    # Parse command-line arguments
    args = parse_arguments()
    
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
    
    if args.clear_cache:
        from cache_manager import clear_cache
        num_cleared = clear_cache()
        console.print(f"[yellow]Cache ryddet: {num_cleared} filer slettet[/yellow]")
    
    if args.reset_circuits:
        from retry_manager import reset_circuit
        reset_circuit("perplexity_api")
        reset_circuit("openai_api")
        console.print("[yellow]Alle circuit breakers nulstillet[/yellow]")

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
    table.add_row("Antal fokusområder", str(len(profile.fokusOmrader)))     # Updated attribute name
    table.add_row("Antal no-go områder", str(len(profile.noGoOmrader)))         # Updated attribute name
    
    console.print(table)
    
    # Get information about the topic with progress spinner
    topic_info = None
    with Progress(
        SpinnerColumn(),
        TextColumn(f"[bold blue]Henter information om \"{args.emne}\"..."),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("Researching", total=None)
        topic_info = await fetch_topic_information(args.emne, dev_mode=args.dev_mode, bypass_cache=args.bypass_cache)
        progress.update(task, completed=True)
    
    if topic_info:
        console.print("[green]✓[/green] Baggrundsinformation indhentet")
    else:
        console.print("[yellow]⚠️ Kunne ikke indhente detaljeret baggrundsinformation[/yellow]")
        console.print("[yellow]  Fortsætter med begrænset kontekst[/yellow]")
        # Set a minimal fallback for topic_info
        topic_info = f"Emnet handler om {args.emne}. Ingen yderligere baggrundsinformation tilgængelig."
    
    # Generate angles with progress spinner
    angles = None
    with Progress(
        SpinnerColumn(),
        TextColumn(f"[bold blue]Genererer vinkler..."),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("Generating", total=None)
        
        # Direct OpenAI API call to bypass any decorators that might inject 'proxies'
        try:
            # Import directly to ensure we're using the correct version
            import os
            import sys
            from prompt_engineering import construct_angle_prompt, parse_angles_from_response
            
            # Use Claude API instead of OpenAI
            print("DEBUG - Using Claude API instead of OpenAI")
            
            # Convert profile into strings for prompt construction
            principper = "\n".join([f"- {p}" for p in profile.kerneprincipper])
            nyhedskriterier = "\n".join([f"- {k}: {v}" for k, v in profile.nyhedsprioritering.items()])
            fokusomrader = "\n".join([f"- {f}" for f in profile.fokusOmrader])
            nogo_omrader = "\n".join([f"- {n}" for n in profile.noGoOmrader]) if profile.noGoOmrader else "Ingen"
            
            # Create the prompt
            prompt = construct_angle_prompt(
                args.emne,
                topic_info,
                principper,
                profile.tone_og_stil,
                fokusomrader,
                nyhedskriterier,
                nogo_omrader
            )
            
            # Use Claude API instead
            import requests
            from config import ANTHROPIC_API_KEY
            
            # Claude API call
            claude_response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01"
                },
                json={
                    "model": "claude-3-opus-20240229",
                    "max_tokens": 2500,
                    "temperature": 0.7,
                    "system": "Du er en erfaren journalist med ekspertise i at udvikle kreative og relevante nyhedsvinkler.",
                    "messages": [{"role": "user", "content": prompt}],
                }
            )
            
            # Parse Claude response
            if claude_response.status_code != 200:
                print(f"Claude API fejl: {claude_response.status_code}: {claude_response.text}")
                raise ValueError(f"Claude API fejl: {claude_response.status_code}")
                
            response_data = claude_response.json()
            print(f"DEBUG - Claude API response: {response_data}")
            response_text = response_data['content'][0]['text']
            print(f"DEBUG - Claude API response text: {response_text[:500]}...")
            angles = parse_angles_from_response(response_text)
            
            # Add perplexity information to each angle
            if angles and isinstance(angles, list):
                perplexity_extract = topic_info[:1000] + ("..." if len(topic_info) > 1000 else "")
                
                # Generate source suggestions using Claude
                source_suggestions_prompt = f"""
                Baseret på emnet '{args.emne}', giv en kort liste med 3-5 relevante og troværdige danske kilder, 
                som en journalist kunne bruge til research. Inkluder officielle hjemmesider, forskningsinstitutioner, 
                eksperter og organisationer. Formater som en simpel punktopstilling med korte beskrivelser på dansk.
                Hold dit svar under 250 ord og fokuser kun på de mest pålidelige kilder.
                """
                
                # Claude API call for source suggestions
                try:
                    source_response = requests.post(
                        "https://api.anthropic.com/v1/messages",
                        headers={
                            "Content-Type": "application/json",
                            "x-api-key": ANTHROPIC_API_KEY,
                            "anthropic-version": "2023-06-01"
                        },
                        json={
                            "model": "claude-3-haiku-20240307",
                            "max_tokens": 500,
                            "temperature": 0.2,
                            "system": "Du er en hjælpsom researchassistent med stort kendskab til troværdige danske kilder. Du svarer altid på dansk.",
                            "messages": [{"role": "user", "content": source_suggestions_prompt}],
                        }
                    )
                    
                    if source_response.status_code == 200:
                        source_data = source_response.json()
                        source_text = source_data['content'][0]['text']
                        
                        # Add both perplexity info and source suggestions to each angle
                        for angle in angles:
                            if isinstance(angle, dict):
                                angle['perplexityInfo'] = perplexity_extract
                                angle['kildeForslagInfo'] = source_text
                    else:
                        # If source generation fails, just add perplexity info
                        print(f"Failed to generate source suggestions: {source_response.status_code}: {source_response.text}")
                        for angle in angles:
                            if isinstance(angle, dict):
                                angle['perplexityInfo'] = perplexity_extract
                except Exception as e:
                    print(f"Error generating source suggestions: {e}")
                    # If there's an error, just add perplexity info
                    for angle in angles:
                        if isinstance(angle, dict):
                            angle['perplexityInfo'] = perplexity_extract
        except Exception as e:
            console.print(f"[bold red]Error during direct API call:[/bold red] {e}")
            # Fall back to the regular function for logging purposes
            angles = generate_angles(args.emne, topic_info, profile, bypass_cache=args.bypass_cache)
        
        progress.update(task, completed=True)
    
    # Check if we have any angles
    if not angles:
        console.print("[bold red]Ingen vinkler kunne genereres.[/bold red]")
        console.print("[yellow]Mulige årsager:[/yellow]")
        console.print("  - API-fejl ved forbindelse til OpenAI")
        console.print("  - Emnet er for specifikt eller ukendt")
        console.print("  - Profilen er for restriktiv")
        console.print("[yellow]Prøv et andet emne eller kontrollér API-nøglen.[/yellow]")
        sys.exit(1)
    
    console.print(f"[green]✓[/green] Genereret {len(angles)} råvinkler")
    
    # Filter and rank angles with progress spinner
    ranked_angles = None
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Filtrerer og rangerer vinkler..."),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("Filtering", total=None)
        try:
            ranked_angles = safe_process_angles(angles, profile, 5)
            progress.update(task, completed=True)
        except Exception as e:
            progress.update(task, completed=True)
            console.print(f"[bold red]Fejl ved filtrering af vinkler:[/bold red] {e}")
            console.print("[yellow]Forsøger at fortsætte med ufiltrerede vinkler...[/yellow]")
            # Fallback: use the first 5 angles or all if less than 5
            ranked_angles = angles[:min(5, len(angles))]
    
    if not ranked_angles:
        console.print("[bold red]Ingen vinkler tilbage efter filtrering.[/bold red]")
        console.print("[yellow]Emnet matcher muligvis ikke mediets profil, eller alle genererede vinkler rammer no-go områder.[/yellow]")
        sys.exit(1)
    
    console.print(f"[green]✓[/green] Rangeret og filtreret til {len(ranked_angles)} vinkler")
    
    # Present results with nice formatting
    console.print("\n[bold blue]🎯 Anbefalede vinkler:[/bold blue]")
    for i, angle in enumerate(ranked_angles, 1):
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
                ranked_angles, 
                format_type=args.format,
                profile_name=profile_name,
                topic=args.emne,
                output_path=args.output
            )
            
            console.print(f"\n[green]✓[/green] Resultater gemt i {args.output} ({args.format} format)")
        except ImportError:
            # Fallback to JSON if formatter module not available
            with open(args.output, 'w', encoding='utf-8') as outfile:
                json.dump(ranked_angles, outfile, ensure_ascii=False, indent=2)
            console.print(f"\n[green]✓[/green] Resultater gemt i {args.output} (JSON format)")
        except IOError as e:
            console.print(f"\n[bold red]Fejl ved skrivning til fil:[/bold red] {e}")
            console.print(f"[yellow]Tjek om stien eksisterer og om du har skriverettigheder.[/yellow]")
        except Exception as e:
            console.print(f"\n[bold red]Uventet fejl ved skrivning til fil:[/bold red] {e}")
    
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


def main() -> None:
    """
    Main function that orchestrates the application flow.
    """
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        console.print("\n[yellow]Program afbrudt af bruger.[/yellow]")
        sys.exit(0)
    except ValueError as e:
        # Håndter ValueError separat, da disse ofte er forventede fejl
        console.print(f"\n[bold red]Fejl:[/bold red] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]Uventet fejl:[/bold red] {e}")
        console.print("[yellow]Dette er sandsynligvis en bug i programmet. Indsend venligst en fejlrapport.[/yellow]")
        sys.exit(1)


if __name__ == "__main__":
    main()