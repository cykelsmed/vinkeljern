"""
Main entry point for the Vinkeljernet application.

This module handles command-line arguments and orchestrates the application flow.
"""

import argparse
from argparse import Namespace
import sys
import yaml
from pathlib import Path
from rich import print as rprint

# Import configuration components
try:
    from config import OPENAI_API_KEY, PERPLEXITY_API_KEY
except ValueError as e:
    rprint(f"[bold red]Fejl ved indlæsning af API nøgler:[/bold red] {e}")
    sys.exit(1)

from config_loader import load_and_validate_profile


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
    
    return parser.parse_args()


def main() -> None:
    """
    Main function that orchestrates the application flow.
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # Print start message
    rprint("[bold blue]Starter Vinkeljernet...[/bold blue]")
    
    # Display received arguments
    rprint("[bold blue]Vinkeljernet starter med følgende parametre:[/bold blue]")
    rprint(f"  [green]Nyhedsemne:[/green] {args.emne}")
    rprint(f"  [green]Profil fil:[/green] {args.profil}")
    rprint(f"  [green]Output fil:[/green] {args.output if args.output else 'Ingen (viser i terminalen)'}")
    
    # Confirm API keys are loaded
    rprint("[green]✓[/green] API nøgler er indlæst")
    
    # Load and validate profile
    profile_path = Path(args.profil)
    try:
        rprint(f"[blue]Indlæser profil fra {profile_path}...[/blue]")
        profile = load_and_validate_profile(profile_path)
        rprint("[green]✓[/green] Profil er indlæst og valideret")
        
        # Display some profile information as confirmation
        rprint("[blue]Profil information:[/blue]")
        rprint(f"  [green]Kerneprincipper:[/green]")
        for p in profile.kerneprincipper:
            for k, v in p.items():
                rprint(f"    - {k}: {v}")
        rprint(f"  [green]Tone og stil:[/green] {profile.tone_og_stil}")
        rprint(f"  [green]Antal nyhedsprioriteringer:[/green] {len(profile.nyhedsprioritering)}")
        rprint(f"  [green]Antal fokusområder:[/green] {len(profile.fokusområder)}")
    
    except FileNotFoundError as e:
        rprint(f"[bold red]Fejl:[/bold red] Profil filen blev ikke fundet: {e}")
        sys.exit(1)
    except yaml.YAMLError as e:
        rprint(f"[bold red]Fejl:[/bold red] Kunne ikke læse YAML profil: {e}")
        sys.exit(1)
    except (ValueError, TypeError) as e:
        rprint(f"[bold red]Fejl:[/bold red] Profilvalidering fejlede: {e}")
        sys.exit(1)
    
    # Placeholders for next steps
    rprint("[blue]TODO: Indhent information om emnet...[/blue]")
    rprint("[blue]TODO: Generer vinkler baseret på emne og profil...[/blue]")
    rprint("[blue]TODO: Præsenter resultater...[/blue]")


if __name__ == "__main__":
    main()