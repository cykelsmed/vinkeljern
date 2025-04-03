"""
Command-line interface for the Vinkeljernet application.

This module handles command-line arguments and dispatches to the appropriate mode.
"""

import argparse
import asyncio
import logging
import sys
from argparse import Namespace
from pathlib import Path
from typing import Optional

from rich import print as rprint

from vinkeljernet.standard_mode import run_standard_mode
from vinkeljernet.interactive_mode import run_interactive_cli, run_simple_cli
from vinkeljernet.utils import setup_logging

# Configure logger
logger = logging.getLogger("vinkeljernet.cli")


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


def main() -> None:
    """
    Main function that orchestrates the application flow.
    """
    try:
        # Parse command-line arguments
        args = parse_arguments()
        
        # Set up logging
        setup_logging(debug=args.debug)
        logger.info("Vinkeljernet starting up")
        
        # Check if we should run in interactive mode
        if args.interactive:
            asyncio.run(run_interactive_cli(args))
        else:
            # Traditional CLI mode
            asyncio.run(run_standard_mode(args))
            
    except KeyboardInterrupt:
        rprint("\n[yellow]Program afbrudt af bruger.[/yellow]")
        sys.exit(0)
    except ValueError as e:
        # Handle ValueError separately, as these are often expected errors
        rprint(f"\n[bold red]Fejl:[/bold red] {e}")
        sys.exit(1)
    except Exception as e:
        rprint(f"\n[bold red]Uventet fejl:[/bold red] {e}")
        rprint("[yellow]Dette er sandsynligvis en bug i programmet. Indsend venligst en fejlrapport.[/yellow]")
        sys.exit(1)


if __name__ == "__main__":
    main()