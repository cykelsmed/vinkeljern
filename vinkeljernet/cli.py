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
        
        # Import API client wrapper functions
        from api_clients_wrapper import initialize_api_client, shutdown_api_client, ensure_event_loop
        
        # Create and manage a single event loop for the entire application
        loop = ensure_event_loop()
        
        # Check if we should run in interactive mode
        if args.interactive:
            try:
                # Initialize the API client in this loop
                loop.run_until_complete(initialize_api_client())
                
                # Run the interactive CLI in the same loop
                loop.run_until_complete(run_interactive_cli(args))
            finally:
                # Make sure we clean up properly
                loop.run_until_complete(shutdown_api_client())
                # Don't close the loop yet, as we might use it again
        else:
            try:
                # Initialize the API client in this loop
                loop.run_until_complete(initialize_api_client())
                
                # Run standard mode in the same loop
                loop.run_until_complete(run_standard_mode(args))
            finally:
                # Make sure we clean up properly
                loop.run_until_complete(shutdown_api_client())
                # Don't close the loop yet, as we might use it again
                
        # Now we can safely close the loop
        if not loop.is_closed():
            loop.close()
            
    except KeyboardInterrupt:
        # Handle keyboard interrupt
        rprint("\n[yellow]Program afbrudt af bruger.[/yellow]")
        
        # Clean up sessions even on keyboard interrupt
        try:
            from api_clients_wrapper import shutdown_api_client, ensure_event_loop
            loop = ensure_event_loop()
            if not loop.is_closed():
                loop.run_until_complete(shutdown_api_client())
                loop.close()
        except Exception as e:
            logger.error(f"Error during cleanup after keyboard interrupt: {e}")
            
        sys.exit(0)
    except ValueError as e:
        # Handle ValueError separately, as these are often expected errors
        rprint(f"\n[bold red]Fejl:[/bold red] {e}")
        sys.exit(1)
    except Exception as e:
        rprint(f"\n[bold red]Uventet fejl:[/bold red] {e}")
        rprint("[yellow]Dette er sandsynligvis en bug i programmet. Indsend venligst en fejlrapport.[/yellow]")
        logger.exception("Unexpected error in main")
        
        # Try to clean up sessions even on unexpected errors
        try:
            from api_clients_wrapper import shutdown_api_client, ensure_event_loop
            loop = ensure_event_loop()
            if not loop.is_closed():
                loop.run_until_complete(shutdown_api_client())
                loop.close()
        except Exception as cleanup_error:
            logger.error(f"Error during cleanup after exception: {cleanup_error}")
            
        sys.exit(1)


if __name__ == "__main__":
    main()