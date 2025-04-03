"""
Interactive CLI mode for the Vinkeljernet application.

This module provides an interactive CLI interface using prompt_toolkit.
"""

import asyncio
import logging
import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union

from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter, FuzzyCompleter
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.shortcuts import clear, message_dialog, button_dialog
from prompt_toolkit.application import run_in_terminal

from models import RedaktionelDNA
from config_loader import load_and_validate_profile
from config_manager import get_config
from vinkeljernet.core import process_generation_request
from vinkeljernet.ui_utils import (
    display_welcome_panel,
    display_profile_info,
    display_angles_panels,
    display_help_text
)
from vinkeljernet.utils import (
    get_available_profiles,
    get_profile_names,
    get_default_profile,
    clear_api_cache,
    reset_circuit_breakers,
    get_circuit_stats
)

# Configure logger
logger = logging.getLogger("vinkeljernet.interactive_mode")

# Create console
console = Console()


def display_welcome_message() -> None:
    """Display a welcome message for the interactive CLI."""
    clear()
    display_welcome_panel("Vinkeljernet - Interactive")
    console.print(
        "\n[bold]Welcome to Vinkeljernet CLI![/bold]\n\n"
        "This tool helps generate news angles based on editorial DNA profiles.\n"
        "Use commands to explore and generate angles. Type [bold]help[/bold] to see available commands.\n"
    )


def display_help() -> None:
    """Display help information for the interactive CLI."""
    help_text = """
    # Vinkeljernet Commands
    
    ## Basic
    - `help` - Show this help text
    - `exit` or `quit` - Exit the program
    - `clear` - Clear the screen
    
    ## Profiles
    - `profiles` - Show available editorial profiles
    - `show profile <name>` - Show details about a specific profile
    
    ## Generation
    - `generate <topic> <profile>` - Generate angles for a topic with a specific profile
       Example: `generate climate_change dr_profil`
    
    ## Configuration
    - `toggle debug` - Toggle debug mode
    - `toggle cache` - Toggle caching
    - `clear cache` - Clear cache files
    
    ## System
    - `status` - Show system status including circuit breakers
    """
    
    display_help_text(help_text)


async def run_interactive_cli(args: Namespace) -> None:
    """
    Run the interactive CLI interface.
    
    Args:
        args: Command-line arguments
    """
    # Setup the prompt session with history
    history_file = Path.home() / ".vinkeljernet_history"
    
    # Check if we're running in an environment that doesn't support prompt_toolkit
    try:
        session = PromptSession(history=FileHistory(str(history_file)))
    except Exception as e:
        # Fallback simple mode for non-interactive environments
        console.print(f"[yellow]Could not start interactive mode: {e}[/yellow]")
        console.print("[yellow]Using simple command-line mode instead.[/yellow]")
        return await run_simple_cli(args)
    
    # Set up the style
    style = Style.from_dict({
        'prompt': 'ansicyan bold',
        'command': 'ansigreen',
        'param': 'ansiyellow',
    })
    
    # Set up auto-completion
    commands = ['help', 'exit', 'quit', 'clear', 'profiles', 
                'show profile', 'generate', 'toggle debug', 
                'toggle cache', 'clear cache', 'status']
    profile_names = get_profile_names()
    all_words = commands + profile_names + ['climate_change', 'politics', 'economy', 'technology']
    word_completer = WordCompleter(all_words, ignore_case=True)
    
    # Cache some settings
    debug_mode = args.debug
    bypass_cache = args.bypass_cache
    
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
            if command in ['exit', 'quit']:
                console.print("[yellow]Exiting Vinkeljernet CLI. Goodbye![/yellow]")
                break
                
            # Handle help command
            elif command == 'help':
                display_help()
                
            # Handle clear screen
            elif command == 'clear':
                clear()
                display_welcome_message()
                
            # Handle show profiles
            elif command == 'profiles':
                display_profiles()
                    
            # Handle view profile
            elif command == 'show' and len(cmd_parts) >= 3 and cmd_parts[1] == 'profile':
                profile_name = cmd_parts[2]
                display_profile(profile_name)
                    
            # Handle generate command
            elif command == 'generate' and len(cmd_parts) >= 3:
                topic = cmd_parts[1]
                profile_name = cmd_parts[2]
                await generate_angles_interactive(topic, profile_name, debug_mode, bypass_cache)
            
            # Handle toggle debug command
            elif user_input.strip().lower() == 'toggle debug':
                debug_mode = not debug_mode
                console.print(f"[blue]Debug mode is now {'[green]enabled' if debug_mode else '[red]disabled'}[/blue]")
                
            # Handle toggle cache command
            elif user_input.strip().lower() == 'toggle cache':
                bypass_cache = not bypass_cache
                console.print(f"[blue]Cache bypass is now {'[green]enabled' if bypass_cache else '[red]disabled'}[/blue]")
                
            # Handle clear cache command
            elif user_input.strip().lower() == 'clear cache':
                num_cleared = clear_api_cache()
                console.print(f"[yellow]Cache cleared: {num_cleared} files deleted[/yellow]")
                
            # Handle status command
            elif command == 'status':
                display_status(debug_mode, bypass_cache)
                
            # Unknown command
            else:
                console.print(f"[yellow]Unknown command: {user_input}[/yellow]")
                console.print("Type [bold]help[/bold] to see available commands.")
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Operation interrupted. Type 'exit' to close the program.[/yellow]")
            continue
        except EOFError:
            console.print("\n[yellow]Exiting Vinkeljernet CLI. Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")


def display_profiles() -> None:
    """Display available profiles."""
    profiles = get_profile_names()
    
    if not profiles:
        console.print("[yellow]No profiles found. Check the config/ directory.[/yellow]")
    else:
        table = Table(title="Available Profiles")
        table.add_column("Profile Name", style="cyan")
        for profile in profiles:
            table.add_row(profile)
        console.print(table)


def display_profile(profile_name: str) -> None:
    """
    Display details about a specific profile.
    
    Args:
        profile_name: Name of the profile to display
    """
    profile_path = f"config/{profile_name}.yaml"
    if not profile_name.endswith('.yaml'):
        profile_path = f"config/{profile_name}.yaml"
    
    try:
        profile = load_and_validate_profile(Path(profile_path))
        
        # Display profile
        console.print(f"[bold blue]Profile:[/bold blue] {profile_name}")
        display_profile_info(profile)
        
    except FileNotFoundError:
        console.print(f"[bold red]Error:[/bold red] Profile '{profile_name}' not found")
    except Exception as e:
        console.print(f"[bold red]Error loading profile:[/bold red] {e}")


async def generate_angles_interactive(
    topic: str,
    profile_name: str,
    debug_mode: bool = False,
    bypass_cache: bool = False
) -> None:
    """
    Generate angles interactively.
    
    Args:
        topic: The news topic
        profile_name: The profile name
        debug_mode: Whether debug mode is enabled
        bypass_cache: Whether to bypass cache
    """
    # Check if this is a valid profile
    if not profile_name.endswith('.yaml'):
        profile_path = f"config/{profile_name}.yaml"
    else:
        profile_path = profile_name
    
    try:
        # Create a simple args-like object
        class Args:
            pass
        
        args = Args()
        args.emne = topic
        args.profil = profile_path
        args.output = None
        args.format = "json"
        args.dev_mode = False
        args.clear_cache = False
        args.bypass_cache = bypass_cache
        args.debug = debug_mode
        
        # Show progress message
        console.print(f"[bold blue]Generating angles for topic '{topic}' with profile '{profile_name}'...[/bold blue]")
        await run_in_terminal(lambda: console.print("Processing... (this may take a minute)"))
        
        # Generate angles
        ranked_angles, profile, topic_info = await process_generation_request(
            topic=topic,
            profile_path=profile_path,
            dev_mode=False,
            bypass_cache=bypass_cache,
            debug=debug_mode
        )
        
        # Display the results
        display_angles_panels(ranked_angles)
        
        # Ask if user wants to save the results
        ask_save_output(ranked_angles, profile, topic)
        
    except Exception as e:
        console.print(f"[bold red]Error generating angles:[/bold red] {e}")


def display_status(debug_mode: bool, bypass_cache: bool) -> None:
    """
    Display system status.
    
    Args:
        debug_mode: Whether debug mode is enabled
        bypass_cache: Whether cache bypass is enabled
    """
    # Get circuit breaker stats
    stats = get_circuit_stats()
    
    circuit_table = Table(show_header=True, header_style="bold blue")
    circuit_table.add_column("API", style="dim")
    circuit_table.add_column("State")
    circuit_table.add_column("Successful calls")
    circuit_table.add_column("Failed calls")
    circuit_table.add_column("Consecutive failures")
    
    for name, data in stats.items():
        state_style = "green" if data["state"] == "closed" else "red" if data["state"] == "open" else "yellow"
        circuit_table.add_row(
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
    console.print(f"Cache bypass: {'[red]Enabled' if bypass_cache else '[green]Disabled'}")
    console.print(f"Debug mode: {'[green]Enabled' if debug_mode else '[red]Disabled'}")


def ask_save_output(
    angles: List[Dict[str, Any]],
    profile: RedaktionelDNA,
    topic: str
) -> None:
    """
    Ask if the user wants to save the output.
    
    Args:
        angles: The generated angles
        profile: The editorial DNA profile
        topic: The news topic
    """
    try:
        from formatters import format_angles
        
        # Extract profile name
        profile_name = Path(profile.navn).stem if hasattr(profile, 'navn') else "unknown"
        
        # Ask if user wants to save
        save = button_dialog(
            title="Save Result",
            text="Do you want to save the generated angles to a file?",
            buttons=[
                ("Yes", True),
                ("No", False),
            ],
        ).run()
        
        if save:
            # Ask for filename and format
            from prompt_toolkit.shortcuts import input_dialog
            
            filename = input_dialog(
                title="Save File",
                text="Enter the filename:",
                default=f"{topic}_{profile_name}.md"
            ).run()
            
            if filename:
                # Determine format based on extension
                extension = Path(filename).suffix.lower()
                if extension == '.json':
                    format_type = 'json'
                elif extension == '.md':
                    format_type = 'markdown'
                elif extension == '.html':
                    format_type = 'html'
                else:
                    format_type = 'markdown'  # Default
                
                # Save the file
                format_angles(
                    angles, 
                    format_type=format_type,
                    profile_name=profile_name,
                    topic=topic,
                    output_path=filename
                )
                
                console.print(f"\n[green]✓[/green] Results saved to {filename} ({format_type} format)")
    except:
        # If there's an error with the prompt_toolkit dialogs, just skip asking
        pass


async def run_simple_cli(args: Namespace) -> None:
    """
    Run a simplified CLI interface for environments where prompt_toolkit doesn't work.
    
    Args:
        args: Command-line arguments
    """
    display_welcome_message()
    display_help()
    
    # Get available profiles
    profiles = get_profile_names()
    console.print("\n[bold blue]Available profiles:[/bold blue]")
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
            if command in ['exit', 'quit']:
                console.print("[yellow]Exiting Vinkeljernet CLI. Goodbye![/yellow]")
                break
                
            # Handle help command
            elif command == 'help':
                display_help()
                
            # Handle show profiles
            elif command == 'profiles':
                display_profiles()
            
            # Handle generate command with simple mode
            elif command == 'generate' and len(cmd_parts) >= 3:
                topic = cmd_parts[1]
                profile_name = cmd_parts[2]
                
                console.print(f"[bold]Generating angles for '{topic}' with profile '{profile_name}'...[/bold]")
                
                # Construct a proper file path for the profile
                if not profile_name.endswith('.yaml'):
                    profile_path = f"config/{profile_name}.yaml"
                else:
                    profile_path = profile_name
                
                try:
                    # Generate angles
                    ranked_angles, profile, topic_info = await process_generation_request(
                        topic=topic,
                        profile_path=profile_path,
                        dev_mode=args.dev_mode,
                        bypass_cache=args.bypass_cache,
                        debug=args.debug
                    )
                    
                    # Display the results
                    display_angles_panels(ranked_angles)
                except Exception as e:
                    console.print(f"[bold red]Error generating angles:[/bold red] {e}")
            
            # Handle unknown commands
            else:
                console.print(f"[yellow]Unknown command: {user_input}[/yellow]")
                console.print("Type [bold]help[/bold] to see available commands.")
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Operation interrupted. Type 'exit' to close the program.[/yellow]")
            continue
        except EOFError:
            console.print("\n[yellow]Exiting Vinkeljernet CLI. Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")