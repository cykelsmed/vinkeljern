#!/usr/bin/env python3
"""
Command Line Interface for managing editorial DNA profiles
"""

import argparse
import os
import sys
import yaml
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax
from rich.panel import Panel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from profile_manager import ProfileManager
from models import RedaktionelDNA

console = Console()

def main():
    parser = argparse.ArgumentParser(description="Værktøj til håndtering af redaktionelle profiler")
    
    subparsers = parser.add_subparsers(dest="command", help="Kommandoer")
    
    # List profiles command
    list_parser = subparsers.add_parser("list", help="Vis liste over tilgængelige profiler")
    list_parser.add_argument("--dir", type=str, default="config", help="Mappe med profiler")
    
    # Show profile command
    show_parser = subparsers.add_parser("show", help="Vis en profil")
    show_parser.add_argument("profile", type=str, help="Profil at vise")
    
    # Create profile command
    create_parser = subparsers.add_parser("create", help="Opret en ny profil")
    create_parser.add_argument("name", type=str, help="Navn på den nye profil")
    create_parser.add_argument("--template", type=str, choices=["tabloid", "broadsheet"], 
                              default="tabloid", help="Skabelon til ny profil")
    create_parser.add_argument("--output", type=str, default=None, 
                              help="Output fil (standard: config/{name}_profil.yaml)")
    
    # Compare profiles command
    compare_parser = subparsers.add_parser("compare", help="Sammenlign to profiler")
    compare_parser.add_argument("profile1", type=str, help="Første profil")
    compare_parser.add_argument("profile2", type=str, help="Anden profil")
    
    args = parser.parse_args()
    
    if args.command == "list":
        list_profiles(args.dir)
    elif args.command == "show":
        show_profile(args.profile)
    elif args.command == "create":
        create_profile(args.name, args.template, args.output)
    elif args.command == "compare":
        compare_profiles(args.profile1, args.profile2)
    else:
        parser.print_help()
    
def list_profiles(directory="config"):
    """List all available profiles"""
    path = Path(directory)
    
    if not path.exists():
        console.print(f"[red]Fejl:[/red] Mappen {directory} findes ikke")
        return
        
    yaml_files = list(path.glob("*.yaml")) + list(path.glob("*.yml"))
    
    if not yaml_files:
        console.print(f"Ingen profiler fundet i {directory}")
        return
        
    table = Table(title="Tilgængelige profiler")
    table.add_column("Profil", style="cyan")
    table.add_column("Sti", style="green")
    table.add_column("Størrelse", style="blue")
    
    for file in sorted(yaml_files):
        table.add_row(file.stem, str(file), f"{file.stat().st_size} bytes")
        
    console.print(table)

def show_profile(profile_path):
    """Show details of a profile"""
    try:
        profile = ProfileManager.load_profile(profile_path)
        
        # Format the values
        table = Table(title=f"Profil: {profile.navn}")
        table.add_column("Attribut", style="cyan")
        table.add_column("Værdi", style="green")
        
        table.add_row("Navn", profile.navn)
        table.add_row("Beskrivelse", profile.beskrivelse)
        
        # Handle tone_og_stil (updated reference)
        tone_og_stil = getattr(profile, "tone_og_stil", "N/A")
        table.add_row("Tone og stil", tone_og_stil)
        
        table.add_row("Kerneprincipper", "\n".join([f"- {p}" for p in profile.kerneprincipper]))
        table.add_row("Fokusområder", "\n".join([f"- {p}" for p in profile.fokusOmrader]))
        table.add_row("No-go områder", "\n".join([f"- {p}" for p in profile.noGoOmrader]))
        
        # Format news criteria priorities
        priorities = []
        for criterion, weight in profile.nyhedsprioritering.items():
            priorities.append(f"- {criterion}: {weight}")
        table.add_row("Nyhedsprioritering", "\n".join(priorities))
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Fejl ved indlæsning af profilen:[/red] {e}")

# Add more functions for create_profile and compare_profiles

if __name__ == "__main__":
    main()