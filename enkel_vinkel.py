#!/usr/bin/env python
"""
Enkel Vinkelgenerator

Direkte adgang til at generere vinkler uden interaktive prompts.
Bruger vinkeljernet-biblioteket programmatisk.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Opret en console til rich output
from rich.console import Console
console = Console()

# Konfigurer logging
logging.basicConfig(
    filename='vinkeljernet.log',
    filemode='a',  # 'a' to append
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Tilføj projekt-roden til PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import nødvendige moduler
try:
    # Prøv at importere fra de samme steder som main.py
    from models import RedaktionelDNA
    from config_loader import load_and_validate_profile
    from api_clients_wrapper import (
        fetch_topic_information, 
        generate_angles, 
        process_generation_request,
        initialize_api_client,
        shutdown_api_client
    )
    from vinkeljernet.ui_utils import display_angles_panels
except ImportError as e:
    console.print(f"[bold red]Fejl ved import af moduler:[/bold red] {e}")
    console.print("[yellow]Sørg for at køre scriptet fra projektets rodmappe.[/yellow]")
    sys.exit(1)


async def generate_vinkler(emne, profil_sti, fulde_kilder=False, bypass_cache=False, debug=False):
    """
    Generer vinkler for et givet emne med en specifik profil.
    
    Args:
        emne: Emnet at generere vinkler for
        profil_sti: Sti til YAML-profil
        fulde_kilder: Vis detaljerede kildeoplysninger
        bypass_cache: Ignorer cache
        debug: Vis debug output
    """
    try:
        console.print(f"\n🔍 [bold blue]Genererer vinkler for emnet:[/bold blue] {emne}")
        console.print(f"📄 [bold blue]Med redaktionel profil:[/bold blue] {Path(profil_sti).stem}")
        
        # Initialiser API klienten
        await initialize_api_client()
        
        # Indlæs profil
        profil = load_and_validate_profile(profil_sti)
        
        console.print("\n⏳ [bold]Arbejder på at generere vinkler...[/bold] Dette kan tage et par minutter.")
        
        # Brug den forenklede process_generation_request funktion som håndterer alt for os
        vinkler = await process_generation_request(
            emne, 
            profil, 
            bypass_cache=bypass_cache
        )
        
        # Vis resultater
        console.print("\n✅ [bold green]Vinkler genereret![/bold green]")
        display_angles_panels(vinkler, verbose=fulde_kilder)
        
        return vinkler
    
    except Exception as e:
        console.print(f"\n❌ [bold red]Fejl:[/bold red] {e}")
        if debug:
            import traceback
            traceback.print_exc()
        return None
    finally:
        # Oprydning
        await shutdown_api_client()


def main():
    """Hovedfunktion"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Enkel vinkelgenerator uden interaktiv mode"
    )
    
    parser.add_argument(
        "--emne", "-e",
        type=str,
        required=True,
        help="Emnet, der skal genereres vinkler for"
    )
    
    parser.add_argument(
        "--profil", "-p",
        type=str,
        required=True,
        help="Profil til brug (eks: weekendavisen_profil eller komplet sti)"
    )
    
    parser.add_argument(
        "--fulde-kilder",
        action="store_true",
        help="Vis alle detaljer om ekspertkilder"
    )
    
    parser.add_argument(
        "--bypass-cache",
        action="store_true",
        help="Ignorer cache og hent friske data"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Vis debug information"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Gem output til fil"
    )
    
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "markdown", "html"],
        default="markdown",
        help="Format til output fil"
    )
    
    args = parser.parse_args()
    
    # Håndter relative profil-navne
    profil_sti = args.profil
    if not os.path.exists(profil_sti):
        # Prøv at tilføje config/ og .yaml hvis nødvendigt
        if not profil_sti.endswith('.yaml'):
            profil_sti = f"{profil_sti}.yaml"
        if not profil_sti.startswith('config/'):
            profil_sti = f"config/{profil_sti}"
        
        if not os.path.exists(profil_sti):
            console.print(f"[bold red]❌ Profil ikke fundet:[/bold red] {args.profil}")
            console.print(f"[yellow]Prøvede også:[/yellow] {profil_sti}")
            return 1
    
    try:
        # Kør generator
        vinkler = asyncio.run(
            generate_vinkler(
                args.emne, 
                profil_sti, 
                fulde_kilder=args.fulde_kilder,
                bypass_cache=args.bypass_cache,
                debug=args.debug
            )
        )
        
        # Gem output til fil hvis påkrævet
        if vinkler and args.output:
            try:
                from formatters import format_angles
                
                format_angles(
                    vinkler,
                    format_type=args.format,
                    profile_name=Path(profil_sti).stem,
                    topic=args.emne,
                    output_path=args.output
                )
                
                console.print(f"\n[green]✓[/green] Resultater gemt i [bold]{args.output}[/bold] ({args.format} format)")
            except ImportError:
                # Prøv en anden placering
                sys.path.append('./vinkeljernet')
                from vinkeljernet.formatters import format_angles
                
                format_angles(
                    vinkler,
                    format_type=args.format,
                    profile_name=Path(profil_sti).stem,
                    topic=args.emne,
                    output_path=args.output
                )
                
                console.print(f"\n[green]✓[/green] Resultater gemt i [bold]{args.output}[/bold] ({args.format} format)")
        
        return 0
    
    except KeyboardInterrupt:
        console.print("\n[yellow]🛑 Afbrudt af bruger[/yellow]")
        return 130
    
    except Exception as e:
        console.print(f"\n[bold red]❌ Fejl:[/bold red] {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 