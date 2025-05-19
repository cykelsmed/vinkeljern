#!/usr/bin/env python
"""
Wizard til Vinkeljernet

Dette script vejleder brugeren gennem processen med at vælge parametre
og køre vinkeljern med de rigtige argumenter.
"""

import os
import glob
import subprocess
from pathlib import Path

def clear_screen():
    """Ryd terminal-skærmen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Udskriv header til wizarden"""
    print("\n" + "=" * 70)
    print("VINKELJERNET WIZARD".center(70))
    print("=" * 70 + "\n")

def get_available_profiles():
    """Hent alle tilgængelige profiler"""
    profiles = glob.glob("config/*.yaml")
    # Filter konfigurations-filer fra
    profiles = [p for p in profiles if "config.development" not in p 
                and "config.production" not in p 
                and "config.testing" not in p]
    return profiles

def select_profile(profiles):
    """Lad brugeren vælge en profil fra listen"""
    print("\n🔍 VÆLG PROFIL\n")
    
    # Vis alle profiler med numre
    for i, profile in enumerate(profiles, 1):
        # Fjern 'config/' og '.yaml' fra visningen for at gøre det mere læsbart
        display_name = profile.replace("config/", "").replace(".yaml", "")
        print(f"  {i}. {display_name}")
    
    # Lad brugeren vælge ved nummer
    while True:
        try:
            choice = input("\nVælg profilnummer: ")
            idx = int(choice) - 1
            if 0 <= idx < len(profiles):
                return profiles[idx]
            print("Ugyldigt nummer, prøv igen.")
        except ValueError:
            print("Indtast venligst et tal.")

def get_topic():
    """Lad brugeren indtaste et emne"""
    print("\n📝 INDTAST EMNE\n")
    
    while True:
        topic = input("Indtast dit emne (f.eks. 'klimaforandringer', 'kunstig intelligens'): ")
        if topic.strip():
            return topic
        print("Emne kan ikke være tomt, prøv igen.")

def get_additional_options():
    """Lad brugeren vælge ekstra indstillinger"""
    print("\n⚙️ EKSTRA INDSTILLINGER\n")
    
    options = []
    
    # Vælg kørselstilstand
    print("Kørselstilstand:")
    print("  1. Standard (kommandolinje) [ANBEFALET] - Kører automatisk uden yderligere interaktion")
    print("  2. Interaktiv - Starter en interaktiv terminal med flere kommandoer")
    
    while True:
        mode_choice = input("\nVælg kørselstilstand (1/2) [Standard=1]: ")
        if mode_choice == "" or mode_choice == "1":
            # Standard tilstand - ingen flag tilføjes
            break
        elif mode_choice == "2":
            # Interaktiv tilstand
            options.append("--interactive")
            break
        else:
            print("Ugyldigt valg, vælg 1 eller 2.")
    
    # Mulighed for fulde kildedetaljer
    if input("\nVil du se fulde kildedetaljer? (j/n): ").lower().startswith('j'):
        options.append("--fulde-kilder")
    
    # Mulighed for at gemme output til fil
    if input("Vil du gemme resultatet til en fil? (j/n): ").lower().startswith('j'):
        file_format = "markdown"  # Standard
        format_choice = input("Vælg format (markdown, json, html) [standard: markdown]: ").lower()
        if format_choice in ["markdown", "json", "html"]:
            file_format = format_choice
        
        filename = input(f"Indtast filnavn [standard: resultat.{file_format}]: ")
        if not filename:
            filename = f"resultat.{file_format}"
        
        options.append(f"--output {filename}")
        options.append(f"--format {file_format}")
    
    # Mulighed for at ignorere cache
    if input("Vil du ignorere cache og hente frisk data? (j/n): ").lower().startswith('j'):
        options.append("--bypass-cache")
    
    # Mulighed for debug tilstand
    if input("Vil du aktivere debug tilstand? (j/n): ").lower().startswith('j'):
        options.append("--debug")
    
    return options

def build_and_run_command(profile_path, topic, options):
    """Byg den endelige kommando og kør programmet"""
    print("\n🚀 KØRER VINKELJERNET\n")
    
    # Indstil PYTHONPATH for at sikre imports virker
    import sys
    sys.path.append('.')
    
    # Import nødvendige komponenter direkte
    import asyncio
    from vinkeljernet.api_wrapper import initialize_api_client
    from vinkeljernet.redaktionel_dna import RedaktionelDNA
    from vinkeljernet.vinkelgenerator import generate_angles
    from vinkeljernet.ui_utils import display_angles_panels
    
    # Opret en object der indeholder alle de argumenter, vi skal bruge
    class Args:
        pass
    
    args = Args()
    args.emne = topic
    args.profil = profile_path
    args.output = None
    args.dev_mode = False
    args.bypass_cache = "--bypass-cache" in options
    args.format = "json"
    args.debug = "--debug" in options
    args.fulde_kilder = "--fulde-kilder" in options
    
    # For output fil
    for opt in options:
        if opt.startswith("--output "):
            args.output = opt.split(" ", 1)[1].strip()
        elif opt.startswith("--format "):
            args.format = opt.split(" ", 1)[1].strip()
    
    # Vis kommandoen (som hvis den blev kørt fra terminalen)
    cmd_parts = [f"python main.py --emne \"{topic}\" --profil {profile_path}"]
    if options:
        cmd_parts.append(" ".join(options))
    print(f"Kommando der simuleres: {' '.join(cmd_parts)}\n")
    
    # Kør direkte vinkelgenerator uden at gå gennem hovedmenuen
    print("Genererer vinkler - det kan tage et par minutter...\n")
    
    try:
        # Kør programmet
        async def run_generation():
            # Initialiser API klienten
            await initialize_api_client()
            
            # Indlæs profil og generer vinkler
            profile = RedaktionelDNA.from_yaml(profile_path)
            angles = await generate_angles(topic, profile, bypass_cache=args.bypass_cache)
            
            # Vis resultater
            print("\n[bold green]Viser genererede vinkler:[/bold green]")
            display_angles_panels(angles, verbose=args.fulde_kilder)
            
            # Gem til fil hvis angivet
            if args.output:
                from vinkeljernet.formatters import format_angles
                format_angles(
                    angles,
                    format_type=args.format,
                    profile_name=Path(args.profil).stem,
                    topic=args.emne,
                    output_path=args.output
                )
                print(f"\n✓ Resultater gemt i {args.output} ({args.format} format)")
        
        # Kør asynkront
        asyncio.run(run_generation())
        
    except KeyboardInterrupt:
        print("\nAfbrudt af bruger")
    except Exception as e:
        print(f"\n⚠️ Fejl under kørsel: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()

def main():
    """Hovedfunktion der styrer wizard-flowet"""
    try:
        clear_screen()
        print_header()
        
        print("Denne wizard vil guide dig gennem opsætningen af Vinkeljernet.")
        print("Du vil blive bedt om at vælge en profil, indtaste et emne og vælge")
        print("ekstra indstillinger, før programmet køres.")
        
        # Sørg for at vi er i det rette miljø
        if not os.path.exists("main.py"):
            print("\n⚠️ FEJL: main.py blev ikke fundet i den aktuelle mappe.")
            print("Sørg for at køre denne wizard fra projektets rodmappe.")
            return
        
        # Tjek om der er profiler tilgængelige
        profiles = get_available_profiles()
        if not profiles:
            print("\n⚠️ FEJL: Ingen profiler fundet i config/")
            return
        
        # Vælg profil
        profile_path = select_profile(profiles)
        
        # Indtast emne
        topic = get_topic()
        
        # Vælg ekstra indstillinger
        options = get_additional_options()
        
        # Bekræft kørsel
        print("\n✅ OPSÆTNING KOMPLET\n")
        print(f"Profil: {profile_path}")
        print(f"Emne: {topic}")
        print(f"Ekstra indstillinger: {' '.join(options) if options else 'Ingen'}")
        
        if input("\nKlar til at køre Vinkeljernet? (j/n): ").lower().startswith('j'):
            build_and_run_command(profile_path, topic, options)
        else:
            print("\nAfbryder. Du kan starte wizard'en igen når du er klar.")
    
    except KeyboardInterrupt:
        print("\n\nAfbrudt af bruger. Afslutter.")
    except Exception as e:
        print(f"\n⚠️ En uventet fejl opstod: {e}")

if __name__ == "__main__":
    main() 