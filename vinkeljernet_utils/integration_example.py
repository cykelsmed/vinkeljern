"""
Integration eksempel for brug af vinkeljernet_utils i main.py.

Dette eksempel viser, hvordan man integrerer vinkeljernet_utils-pakken i den
eksisterende Vinkeljernet-kodebase. Det demonstrerer både tilføjelsen af de nye
funktioner og opdateringen af eksisterende funktioner.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable

# 1. Importer vinkeljernet_utils-pakken
from vinkeljernet_utils import (
    setup,
    robust_json_parse,
    safe_parse_json,
    get_expert_sources,
    validate_expert_sources,
    get_performance_stats
)

# 2. Importer eksisterende Vinkeljernet-komponenter
# Dette er et eksempel - i den virkelige implementation skal du importere
# de faktiske komponenter fra din kodebase
try:
    # Eksisterende importeringer fra Vinkeljernet
    from config import OPENAI_API_KEY, PERPLEXITY_API_KEY, ANTHROPIC_API_KEY
    from models import RedaktionelDNA
    from config_loader import load_and_validate_profile
    from api_clients_optimized import (
        fetch_topic_information,
        generate_angles,
        process_generation_request,
        initialize_api_client,
        shutdown_api_client
    )
    from angle_processor import filter_and_rank_angles
except ImportError:
    print("Note: Dette er kun et eksempel - virkelige importer ikke tilgængelige")
    # Mock-klasser til eksempel
    class RedaktionelDNA:
        pass


# 3. Konfigurer vinkeljernet_utils ved opstart
def setup_vinkeljernet_utils():
    """Konfigurerér vinkeljernet_utils-pakken."""
    config = setup(
        debug=True,  # Brug debug til udvikling, False til produktion
        log_level="INFO",
        cache_enabled=True,
        timeout=30
    )
    return config


# 4. Opdater process_generation_request funktionen til at bruge robust JSON-parsing
async def enhanced_process_generation_request(
    topic: str,
    profile: RedaktionelDNA,
    bypass_cache: bool = False,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> List[Dict[str, Any]]:
    """
    Forbedret version af process_generation_request med robust JSON-parsing og
    ekspertkilde-håndtering.
    
    Args:
        topic: Nyhedsemne at generere vinkler for
        profile: Redaktionel DNA-profil
        bypass_cache: Om cache skal ignoreres
        progress_callback: Callback-funktion til fremskridtsrapportering
        
    Returns:
        Liste af genererede vinkler
    """
    # Initialiser API-klienten
    await initialize_api_client()
    
    try:
        # Opdater fremskridtsmåler
        if progress_callback:
            await progress_callback(5)
        
        # 1. Indhent baggrundsinformation om emnet
        topic_info = await fetch_topic_information(
            topic,
            bypass_cache=bypass_cache,
            progress_callback=progress_callback
        )
        
        if not topic_info:
            topic_info = f"Emnet handler om {topic}."
        
        if progress_callback:
            await progress_callback(25)
        
        # 2. Generer vinkler med API
        response_text = await generate_angles(
            topic,
            topic_info,
            profile,
            bypass_cache=bypass_cache
        )
        
        if progress_callback:
            await progress_callback(50)
        
        # 3. Brug robust JSON-parsing til at udtrække vinkler fra responsen
        angles, error = robust_json_parse(response_text, context="angle generation")
        
        if error:
            logging.error(f"Error parsing angles: {error}")
            # Prøv at redde situationen med en mere aggressiv parsing
            logging.info("Attempting to recover with more aggressive parsing...")
            
            # Brug safe_parse_json som sidste udvej
            fallback_data = safe_parse_json(
                response_text,
                context="angle generation fallback",
                fallback={"vinkler": []}
            )
            
            if "vinkler" in fallback_data and isinstance(fallback_data["vinkler"], list):
                angles = fallback_data["vinkler"]
            elif "angles" in fallback_data and isinstance(fallback_data["angles"], list):
                angles = fallback_data["angles"]
            else:
                # Lav en nødløsning hvis vi ikke kan parse
                angles = [
                    {
                        "overskrift": f"Vinkel om {topic}",
                        "beskrivelse": "Kunne ikke generere detaljeret vinkel."
                    }
                ]
        
        if progress_callback:
            await progress_callback(60)
        
        # 4. Filtrér og rangér vinkler
        ranked_angles = filter_and_rank_angles(angles, profile, 5)
        
        if progress_callback:
            await progress_callback(70)
        
        # 5. Beriging af vinklerne med ekspertkilder og baggrundsinformation
        expert_source_tasks = []
        
        # Tilføj baggrundsinformation
        for angle in ranked_angles:
            angle["baggrundsinformation"] = topic_info[:1000]
        
        # Start ekspertkilde-opgaver for top-3 vinkler
        for i, angle in enumerate(ranked_angles[:3]):
            task = asyncio.create_task(
                get_expert_sources(
                    topic=topic,
                    angle_headline=angle.get("overskrift", ""),
                    angle_description=angle.get("beskrivelse", ""),
                    api_client=None,  # Skal erstattes med din API-klient
                    bypass_cache=bypass_cache
                )
            )
            expert_source_tasks.append((i, task))
        
        # Vent på at ekspertkilde-opgaverne fuldføres
        for i, task in enumerate(expert_source_tasks):
            idx, coroutine = task
            try:
                expert_sources = await coroutine
                
                # Konverter til dictionary og tilføj til vinkel
                if expert_sources and idx < len(ranked_angles):
                    # Valider resultatet
                    valid, issues = validate_expert_sources(expert_sources)
                    if not valid:
                        logging.warning(f"Expert source validation issues: {issues}")
                    
                    # Tilføj til vinkel
                    ranked_angles[idx]["ekspertKilder"] = expert_sources.to_dict()
                    ranked_angles[idx]["harEkspertKilder"] = True
            except Exception as e:
                logging.error(f"Error fetching expert sources: {e}")
                # Tilføj tomme kildelister
                if idx < len(ranked_angles):
                    ranked_angles[idx]["ekspertKilder"] = {
                        "experts": [],
                        "institutions": [],
                        "data_sources": [],
                        "error": str(e)
                    }
                    ranked_angles[idx]["harEkspertKilder"] = False
        
        if progress_callback:
            await progress_callback(100)
        
        # 6. Vis performance-statistik
        stats = get_performance_stats()
        logging.info(f"JSON Parsing success rate: {stats.json_parsing_success_rate:.1f}%")
        logging.info(f"Cache hit rate: {stats.cache_hit_rate:.1f}%")
        logging.info(f"Expert sources success rate: {stats.expert_sources_success_rate:.1f}%")
        
        return ranked_angles
        
    finally:
        # Ryd op og luk forbindelser
        await shutdown_api_client()


# Eksempelkode til at køre funktionen
async def example_usage():
    """Eksempel på brug af forbedret funktionalitet."""
    # Konfigurer vinkeljernet_utils
    setup_vinkeljernet_utils()
    
    # Mock en simpel RedaktionelDNA-instans (erstat med rigtige data)
    profile = RedaktionelDNA()
    setattr(profile, "navn", "Eksempelprofil")
    
    # Definer en simpel fremskridtsmåler
    async def progress_callback(percent):
        print(f"Progress: {percent}%")
    
    try:
        # Kald den forbedrede funktion
        result = await enhanced_process_generation_request(
            topic="Klimaforandringer",
            profile=profile,
            progress_callback=progress_callback
        )
        
        print(f"Generated {len(result)} angles")
        
        # Vis det første resultat
        if result:
            print(f"First angle: {result[0].get('overskrift', 'No headline')}")
            
            # Vis ekspertkilder hvis tilgængelige
            if "ekspertKilder" in result[0]:
                experts = result[0]["ekspertKilder"].get("experts", [])
                print(f"Expert sources: {len(experts)}")
                for expert in experts:
                    print(f"- {expert.get('name', 'Unknown')}, {expert.get('title', 'Unknown')}")
        
    except Exception as e:
        print(f"Error in example: {e}")


# Kør eksemplet hvis scriptet køres direkte
if __name__ == "__main__":
    try:
        asyncio.run(example_usage())
    except KeyboardInterrupt:
        print("\nAfbrudt af bruger")
    except Exception as e:
        print(f"Fejl: {e}")