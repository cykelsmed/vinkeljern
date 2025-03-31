"""
Profile management module for Vinkeljernet project.

This module provides functionality to create, validate, compare and
inherit from editorial DNA profiles.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from models import RedaktionelDNA
from pydantic import ValidationError

class ProfileManager:
    """Manager for editorial DNA profiles."""
    
    @staticmethod
    def load_profile(path: Union[str, Path]) -> RedaktionelDNA:
        """
        Load a profile from a YAML file.
        
        Args:
            path: Path to the YAML file
            
        Returns:
            RedaktionelDNA: The loaded profile
        """
        path = Path(path) if isinstance(path, str) else path
        
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            
        try:
            return RedaktionelDNA(**data)
        except ValidationError as e:
            print(f"Error validating profile {path.name}:")
            for error in e.errors():
                print(f"- {error['loc'][0]}: {error['msg']}")
            raise
    
    @staticmethod
    def save_profile(profile: RedaktionelDNA, path: Union[str, Path]) -> None:
        """
        Save a profile to a YAML file.
        
        Args:
            profile: The profile to save
            path: Path to save the profile to
        """
        path = Path(path) if isinstance(path, str) else path
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(profile.dict(), f, allow_unicode=True, sort_keys=False)
    
    @staticmethod
    def merge_profiles(base: RedaktionelDNA, override: RedaktionelDNA) -> RedaktionelDNA:
        """
        Merge two profiles, with the override profile taking precedence.
        
        Args:
            base: Base profile
            override: Profile to override base values
            
        Returns:
            RedaktionelDNA: The merged profile
        """
        # Start with the base profile as a dictionary
        merged_dict = base.dict()
        override_dict = override.dict()
        
        # Override simple fields (update key from toneOgStil to tone_og_stil)
        for field in ["navn", "beskrivelse", "tone_og_stil"]:
            if override_dict.get(field):
                merged_dict[field] = override_dict[field]
        
        # Merge lists with deduplication
        for field in ["kerneprincipper", "fokusOmrader", "noGoOmrader"]:
            if override_dict.get(field):
                # Add override items first, then base items if not already present
                merged_list = list(override_dict[field])
                for item in base.dict().get(field, []):
                    if item not in merged_list:
                        merged_list.append(item)
                merged_dict[field] = merged_list
        
        # Merge dictionaries
        for field in ["nyhedsprioritering"]:
            if override_dict.get(field):
                # Start with base dict, then override with specific values
                merged_dict[field] = {**merged_dict.get(field, {}), **override_dict[field]}
        
        return RedaktionelDNA(**merged_dict)
    
    @staticmethod
    def create_sample_profile(profile_type: str, name: str) -> RedaktionelDNA:
        """
        Create a sample profile based on a predefined type.
        
        Args:
            profile_type: Type of profile to create 
                         ("tabloid", "broadsheet", "tv", "radio", "magazine")
            name: Name for the profile
            
        Returns:
            RedaktionelDNA: A sample profile
        """
        samples = {
            "tabloid": {
                "navn": name,
                "beskrivelse": "Tabloid profil med fokus på sensationelle historier",
                "kerneprincipper": [
                    "Sensationspræget journalistik",
                    "Fokus på personer og enkeltskæbner",
                    "Klart og letforståeligt sprog",
                    "Visuel og fængende præsentation"
                ],
                "tone_og_stil": "Direkte, personlig og følelsesladet. Bruger stærke ord og vendinger.",
                "nyhedsprioritering": {
                    "sensation": 10,
                    "konflikt": 8,
                    "identifikation": 7,
                    "aktualitet": 5,
                    "væsentlighed": 3
                },
                "fokusOmrader": [
                    "Kendte og kongelige",
                    "Kriminalstof",
                    "Personlige tragedier",
                    "Forbrugerrettigheder",
                    "Sport"
                ],
                "noGoOmrader": [
                    "Komplekst stof uden personlig vinkel",
                    "Lange tekniske udredninger",
                    "Emner uden klar konflikt"
                ]
            },
            "broadsheet": {
                "navn": name,
                "beskrivelse": "Seriøs omnibusavis med vægt på samfundsdebat",
                "kerneprincipper": [
                    "Grundig og faktabaseret journalistik",
                    "Samfundsansvarlig og seriøs vinkling",
                    "Nuanceret fremstilling",
                    "Adskillelse af news og views"
                ],
                "tone_og_stil": "Saglig, præcis og afbalanceret. Sprogligt korrekt med plads til nuancer.",
                "nyhedsprioritering": {
                    "væsentlighed": 10,
                    "aktualitet": 7,
                    "konflikt": 5,
                    "identifikation": 4,
                    "sensation": 2
                },
                "fokusOmrader": [
                    "Politik og samfund",
                    "Økonomi og erhverv",
                    "International politik",
                    "Kultur og debat",
                    "Videnskab og forskning"
                ],
                "noGoOmrader": [
                    "Overdreven sensationsjournalistik",
                    "Uvedkommende privatlivsstof",
                    "Spekulationer uden faktuelt grundlag"
                ]
            }
            # Additional types could be defined here
        }
        
        if profile_type not in samples:
            raise ValueError(f"Unknown profile type: {profile_type}. Available types: {', '.join(samples.keys())}")
        
        return RedaktionelDNA(**samples[profile_type])
    
    @staticmethod
    def compare_profiles(profile1: RedaktionelDNA, profile2: RedaktionelDNA) -> Dict[str, Any]:
        """
        Compare two profiles and return the differences
        
        Args:
            profile1: First profile to compare
            profile2: Second profile to compare
            
        Returns:
            Dict with differences between profiles
        """
        result = {"matches": [], "differences": {}}
        
        # Compare simple fields
        for field in ["navn", "beskrivelse"]:
            val1 = getattr(profile1, field)
            val2 = getattr(profile2, field)
            if val1 == val2:
                result["matches"].append(field)
            else:
                result["differences"][field] = {"profile1": val1, "profile2": val2}
        
        # Compare lists
        for field in ["kerneprincipper", "fokusOmrader", "noGoOmrader"]:
            list1 = getattr(profile1, field, [])
            list2 = getattr(profile2, field, [])
            
            common = set(list1) & set(list2)
            only_in_1 = set(list1) - set(list2)
            only_in_2 = set(list2) - set(list1)
            
            if not only_in_1 and not only_in_2:
                result["matches"].append(field)
            else:
                result["differences"][field] = {
                    "common": list(common),
                    "only_in_profile1": list(only_in_1),
                    "only_in_profile2": list(only_in_2)
                }
        
        # Compare dictionaries
        if profile1.nyhedsprioritering == profile2.nyhedsprioritering:
            result["matches"].append("nyhedsprioritering")
        else:
            # Find keys with different values
            diff_keys = {}
            all_keys = set(profile1.nyhedsprioritering.keys()) | set(profile2.nyhedsprioritering.keys())
            
            for key in all_keys:
                val1 = profile1.nyhedsprioritering.get(key)
                val2 = profile2.nyhedsprioritering.get(key)
                if val1 != val2:
                    diff_keys[key] = {"profile1": val1, "profile2": val2}
            
            result["differences"]["nyhedsprioritering"] = diff_keys
        
        return result
    
    @staticmethod
    def create_profile_template() -> Dict[str, Any]:
        """
        Generate a template dictionary for creating a new profile
        
        Returns:
            Dict with empty profile structure
        """
        return {
            "navn": "Ny redaktionel profil",
            "beskrivelse": "Beskrivelse af din medies redaktionelle DNA",
            "kerneprincipper": [
                "Princip 1: Beskriv det første kerneprincip",
                "Princip 2: Beskriv det andet kerneprincip"
            ],
            "tone_og_stil": "Beskriv din tone og stil her",
            "nyhedsprioritering": {
                "sensation": 5,
                "identifikation": 5,
                "konflikt": 5,
                "aktualitet": 5,
                "væsentlighed": 5
            },
            "fokusOmrader": [
                "Område 1",
                "Område 2"
            ],
            "noGoOmrader": [
                "Område der ikke dækkes 1",
                "Område der ikke dækkes 2"
            ]
        }

def print_profile_details(profile: RedaktionelDNA):
    print("Profil:", profile.navn)
    print("Beskrivelse:", profile.beskrivelse)
    print("Tone og stil:", profile.tone_og_stil)