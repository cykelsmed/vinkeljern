"""
Prompt engineering module for Vinkeljernet project.

This module handles the construction of prompts for the OpenAI API
to generate angles based on topic information and editorial profiles.
"""

import json
from typing import Dict, Any, List

def construct_angle_prompt(topic: str, topic_info: str, profile: Dict[str, Any]) -> str:
    """
    Construct a detailed prompt for angle generation.
    
    Args:
        topic: The news topic
        topic_info: Information about the topic
        profile: Dictionary representation of RedaktionelDNA
        
    Returns:
        str: Formatted prompt
    """
    # Format the profile data for easier reading in the prompt
    principles = "\n".join([f"- {list(p.keys())[0]}: {list(p.values())[0]}" for p in profile["kerneprincipper"]])
    priorities = "\n".join([f"- {key}: {value}/5" for key, value in profile["nyhedsprioritering"].items()])
    focus_areas = "\n".join([f"- {area}" for area in profile["fokusområder"]])
    nogo_areas = "\n".join([f"- {area}" for area in profile["nogo_områder"]])
    
    # Construct the prompt
    prompt = f"""
OPGAVE: Generér journalistiske vinkler til emnet "{topic}" baseret på den redaktionelle DNA profil beskrevet nedenfor.

## EMNE INFORMATION:
{topic_info}

## REDAKTIONEL DNA PROFIL:

### Kerneprincipper:
{principles}

### Nyhedsprioritering:
{priorities}

### Tone og stil:
{profile["tone_og_stil"]}

### Fokusområder:
{focus_areas}

### No-go områder:
{nogo_areas}

## INSTRUKTIONER:
1. Generér 10-15 potentielle vinkler på emnet der passer til den redaktionelle profil
2. Sørg for diversitet i perspektiver (menneskelig, politisk, økonomisk, konsekvens)
3. Undgå klichéer og standard-vinkler
4. For hver vinkel, identificér hvilke nyhedskriterier fra profilen der primært opfyldes
5. Flag eventuelle usikkerheder eller modstridende information i emne-beskrivelsen
6. Undgå helt vinkler der rammer no-go områderne

## OUTPUT FORMAT:
Returner din output som en JSON-array af objekter med følgende struktur:
```
[
  {{
    "overskrift": "Fængende overskrift her",
    "beskrivelse": "2-3 sætninger der uddyber vinklen",
    "nyhedskriterier": ["sensation", "identifikation"],  // fra nyhedsprioritering
    "kriterieScore": 8,  // Samlet score baseret på nyhedsprioriteringsvægte
    "startSpørgsmål": ["Spørgsmål 1?", "Spørgsmål 2?", "Spørgsmål 3?"],
    "flags": []  // evt. usikkerheder, kan være tom
  }},
  // ... flere vinkler
]
```

Svar KUN med JSON-array og ingen andre forklaringer eller text.
"""
    return prompt

def parse_angles_from_response(response_text: str) -> List[Dict[str, Any]]:
    """
    Parse the angles from the LLM response.
    
    Args:
        response_text: The raw response from the LLM
        
    Returns:
        List[Dict]: List of parsed angle objects
    """
    # Remove any markdown code block indicators
    cleaned_text = response_text.strip()
    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text[7:]
    if cleaned_text.endswith("```"):
        cleaned_text = cleaned_text[:-3]
    cleaned_text = cleaned_text.strip()
    
    try:
        # Parse the JSON
        angles = json.loads(cleaned_text)
        return angles
    except json.JSONDecodeError as e:
        print(f"Error parsing response as JSON: {e}")
        # Fallback: Try to salvage some content if possible
        print("Attempting to recover partial content...")
        try:
            # Try finding anything that looks like JSON array with objects
            import re
            pattern = r'\[\s*\{.*?\}\s*(?:,\s*\{.*?\}\s*)*\]'
            match = re.search(pattern, cleaned_text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return []
        except Exception:
            return []