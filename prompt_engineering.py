"""
Prompt engineering module for Vinkeljernet project.

This module handles the construction of prompts for the OpenAI API
to generate angles based on topic information and editorial profiles.
"""

import json
from typing import Dict, Any, List, Optional
from models import VinkelForslag

def construct_angle_prompt(
    topic: str, 
    topic_info: str,
    principper: str,
    tone_og_stil: str,
    fokusområder: str,
    nyhedskriterier: str,
    nogo_områder: str
) -> str:
    """
    Construct a prompt for the angle generation API.
    
    Args:
        topic: The news topic
        topic_info: Research information about the topic
        principper: Editorial principles
        tone_og_stil: Editorial tone and style
        fokusområder: Focus areas
        nyhedskriterier: News criteria
        nogo_områder: No-go areas
        
    Returns:
        str: The formatted prompt
    """
    return f"""
    Du er en erfaren journalist med ekspertise i at udvikle nyhedsvinkler.
    
    # Nyhedsemne:
    {topic}
    
    # Baggrundsinformation:
    {topic_info}
    
    # Redaktionel DNA-profil:
    ## Kerneprincipper:
    {principper}
    
    ## Tone og stil:
    {tone_og_stil}
    
    ## Fokusområder:
    {fokusområder}
    
    ## Nyhedskriterier vi prioriterer:
    {nyhedskriterier}
    
    ## No-go områder:
    {nogo_områder}
    
    # Opgave:
    Generer 8 forskellige vinkler på nyhedsemnet, der passer til vores redaktionelle DNA.
    For hver vinkel, angiv:
    1. En præcis overskrift (maks 10 ord)
    2. En kort beskrivelse af vinklen (2-3 sætninger)
    3. En begrundelse for hvorfor vinklen passer til vores profil (2-3 sætninger)
    4. Liste af nyhedskriterier som vinklen rammer (vælg fra: {nyhedskriterier})
    5. Tre gode startspørgsmål til interviews inden for denne vinkel
    
    Dit svar skal være i dette JSON-format:
    ```json
    [
      {{
        "overskrift": "Overskrift på vinkel",
        "beskrivelse": "Kort beskrivelse af vinklen",
        "begrundelse": "Begrundelse for valg af vinkel",
        "nyhedskriterier": ["kriterium1", "kriterium2"],
        "startSpørgsmål": ["Spørgsmål 1?", "Spørgsmål 2?", "Spørgsmål 3?"]
      }},
      ...flere vinkler...
    ]
    ```
    
    Vær kreativ og nuanceret, men hold dig til vores redaktionelle DNA. Undgå at foreslå vinkler, der falder under vores no-go områder.
    """

def parse_angles_from_response(response_text: str) -> List[Dict[str, Any]]:
    """
    Parse angles from the API response text.
    
    Args:
        response_text: The text response from the API
        
    Returns:
        List[Dict]: The parsed angles
    """
    try:
        # Try to parse the response as JSON
        data = json.loads(response_text)
        angles = []
        
        # Handle different response formats
        if isinstance(data, list):
            raw_angles = data
        elif isinstance(data, dict) and "angles" in data:
            raw_angles = data["angles"]
        elif isinstance(data, dict) and all(k.isdigit() for k in data.keys() if k):
            # Sometimes OpenAI returns {"0": {...}, "1": {...}} format
            raw_angles = [data[k] for k in sorted(data.keys(), key=lambda x: int(x) if x.isdigit() else 0)]
        else:
            # If we don't recognize the format, just try to use the whole response
            raw_angles = [data]
        
        # Validate and standardize each angle
        for raw_angle in raw_angles:
            try:
                # Ensure startSpørgsmål is present (it's new)
                if "startSpørgsmål" not in raw_angle and "startspørgsmål" in raw_angle:
                    raw_angle["startSpørgsmål"] = raw_angle["startspørgsmål"]
                elif "startSpørgsmål" not in raw_angle:
                    raw_angle["startSpørgsmål"] = [
                        f"Hvordan påvirker {raw_angle.get('overskrift', 'dette emne')} almindelige mennesker?",
                        "Hvad mener eksperterne om denne problemstilling?"
                    ]
                
                # Validate with Pydantic model
                validated_angle = VinkelForslag(**raw_angle).dict()
                angles.append(validated_angle)
            except Exception as e:
                # If a single angle fails validation, log it but continue with others
                print(f"Warning: Skipping invalid angle - {str(e)}")
                continue
        
        return angles
    
    except json.JSONDecodeError as e:
        print(f"Error: Response is not valid JSON: {e}")
        # Try to find where JSON might start in the response
        try:
            json_start = response_text.find("```json")
            json_end = response_text.rfind("```")
            if json_start != -1 and json_end != -1 and json_end > json_start:
                # Extract JSON from code block
                json_content = response_text[json_start+7:json_end].strip()
                print(f"Attempting to parse extracted JSON: {json_content[:100]}...")
                return json.loads(json_content)
        except Exception as e2:
            print(f"Failed to extract JSON from markdown blocks: {e2}")
        return []
    except Exception as e:
        print(f"Error parsing angles: {str(e)}")
        return []