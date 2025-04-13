"""
Prompt engineering module for Vinkeljernet project.

This module handles the construction of prompts for the OpenAI API
to generate angles based on topic information and editorial profiles.
"""

import json
from typing import Dict, Any, List, Optional
from models import VinkelForslag, RedaktionelDNA

def create_angle_generation_prompt(topic: str, distillate: Dict[str, Any], profile: RedaktionelDNA) -> str:
    """
    Create a more robust and simpler prompt for angle generation that produces cleaner JSON.
    
    Args:
        topic: The news topic
        distillate: The knowledge distillate
        profile: The editorial DNA profile
        
    Returns:
        str: A prompt focused on generating well-formed JSON angles
    """
    # Extract key profile elements
    kerneprincipper = ", ".join([list(p.keys())[0] for p in profile.kerneprincipper]) if isinstance(profile.kerneprincipper, list) else "Ingen specificeret"
    nyhedskriterier = ", ".join(profile.nyhedskriterier) if isinstance(profile.nyhedskriterier, list) else "aktualitet, identifikation, sensation"
    fokusomrader = ", ".join(profile.fokusområder) if isinstance(profile.fokusområder, list) else "generelle nyhedsområder"
    nogo = ", ".join(profile.noGoOmrader) if isinstance(profile.noGoOmrader, list) else "Ingen specificeret"
    
    # Clean distillate for prompt
    try:
        clean_distillate = {}
        if "hovedpunkter" in distillate and isinstance(distillate["hovedpunkter"], list):
            clean_distillate["hovedpunkter"] = distillate["hovedpunkter"]
        if "noegletal" in distillate and isinstance(distillate["noegletal"], list):
            clean_distillate["nøgletal"] = [f"{item.get('tal', '')}: {item.get('beskrivelse', '')}" 
                                          for item in distillate["noegletal"] if isinstance(item, dict)]
        if "centralePaastand" in distillate and isinstance(distillate["centralePaastand"], list):
            clean_distillate["modsætninger"] = [item.get("paastand", "") 
                                             for item in distillate["centralePaastand"] if isinstance(item, dict)]
    except Exception:
        # Fallback to simpler format
        clean_distillate = {"hovedpunkter": ["Kunne ikke processere videndistillat"]}

    # Create a simple, robust prompt    
    return f"""
    # INPUT
    Emne: {topic}
    
    # VIDENDISTILLAT
    {json.dumps(clean_distillate, indent=2, ensure_ascii=False)}
    
    # REDAKTIONEL PROFIL
    Kerneprincipper: {kerneprincipper}
    Nyhedskriterier: {nyhedskriterier}
    Fokusområder: {fokusomrader}
    No-go områder: {nogo}
    Tone og stil: {profile.tone_og_stil if profile.tone_og_stil else "Professionel og neutral"}
    
    # OPGAVE
    Du er ekspert i journalistisk vinkling og skal generere 5-8 forskellige vinkler på det givne emne, 
    der passer til den redaktionelle profil. Fokuser på at skabe klare, præcise og velformulerede vinkler.
    
    # OUTPUT FORMAT
    Returner KUN et JSON-array med objekter, der hver repræsenterer en vinkel:
    [
      {{
        "overskrift": "Fængende overskrift der opsummerer vinklen",
        "beskrivelse": "2-3 sætninger der uddyber vinklen",
        "begrundelse": "Kort begrundelse for hvorfor denne vinkel er relevant ift. redaktionel profil",
        "nyhedskriterier": ["kriterie1", "kriterie2"],
        "startSpørgsmål": ["Spørgsmål 1?", "Spørgsmål 2?"]
      }}
    ]
    
    VIGTIGT:
    - Returner KUN et velformateret JSON-array uden nogen indledende eller afsluttende tekst
    - Alle felter er påkrævede og skal være korrekt udfyldt
    - Brug nøjagtigt de feltnavne der er angivet ovenfor, inklusive startSpørgsmål med stort S
    - Brug dobbelte anførselstegn for alle strenge
    - Brug dansk sprog
    - Hold svaret kort og overskueligt (maks 8 vinkler)
    """

def construct_angle_prompt(
    topic: str, 
    topic_info: str,
    principper: str,
    tone_og_stil: str,
    fokusområder: str,
    nyhedskriterier: str,
    nogo_områder: str,
    additional_context: str = ""  # Add this parameter with default empty string
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
        additional_context: Optional additional context like knowledge distillate
        
    Returns:
        str: The formatted prompt
    """
    prompt = f"""
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
    
    Baseret på informationen, generer 8 forskellige vinkler på nyhedsemnet, der passer til vores redaktionelle DNA.
    For hver vinkel, angiv:
    1. En præcis overskrift (maks 10 ord)
    2. En kort beskrivelse af vinklen (2-3 sætninger)
    3. En begrundelse for hvorfor vinklen passer til vores profil (2-3 sætninger)
    4. Liste af nyhedskriterier som vinklen rammer (vælg fra: {nyhedskriterier})
    5. Tre gode startspørgsmål til interviews inden for denne vinkel
    
    VIGTIGT: Dit svar SKAL være i JSON-format og MÅ IKKE indeholde nogen form for indledende eller afsluttende tekst. Start direkte med JSON-objektet.
    
    Returnér resultat i dette JSON-format og intet andet:
    [
      {{
        "overskrift": "Overskrift på vinkel",
        "beskrivelse": "Kort beskrivelse af vinklen",
        "begrundelse": "Begrundelse for valg af vinkel",
        "nyhedskriterier": ["kriterium1", "kriterium2"],
        "startSpørgsmål": ["Spørgsmål 1?", "Spørgsmål 2?", "Spørgsmål 3?"]
      }},
      {{
        "overskrift": "Overskrift på en anden vinkel",
        "beskrivelse": "Kort beskrivelse af vinklen",
        "begrundelse": "Begrundelse for valg af vinkel",
        "nyhedskriterier": ["kriterium1", "kriterium3"],
        "startSpørgsmål": ["Spørgsmål 1?", "Spørgsmål 2?", "Spørgsmål 3?"]
      }},
      // flere vinkler...
    ]
    
    HUSK: Begynd dit svar med [ og afslut med ] - returnér KUN JSON-data og absolut ingen anden tekst.
    """
    
    # Add additional context to the prompt if provided
    if additional_context:
        prompt += f"\n\n# YDERLIGERE KONTEKST:\n{additional_context}\n"
        
    return prompt

def parse_angles_from_response(response_text: str) -> List[Dict[str, Any]]:
    """
    Parse angles from the API response text with better handling for direct list format.
    
    Args:
        response_text: The text response from the API
        
    Returns:
        List[Dict]: The parsed angles
    """
    import logging
    logger = logging.getLogger(__name__)
    
    if not response_text:
        logger.error("Empty response received from API")
        return []
        
    # Log a preview of the response for debugging
    log_preview = response_text[:200] + ("..." if len(response_text) > 200 else "")
    logger.debug(f"Parsing angles from response (preview): {log_preview}")
        
    try:
        # First try using the more robust parser from json_parser if available
        try:
            from json_parser import safe_parse_json
            
            # Use safe_parse_json which is more robust
            data = safe_parse_json(
                response_text, 
                context="angle generation",
                fallback=[]
            )
            
            if not data and not isinstance(data, (list, dict)):
                raise ValueError("No valid data parsed")
                
        except ImportError:
            # Fall back to standard JSON parsing if json_parser module not available
            data = json.loads(response_text)
        
        angles = []
        
        # Handle different response formats
        if isinstance(data, list):
            # Direct list of angles (our simplified format)
            raw_angles = data
            logger.debug(f"Found direct list format with {len(raw_angles)} items")
        elif isinstance(data, dict) and "vinkler" in data:
            # Format with wrapper object containing vinkler array
            raw_angles = data["vinkler"]
            logger.debug(f"Found 'vinkler' key with {len(raw_angles)} items")
        elif isinstance(data, dict) and "videnDistillat" in data and "vinkler" in data:
            # Format with knowledge distillate and angles in same object
            raw_angles = data["vinkler"]
            logger.debug(f"Found new format with videnDistillat and {len(raw_angles)} vinkler")
        elif isinstance(data, dict):
            # Single angle as dictionary
            raw_angles = [data]
            logger.debug("Using single dictionary as an angle")
        else:
            logger.error(f"Unrecognized response format: {type(data)}")
            raw_angles = []
        
        # If raw_angles is not a list, handle that case
        if not isinstance(raw_angles, list):
            logger.warning(f"Expected list of angles but got {type(raw_angles)}, attempting to convert")
            try:
                if raw_angles is None:
                    raw_angles = []
                else:
                    raw_angles = [raw_angles]
            except Exception as e:
                logger.error(f"Could not convert to list: {e}")
                raw_angles = []
        
        # Process each angle
        for raw_angle in raw_angles:
            if not isinstance(raw_angle, dict):
                logger.warning(f"Expected dict but got {type(raw_angle)}, skipping")
                continue
                
            # Create a copy to avoid modifying the original
            angle_data = raw_angle.copy()
            
            # Ensure required fields exist with defaults if missing
            required_fields = ["overskrift", "beskrivelse", "begrundelse", "nyhedskriterier"]
            for field in required_fields:
                if field not in angle_data:
                    if field == "overskrift":
                        angle_data[field] = "Ubenævnt vinkel"
                    elif field == "beskrivelse":
                        angle_data[field] = "Ingen beskrivelse tilgængelig"
                    elif field == "begrundelse":
                        angle_data[field] = "Ingen begrundelse angivet"
                    elif field == "nyhedskriterier":
                        angle_data[field] = ["aktualitet"]
            
            # Handle different formats of startSpørgsmål
            if "startSpørgsmål" not in angle_data:
                if "startspørgsmål" in angle_data:  # Common case-sensitivity issue
                    angle_data["startSpørgsmål"] = angle_data["startspørgsmål"]
                elif "spørgsmål" in angle_data:  # Another common variation 
                    angle_data["startSpørgsmål"] = angle_data["spørgsmål"]
                else:
                    # Create default questions based on the headline
                    angle_data["startSpørgsmål"] = [
                        f"Hvordan påvirker {angle_data['overskrift']} almindelige mennesker?",
                        f"Hvad er de vigtigste aspekter af {angle_data['overskrift']}?",
                        "Hvad mener eksperterne om denne problemstilling?"
                    ]
            
            # Ensure nyhedskriterier is a list
            if not isinstance(angle_data["nyhedskriterier"], list):
                if isinstance(angle_data["nyhedskriterier"], str):
                    # Split string by common separators
                    angle_data["nyhedskriterier"] = [k.strip() for k in 
                                                   angle_data["nyhedskriterier"].replace(',', ' ')
                                                   .replace(';', ' ').split() if k.strip()]
                    if not angle_data["nyhedskriterier"]:
                        angle_data["nyhedskriterier"] = ["aktualitet"]
                else:
                    angle_data["nyhedskriterier"] = ["aktualitet"]
            
            # Ensure startSpørgsmål is a list
            if not isinstance(angle_data["startSpørgsmål"], list):
                if isinstance(angle_data["startSpørgsmål"], str):
                    angle_data["startSpørgsmål"] = [angle_data["startSpørgsmål"]]
                else:
                    angle_data["startSpørgsmål"] = ["Hvad er de vigtigste aspekter ved denne sag?"]
            
            try:
                # Use Pydantic model for validation if possible
                from models import VinkelForslag
                validated_angle = VinkelForslag(**angle_data).dict()
                angles.append(validated_angle)
                logger.debug(f"Successfully validated angle: {validated_angle['overskrift']}")
            except Exception as validation_error:
                logger.warning(f"Validation error for angle, using raw data: {str(validation_error)}")
                # If validation fails, just use the cleaned data directly
                angles.append(angle_data)
        
        # If we couldn't parse any angles, create a default error angle
        if not angles:
            error_angle = {
                "overskrift": "Fejl under generering af vinkler",
                "beskrivelse": "Kunne ikke generere vinkler fra AI-svaret. Dette kan skyldes ændringer i formatet.",
                "begrundelse": "Systemfejl - parsing af svar fejlede",
                "nyhedskriterier": ["aktualitet"],
                "startSpørgsmål": ["Hvad er de vigtigste aspekter af denne sag?"]
            }
            angles = [error_angle]
            logger.warning("Created error angle due to parsing issues")
        
        logger.info(f"Successfully parsed {len(angles)} angles")
        return angles
    
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        
        # Try to extract JSON from code blocks
        try:
            import re
            # Look for code blocks or JSON-like structures
            json_pattern = r'```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```'
            match = re.search(json_pattern, response_text, re.DOTALL)
            
            if match:
                potential_json = match.group(1).strip()
                logger.info("Extracted JSON from code block, attempting to parse")
                return parse_angles_from_response(potential_json)
        except Exception as e2:
            logger.error(f"Failed to extract JSON from code blocks: {e2}")
        
        # Create an error angle as fallback
        error_angle = {
            "overskrift": "Fejl under generering af vinkler",
            "beskrivelse": f"JSON parsing fejl: {str(e)}",
            "begrundelse": "Systemfejl - JSON parsing fejlede",
            "nyhedskriterier": ["aktualitet"],
            "startSpørgsmål": ["Hvad er de vigtigste aspekter af denne sag?"]
        }
        return [error_angle]
            
    except Exception as e:
        logger.error(f"Unexpected error parsing angles: {str(e)}")
        
        # Create an error angle as fallback
        error_angle = {
            "overskrift": "Fejl under generering af vinkler",
            "beskrivelse": f"Uventet fejl under parsing: {str(e)}",
            "begrundelse": "Systemfejl",
            "nyhedskriterier": ["aktualitet"],
            "startSpørgsmål": ["Hvad er de vigtigste aspekter af denne sag?"]
        }
        return [error_angle]