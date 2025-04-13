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
    
    ## Del 1: Videndistillat
    Baseret på informationen, generer først et videndistillat på 400-600 ord, der opsummerer den vigtigste viden om emnet. Videndistillatet skal:

    1. Indeholde 5-7 hovedpunkter, der opsummerer de vigtigste fakta og nuancer om emnet
    2. Inkludere en kort tidslinje over relevante begivenheder (hvis tilgængeligt)
    3. Fremhæve 3-5 nøglestatistikker eller tal, der er relevante for forståelsen af emnet
    4. Identificere 2-4 centrale modsætningsforhold eller konflikter relateret til emnet
    5. Være objektivt og faktuelt, men med øje for de perspektiver og vinkler, der ville være relevante for mediets redaktionelle linje

    Videndistillatet skal være struktureret, præcist og dække bredden af den indsamlede information.
    
    ## Del 2: Vinkelgenerering
    Efter at have bearbejdet og destilleret informationen, generer 8 forskellige vinkler på nyhedsemnet, der passer til vores redaktionelle DNA.
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
        "videnDistillat": {{
          "hovedpunkter": ["Hovedpunkt 1", "Hovedpunkt 2", "Hovedpunkt 3", "Hovedpunkt 4", "Hovedpunkt 5"],
          "tidslinje": [
            {{"dato": "YYYY-MM-DD", "begivenhed": "Beskrivelse af begivenhed"}},
            {{"dato": "YYYY-MM-DD", "begivenhed": "Beskrivelse af begivenhed"}}
          ],
          "nøglestatistikker": {{
            "Statistik 1": "Værdi 1",
            "Statistik 2": "Værdi 2",
            "Statistik 3": "Værdi 3"
          }},
          "centrale_modsætninger": ["Modsætning 1", "Modsætning 2", "Modsætning 3"]
        }},
        "vinkler": [
          {{
            "overskrift": "Overskrift på vinkel",
            "beskrivelse": "Kort beskrivelse af vinklen",
            "begrundelse": "Begrundelse for valg af vinkel",
            "nyhedskriterier": ["kriterium1", "kriterium2"],
            "startSpørgsmål": ["Spørgsmål 1?", "Spørgsmål 2?", "Spørgsmål 3?"]
          }},
          // ...flere vinkler...
        ]
      }}
    ]
    
    HUSK: Begynd dit svar med [ og afslut med ] - returnér KUN JSON-data og absolut ingen anden tekst.
    """
    
    # Add additional context to the prompt if provided
    if additional_context:
        prompt += f"\n\nYDERLIGERE KONTEKST:\n{additional_context}\n"
        
    return prompt

def parse_angles_from_response(response_text: str) -> List[Dict[str, Any]]:
    """
    Parse angles and knowledge distillate from the API response text.
    
    Args:
        response_text: The text response from the API
        
    Returns:
        List[Dict]: The parsed angles with knowledge distillate attached
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
        # Use the more robust parse function from json_parser if available
        try:
            from json_parser import robust_json_parse
            parsed_data, error = robust_json_parse(response_text, "angle generation")
            if error:
                logger.warning(f"Robust parser reported issues: {error}")
                if not parsed_data:
                    raise ValueError(f"Failed to parse JSON: {error}")
                # Continue with partial data if available
                data = parsed_data[0] if isinstance(parsed_data, list) and parsed_data else parsed_data
            else:
                # Handle the case where robust_json_parse returns a list as the first item
                if isinstance(parsed_data, list):
                    if len(parsed_data) == 1 and isinstance(parsed_data[0], dict):
                        data = parsed_data[0]
                    else:
                        data = parsed_data
                else:
                    data = parsed_data
        except ImportError:
            # Fall back to standard JSON parsing if json_parser module not available
            data = json.loads(response_text)
        
        angles = []
        videndistillat = None
        
        # Handle the new format that includes videnDistillat
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and "videnDistillat" in data[0] and "vinkler" in data[0]:
            # Extract videndistillat from the new format
            videndistillat = data[0].get("videnDistillat", None)
            raw_angles = data[0].get("vinkler", [])
            logger.debug(f"Found new format with videnDistillat and {len(raw_angles)} vinkler")
        # Handle older and alternative formats
        elif isinstance(data, list):
            raw_angles = data
            logger.debug(f"Found list format with {len(raw_angles)} items")
        elif isinstance(data, dict) and "angles" in data:
            raw_angles = data["angles"]
            logger.debug(f"Found 'angles' key with {len(raw_angles)} items")
        elif isinstance(data, dict) and "vinkler" in data:
            raw_angles = data["vinkler"]
            logger.debug(f"Found 'vinkler' key with {len(raw_angles)} items")
        elif isinstance(data, dict) and all(k.isdigit() for k in data.keys() if k):
            # Sometimes OpenAI returns {"0": {...}, "1": {...}} format
            raw_angles = [data[k] for k in sorted(data.keys(), key=lambda x: int(x) if x.isdigit() else 0)]
            logger.debug(f"Found numbered dictionary format with {len(raw_angles)} items")
        else:
            # If we don't recognize the format, just try to use the whole response
            if isinstance(data, dict):
                raw_angles = [data]
                logger.debug("Using single dictionary as an angle")
            else:
                logger.error(f"Unrecognized response format: {type(data)}")
                raw_angles = []
        
        # If raw_angles is not a list, try to convert it
        if not isinstance(raw_angles, list):
            logger.warning(f"Expected list of angles but got {type(raw_angles)}, attempting to convert")
            if raw_angles is None:
                raw_angles = []
            else:
                try:
                    raw_angles = [raw_angles]
                except Exception as e:
                    logger.error(f"Could not convert to list: {e}")
                    raw_angles = []
        
        # Validate and standardize each angle
        for raw_angle in raw_angles:
            try:
                if not isinstance(raw_angle, dict):
                    logger.warning(f"Expected dict but got {type(raw_angle)}, skipping")
                    continue
                
                # Create a copy to avoid modifying the original
                angle_data = raw_angle.copy()
                
                # Ensure required fields exist
                required_fields = ["overskrift", "beskrivelse", "begrundelse", "nyhedskriterier"]
                missing_fields = [f for f in required_fields if f not in angle_data]
                if missing_fields:
                    logger.warning(f"Angle missing required fields: {', '.join(missing_fields)}")
                    
                    # Try to add reasonable defaults for missing fields
                    if "overskrift" not in angle_data:
                        angle_data["overskrift"] = "Mulig vinkel på emnet"
                    if "beskrivelse" not in angle_data:
                        angle_data["beskrivelse"] = "Genereret vinkel uden beskrivelse"
                    if "begrundelse" not in angle_data:
                        angle_data["begrundelse"] = "Ingen begrundelse"
                    if "nyhedskriterier" not in angle_data:
                        angle_data["nyhedskriterier"] = ["aktualitet"]
                
                # Handle different formats of startSpørgsmål
                if "startSpørgsmål" not in angle_data and "startspørgsmål" in angle_data:
                    angle_data["startSpørgsmål"] = angle_data["startspørgsmål"]
                elif "startSpørgsmål" not in angle_data:
                    angle_data["startSpørgsmål"] = [
                        f"Hvordan påvirker {angle_data.get('overskrift', 'dette emne')} almindelige mennesker?",
                        "Hvad mener eksperterne om denne problemstilling?"
                    ]
                
                # Ensure nyhedskriterier is a list
                if "nyhedskriterier" in angle_data and not isinstance(angle_data["nyhedskriterier"], list):
                    if isinstance(angle_data["nyhedskriterier"], str):
                        # Try to split by common separators
                        angle_data["nyhedskriterier"] = [k.strip() for k in angle_data["nyhedskriterier"].replace(',', ' ').replace(';', ' ').split()]
                    else:
                        angle_data["nyhedskriterier"] = ["aktualitet"]
                        
                # Ensure startSpørgsmål is a list
                if "startSpørgsmål" in angle_data and not isinstance(angle_data["startSpørgsmål"], list):
                    if isinstance(angle_data["startSpørgsmål"], str):
                        angle_data["startSpørgsmål"] = [angle_data["startSpørgsmål"]]
                    else:
                        angle_data["startSpørgsmål"] = ["Hvad er de vigtigste aspekter ved denne sag?"]
                
                try:
                    # Validate with Pydantic model
                    validated_angle = VinkelForslag(**angle_data).dict()
                    
                    # Add the videndistillat to each angle if it exists
                    if videndistillat:
                        validated_angle["videnDistillat"] = videndistillat
                        validated_angle["harVidenDistillat"] = True
                    
                    angles.append(validated_angle)
                    logger.debug(f"Successfully validated angle: {validated_angle['overskrift']}")
                except Exception as validation_error:
                    logger.error(f"Validation error for angle: {str(validation_error)}")
                    
                    # If validation failed, try to create a fallback angle with the available data
                    try:
                        fallback_angle = {
                            "overskrift": angle_data.get("overskrift", "Genereret vinkel"),
                            "beskrivelse": angle_data.get("beskrivelse", "Beskrivelse ikke tilgængelig"),
                            "begrundelse": angle_data.get("begrundelse", "Ingen begrundelse"),
                            "nyhedskriterier": angle_data.get("nyhedskriterier", ["aktualitet"]),
                            "startSpørgsmål": angle_data.get("startSpørgsmål", ["Hvad er de vigtigste aspekter af denne sag?"])
                        }
                        
                        # Validate with Pydantic model
                        validated_fallback = VinkelForslag(**fallback_angle).dict()
                        angles.append(validated_fallback)
                        logger.debug(f"Added fallback angle: {fallback_angle['overskrift']}")
                    except Exception as fallback_error:
                        logger.error(f"Failed to create fallback angle: {str(fallback_error)}")
            except Exception as e:
                # If a single angle fails validation, log it but continue with others
                logger.error(f"Error processing angle: {str(e)}")
                continue
        
        # If we couldn't parse any angles, create a default error angle
        if not angles:
            error_angle = {
                "overskrift": "Fejl under generering af vinkler",
                "beskrivelse": "Kunne ikke generere vinkler fra AI-svaret. Se log for detaljer.",
                "begrundelse": "Ingen begrundelse",
                "nyhedskriterier": ["aktualitet"],
                "startSpørgsmål": ["Hvad er de vigtigste aspekter af denne sag?"]
            }
            validated_error = VinkelForslag(**error_angle).dict()
            angles = [validated_error]
            logger.warning("Created error angle due to parsing issues")
        
        logger.info(f"Successfully parsed {len(angles)} angles")
        return angles
    
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        logger.debug(f"Response preview: {response_text[:500]}...")
        logger.debug(f"Response suffix: ...{response_text[-500:] if len(response_text) > 500 else response_text}")
        
        # Try to find where JSON might start in the response
        try:
            json_start = response_text.find("```json")
            json_end = response_text.rfind("```")
            if json_start != -1 and json_end != -1 and json_end > json_start:
                # Extract JSON from code block
                json_content = response_text[json_start+7:json_end].strip()
                logger.info(f"Attempting to parse extracted JSON from code block")
                return parse_angles_from_response(json_content)  # Recursive call with extracted JSON
        except Exception as e2:
            logger.error(f"Failed to extract JSON from markdown blocks: {e2}")
            
        # Create an error angle as fallback
        error_angle = {
            "overskrift": "Fejl under generering af vinkler",
            "beskrivelse": f"JSON parsing fejl: {e}",
            "begrundelse": "Ingen begrundelse",
            "nyhedskriterier": ["aktualitet"],
            "startSpørgsmål": ["Hvad er de vigtigste aspekter af denne sag?"]
        }
        
        try:
            validated_error = VinkelForslag(**error_angle).dict()
            return [validated_error]
        except Exception:
            logger.error("Failed to create error angle")
            return []
            
    except Exception as e:
        logger.error(f"Unexpected error parsing angles: {str(e)}")
        
        # Create an error angle as fallback
        error_angle = {
            "overskrift": "Fejl under generering af vinkler",
            "beskrivelse": f"Uventet fejl: {e}",
            "begrundelse": "Ingen begrundelse",
            "nyhedskriterier": ["aktualitet"],
            "startSpørgsmål": ["Hvad er de vigtigste aspekter af denne sag?"]
        }
        
        try:
            validated_error = VinkelForslag(**error_angle).dict()
            return [validated_error]
        except Exception:
            logger.error("Failed to create error angle")
            return []