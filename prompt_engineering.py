"""
Prompt engineering module for Vinkeljernet project.

This module handles the construction of prompts for the OpenAI API
to generate angles based on topic information and editorial profiles.
"""

import json
from typing import Dict, Any, List, Optional
from models import VinkelForslag, RedaktionelDNA
import logging

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

def construct_json_structure_prompt(base_prompt: str, struct_type: str = "angles") -> str:
    """
    Tilføjer instruktioner til en prompt for at sikre korrekt JSON-formatering.
    Denne funktion er specielt designet til at hjælpe LLM'er med at generere velformateret JSON.
    
    Args:
        base_prompt: Den grundlæggende prompt der skal udvides
        struct_type: Typen af struktur ("angles", "experts", "knowledge")
        
    Returns:
        str: Udvidet prompt med JSON-formaterings instruktioner
    """
    # Struktur-specifikke instruktioner
    json_structures = {
        "angles": """
        Dit svar skal være et gyldigt JSON-array med vinkelobjekter, der har følgende struktur:
        [
          {
            "overskrift": "Tydelig overskrift",
            "beskrivelse": "Detaljeret beskrivelse af vinklen",
            "begrundelse": "Journalistisk begrundelse for vinklen",
            "nyhedskriterier": ["aktualitet", "væsentlighed"],
            "startSpørgsmål": ["Konkret spørgsmål 1?", "Konkret spørgsmål 2?"]
          },
          {
            // Næste vinkel...
          }
        ]
        """,
        "experts": """
        Dit svar skal være et gyldigt JSON-objekt med ekspertkilder, der har følgende struktur:
        {
          "eksperter": [
            {
              "navn": "Fuldt navn",
              "titel": "Stillingsbetegnelse",
              "organisation": "Arbejdssted",
              "ekspertise": "Beskrivelse af ekspertområde",
              "kontaktInfo": "Email eller anden kontaktinformation hvis relevant"
            }
          ],
          "institutioner": [
            {
              "navn": "Institutionens fulde navn",
              "type": "Type institution",
              "relevans": "Hvorfor denne institution er relevant"
            }
          ]
        }
        """,
        "knowledge": """
        Dit svar skal være et gyldigt JSON-objekt med videndistillat, der har følgende struktur:
        {
          "hovedpunkter": ["Punkt 1", "Punkt 2", "Punkt 3"],
          "noegletal": [
            {
              "tal": "42%",
              "beskrivelse": "Beskrivelse af hvad tallet betyder",
              "kilde": "Kilden til tallet"
            }
          ],
          "centralePaastand": [
            {
              "paastand": "Den centrale påstand",
              "kilde": "Kilden til påstanden"
            }
          ]
        }
        """
    }
    
    # Generelle JSON-formaterings instruktioner
    json_instructions = """
    VIGTIGT OM JSON-FORMATERING:
    1. Svar KUN med gyldig JSON, uden forklarende tekst før eller efter
    2. Brug dobbelte anførselstegn (") omkring keys og string-værdier
    3. Brug ikke enkelte anførselstegn (')
    4. Brug ikke trailing komma i arrays eller objekter
    5. Brug kun boolske værdier som true/false (ikke True/False)
    6. Brug null for manglende værdier (ikke None)
    7. Indlejr ikke andre formater i JSON (markdown, kommentarer, etc.)
    8. Sørg for at alle brackets {} og [] er korrekt afsluttet 
    
    ### FORVENTET FORMAT:
    """
    
    # Hent den struktur-specifikke instruktion
    specific_structure = json_structures.get(
        struct_type, 
        json_structures["angles"]  # Brug angles som default
    )
    
    # Sammensæt den endelige prompt
    final_prompt = f"{base_prompt}\n\n{json_instructions}{specific_structure}\n\nHusk: Dit svar skal KUN være gyldig JSON uden forklarende tekst før, efter eller i midten af JSON-strukturen."
    
    return final_prompt

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

def construct_expert_prompt(topic: str, angle_headline: str, angle_description: str) -> str:
    """
    Konstruerer en prompt for ekspertkildeforslag med forbedret JSON-struktur instruktion.
    
    Args:
        topic: Hovedemnet
        angle_headline: Vinkel-overskrift
        angle_description: Beskrivelse af vinklen
        
    Returns:
        str: Optimeret prompt til ekspertkilder med JSON-instruktioner
    """
    base_prompt = f"""
    Find konkrete ekspertkilder, institutioner og datakilder til følgende journalistiske vinkel:
    
    EMNE: {topic}
    OVERSKRIFT: {angle_headline}
    BESKRIVELSE: {angle_description}
    
    Jeg har brug for:
    1. 4-6 konkrete eksperter med navn, titel og organisation
    2. 3-5 relevante organisationer/institutioner
    3. 2-4 specifikke datakilder der kan bruges i researchen
    
    Eksperterne skal være reelle personer med korrekte titler og institutioner. Prioriter danske eksperter.
    """
    
    return construct_json_structure_prompt(base_prompt, "experts")

def construct_knowledge_prompt(topic: str, topic_info: str) -> str:
    """
    Konstruerer en prompt for videndistillat med forbedret JSON-struktur instruktion.
    
    Args:
        topic: Emnet
        topic_info: Baggrundsinformation om emnet
        
    Returns:
        str: Optimeret prompt til videndistillat med JSON-instruktioner
    """
    # Begræns længden af baggrundsinformation
    if len(topic_info) > 3000:
        topic_info = topic_info[:2997] + "..."
    
    base_prompt = f"""
    Analyser denne information om '{topic}' og lav et videndistillat med de vigtigste fakta.
    
    BAGGRUNDSINFORMATION:
    {topic_info}
    
    Jeg har brug for følgende information:
    1. 4 hovedpunkter fra materialet
    2. 3 nøgletal med tal, beskrivelse og kilde
    3. 3 centrale påstande med kilde
    
    Brug KUN information fra baggrundsmaterialet.
    """
    
    return construct_json_structure_prompt(base_prompt, "knowledge")

def parse_angles_from_response(response_text: str) -> List[Dict[str, Any]]:
    """
    Parse angles from the API response text with better handling for direct list format.
    
    Args:
        response_text: The text response from the API
        
    Returns:
        List[Dict]: The parsed angles
    """
    logger = logging.getLogger(__name__)
    
    if not response_text:
        logger.error("Empty response received from API")
        return []
        
    # Log the entire raw response for debugging
    logger.info(f"[ANGLE_VALIDATION] Raw LLM response: {response_text}")
    
    # Log a preview of the response for debugging (for backward compatibility)
    log_preview = response_text[:200] + ("..." if len(response_text) > 200 else "")
    logger.debug(f"Parsing angles from response (preview): {log_preview}")
        
    try:
        # First try using the more robust parser from json_parser if available
        try:
            from json_parser import robust_json_parse
            # Use robust_json_parse to get all possible angle objects
            parsed_data, error = robust_json_parse(
                response_text,
                context="angle generation response"
            )
            if not parsed_data:
                logger.error(f"Failed to parse angles from LLM response: {error}")
                return []
            raw_angles = parsed_data
        except ImportError:
            # Fall back to standard JSON parsing if json_parser module not available
            logger.info("[ANGLE_VALIDATION] json_parser module not available, falling back to standard JSON parsing")
            data = json.loads(response_text)
            if isinstance(data, list):
                raw_angles = data
            elif isinstance(data, dict):
                raw_angles = [data]
            else:
                raw_angles = []

        # If raw_angles is not a list, handle that case
        if not isinstance(raw_angles, list):
            logger.warning(f"[ANGLE_VALIDATION] Expected list of angles but got {type(raw_angles)}, attempting to convert")
            try:
                if raw_angles is None:
                    raw_angles = []
                else:
                    raw_angles = [raw_angles]
            except Exception as e:
                logger.error(f"[ANGLE_VALIDATION] Could not convert to list: {e}")
                raw_angles = []

        angles = []
        for i, raw_angle in enumerate(raw_angles):
            if not isinstance(raw_angle, dict):
                logger.warning(f"[ANGLE_VALIDATION] Angle {i+1}: Expected dict but got {type(raw_angle)}, skipping")
                continue
                
            # Log the raw angle object
            logger.info(f"[ANGLE_VALIDATION] Angle {i+1} raw object: {json.dumps(raw_angle, ensure_ascii=False)}")
            
            # Create a copy to avoid modifying the original
            angle_data = raw_angle.copy()
            
            # Track validation fixes for detailed logging
            validation_fixes = []
            
            # Ensure required fields exist with defaults if missing
            required_fields = ["overskrift", "beskrivelse", "begrundelse", "nyhedskriterier"]
            for field in required_fields:
                if field not in angle_data:
                    if field == "overskrift":
                        angle_data[field] = "Ubenævnt vinkel"
                        validation_fixes.append(f"Missing required field '{field}', added default value")
                    elif field == "beskrivelse":
                        angle_data[field] = "Ingen beskrivelse tilgængelig"
                        validation_fixes.append(f"Missing required field '{field}', added default value")
                    elif field == "begrundelse":
                        angle_data[field] = "Ingen begrundelse angivet"
                        validation_fixes.append(f"Missing required field '{field}', added default value")
                    elif field == "nyhedskriterier":
                        angle_data[field] = ["aktualitet"]
                        validation_fixes.append(f"Missing required field '{field}', added default value")
            
            # Handle different formats of startSpørgsmål
            if "startSpørgsmål" not in angle_data:
                if "startspørgsmål" in angle_data:  # Common case-sensitivity issue
                    angle_data["startSpørgsmål"] = angle_data["startspørgsmål"]
                    validation_fixes.append("Fixed case sensitivity issue with 'startspørgsmål'")
                elif "spørgsmål" in angle_data:  # Another common variation 
                    angle_data["startSpørgsmål"] = angle_data["spørgsmål"]
                    validation_fixes.append("Used 'spørgsmål' field as 'startSpørgsmål'")
                else:
                    # Create default questions based on the headline
                    angle_data["startSpørgsmål"] = [
                        f"Hvordan påvirker {angle_data['overskrift']} almindelige mennesker?",
                        f"Hvad er de vigtigste aspekter af {angle_data['overskrift']}?",
                        "Hvad mener eksperterne om denne problemstilling?"
                    ]
                    validation_fixes.append("Missing 'startSpørgsmål', created default questions")
            
            # Ensure nyhedskriterier is a list
            if not isinstance(angle_data["nyhedskriterier"], list):
                if isinstance(angle_data["nyhedskriterier"], str):
                    # Split string by common separators
                    angle_data["nyhedskriterier"] = [k.strip() for k in 
                                                   angle_data["nyhedskriterier"].replace(',', ' ')
                                                   .replace(';', ' ').split() if k.strip()]
                    if angle_data["nyhedskriterier"]:
                        validation_fixes.append("Converted 'nyhedskriterier' from string to list")
                    else:
                        angle_data["nyhedskriterier"] = ["aktualitet"]
                        validation_fixes.append("Empty 'nyhedskriterier' string, added default value")
                else:
                    angle_data["nyhedskriterier"] = ["aktualitet"]
                    validation_fixes.append(f"Incorrect type for 'nyhedskriterier' ({type(angle_data['nyhedskriterier']).__name__}), set to default")
            
            # Ensure startSpørgsmål is a list
            if not isinstance(angle_data["startSpørgsmål"], list):
                if isinstance(angle_data["startSpørgsmål"], str):
                    angle_data["startSpørgsmål"] = [angle_data["startSpørgsmål"]]
                    validation_fixes.append("Converted 'startSpørgsmål' from string to list")
                else:
                    angle_data["startSpørgsmål"] = ["Hvad er de vigtigste aspekter ved denne sag?"]
                    validation_fixes.append(f"Incorrect type for 'startSpørgsmål' ({type(angle_data['startSpørgsmål']).__name__}), set to default")
            
            # Log validation fixes if any were made
            if validation_fixes:
                logger.info(f"[ANGLE_VALIDATION] Angle {i+1} ({angle_data.get('overskrift', 'Unnamed')}): Applied fixes: {'; '.join(validation_fixes)}")
            else:
                logger.info(f"[ANGLE_VALIDATION] Angle {i+1} ({angle_data.get('overskrift', 'Unnamed')}): No validation fixes needed")
            
            try:
                # Use Pydantic model for validation if possible
                from models import VinkelForslag
                validated_angle = VinkelForslag(**angle_data).dict()
                angles.append(validated_angle)
                logger.info(f"[ANGLE_VALIDATION] Angle {i+1} ({validated_angle['overskrift']}): PASSED validation")
            except Exception as validation_error:
                validation_error_str = str(validation_error)
                logger.warning(f"[ANGLE_VALIDATION] Angle {i+1} ({angle_data.get('overskrift', 'Unnamed')}): FAILED validation: {validation_error_str}")
                
                # Attempt to extract specific validation errors
                error_details = []
                if "overskrift" in validation_error_str.lower():
                    error_details.append("Invalid headline")
                if "beskrivelse" in validation_error_str.lower():
                    error_details.append("Invalid description")
                if "nyhedskriterier" in validation_error_str.lower():
                    error_details.append("Invalid news criteria")
                if "startspørgsmål" in validation_error_str.lower():
                    error_details.append("Invalid questions")
                if not error_details:
                    error_details.append("Unknown validation error")
                
                logger.warning(f"[ANGLE_VALIDATION] Angle {i+1}: Specific errors: {', '.join(error_details)}")
                
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
            logger.warning("[ANGLE_VALIDATION] Created error angle due to parsing issues")
        
        logger.info(f"[ANGLE_VALIDATION] Final validation result: {len(angles)} valid angles out of {len(raw_angles)} raw angles")
        return angles
    
    except json.JSONDecodeError as e:
        logger.error(f"[ANGLE_VALIDATION] JSON decode error: {e}")
        
        # Try to extract JSON from code blocks
        try:
            import re
            # Look for code blocks or JSON-like structures
            json_pattern = r'```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```'
            match = re.search(json_pattern, response_text, re.DOTALL)
            
            if match:
                potential_json = match.group(1).strip()
                logger.info("[ANGLE_VALIDATION] Extracted JSON from code block, attempting to parse")
                return parse_angles_from_response(potential_json)
        except Exception as e2:
            logger.error(f"[ANGLE_VALIDATION] Failed to extract JSON from code blocks: {e2}")
        
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
        logger.error(f"[ANGLE_VALIDATION] Unexpected error parsing angles: {str(e)}")
        
        # Create an error angle as fallback
        error_angle = {
            "overskrift": "Fejl under generering af vinkler",
            "beskrivelse": f"Uventet fejl under parsing: {str(e)}",
            "begrundelse": "Systemfejl",
            "nyhedskriterier": ["aktualitet"],
            "startSpørgsmål": ["Hvad er de vigtigste aspekter af denne sag?"]
        }
        return [error_angle]

class OptimizedPromptEngineering:
    """
    Optimeret prompt engineering for hurtigere LLM-svar.
    
    Denne klasse indeholder metoder til at reducere størrelsen på prompts,
    optimere kontekst-kompression, og bruge token-besparende teknikker for at
    få hurtigere svar fra LLM-modeller.
    """
    
    # Maksimale tegn for forskellige prompt-dele
    MAX_TOPIC_INFO_CHARS = 2000  # Maks tegn for baggrundsinformation
    MAX_PROFILE_CHARS = 1000     # Maks tegn for profilbeskrivelse
    MAX_PRINCIPLES_CHARS = 500   # Maks tegn for kerneprincipper
    
    # Token-besparende dele der kan fjernes
    REMOVABLE_PARTS = [
        "# Baggrundsinformation:",
        "# Redaktionel DNA-profil:",
        "## Kerneprincipper:",
        "## Tone og stil:",
        "## Fokusområder:",
        "## Nyhedskriterier vi prioriterer:",
        "## No-go områder:",
        "# Opgave:"
    ]
    
    @staticmethod
    def truncate_section(section: str, max_chars: int) -> str:
        """Afkorter en sektion til max_chars med intelligent afkortning."""
        if not section or len(section) <= max_chars:
            return section
            
        # Hvis det er en liste, behold så mange hele punkter som muligt
        if section.strip().startswith('-'):
            lines = section.strip().split('\n')
            result = []
            current_length = 0
            
            for line in lines:
                if current_length + len(line) + 1 <= max_chars:
                    result.append(line)
                    current_length += len(line) + 1  # +1 for newline
                else:
                    break
                    
            return '\n'.join(result)
        else:
            # Ellers afkort til max_chars og tilføj ellipsis
            return section[:max_chars-3] + "..."
    
    @staticmethod
    def optimize_prompt(prompt: str, target_length: int = 2500) -> str:
        """
        Optimerer en prompt ved at reducere længden intelligent.
        
        Args:
            prompt: Original prompt
            target_length: Mål-længde i tegn
            
        Returns:
            str: Optimeret prompt
        """
        if len(prompt) <= target_length:
            return prompt
            
        # Del prompt op i sektioner
        sections = {}
        current_section = "intro"
        lines = prompt.split('\n')
        sections[current_section] = []
        
        for line in lines:
            if line.startswith('#'):
                current_section = line.strip('# ')
                sections[current_section] = [line]
            else:
                if current_section in sections:
                    sections[current_section].append(line)
                else:
                    sections[current_section] = [line]
        
        # Prioriter sektioner for bevarelse af vigtig information
        priority_order = [
            "Opgave:", 
            "Baggrundsinformation:", 
            "Fokusområder:", 
            "Nyhedskriterier vi prioriterer:", 
            "Tone og stil:", 
            "Kerneprincipper:", 
            "No-go områder:"
        ]
        
        # Maks længder for forskellige sektioner
        max_lengths = {
            "Baggrundsinformation:": OptimizedPromptEngineering.MAX_TOPIC_INFO_CHARS,
            "Kerneprincipper:": OptimizedPromptEngineering.MAX_PRINCIPLES_CHARS,
            "Tone og stil:": 300,
            "Fokusområder:": 300,
            "Nyhedskriterier vi prioriterer:": 300,
            "No-go områder:": 200,
            "Opgave:": 500,
            "intro": 200
        }
        
        # Rekonstruer prompt med afkortede sektioner
        optimized_sections = {}
        for section, content in sections.items():
            content_str = '\n'.join(content)
            max_len = max_lengths.get(section, 200)
            optimized_sections[section] = OptimizedPromptEngineering.truncate_section(
                content_str, max_len
            )
        
        # Sammensæt i prioriteret rækkefølge
        result = []
        if "intro" in optimized_sections:
            result.append(optimized_sections["intro"])
            
        for section in priority_order:
            if section in optimized_sections:
                result.append(optimized_sections[section])
        
        # Tilføj evt. resterende sektioner
        for section, content in optimized_sections.items():
            if section != "intro" and section not in priority_order:
                result.append(content)
                
        return "\n".join(result)
    
    @staticmethod
    def compress_context(context: str, max_length: int = 1500) -> str:
        """
        Komprimerer kontekst ved at fjerne redundans og unødvendig information.
        
        Args:
            context: Original kontekst tekst
            max_length: Maksimal længde i tegn
            
        Returns:
            str: Komprimeret kontekst
        """
        if not context or len(context) <= max_length:
            return context
            
        # Del kontekst op i afsnit
        paragraphs = context.split('\n\n')
        
        # Hvis få afsnit, afkort hvert afsnit proportionalt
        if len(paragraphs) <= 3:
            total_reduction = len(context) - max_length
            if total_reduction <= 0:
                return context
                
            # Beregn hvor meget hvert afsnit skal reduceres
            total_chars = sum(len(p) for p in paragraphs)
            result = []
            
            for p in paragraphs:
                # Proportional reduktion baseret på afsnittets længde
                reduction_ratio = len(p) / total_chars
                target_p_length = max(50, int(len(p) - (total_reduction * reduction_ratio)))
                
                if len(p) > target_p_length:
                    # Afkort afsnit
                    p = p[:target_p_length-3] + "..."
                    
                result.append(p)
                
            return '\n\n'.join(result)
            
        else:
            # Ved mange afsnit, behold de vigtigste
            # Prioriter første og sidste afsnit + nogle i midten
            
            # Behold første og sidste afsnit
            essential = [paragraphs[0], paragraphs[-1]]
            
            # Find "vigtige" afsnit ved at se på nøgleord
            important_keywords = ["vigtig", "central", "afgørende", "kritisk", 
                                "hovedpunkt", "nøgle", "primær", "essentiel"]
            
            middle_paragraphs = paragraphs[1:-1]
            scored_paragraphs = []
            
            for i, p in enumerate(middle_paragraphs):
                # Score baseret på position (midten er mindre vigtig)
                pos_score = 1.0 - abs((i - len(middle_paragraphs)/2) / (len(middle_paragraphs)/2))
                
                # Score baseret på nøgleord
                keyword_score = sum(1 for kw in important_keywords if kw.lower() in p.lower()) * 0.2
                
                # Samlet score
                total_score = pos_score + keyword_score
                scored_paragraphs.append((p, total_score))
                
            # Sorter efter score (højest først)
            scored_paragraphs.sort(key=lambda x: x[1], reverse=True)
            
            # Tilføj højest-scorede afsnit indtil vi når maks længde
            current_length = sum(len(p) for p in essential) + (len(essential) - 1) * 2  # +2 for '\n\n'
            
            for p, _ in scored_paragraphs:
                if current_length + len(p) + 2 <= max_length:
                    essential.append(p)
                    current_length += len(p) + 2
                else:
                    # Hvis vi ikke kan tilføje hele afsnittet, kan vi evt. tilføje en del
                    space_left = max_length - current_length
                    if space_left > 50:  # Kun hvis der er plads til et meningsfuldt uddrag
                        essential.append(p[:space_left-3] + "...")
                    break
            
            # Sorter afsnit i oprindelig rækkefølge
            essential_set = set(essential)
            result = [p for p in paragraphs if p in essential_set]
            
            return '\n\n'.join(result)
    
    @staticmethod
    def create_efficient_angle_prompt(
        topic: str,
        topic_info: str,
        profile: dict,
        include_knowledge_distillate: bool = True
    ) -> str:
        """
        Skaber en effektiv prompt til vinkelgenerering med minimalt token-forbrug.
        
        Args:
            topic: Nyhedsemnet
            topic_info: Baggrundsinformation om emnet
            profile: Redaktionel profil som dictionary
            include_knowledge_distillate: Om videndistillat skal inkluderes
            
        Returns:
            str: Optimeret prompt
        """
        # Komprimér baggrundsinformation
        compressed_info = OptimizedPromptEngineering.compress_context(
            topic_info,
            OptimizedPromptEngineering.MAX_TOPIC_INFO_CHARS
        )
        
        # Pak profil-information sammen
        profile_info = []
        
        # Ekstraher og afkort information fra profilen
        principles = profile.get("kerneprincipper", [])
        principles_str = "\n".join([f"- {p}" for p in principles[:5]])
        
        tone = profile.get("tone_og_stil", "")
        if len(tone) > 300:
            tone = tone[:297] + "..."
            
        focus_areas = profile.get("fokusOmrader", [])
        focus_str = "\n".join([f"- {f}" for f in focus_areas[:5]])
        
        criteria = profile.get("nyhedsprioritering", {})
        criteria_str = "\n".join([f"- {k}: {v}" for k, v in list(criteria.items())[:5]])
        
        nogo = profile.get("noGoOmrader", [])
        nogo_str = "\n".join([f"- {n}" for n in nogo[:3]])
        
        # Byg en kompakt prompt
        prompt = f"""Du er en erfaren nyhedsjournalist. Generer vinkler på emnet "{topic}".

BAGGRUND:
{compressed_info}

PROFIL:
- Principper: {principles_str.replace('\n', '; ')}
- Tone: {tone}
- Fokus: {focus_str.replace('\n', '; ')}
- Vigtige nyhedskriterier: {criteria_str.replace('\n', '; ')}
- No-go: {nogo_str.replace('\n', '; ')}

OPGAVE:
Lav 5-7 vinkler som JSON-array med følgende struktur:
[
  {{
    "overskrift": "Kort og præcis overskrift",
    "beskrivelse": "2-3 sætninger om vinklen",
    "begrundelse": "Hvorfor denne vinkel passer til profilen",
    "nyhedskriterier": ["kriterie1", "kriterie2"],
    "startSpørgsmål": ["Spørgsmål 1", "Spørgsmål 2"]
  }}
]

VIGTIGT: Returner KUN et JSON-array med vinkelobjekter.
"""
        
        return prompt
    
    @staticmethod
    def create_expert_sources_prompt(
        topic: str, 
        angle_headline: str,
        angle_description: str
    ) -> str:
        """
        Skaber en effektiv prompt til ekspert-kildeforslag med minimalt token-forbrug og stærk vægt på kritisk/alternativ vinkel.
        
        Args:
            topic: Nyhedsemnet
            angle_headline: Vinkel-overskrift
            angle_description: Beskrivelse af vinklen
            
        Returns:
            str: Optimeret prompt
        """
        return f"""Find ekspertkilder til vinkel: \"{angle_headline}\" på emnet \"{topic}\".\n\nVINKEL BESKRIVELSE: {angle_description}\n\nVIGTIGT:\n- Foreslå kun eksperter, institutioner og datakilder, der er relevante for den specifikke kritiske, alternative eller anti-autoritære vinkel i overskriften og beskrivelsen – ikke bare generelle eksperter på emnet.\n- Prioritér kilder fra undergrunden, uafhængige forskere, aktivister, whistleblowere, alternative medier, samfundskritikere og andre, der udfordrer magtstrukturer, autoriteter eller det etablerede narrativ.\n- Undgå mainstream- eller autoritetsbærende eksperter, medmindre de er kendt for at have et kritisk eller alternativt synspunkt.\n- Svar i en stil der matcher et anti-autoritært, undergrunds- og samfundskritisk medie (råt, direkte, uformelt, evt. ironisk).\n- For hver ekspert: Giv en kort forklaring på, hvorfor de er relevante for netop denne vinkel.\n- Angiv et link til deres profilside (fx uafhængigt medie, aktivistisk organisation, alternativt forskningsmiljø) eller en central publikation, hvis muligt. Hvis intet link findes, angiv blot navn, titel og organisation.\n\nReturner JSON med dette format:\n{{\n  \"eksperter\": [\n    {{\n      \"navn\": \"Fulde navn\",\n      \"titel\": \"Stillingsbetegnelse\",\n      \"organisation\": \"Arbejdssted/tilhørsforhold\",\n      \"ekspertise\": \"Relevant ekspertområde\",\n      \"relevans\": \"Kort forklaring på relevans ift. vinkel\",\n      \"link\": \"URL til profil eller publikation, hvis muligt\"\n    }}\n  ],\n  \"institutioner\": [\n    {{\n      \"navn\": \"Institutionens navn\",\n      \"type\": \"Type institution\",\n      \"relevans\": \"Hvorfor relevant\",\n      \"link\": \"URL til institutionens relevante side, hvis muligt\"\n    }}\n  ],\n  \"datakilder\": [\n    {{\n      \"titel\": \"Datakilde titel\",\n      \"udgiver\": \"Udgiver\",\n      \"beskrivelse\": \"Kort beskrivelse\",\n      \"link\": \"URL til datakilden, hvis muligt\"\n    }}\n  ]\n}}\n\nKUN JSON, ingen forklarende tekst.\n"""
    
    @staticmethod
    def create_knowledge_distillate_prompt(topic: str, topic_info: str) -> str:
        """
        Skaber en effektiv prompt til videndistillat med minimalt token-forbrug.
        
        Args:
            topic: Nyhedsemnet
            topic_info: Baggrundsinformation om emnet
            
        Returns:
            str: Optimeret prompt
        """
        # Komprimér baggrundsinformation
        compressed_info = OptimizedPromptEngineering.compress_context(topic_info, 2000)
        
        return f"""Uddrag nøglefakta fra denne information om "{topic}".

BAGGRUND:
{compressed_info}

Returner som JSON:
{{
  "hovedpunkter": ["Punkt 1", "Punkt 2", "Punkt 3", "Punkt 4"],
  "noegletal": [
    {{ "tal": "42%", "beskrivelse": "Af noget", "kilde": "Kilde" }}
  ],
  "centralePaastand": [
    {{ "paastand": "Hovedpåstand", "kilde": "Kilde" }}
  ]
}}

KUN JSON, ingen forklarende tekst."""