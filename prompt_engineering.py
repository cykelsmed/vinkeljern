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
    additional_context: str = "",
    output_format: str = "array"  # 'array' eller 'jsonl'
) -> str:
    """
    Construct a prompt for the angle generation API.
    Args:
        topic: The news topic
        topic_info: Research information about the topic
        principper: Editorial principles
        tone_og_stil: Editorial tone and style
        fokusområder: Editorial focus areas
        nyhedskriterier: News criteria
        nogo_områder: No-go areas
        additional_context: Extra context (e.g. knowledge distillate)
        output_format: 'array' (default) or 'jsonl'
    Returns:
        str: Prompt for LLM
    """
    prompt = f"""
# EMNE
{topic}

# BAGGRUNDSINFORMATION
{topic_info}

# VIDENDISTILLAT / EKSTRA KONTEKST
{additional_context}

# REDAKTIONEL PROFIL
Kerneprincipper: {principper}
Nyhedskriterier: {nyhedskriterier}
Fokusområder: {fokusområder}
No-go områder: {nogo_områder}
Tone og stil: {tone_og_stil}

# OPGAVE
Du er en erfaren dansk journalist. Generér 5-8 forskellige, relevante vinkler på emnet, der matcher profilen. Hver vinkel skal være detaljeret, original og målrettet målgruppen.

# OUTPUTFORMAT
Returnér KUN et gyldigt JSON-array (liste) hvor hvert objekt har PRÆCIS disse felter:
- "overskrift": string (fængende, dækkende overskrift)
- "beskrivelse": string (detaljeret uddybning af vinklen)
- "begrundelse": string (hvorfor er denne vinkel relevant for målgruppen/profilen?)
- "startSpørgsmål": liste af strings (2-3 gode, konkrete spørgsmål til research/interview)
- "nyhedskriterier": liste af strings (1-3 relevante nyhedskriterier)

Eksempel på format:
[
  {{
    "overskrift": "Danskerne sorterer mere affald – men hvad sker der med komposten?",
    "beskrivelse": "En dybdegående analyse af hvordan øget affaldssortering har påvirket mængden og kvaliteten af kompost i Danmark. Fokus på både miljømæssige og økonomiske konsekvenser.",
    "begrundelse": "Relevant for læsere, der interesserer sig for miljø og bæredygtighed. Matcher profilens fokus på oplysning og samfundsperspektiv.",
    "startSpørgsmål": [
      "Hvordan har affaldssortering ændret sig de seneste 5 år?",
      "Hvilke udfordringer oplever kommunerne med kompostkvalitet?"
    ],
    "nyhedskriterier": ["aktualitet", "væsentlighed"]
  }}
]

VIGTIGT:
- Returnér KUN et gyldigt JSON-array uden nogen forklarende tekst før eller efter
- Brug nøjagtigt de feltnavne der er angivet ovenfor
- Alle felter er påkrævede og skal være udfyldt
- Brug dobbelte anførselstegn for alle strenge
- Svar på dansk
- Maks 8 vinkler
"""
    if output_format == "jsonl":
        prompt += "\nHvis du ikke kan returnere hele listen som ét JSON-array, så returnér én gyldig JSON pr. linje (JSON Lines format)."
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
    logger.debug(f"Raw LLM response: {response_text}")
        
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
        for raw_angle in raw_angles:
            try:
                # Validate with Pydantic model
                validated_angle = VinkelForslag(**raw_angle).dict()
                angles.append(validated_angle)
            except Exception as e:
                # Log the raw angle that caused the issue for better debugging
                angle_preview = str(raw_angle)[:100] + "..." if len(str(raw_angle)) > 100 else str(raw_angle)
                logger.warning(f"Skipping invalid angle ({str(e)}): {angle_preview}")
                continue

        logger.info(f"Successfully parsed and validated {len(angles)} angles from response")
        return angles

    except Exception as e:
        logger.error(f"Error parsing angles from response: {e}")
        return []

async def generate_missing_fields_with_llm(angle: dict, missing_fields: list) -> dict:
    """
    Brug Claude Haiku til at generere manglende felter ('begrundelse', 'startSpørgsmål') for en vinkel.
    """
    import aiohttp
    import os
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
    ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01"
    }
    prompt = f"""
Du får en vinkel til et journalistisk emne. Udfyld de manglende felter:

Overskrift: {angle.get('overskrift', '')}
Beskrivelse: {angle.get('beskrivelse', '')}

Returnér kun de manglende felter som gyldig JSON. Brug dansk.
"""
    if "begrundelse" in missing_fields:
        prompt += "\nGenerér en kort, præcis begrundelse for hvorfor denne vinkel er relevant for målgruppen/profilen."
    if "startSpørgsmål" in missing_fields:
        prompt += "\nGenerér 2-3 gode, konkrete startspørgsmål til research/interview som en liste af strings."
    payload = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 256,
        "temperature": 0.2,
        "system": "Du er en hjælpsom dansk journalistisk assistent.",
        "messages": [{"role": "user", "content": prompt}],
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(ANTHROPIC_API_URL, json=payload, headers=headers) as response:
            if response.status != 200:
                return {}
            data = await response.json()
            text = data['content'][0]['text'] if 'content' in data and data['content'] and 'text' in data['content'][0] else None
            if not text:
                return {}
            try:
                import json
                result = json.loads(text)
                return result if isinstance(result, dict) else {}
            except Exception:
                return {}

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
        principles_str_flat = principles_str.replace('\n', '; ')
        
        tone = profile.get("tone_og_stil", "")
        if len(tone) > 300:
            tone = tone[:297] + "..."
        
        focus_areas = profile.get("fokusOmrader", [])
        focus_str = "\n".join([f"- {f}" for f in focus_areas[:5]])
        focus_str_flat = focus_str.replace('\n', '; ')
        
        criteria = profile.get("nyhedsprioritering", {})
        criteria_str = "\n".join([f"- {k}: {v}" for k, v in list(criteria.items())[:5]])
        criteria_str_flat = criteria_str.replace('\n', '; ')
        
        nogo = profile.get("noGoOmrader", [])
        nogo_str = "\n".join([f"- {n}" for n in nogo[:3]])
        nogo_str_flat = nogo_str.replace('\n', '; ')
        
        # Byg en kompakt prompt
        prompt = f"""Du er en erfaren nyhedsjournalist. Generer vinkler på emnet \"{topic}\".\n\nBAGGRUND:\n{compressed_info}\n\nPROFIL:\n- Principper: {principles_str_flat}\n- Tone: {tone}\n- Fokus: {focus_str_flat}\n- Vigtige nyhedskriterier: {criteria_str_flat}\n- No-go: {nogo_str_flat}\n\nOPGAVE:\nLav 5-7 vinkler som JSON-array med følgende struktur:\n[\n  {{\n    \"overskrift\": \"Kort og præcis overskrift\",\n    \"beskrivelse\": \"2-3 sætninger om vinklen\",\n    \"begrundelse\": \"Hvorfor denne vinkel passer til profilen\",\n    \"nyhedskriterier\": [\"kriterie1\", \"kriterie2\"],\n    \"startSpørgsmål\": [\"Spørgsmål 1\", \"Spørgsmål 2\"]\n  }}\n]\n\nVIGTIGT: Returner KUN et JSON-array med vinkelobjekter.\n"""
        
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