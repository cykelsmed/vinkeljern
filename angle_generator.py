"""
Angle generator module for Vinkeljernet project.

This module handles the core functionality of generating news angles
based on topic information and editorial DNA profiles.
"""

import os
import json
from typing import List, Dict, Any, Optional
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.llms import OpenAI
from rich import print as rprint
from pydantic import BaseModel, Field

from api_clients import fetch_topic_information
from models import RedaktionelDNA


class VinkelForslag(BaseModel):
    """Model representing a suggested news angle."""
    overskrift: str = Field(description="Catchy headline for the news angle")
    beskrivelse: str = Field(description="Brief description of the angle (2-3 sentences)")
    nyhedskriterier: List[str] = Field(description="Primary news criteria this angle fulfills")
    begrundelse: str = Field(description="Rationale linking to editorial DNA")
    start_sporgsmaal: List[str] = Field(description="2-3 key interview/research questions")


class VinkelForslagListe(BaseModel):
    """Model representing a list of news angle suggestions."""
    vinkler: List[VinkelForslag] = Field(description="List of news angle suggestions")
    kilder: List[str] = Field(description="Sources used for information")


async def get_topic_information(topic: str) -> Optional[str]:
    """
    Fetch information about a topic using Perplexity API.
    
    Args:
        topic: The news topic to fetch information about
        
    Returns:
        Optional[str]: Information about the topic or None if the API call fails
    """
    rprint(f"[blue]Indhenter information om emnet: '{topic}'...[/blue]")
    
    # Enhance the query to get comprehensive information
    enhanced_query = f"""
    Giv en grundig, faktuel og opdateret oversigt over nyhedsemnet: {topic}.
    Inkluder vigtige fakta, aktører, baggrund, aktuelle udviklinger og forskellige perspektiver.
    Vær objektiv og omfattende. Angiv primære kilder til slut.
    """
    
    information = fetch_topic_information(enhanced_query)
    
    if not information:
        rprint("[red]Kunne ikke indhente information om emnet fra Perplexity API.[/red]")
        return None
        
    rprint("[green]✓[/green] Information indhentet")
    return information


def create_prompt_template() -> PromptTemplate:
    """
    Create the prompt template for generating news angles.
    
    Returns:
        PromptTemplate: The template to use for LLM prompting
    """
    # Design a comprehensive prompt that includes all necessary context and instructions
    template = """
    Du er en erfaren journalist og redaktør med ekspertise i at udvikle nyhedsvinkler.
    Din opgave er at generere nyhedsvinkler for følgende emne, baseret på den redaktionelle DNA og den aktuelle information.
    
    ### NYHEDSEMNE:
    {emne}
    
    ### AKTUEL INFORMATION OM EMNET:
    {information}
    
    ### REDAKTIONEL DNA:
    Kerneprincipper: {kerneprincipper}
    Nyhedsprioritering: {nyhedsprioritering}
    Tone og stil: {tone_og_stil}
    Fokusområder: {fokusomraader}
    No-go områder: {nogo_omraader}
    
    ### INSTRUKTIONER:
    1. Generer 10-15 forskellige potentielle vinkler på nyhedsemnet.
    2. Hver vinkel skal være:
       - Original og ikke-kliché
       - I overensstemmelse med mediets redaktionelle DNA
       - Baseret på den aktuelle information
       - Journalistisk realiserbar (ikke spekulativ)
    3. Sikr diversitet i perspektiver (menneskeligt, politisk, økonomisk, konsekvens, etc.)
    4. Undgå vinkler der falder inden for no-go områderne.
    5. For hver vinkel, identificer de primære nyhedskriterier den opfylder (sensation, identifikation, konflikt, etc.)
    6. Flag eventuelle modsigelser eller usikkerheder i informationen.
    
    ### OUTPUT FORMAT:
    {format_instructions}
    """
    
    # Create a PydanticOutputParser for the VinkelForslagListe model
    parser = PydanticOutputParser(pydantic_object=VinkelForslagListe)
    
    return PromptTemplate(
        template=template,
        input_variables=[
            "emne",
            "information",
            "kerneprincipper",
            "nyhedsprioritering", 
            "tone_og_stil",
            "fokusomraader",
            "nogo_omraader"
        ],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )


def score_and_filter_angles(all_angles: List[VinkelForslag], 
                           nyhedsprioritering: Dict[str, int], 
                           antal: int = 5) -> List[VinkelForslag]:
    """
    Score, filter and select the best news angles based on editorial DNA.
    
    Args:
        all_angles: List of all generated angle suggestions
        nyhedsprioritering: Dictionary of news criteria priorities from the profile
        antal: Number of angles to return (default: 5)
        
    Returns:
        List[VinkelForslag]: The top selected angles
    """
    scored_angles = []
    
    for angle in all_angles:
        # Calculate score based on news criteria that match priorities
        score = 0
        matching_criteria = []
        
        for criterion in angle.nyhedskriterier:
            criterion_lower = criterion.lower()
            for priority_key, priority_value in nyhedsprioritering.items():
                if priority_key in criterion_lower:
                    score += priority_value
                    matching_criteria.append(priority_key)
        
        # Add diversity bonus (to ensure we don't just get angles with the same criteria)
        if len(matching_criteria) > 1:
            score += len(set(matching_criteria))
            
        scored_angles.append((angle, score))
    
    # Sort by score (descending) and return the top 'antal' angles
    scored_angles.sort(key=lambda x: x[1], reverse=True)
    
    # Try to ensure diversity in the selected angles
    selected_angles = []
    used_criteria_sets = []
    
    for angle, _ in scored_angles:
        # Create a frozen set of the angle's criteria for comparison
        criteria_set = frozenset([c.lower() for c in angle.nyhedskriterier])
        
        # If we already have enough angles, break
        if len(selected_angles) >= antal:
            break
            
        # If we have less than half our target, accept any high-scoring angle
        if len(selected_angles) < antal // 2:
            selected_angles.append(angle)
            used_criteria_sets.append(criteria_set)
        # Otherwise, try to diversify by avoiding exact criteria matches
        elif criteria_set not in used_criteria_sets:
            selected_angles.append(angle)
            used_criteria_sets.append(criteria_set)
    
    # If we still need more angles, just take the highest scoring remaining ones
    remaining_slots = antal - len(selected_angles)
    if remaining_slots > 0:
        for angle, _ in scored_angles:
            if angle not in selected_angles:
                selected_angles.append(angle)
                if len(selected_angles) >= antal:
                    break
    
    return selected_angles


async def generate_angles(topic: str, profile: RedaktionelDNA) -> Optional[List[VinkelForslag]]:
    """
    Generate news angles based on topic and editorial profile.
    
    Args:
        topic: News topic to generate angles for
        profile: Editorial DNA profile
        
    Returns:
        Optional[List[VinkelForslag]]: List of generated news angles or None if generation fails
    """
    # Check if profile is a string and try to handle it
    if isinstance(profile, str):
        rprint(f"[yellow]WARNING: profile parameter was passed as a string. Attempting to handle.[/yellow]")
        try:
            from config_manager import load_profile
            profile = load_profile(profile)
        except Exception as e:
            rprint(f"[red]Error converting profile string to RedaktionelDNA: {e}[/red]")
            return None
            
    # Check if profile is a RedaktionelDNA object
    if not isinstance(profile, RedaktionelDNA):
        rprint(f"[red]Error: profile must be a RedaktionelDNA object, got {type(profile)}[/red]")
        return None
        
    # Get information about the topic
    information = await get_topic_information(topic)
    if not information:
        return None
    
    # Prepare input for the LLM
    rprint("[blue]Genererer vinkler baseret på emne og redaktionsprofil...[/blue]")
    
    # Format kerneprincipper for prompt
    try:
        kerneprincipper_str = ", ".join([list(p.keys())[0] for p in profile.kerneprincipper])
    except (AttributeError, TypeError) as e:
        rprint(f"[red]Error accessing kerneprincipper: {e}[/red]")
        rprint(f"[red]Profile type: {type(profile)}[/red]")
        return None
    
    # DEAKTIVERET: Langchain har kompatibilitetsproblemer med nyeste OpenAI API
    # Dette vil kræve en opdatering af Langchain
    # I stedet for at bruge Langchain, bruger vi direkte OpenAI API i api_clients.py
    
    # Fortæl brugeren at vi ikke understøtter denne metode længere og returner None
    rprint("[yellow]Angle generator modulet er ikke længere kompatibelt med nyeste version af OpenAI API. "
           "Brug venligst main.py med de anbefalede kommandoer i stedet.[/yellow]")
    return None


def format_angles_for_output(angles: List[VinkelForslag], kilder: List[str]) -> str:
    """
    Format angles for text output.
    
    Args:
        angles: List of angle suggestions
        kilder: List of sources
        
    Returns:
        str: Formatted text output
    """
    output = []
    
    output.append("# VINKELFORSLAG\n")
    
    for i, angle in enumerate(angles, 1):
        output.append(f"## Vinkel {i}: {angle.overskrift}")
        output.append(f"\n{angle.beskrivelse}")
        output.append(f"\n**Nyhedskriterier:** {', '.join(angle.nyhedskriterier)}")
        output.append(f"\n**Begrundelse:** {angle.begrundelse}")
        output.append(f"\n**Startspørgsmål:**")
        for q in angle.start_sporgsmaal:
            output.append(f"\n- {q}")
        output.append("\n" + "-"*40 + "\n")
    
    output.append("# KILDER")
    output.append("\nInformation indhentet via Perplexity AI med følgende primære kilder:")
    for kilde in kilder:
        output.append(f"\n- {kilde}")
    
    return "\n".join(output)