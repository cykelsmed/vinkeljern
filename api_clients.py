"""
API client module for Vinkeljernet project.

This module provides functions to interact with external APIs such as
Perplexity for information retrieval and OpenAI for generating angles.
"""

# At the top of your file:
try:
    import aiohttp
except ImportError:
    print("Error: aiohttp package not found. Installing now...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "aiohttp==3.9.1"])
    import aiohttp

import requests
import sys
import json
from rich import print as rprint
from typing import Optional, Dict, Any, List
from openai import OpenAI
from config import PERPLEXITY_API_KEY, OPENAI_API_KEY
from models import RedaktionelDNA
from prompt_engineering import construct_angle_prompt, parse_angles_from_response

PERPLEXITY_API_URL = 'https://api.perplexity.ai/chat/completions'

async def fetch_topic_information(topic: str) -> Optional[str]:
    """
    Fetch information about a topic using the Perplexity API asynchronously.
    
    Args:
        topic: The news topic to fetch information about
        
    Returns:
        Optional[str]: The information retrieved or None if failed
    """
    if not PERPLEXITY_API_KEY:
        rprint("[red]Error: PERPLEXITY_API_KEY is not set. Please configure it in the .env file.[/red]")
        sys.exit(1)

    headers = {
        'Authorization': f'Bearer {PERPLEXITY_API_KEY}',
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    payload = {
        'model': 'llama-3-sonar-large-32k-online',
        'messages': [
            {"role": "system", "content": "Du er en professionel journalist, der skal give koncis og faktabaseret information om et aktuelt nyhedsemne. Giv grundige, men velstrukturerede fakta, baggrund og kontekst. Inkluder omhyggeligt datoer, tal og faktuelle detaljer, der er relevante for emnet. Undgå at udelade væsentlig information. Inkluder også kildereferencer, hvis muligt, i formatet [1] URL, [2] URL, osv."},
            {"role": "user", "content": f"Giv mig en grundig, men velstruktureret oversigt over den aktuelle situation vedrørende følgende nyhedsemne: {topic}. Inkluder relevante fakta, baggrund, kontekst og eventuelle nylige udviklinger. Vær præcis og faktabaseret. Nævn kildereferencer, hvis muligt."}
        ],
        'max_tokens': 1200,
        'temperature': 0.2
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(PERPLEXITY_API_URL, headers=headers, json=payload, timeout=45) as response:
                response.raise_for_status()
                data = await response.json()
                try:
                    return data['choices'][0]['message']['content']
                except (KeyError, IndexError):
                    rprint("[yellow]Warning: Unexpected response format from Perplexity API.[/yellow]")
                    return None
    except aiohttp.ClientError as e:
        rprint(f"[red]Error: Failed to fetch data from Perplexity API. Details: {e}[/red]")
        return None


def generate_angles(topic: str, topic_info: str, profile: RedaktionelDNA) -> List[Dict[str, Any]]:
    """
    Generate news angles based on topic information and editorial profile using OpenAI.
    
    Args:
        topic: The news topic
        topic_info: Information about the topic (from Perplexity)
        profile: RedaktionelDNA profile object
    
    Returns:
        List[Dict]: List of generated angle objects
    """
    if not OPENAI_API_KEY:
        rprint("[red]Error: OPENAI_API_KEY is not set. Please configure it in the .env file.[/red]")
        sys.exit(1)
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Convert profile to a structured format for the prompt
    profile_data = {
        "kerneprincipper": profile.kerneprincipper,
        "nyhedsprioritering": profile.nyhedsprioritering,
        "tone_og_stil": profile.tone_og_stil,
        "fokusområder": profile.fokusområder,
        "nogo_områder": profile.nogo_områder
    }
    
    # Create the prompt
    prompt = construct_angle_prompt(topic, topic_info, profile_data)
    
    # Call OpenAI API
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Du er en erfaren redaktør og vinkel-ekspert med dyb forståelse for journalistiske nyhedskriterier og medieprofiler. Du hjælper journalister med at udvikle skarpe, relevante og varierede vinkler på nyhedsemner baseret på mediers redaktionelle DNA."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2500
        )
        
        # Parse and return the angles
        return parse_angles_from_response(response.choices[0].message.content)
    except Exception as e:
        rprint(f"[red]Error: Failed to generate angles with OpenAI. Details: {e}[/red]")
        return []