"""
Common pytest fixtures for Vinkeljernet tests.
"""

import sys
import os
import json
import pytest
import yaml
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import RedaktionelDNA, VinkelForslag


@pytest.fixture
def sample_yaml_path(tmp_path):
    """Create a sample YAML config file for testing."""
    content = """
kerneprincipper:
  - Afslørende: Vi graver efter sandheden, især når den er ubehagelig for magthaverne.
  - Folkelig relevans: Historier skal ramme og betyde noget for almindelige danskere.
tone_og_stil: Direkte, skarp, kontant og letforståelig tone.
nyhedsprioritering:
  Aktualitet: 5
  Væsentlighed: 4
  Konflikt: 4
  Identifikation: 3
  Sensation: 2
fokusområder:
  - Politik og magt
  - Økonomi og arbejdsmarked
  - Forbruger og hverdagsliv
nogo_områder:
  - Indhold der krænker privatlivets fred uden samfundsmæssig relevans
  - Historier baseret på anonyme kilder uden faktuel verifikation
    """
    yaml_file = tmp_path / "test_profile.yaml"
    yaml_file.write_text(content)
    return yaml_file


@pytest.fixture
def sample_profile():
    """Return a sample RedaktionelDNA profile object."""
    return RedaktionelDNA(
        kerneprincipper=[
            {"Afslørende": "Vi graver efter sandheden, især når den er ubehagelig for magthaverne."},
            {"Folkelig relevans": "Historier skal ramme og betyde noget for almindelige danskere."}
        ],
        tone_og_stil="Direkte, skarp, kontant og letforståelig tone.",
        nyhedsprioritering={
            "Aktualitet": 5,
            "Væsentlighed": 4,
            "Konflikt": 4,
            "Identifikation": 3,
            "Sensation": 2
        },
        fokusområder=[
            "Politik og magt",
            "Økonomi og arbejdsmarked",
            "Forbruger og hverdagsliv"
        ],
        nogo_områder=[
            "Indhold der krænker privatlivets fred uden samfundsmæssig relevans",
            "Historier baseret på anonyme kilder uden faktuel verifikation"
        ]
    )


@pytest.fixture
def sample_topic_info():
    """Return sample topic information for testing."""
    return """
    Klimaforandringer er en global udfordring, der påvirker vejrmønstre, økosystemer og samfund verden over. 
    Nylige rapporter fra FN's klimapanel (IPCC) har understreget, at verden er på vej mod en temperaturstigning 
    på over 1,5 grader Celsius medmindre der implementeres drastiske reduktioner i CO2-udledninger inden 2030. 
    I Danmark har man oplevet flere ekstreme vejrhændelser, herunder kraftige skybrud og oversvømmelser, 
    der koster samfundet milliarder i skader og tilpasninger. Den danske regering har forpligtet sig til en 
    reduktion på 70% af drivhusgasser inden 2030 sammenlignet med 1990-niveauet, men flere eksperter peger på, 
    at der mangler konkrete handlingsplaner for at nå dette mål. Landbruget, transportsektoren og industrien 
    står for de største udledninger i Danmark.
    """


@pytest.fixture
def sample_angles():
    """Return sample angles for testing."""
    return [
        {
            "overskrift": "Klimaforandringer truer dansk landbrug: Hver tredje landmand frygter konkurs",
            "beskrivelse": "Nye tal viser, at danske landmænd i stigende grad bliver økonomisk presset af klimaforandringer. Tørke og oversvømmelser har reduceret udbyttet med gennemsnitligt 22% i 2023.",
            "nyhedskriterier": ["Væsentlighed", "Konflikt", "Aktualitet"],
            "begrundelse": "Vinklen kombinerer de folkelige konsekvenser med et aktuelt problem og rammer vores fokus på almindelige danskeres udfordringer.",
            "startSpørgsmål": [
                "Hvordan har klimaforandringerne konkret påvirket din drift det seneste år?",
                "Hvad mener du, politikerne bør gøre for at hjælpe landbruget gennem klimakrisen?"
            ]
        },
        {
            "overskrift": "Regeringens klimaplan mangler konkrete tiltag trods løfter",
            "beskrivelse": "Dokumenter viser, at regeringens klimaplan er forsinket og mangler konkrete tiltag. Eksperter kalder det 'greenwashing' og advarer om, at 2030-målene ikke kan indfries.",
            "nyhedskriterier": ["Konflikt", "Væsentlighed", "Aktualitet"],
            "begrundelse": "Vinklen afslører, at der er forskel på politikernes udmeldinger og faktiske handling, hvilket matcher vores afslørende kerneprincip.",
            "startSpørgsmål": [
                "Hvorfor er konkrete klimatiltag blevet forsinket?",
                "Hvilke konkrete tiltag mangler der i planen?"
            ]
        },
    ]


@pytest.fixture
def mock_perplexity_response():
    """Mock a successful response from the Perplexity API."""
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={
        "choices": [
            {
                "message": {
                    "content": "Sample research content about climate change",
                    "role": "assistant"
                }
            }
        ]
    })
    return mock_response


@pytest.fixture
def mock_perplexity_error_response():
    """Mock an error response from the Perplexity API."""
    mock_response = AsyncMock()
    mock_response.status = 401
    mock_response.text = AsyncMock(return_value=json.dumps({"error": "Unauthorized"}))
    mock_response.json = AsyncMock(side_effect=json.JSONDecodeError("Invalid JSON", "", 0))
    return mock_response


@pytest.fixture
def mock_openai_response():
    """Mock a successful response from the OpenAI API."""
    class MockChoice:
        class MockMessage:
            content = json.dumps([
                {
                    "overskrift": "Test Headline",
                    "beskrivelse": "Test description",
                    "begrundelse": "Test rationale",
                    "nyhedskriterier": ["Aktualitet", "Konflikt"],
                    "startSpørgsmål": ["Question 1?", "Question 2?"]
                }
            ])
        
        message = MockMessage()
    
    class MockResponse:
        choices = [MockChoice()]
    
    mock = MagicMock()
    mock.chat.completions.create.return_value = MockResponse()
    return mock


@pytest.fixture
def mock_openai_error():
    """Mock an error response from OpenAI."""
    mock = MagicMock()
    mock.chat.completions.create.side_effect = Exception("OpenAI API Error")
    return mock