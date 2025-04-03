# Udviklerguide til Vinkeljernet

Denne guide giver information om, hvordan du kan udvide og tilpasse Vinkeljernet-systemet. Den er rettet mod udviklere, der ønsker at tilføje funktioner, ændre eksisterende funktionalitet eller integrere systemet med andre værktøjer.

## Indholdsfortegnelse

1. [Arkitektur](#arkitektur)
2. [Udviklingsmiljø](#udviklingsmiljø)
3. [Kernekomponenter](#kernekomponenter)
4. [Udvidelsespunkter](#udvidelsespunkter)
5. [Test](#test)
6. [Bedste praksis](#bedste-praksis)
7. [Almindelige opgaver](#almindelige-opgaver)

## Arkitektur

Vinkeljernet følger en modulær arkitektur opdelt i flere uafhængige komponenter:

![Vinkeljernet Arkitektur](https://via.placeholder.com/800x500?text=Vinkeljernet+Arkitektur)

### Arkitekturlag

1. **Brugergrænseflader**
   - `app.py` - Flask webapplikation 
   - `main.py` - Kommandolinjegrænseflade

2. **Kernefunktionalitet**
   - `api_clients.py` - Integration med eksterne AI-tjenester
   - `angle_processor.py` - Filtrering og rangering af vinkler
   - `prompt_engineering.py` - Promptskabeloner og parsing

3. **Infrastruktur**
   - `config_loader.py` - Profil-indlæsning
   - `models.py` - Datamodeller
   - `error_handling.py` - Fejlhåndtering
   - `cache_manager.py` - API-caching

## Udviklingsmiljø

### Opsætning af udviklingsmiljø

```bash
# Klon repository
git clone https://github.com/yourusername/vinkeljernet.git
cd vinkeljernet

# Opret og aktiver virtuelt miljø
python -m venv venv
source venv/bin/activate  # På Windows: venv\Scripts\activate

# Installer udviklingsafhængigheder
pip install -r requirements-dev.txt

# Installer pre-commit hooks
pre-commit install
```

### Kodekvalitetsværktøjer

Vinkeljernet bruger følgende værktøjer til at opretholde kodekvalitet:

- **Black**: Automatisk formatering
- **Flake8**: Linting
- **isort**: Import sortering
- **mypy**: Type tjek
- **pytest**: Testing

Du kan køre disse værktøjer med følgende kommandoer:

```bash
# Formatering
black .

# Linting
flake8

# Import sortering
isort .

# Type tjek
mypy .

# Test
pytest
```

## Kernekomponenter

### Datamodeller (`models.py`)

Datamodeller bruger Pydantic til validering og typekonvertering. Hvis du tilføjer nye funktioner, skal du overveje at udvide de eksisterende modeller eller oprette nye.

```python
class NyModel(BaseModel):
    felt1: str
    felt2: int
    valgfrit_felt: Optional[List[str]] = None
    
    # Validatorer
    @field_validator('felt1')
    def validate_felt1(cls, value):
        if len(value) < 3:
            raise ValueError("felt1 skal være mindst 3 tegn")
        return value
```

### Profilhåndtering (`config_loader.py`)

Profilhåndtering er ansvarlig for at indlæse og validere YAML-profiler. Hvis du ændrer profilformatet, skal du opdatere:

1. `RedaktionelDNA`-modellen i `models.py`
2. Validerings- og indlæsningsfunktioner i `config_loader.py`
3. Dokumentation i `PROFIL_FORMAT.md`

### AI-Integration (`api_clients.py`)

AI-integrationsmodulet håndterer kommunikation med eksterne AI-tjenester:

- **Perplexity API**: Til baggrundsinformation om emner
- **Claude/Anthropic API**: Til vinkelgenerering
- **OpenAI API**: Alternativ for vinkelgenerering

For at tilføje en ny AI-provider:

1. Tilføj konfiguration i `config.py`
2. Implementer funktioner i `api_clients.py`
3. Udvid prompt-skabeloner i `prompt_engineering.py`

### Promptkonstruktion (`prompt_engineering.py`)

Dette modul håndterer konstruktionen af prompts til AI-modellerne og parsing af resultater.

Nøglefunktioner:
- `construct_angle_prompt()`: Bygger prompten baseret på emne og profil
- `parse_angles_from_response()`: Konverterer AI-svar til strukturerede objekter

## Udvidelsespunkter

### Tilføjelse af nye AI-modeller

For at tilføje støtte for en ny AI-model:

1. Tilføj nødvendige API-nøgler til `.env` og `config.py`
2. Implementer API-klient i `api_clients.py`
3. Tilføj promptskabelon i `prompt_engineering.py`
4. Tilføj respons-parsing, hvis nødvendigt

Eksempel på implementering af en ny API-klient:

```python
async def call_new_ai_provider(prompt: str) -> str:
    """Kalder en ny AI-provider med den givne prompt."""
    # Implementation af API-kald
    return response_text
```

### Tilføjelse af nye eksportformater

For at tilføje et nyt eksportformat:

1. Tilføj en formatfunktion i `formatters.py`
2. Opdater CLI-argumentparseren i `main.py`
3. Tilføj en downloadrute i `app.py`, hvis nødvendigt

Eksempel på implementering af et nyt formatmodul:

```python
def format_angles_as_new_format(
    angles: List[Dict], 
    profile_name: str, 
    topic: str
) -> str:
    """Formaterer vinkler i et nyt format."""
    # Implementation af formatering
    return formatted_content
```

### Udvidelse af webgrænsefladen

For at tilføje nye funktioner til webgrænsefladen:

1. Tilføj nye ruter i `app.py`
2. Opret nye HTML-skabeloner i `templates/`
3. Udvid JavaScript/CSS efter behov

### Tilføjelse af nye nyhedskriterier

For at tilføje et nyt nyhedskriterium:

1. Opdater `RedaktionelDNA`-modellen i `models.py` (hvis validation er nødvendig)
2. Udvid `calculate_criteria_score()` i `angle_processor.py`
3. Opdater promptskabelonen i `prompt_engineering.py`
4. Opdater dokumentation i `PROFIL_FORMAT.md`

## Test

### Teststruktur

Vinkeljernet bruger pytest som testframework. Tests er organiseret i følgende kategorier:

- **Unittest**: Isolerede tests af enkelte funktioner eller klasser
- **Integrationstests**: Tests af samspillet mellem flere komponenter
- **End-to-end tests**: Tests af hele systemet fra input til output

### Kørsel af tests

```bash
# Kør alle tests
pytest

# Kør specifikke testtyper
pytest tests/unit/
pytest tests/integration/

# Kør specifikke tests
pytest tests/unit/test_config_loader.py

# Kør med coverage
pytest --cov=.
```

### Skrivning af tests

Eksempel på en unittest:

```python
import pytest
from config_loader import load_and_validate_profile
from pathlib import Path

def test_load_valid_profile(tmp_path):
    """Test loading a valid profile."""
    # Opsæt testdata
    profile_path = tmp_path / "test_profile.yaml"
    with open(profile_path, 'w') as f:
        f.write("""
        navn: "Test Medie"
        beskrivelse: "Et testmedie"
        kerneprincipper:
          - "Princip 1"
        tone_og_stil: "Neutral tone"
        nyhedsprioritering:
          Aktualitet: 3
        fokusOmrader:
          - "Område 1"
        """)
    
    # Kør funktion der testes
    profile = load_and_validate_profile(profile_path)
    
    # Tjek resultater
    assert profile.navn == "Test Medie"
    assert len(profile.kerneprincipper) == 1
    assert profile.nyhedsprioritering["Aktualitet"] == 3
```

## Bedste praksis

### Kodestil

- Følg PEP 8 for Python-kodestil
- Brug Black til automatisk formatering
- Inkludér docstrings for alle offentlige funktioner og klasser
- Brug typehints for bedre type safety

### Fejlhåndtering

- Brug `try`/`except` til at fange specifikke fejl
- Log fejl med detaljer for debugging
- Konvertér tekniske fejlbeskeder til brugervenlige beskeder
- Brug custom exceptions for domænespecifikke fejl

### Asynkron kode

En del af Vinkeljernet er asynkron for bedre ydelse ved API-kald:

- Brug `async`/`await` konsistent
- Undgå at blokere event loop
- Brug proper error handling i asynkrone funktioner
- Overvej at bruge `asyncio.gather()` for parallelle API-kald

## Almindelige opgaver

### Tilføjelse af nye profiler

1. Opret en ny YAML-fil i `config/` mappen
2. Følg formatet beskrevet i `PROFIL_FORMAT.md`
3. Test profilen med validering: `python -c "from config_loader import validate_all_profiles; validate_all_profiles()"`

### Opdatering af AI-prompt

1. Åbn `prompt_engineering.py`
2. Ændr `construct_angle_prompt()` funktionen
3. Test med flere forskellige profiler og emner for at sikre gode resultater

### Ændring af vinkelrangeringen

1. Åbn `angle_processor.py`
2. Ændr `calculate_criteria_score()` eller `filter_and_rank_angles()` funktionerne
3. Test med forskellige profiler for at sikre, at rangeringen fungerer efter hensigten

### Tilføjelse af en ny CLI-kommando

1. Åbn `main.py`
2. Tilføj nye kommandolinjeargumenter i `parse_arguments()`
3. Implementer logikken for den nye kommando
4. Opdater dokumentationen

### Implementering af en ny cache-strategi

1. Ændr `cache_manager.py` med din nye cache-implementering
2. Opdater `cached_api` dekoratoren
3. Tilføj rydningslogik hvis nødvendigt