# Vinkeljernet API Dokumentation

Dette dokument beskriver de vigtigste moduler og funktioner i Vinkeljernet-systemet.

## Indholdsfortegnelse

1. [Models](#models)
2. [Config Loader](#config-loader)
3. [API Clients](#api-clients)
4. [Angle Processor](#angle-processor)
5. [Prompt Engineering](#prompt-engineering)
6. [Web Interface (app.py)](#web-interface)
7. [Command Line Interface (main.py)](#command-line-interface)

## Models

Modulet `models.py` definerer datamodeller og validering for Vinkeljernet.

### RedaktionelDNA

```python
class RedaktionelDNA(BaseModel):
    navn: str
    beskrivelse: str
    kerneprincipper: List[str]
    tone_og_stil: str
    nyhedsprioritering: Dict[str, int]
    fokusOmrader: List[str]
    noGoOmrader: List[str] = []
```

Dette er hovedmodellen for en redaktionel DNA-profil:
- `navn`: Mediets navn
- `beskrivelse`: Kort beskrivelse af mediet
- `kerneprincipper`: Liste af kerneværdier og principper
- `tone_og_stil`: Tekstbeskrivelse af tone, sprog og stil
- `nyhedsprioritering`: Ordbog med nyhedskriterier og deres vægtning (1-5)
- `fokusOmrader`: Liste af prioriterede dækningsområder
- `noGoOmrader`: Liste af områder eller vinkler, mediet undgår

### VinkelForslag

```python
class VinkelForslag(BaseModel):
    overskrift: str
    beskrivelse: str
    nyhedskriterier: List[str]
    begrundelse: str
    startSpørgsmål: List[str]
    flags: Optional[List[str]] = None
    kriterieScore: Optional[float] = None
```

Denne model repræsenterer en genereret vinkel:
- `overskrift`: Kort, præcis overskrift/rubrik
- `beskrivelse`: Vinklens beskrivelse (2-3 sætninger)
- `nyhedskriterier`: Liste af nyhedskriterier vinklen opfylder
- `begrundelse`: Begrundelse for hvorfor vinklen passer til profilen
- `startSpørgsmål`: Mulige spørgsmål til interviews
- `flags`: Specielle markører (valgfrit)
- `kriterieScore`: Beregnet score baseret på nyhedsprioritering (valgfrit)

## Config Loader

Modulet `config_loader.py` håndterer indlæsning og validering af YAML-profiler.

### Hovedfunktioner

```python
def load_and_validate_profile(profile_path: str | Path) -> RedaktionelDNA:
    """
    Indlæser og validerer en YAML-profil mod RedaktionelDNA-modellen.
    
    Args:
        profile_path: Sti til YAML-profilen (streng eller Path-objekt)
        
    Returns:
        RedaktionelDNA: Valideret profil som et RedaktionelDNA-objekt
        
    Raises:
        FileNotFoundError: Hvis profilen ikke findes
        yaml.YAMLError: Hvis der er fejl i YAML-formateringen
        ValueError: Hvis profilen ikke validerer mod RedaktionelDNA-modellen
    """
```

```python
def validate_all_profiles(config_dir: str = "config") -> None:
    """
    Validerer alle YAML-profiler i den angivne config-mappe.
    
    Tjekker især at 'tone_og_stil' er en simpel streng og ikke et nested objekt.
    """
```

## API Clients

Modulet `api_clients.py` implementerer integration med eksterne AI-tjenester.

### Hovedfunktioner

```python
async def fetch_topic_information(
    topic: str, 
    dev_mode: bool = False, 
    bypass_cache: bool = False,
    progress_callback = None
) -> str:
    """
    Henter baggrundsinformation om et emne via Perplexity API.
    
    Args:
        topic: Emnet at søge information om
        dev_mode: Hvis True, deaktiveres SSL-verifikation
        bypass_cache: Hvis True, ignoreres cache
        progress_callback: Valgfri callback for fremskridtsrapportering
        
    Returns:
        str: Formateret baggrundsinformation
    """
```

```python
def process_generation_request(
    topic: str,
    profile: RedaktionelDNA,
    bypass_cache: bool = False
) -> List[Dict[str, Any]]:
    """
    Synkron version af vinkelgenerering - beregnet til Flask-applikationen.
    
    Args:
        topic: Nyhedsemnet at generere vinkler for
        profile: Det redaktionelle DNA-profil at bruge
        bypass_cache: Om cache skal ignoreres
        
    Returns:
        List[Dict]: Liste af genererede vinkler
    """
```

```python
async def generate_angles(
    topic: str,
    topic_info: str,
    profile: RedaktionelDNA,
    bypass_cache: bool = False
) -> List[Dict[str, Any]]:
    """
    Genererer nyhedsvinkler baseret på emne, baggrundsinformation og profil.
    
    Args:
        topic: Nyhedsemnet
        topic_info: Baggrundsinformation om emnet
        profile: RedaktionelDNA-profil
        bypass_cache: Om cache skal ignoreres
        
    Returns:
        List[Dict]: Liste af genererede vinkelforslagsobjekter
    """
```

## Angle Processor

Modulet `angle_processor.py` håndterer filtrering, evaluering og rangering af vinkler.

### Hovedfunktioner

```python
def filter_and_rank_angles(
    angles: List[Dict[str, Any]],
    profile: RedaktionelDNA,
    num_angles: int = 5
) -> List[Dict[str, Any]]:
    """
    Filtrerer og rangerer vinkler baseret på profilmatch og nyhedskriterier.
    
    Args:
        angles: Liste af råvinkler fra LLM
        profile: RedaktionelDNA-profil
        num_angles: Antal vinkler at returnere
        
    Returns:
        List[Dict]: Rangeret liste af de bedste vinkler
    """
```

```python
def calculate_criteria_score(
    angle: Dict[str, Any],
    profile: RedaktionelDNA
) -> float:
    """
    Beregner en score for en vinkel baseret på nyhedskriterier i profilen.
    
    Args:
        angle: Vinkel med nyhedskriterier
        profile: RedaktionelDNA med nyhedsprioritering
        
    Returns:
        float: Vinkelscore (højere er bedre)
    """
```

```python
def is_angle_in_nogo_area(
    angle: Dict[str, Any],
    profile: RedaktionelDNA
) -> bool:
    """
    Kontrollerer om en vinkel falder inden for profilernes no-go områder.
    
    Args:
        angle: Vinkel at evaluere
        profile: RedaktionelDNA med no-go områder
        
    Returns:
        bool: True hvis vinklen er i et no-go område
    """
```

## Prompt Engineering

Modulet `prompt_engineering.py` håndterer konstruktion af prompts til LLM-modellerne.

### Hovedfunktioner

```python
def construct_angle_prompt(
    topic: str,
    topic_info: str,
    principper: str,
    tone_og_stil: str,
    fokusomrader: str,
    nyhedskriterier: str,
    nogo_omrader: str
) -> str:
    """
    Konstruerer en prompt til AI-modellen med instruktioner om at generere vinkler.
    
    Args:
        topic: Nyhedsemnet
        topic_info: Baggrundsinformation om emnet
        principper: Mediets kerneprincipper (formateret)
        tone_og_stil: Beskrivelse af tone og stil
        fokusomrader: Mediets fokusområder (formateret)
        nyhedskriterier: Nyhedsprioritering (formateret)
        nogo_omrader: No-go områder (formateret)
        
    Returns:
        str: Komplet prompt til AI-modellen
    """
```

```python
def parse_angles_from_response(response_text: str) -> List[Dict[str, Any]]:
    """
    Analyserer LLM-response og konverterer til en liste af vinkelobjekter.
    
    Args:
        response_text: Rå tekstsvar fra LLM
        
    Returns:
        List[Dict]: Liste af strukturerede vinkelobjekter
    """
```

## Web Interface

Modulet `app.py` implementerer webgrænsefladen til Vinkeljernet.

### Hovedruter

```python
@app.route('/', methods=['GET', 'POST'])
@rate_limit(limit=20, window=60)
def index():
    """Forside med formular til vinkelgenerering."""
```

```python
@app.route('/generate')
@rate_limit(limit=5, window=60)
def generate():
    """Generer vinkler og viser resultater."""
```

```python
@app.route('/detailed_report')
@rate_limit(limit=10, window=60)
def detailed_report():
    """Vis detaljeret rapport med omfattende baggrundsinformation."""
```

```python
@app.route('/download_report')
@rate_limit(limit=5, window=300)
def download_report():
    """Download den fulde detaljerede rapport som PDF."""
```

```python
@app.route('/regenerate_considerations', methods=['POST'])
@rate_limit(limit=5, window=60)
def regenerate_considerations():
    """Regenerer redaktionelle overvejelser."""
```

```python
@app.route('/regenerate_background', methods=['POST'])
@rate_limit(limit=5, window=60)
def regenerate_background():
    """Regenerer baggrundsinformation."""
```

### Hjælpefunktioner

```python
def get_topic_info_sync(topic: str, detailed: bool = False) -> str:
    """Synkron version af fetch_topic_information."""
```

```python
def generate_editorial_considerations(
    topic: str, 
    profile_name: str, 
    angles: List[Dict]
) -> str:
    """Generer redaktionelle overvejelser for de givne vinkler via Claude API."""
```

```python
def generate_report_html(results: Dict) -> str:
    """Generer HTML til den fulde detaljerede PDF-rapport."""
```

## Command Line Interface

Modulet `main.py` implementerer kommandolinjegrænsefladen.

### Hovedfunktioner

```python
def parse_arguments() -> Namespace:
    """Parser kommandolinjeargumenter for Vinkeljernet-applikationen."""
```

```python
async def main_async() -> None:
    """Asynkron hovedfunktion der organiserer applikationsflow."""
```

```python
async def run_interactive_cli() -> None:
    """Kører den interaktive CLI-grænseflade."""
```

```python
def main() -> None:
    """Hovedfunktion der organiserer applikationsflow."""
```

## Yderligere Information

For mere detaljeret dokumentation af den enkelte funktion, se kildekoden for hvert modul med indbyggede docstrings og kommentarer.