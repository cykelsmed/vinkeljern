# Vinkeljernet Utils

En integreret løsningspakke til Vinkeljernet, der kombinerer robust JSON-parsing og ekspertkilde-håndtering
i en samlet pakke, der er nem at integrere i den eksisterende kodebase.

## Funktionaliteter

- **Robust JSON-parsing**: Håndterer malformeret JSON fra LLM-modeller med multiple recovery-strategier
- **Ekspertkilde-håndtering**: Strukturerede modeller og funktioner til at arbejde med ekspertkilder
- **Avanceret fejlhåndtering**: Graceful degradation ved API-fejl
- **Caching**: Intelligent caching med TTL for bedre ydelse
- **Performance-sporing**: Grundig statistikindsamling til monitorering
- **Testværktøjer**: Omfattende testsuite til validering af funktionalitet

## Installation

Pakken er designet til at kunne bruges direkte i Vinkeljernet-projektet.
Alle moduler er synlige igennem toppen af pakken:

```python
from vinkeljernet_utils import robust_json_parse, get_expert_sources, setup
```

## Konfiguration

Start med at konfigurere pakken i starten af din applikation:

```python
from vinkeljernet_utils import setup

config = setup(
    debug=True,  # Aktivér debug-mode
    log_level="INFO",  # Sæt log-niveau
    cache_enabled=True,  # Aktivér caching
    timeout=30  # Standard timeout for API-kald i sekunder
)
```

## JSON-parsing

Robust parsing af JSON fra diverse API'er:

```python
from vinkeljernet_utils import robust_json_parse, safe_parse_json

# Robust parsing med detaljeret fejlsporing
result, error = robust_json_parse(response_text, context="angle generation")
if error:
    print(f"Parsing error: {error}")
else:
    print(f"Found {len(result)} objects")

# Sikker parsing, der aldrig kaster exceptions
parsed_data = safe_parse_json(
    response_text, 
    context="knowledge distillate",
    fallback={"key_statistics": [], "key_claims": []}
)
```

## Ekspertkilder

Håndtering af ekspertkilder med validering:

```python
from vinkeljernet_utils import get_expert_sources, validate_expert_sources
from vinkeljernet_utils.expertise import ExpertSourceResult

# Indhent ekspertkilder
sources = await get_expert_sources(
    topic="Økonomisk vækst",
    angle_headline="Ny dansk vækst overgår EU-gennemsnit",
    angle_description="Danmark oplever højere økonomisk vækst end EU-gennemsnittet",
    api_client=api_client  # Din API-klient (valgfri)
)

# Valider kildedata
valid, issues = validate_expert_sources(sources)
if not valid:
    print(f"Issues with expert sources: {issues}")

# Udpak og vis data
for expert in sources.experts:
    print(f"Expert: {expert.name}, {expert.title} at {expert.organization}")
```

## Performance-sporing

Spor API-ydelse og cachehits:

```python
from vinkeljernet_utils import get_performance_stats

stats = get_performance_stats()
print(f"JSON parsing success rate: {stats.json_parsing_success_rate:.1f}%")
print(f"Cache hit rate: {stats.cache_hit_rate:.1f}%")
print(f"Expert sources success rate: {stats.expert_sources_success_rate:.1f}%")
```

## Kørsel af tests

Pakken har en integreret testsuite:

```python
from vinkeljernet_utils.testing import run_tests

success, message = run_tests(verbose=True)
print(message)
```

## Integration med Vinkeljernet

For at integrere løsningen i dit projekt:

1. Importer pakken i din applikation:
   ```python
   from vinkeljernet_utils import setup, robust_json_parse, get_expert_sources
   ```

2. Konfigurer pakken ved opstart:
   ```python
   setup(debug=True, log_level="INFO")
   ```

3. Erstat eksisterende JSON-parsing med de robuste funktioner:
   ```python
   # I stedet for:
   # parsed_data = json.loads(response_text)
   
   # Brug:
   parsed_data, error = robust_json_parse(response_text, context="angle generation")
   if error:
       logger.error(f"JSON parsing error: {error}")
       # Håndtér fejlen...
   ```

4. Brug ekspertkilde-funktionerne i din applikation:
   ```python
   expert_sources = await get_expert_sources(
       topic=topic,
       angle_headline=angle.overskrift,
       angle_description=angle.beskrivelse,
       api_client=your_api_client
   )
   
   angle.ekspertKilder = expert_sources.to_dict()
   ```

## Logning

Pakken bruger standardbiblioteket logging. For at tilpasse logning:

```python
import logging

# Konfigurér pakkeloggere
logging.getLogger("vinkeljernet_utils").setLevel(logging.DEBUG)
logging.getLogger("vinkeljernet_utils.json_parsing").setLevel(logging.DEBUG)
```

## Udvidelse

Pakken er designet til at være udvidelig:

- Tilføj nye recovery-strategier i `json_parsing/__init__.py`
- Tilføj nye modeller i `expertise/__init__.py`
- Udvid testsuite'en i `testing/__init__.py`

## License

Samme licens som Vinkeljernet-projektet.