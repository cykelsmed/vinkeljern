"""
Vinkeljernet Utils - En integreret lÃ¸sningspakke til Vinkeljernet

Denne pakke kombinerer robust JSON-parsing og ekspertkilde-hÃ¥ndtering i en samlet lÃ¸sning
der er designet til at vÃ¦re enkelt at integrere i den eksisterende Vinkeljernet-kodebase.

Modulet indeholder:
- Robuste JSON-parsere der kan hÃ¥ndtere malformeret JSON fra API'er
- Ekspertkilde-hÃ¥ndtering med caching og validering
- Avancerede optimeringer til parallel databehandling
- FejlhÃ¥ndtering og fallbacks
- TestvÃ¦rktÃ¸jer til validering

Brug:
```python
from vinkeljernet_utils import setup_json_handlers, get_expert_sources, parse_json_safely
from vinkeljernet_utils.json_parsing import robust_json_parse
from vinkeljernet_utils.expertise import ExpertSourceManager
```

For mere detaljerede anvendelsesscenarier, se dokumentationen for hvert submodul.
"""

__version__ = "1.0.0"
__author__ = "Vinkeljernet Team"

# Import central komponenter til toppakken
from vinkeljernet_utils.json_parsing import (
    robust_json_parse,
    safe_parse_json,
    parse_structured_json,
    JSONParseError,
    JSONParsingConfig
)

from vinkeljernet_utils.expertise import (
    ExpertSourceManager,
    ExpertSource,
    get_expert_sources,
    validate_expert_sources
)

from vinkeljernet_utils.common import (
    setup_logging,
    configure_package,
    PackageConfig,
    get_performance_stats
)

# Eksporter centrale funktioner ved toppakken
__all__ = [
    # JSON parsing
    "robust_json_parse",
    "safe_parse_json", 
    "parse_structured_json",
    "JSONParseError",
    "JSONParsingConfig",
    
    # Expertise
    "ExpertSourceManager",
    "ExpertSource",
    "get_expert_sources",
    "validate_expert_sources",
    
    # Common
    "setup_logging",
    "configure_package",
    "PackageConfig",
    "get_performance_stats",
    
    # Module metadata
    "__version__",
    "__author__"
]

# Globale konfigurationer og standardopsÃ¦tning
_config = None

def setup(debug=False, log_level="INFO", cache_enabled=True, timeout=30):
    """
    OpsÃ¦t vinkeljernet_utils-pakken med de angivne indstillinger.
    
    Args:
        debug: AktivÃ©r debug-tilstand for udvidet logning
        log_level: Logniveau ("DEBUG", "INFO", "WARNING", "ERROR")
        cache_enabled: AktivÃ©r caching af resultater
        timeout: Standard timeout for API-kald i sekunder
        
    Returns:
        PackageConfig: Den aktuelle konfiguration
    """
    global _config
    from vinkeljernet_utils.common import configure_package, setup_logging
    
    # Konfigurer logging
    setup_logging(log_level, debug=debug)
    
    # Konfigurer pakken
    _config = configure_package(
        debug=debug,
        cache_enabled=cache_enabled,
        timeout=timeout
    )
    
    return _config


def get_config():
    """
    Hent den aktuelle pakkekonfiguration.
    
    Returns:
        PackageConfig: Den aktuelle konfiguration eller None hvis pakken ikke er initialiseret
    """
    global _config
    return _config


# Automatisk konfiguration fÃ¸rste gang pakken importeres
if _config is None:
    setup()