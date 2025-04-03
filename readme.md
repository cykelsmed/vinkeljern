# Vinkeljernet

Vinkeljernet er en AI-drevet journalistisk vinkelgenerator, der skaber nyhedsvinkler baseret på redaktionelle DNA-profiler. Værktøjet hjælper journalister og redaktører med hurtigt at identificere lovende vinkler på historier, der passer til deres publikations unikke stil og fokusområder.

## Funktioner

- **Profilbaseret vinkelgenerering**: Generer vinkler skræddersyet til en publikations redaktionelle DNA
- **AI-drevet research**: Indsaml automatisk baggrundsinformation om emner
- **Redaktionel analyse**: Få detaljeret analyse af genererede vinkler med redaktionelle overvejelser
- **Multi-format eksport**: Download rapporter som PDF eller tekst
- **Webgrænseflade**: Brugervenlig browserinterface
- **CLI-værktøj**: Kommandolinjegrænseflade til scripting og integration

## Installation

### Forudsætninger

- Python 3.9 eller højere
- pip (Python pakkehåndtering)
- wkhtmltopdf (til PDF-generering)

### Grundlæggende installation

```bash
# Klon repository
git clone https://github.com/yourusername/vinkeljernet.git
cd vinkeljernet

# Opret og aktiver et virtuelt miljø
python -m venv venv
source venv/bin/activate  # På Windows: venv\Scripts\activate

# Installer afhængigheder
pip install -r requirements.txt
```

### Udviklingsinstallation

Hvis du planlægger at bidrage til Vinkeljernet:

```bash
# Installer udviklingsafhængigheder
pip install -r requirements-dev.txt

# Installer pre-commit hooks
pre-commit install
```

### Installation af wkhtmltopdf (til PDF-generering)

PDF-generering kræver wkhtmltopdf på dit system:

- **macOS**: `brew install wkhtmltopdf`
- **Ubuntu/Debian**: `sudo apt-get install wkhtmltopdf`
- **Windows**: Download fra [wkhtmltopdf.org](https://wkhtmltopdf.org/downloads.html)

## Konfiguration

### Miljøvariabler

Vinkeljernet kræver API-nøgler til eksterne tjenester. Opret en `.env` fil i projektets rodmappe:

```
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
PERPLEXITY_API_KEY=your_perplexity_api_key
FLASK_SECRET_KEY=your_flask_secret_key
```

### Redaktionelle profiler

Vinkeljernet bruger YAML-filer til at definere redaktionelle DNA-profiler. Disse profiler findes i `config/` mappen.

For detaljeret information om oprettelse af profiler, se [PROFIL_FORMAT.md](docs/PROFIL_FORMAT.md).

## Anvendelse

### Webgrænseflade

Start webgrænsefladen:

```bash
python app.py
```

Besøg derefter `http://localhost:5000` i din browser.

### Kommandolinjegrænseflade

Vinkeljernet har en kommandolinjegrænseflade til generering af vinkler:

```bash
# Grundlæggende brug
python main.py --emne "klimaforandringer" --profil config/dr_profil.yaml

# Gem output til fil
python main.py --emne "klimaforandringer" --profil config/dr_profil.yaml --output resultater.json

# Interaktiv tilstand
python main.py --interactive
```

#### Kommandolinjeargumenter

Obligatoriske argumenter:
- `--emne` eller `-e`: Det nyhedsemne, der skal genereres vinkler for.
- `--profil` eller `-p`: Stien til YAML-filen med den redaktionelle DNA-profil.

Valgfri argumenter:
- `--output` eller `-o`: Valgfri filsti til at gemme outputtet.
- `--interactive` eller `-i`: Start i interaktiv CLI-tilstand.
- `--dev-mode`: Kør i udviklingstilstand (deaktiverer SSL-verifikation).
- `--clear-cache`: Ryd cache før programmet kører.
- `--bypass-cache`: Ignorer cache og tving friske API-kald.
- `--format`: Format for output (`json`, `markdown`, `html`).
- `--show-circuits`: Vis status for circuit breakers.
- `--reset-circuits`: Nulstil alle circuit breakers.
- `--debug`: Aktiver debug-tilstand med ekstra output.

### Python API

Du kan også bruge Vinkeljernet som et bibliotek i dine Python-projekter:

```python
from config_loader import load_and_validate_profile
from api_clients import process_generation_request
from pathlib import Path

# Indlæs profil
profile = load_and_validate_profile(Path("config/dr_profil.yaml"))

# Generer vinkler
topic = "klimaforandringer"
angles = process_generation_request(topic, profile)

# Bearbejd resultater
for angle in angles:
    print(f"Overskrift: {angle['overskrift']}")
    print(f"Beskrivelse: {angle['beskrivelse']}")
    print("---")
```

## Projektstruktur

```
vinkeljernet/
├── app.py                  # Webgrænseflade
├── main.py                 # CLI-applikation
├── models.py               # Datamodeller
├── config_loader.py        # Profilindlæsning
├── angle_processor.py      # Vinkelgenerering og rangering
├── api_clients.py          # Eksterne API-klienter
├── prompt_engineering.py   # Promptkonstruktion
├── config/                 # Redaktionelle profiler
│   ├── dr_profil.yaml
│   ├── politiken_profil.yaml
│   └── ...
├── templates/              # HTML-skabeloner
└── docs/                   # Dokumentation
```

## Udvidelse af Vinkeljernet

For information om udvidelse og tilpasning af Vinkeljernet, se [UDVIKLER_GUIDE.md](docs/UDVIKLER_GUIDE.md).

## Fejlfinding

For løsninger på almindelige problemer, se [FEJLFINDING.md](docs/FEJLFINDING.md).

## Bidragelse

Vi modtager gerne bidrag! Se [CONTRIBUTING.md](docs/CONTRIBUTING.md) for detaljer.

## Licens

Dette projekt er licenseret under MIT-licensen - se [LICENSE](LICENSE) for detaljer.