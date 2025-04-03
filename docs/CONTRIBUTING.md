# Bidrag til Vinkeljernet

Vi sætter stor pris på bidrag til Vinkeljernet! Denne guide forklarer, hvordan du kan bidrage til projektet, fra små bugfixes til større funktionalitetsudvidelser.

## Indholdsfortegnelse

1. [Adfærdskodeks](#adfærdskodeks)
2. [Kom i gang](#kom-i-gang)
3. [Hvordan du kan bidrage](#hvordan-du-kan-bidrage)
4. [Rapportering af bugs](#rapportering-af-bugs)
5. [Feature requests](#feature-requests)
6. [Pull requests](#pull-requests)
7. [Kodestil](#kodestil)
8. [Test](#test)
9. [Dokumentation](#dokumentation)
10. [Release proces](#release-proces)
11. [Kontakt](#kontakt)

## Adfærdskodeks

Dette projekt følger en adfærdskodeks, der forventer at alle deltagere er respektfulde, imødekommende og samarbejdsvillige. Vi accepterer ikke chikane eller stødende sprogbrug i nogen form.

## Kom i gang

For at komme i gang med at bidrage til Vinkeljernet, følg disse trin:

1. Fork dette repository
2. Klon din fork til din lokale maskine:
   ```bash
   git clone https://github.com/dit-username/vinkeljernet.git
   cd vinkeljernet
   ```
3. Opsæt en udviklingsmiljø:
   ```bash
   python -m venv venv
   source venv/bin/activate  # På Windows: venv\Scripts\activate
   pip install -r requirements-dev.txt
   pre-commit install
   ```
4. Opret en branch for dine ændringer:
   ```bash
   git checkout -b feature/min-nye-feature
   ```

## Hvordan du kan bidrage

Der er mange måder at bidrage til Vinkeljernet:

- **Kode**: Implementer nye funktioner eller fiks bugs
- **Dokumentation**: Forbedre eller udvid dokumentationen
- **Tests**: Tilføj eller forbedre tests
- **Design**: Forbedre brugeroplevelsen eller grænsefladen
- **Ideer**: Foreslå nye funktioner eller forbedringer
- **Profiler**: Bidrag med nye redaktionelle DNA-profiler

## Rapportering af bugs

Hvis du finder en bug, opret en GitHub-issue med følgende information:

- En klar og beskrivende titel
- Trinvis vejledning til at genskabe problemet
- Forventet adfærd og faktisk adfærd
- Systemoplysninger (OS, Python-version, osv.)
- Eventuelle logfiler eller fejlmeddelelser

## Feature Requests

Hvis du har ideer til nye funktioner, opret en GitHub-issue med:

- En klar og beskrivende titel
- Detaljeret beskrivelse af den ønskede funktionalitet
- Eventuelle forslag til implementering
- Hvordan funktionen vil gavne projektet eller brugerne

## Pull Requests

Følg disse trin for at indsende en pull request:

1. Sørg for at din branch er opdateret med `main`:
   ```bash
   git fetch origin
   git rebase origin/main
   ```
2. Kør tests for at sikre, at dine ændringer ikke bryder eksisterende funktionalitet:
   ```bash
   pytest
   ```
3. Følg kodestilsretningslinjerne ved at køre linting:
   ```bash
   black .
   flake8
   isort .
   mypy .
   ```
4. Commit dine ændringer med en beskrivende commit-besked
5. Push til din fork:
   ```bash
   git push -u origin feature/min-nye-feature
   ```
6. Opret en pull request via GitHub's grænseflade

### Pull Request Tjekliste

Før du indsender en PR, kontroller følgende:

- [ ] Kode overholder projektets stilguide
- [ ] Alle tests består
- [ ] Dokumentation er opdateret
- [ ] Commit-beskeder er klare og beskrivende
- [ ] Kode er tilstrækkeligt testet

## Kodestil

Vinkeljernet følger PEP 8 kodestilsguide med nogle tilpasninger defineret i konfigurationsfilerne.

Hovedretningslinjerne er:

- Brug 4 mellemrum til indrykning (ikke tabs)
- Maksimal linjelængde på 100 tegn
- Brug Python typehints
- Inkluder docstrings for alle klasser og funktioner
- Følg [Google-stilen](https://google.github.io/styleguide/pyguide.html) for docstrings

For at kontrollere og formatere din kode:

```bash
# Formater koden
black .

# Sorter imports
isort .

# Kør linter
flake8

# Kør typetjek
mypy .
```

## Test

Alle nye funktioner skal have tilhørende tests. Vi bruger pytest som vores test framework.

Nogle retningslinjer for test:

- En test per funktion eller adfærd
- Navngiv tests beskrivende (f.eks. `test_profile_validation_rejects_invalid_yaml`)
- Brug fixtures og parameterisering til at undgå kodeduplicering
- Brug mocking til at isolere testede komponenter

For at køre tests:

```bash
# Kør alle tests
pytest

# Kør specifikke tests
pytest tests/unit/test_config_loader.py

# Kør med coverage
pytest --cov=.
```

## Dokumentation

Vi sætter stor pris på bidrag til dokumentationen. Vær opmærksom på følgende:

- Dokumentation er skrevet i Markdown
- Hovedfiler skrives på dansk
- Teknisk dokumentation (docstrings) skrives på engelsk
- Alle offentlige funktioner og klasser skal have dokumentation
- Eksempler er meget værdifulde og opmuntres stærkt

## Release Proces

Vinkeljernet følger semantisk versionering (SemVer):

- MAJOR version ved ændringer der bryder bagudkompatibilitet
- MINOR version ved funktionalitetstilføjelser der er bagudkompatible
- PATCH version ved bagudkompatible bugfixes

Releases håndteres af projektvedligeholderne.

## Kontakt

Hvis du har spørgsmål om at bidrage, kan du:

- Åbne en GitHub-issue
- Kontakte projektvedligeholderne via e-mail på [kontakt@vinkeljernet.dk](mailto:kontakt@vinkeljernet.dk)

---

Tak for at overveje at bidrage til Vinkeljernet!