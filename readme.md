# Vinkeljernet

Vinkeljernet er en journalistisk vinkelgenerator, der genererer nyhedsvinkler baseret på et redaktionelt DNA og et givet nyhedsemne. Programmet kører via kommandolinjen og kræver nogle obligatoriske parametre.

## Krav

- Python 3.12+
- Installerede afhængigheder (se requirements.txt)

## Kommando-linje argumenter

Obligatoriske argumenter:
- `--emne` eller `-e`: Det nyhedsemne, der skal genereres vinkler for.
- `--profil` eller `-p`: Stien til YAML-filen med den redaktionelle DNA-profil.

Valgfri argumenter:
- `--output` eller `-o`: Valgfri filsti til at gemme outputtet. Hvis ikke angivet, vises resultatet i terminalen.
- `--dev-mode`: Kør i udviklingstilstand (deaktiverer SSL-verifikation).
- `--clear-cache`: Ryd cache før programmet kører.
- `--bypass-cache`: Ignorer cache og tving friske API-kald.
- `--format`: Format for output, gyldige værdier er `json`, `markdown` og `html` (standard er `json`).
- `--show-circuits`: Vis status for circuit breakers.
- `--reset-circuits`: Nulstil alle circuit breakers.
- `--debug`: Aktiver debug-tilstand med ekstra output.

## Eksempler

### 1. Kør programmet med de obligatoriske parametre

```bash
python [main.py](http://_vscodecontentref_/0) --emne "Politik" --profil config/tvflux_profil.yaml