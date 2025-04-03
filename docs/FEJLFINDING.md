# Fejlfindingsguide til Vinkeljernet

Denne guide hjælper med at løse almindelige problemer, der kan opstå ved brug af Vinkeljernet.

## Indholdsfortegnelse

1. [API Nøgle Problemer](#api-nøgle-problemer)
2. [Profilvalidering](#profilvalidering)
3. [Vinkelgenerering](#vinkelgenerering)
4. [PDF Generering](#pdf-generering)
5. [Web Interface](#web-interface)
6. [Command Line Interface](#command-line-interface)
7. [Cache Problems](#cache-problemer)
8. [Dependency Issues](#afhængighedsproblemer)
9. [Logging og Debugging](#logging-og-debugging)

## API Nøgle Problemer

### Problem: "API key not found" eller lignende fejl

**Årsag**: Vinkeljernet kan ikke finde de nødvendige API-nøgler til OpenAI, Anthropic eller Perplexity.

**Løsning**:
1. Kontroller at du har oprettet en `.env` fil i projektets rodmappe
2. Verificer at `.env` filen indeholder de korrekte nøgler:
   ```
   OPENAI_API_KEY=your_openai_key
   ANTHROPIC_API_KEY=your_anthropic_key
   PERPLEXITY_API_KEY=your_perplexity_key
   ```
3. Sørg for at `.env` filen er i det korrekte format (ingen citationstegn omkring nøgler)
4. Genstart applikationen efter at have opdateret `.env` filen

### Problem: "Authentication Error" eller "Invalid API Key"

**Årsag**: API-nøglerne er inkorrekte eller udløbet.

**Løsning**:
1. Bekræft at dine API-nøgler er gyldige ved at teste dem direkte:
   - For OpenAI: Test via [OpenAI's playground](https://platform.openai.com/playground)
   - For Anthropic: Test via [Anthropic's console](https://console.anthropic.com/)
   - For Perplexity: Test via deres API-testværktøj
2. Generer nye API-nøgler hvis nødvendigt
3. Opdater `.env` filen med de nye nøgler

## Profilvalidering

### Problem: "Profile validation failed" fejl

**Årsag**: YAML-profilen opfylder ikke kravene til en gyldig RedaktionelDNA.

**Løsning**:
1. Tjek fejlmeddelelsen for specifikke problemer med profilen
2. Sammenlign med eksemplerne i `PROFIL_FORMAT.md`
3. Almindelige fejl:
   - `tone_og_stil` er ikke en streng
   - Manglende påkrævede felter
   - `nyhedsprioritering` har ugyldige værdier (skal være 1-5)

### Problem: "File not found" ved indlæsning af profil

**Årsag**: Sti til profil-YAML er forkert.

**Løsning**:
1. Kontroller at profilen eksisterer i den angivne sti
2. Brug absolutte stier eller korrekte relative stier
3. Kontroller om filnavnet har den korrekte endelse (.yaml/.yml)

## Vinkelgenerering

### Problem: "No angles could be generated"

**Årsag**: AI-tjenesten returnerede ikke nogen brugbare vinkler.

**Løsning**:
1. Kontroller at emnet er specifikt og klart defineret
2. Prøv et andet emne for at se om problemet er emne-specifikt
3. Kontroller for prompts med for mange begrænsninger i profilen
4. Tjek at API-tjenesten fungerer korrekt
5. Prøv at bruge `--bypass-cache` flaget

### Problem: Dårlig kvalitet eller irrelevante vinkler

**Årsag**: Enten er emnet for bredt/smalt, eller profilen er ikke specifik nok.

**Løsning**:
1. Forfin emnet til at være mere specifikt
2. Tilpas profilen med mere detaljerede kerneprincipper
3. Juster `tone_og_stil` for at være mere beskrivende
4. Udvid `fokusOmrader` for at give AI bedre kontekst

### Problem: Rate limit eller timeout fejl 

**Årsag**: For mange API-kald eller langsom respons fra AI-tjenester.

**Løsning**:
1. Vent et par minutter og prøv igen
2. Brug `--bypass-cache=False` for at bruge cachede resultater hvis muligt
3. Kontroller om AI-tjenesten rapporterer serviceforstyrrelser

## PDF Generering

### Problem: "wkhtmltopdf not found" eller lignende fejl

**Årsag**: wkhtmltopdf er ikke installeret eller kan ikke findes i PATH.

**Løsning**:
1. Installer wkhtmltopdf:
   - macOS: `brew install wkhtmltopdf`
   - Ubuntu/Debian: `sudo apt-get install wkhtmltopdf`
   - Windows: Download fra [wkhtmltopdf.org](https://wkhtmltopdf.org/downloads.html)
2. Kontroller at wkhtmltopdf er tilgængeligt i PATH:
   - Kør `which wkhtmltopdf` (Unix) eller `where wkhtmltopdf` (Windows)
3. Genstart webserveren efter installation

### Problem: PDF indeholder forkert formateret tekst

**Årsag**: HTML-skabelon eller CSS-problemer.

**Løsning**:
1. Kontroller at HTML-skabelonerne i `templates/` er korrekt formateret
2. Tjek CSS for PDF-generering i `app.py`
3. Prøv at downloade som tekst i stedet for PDF for at se om indholdet er korrekt

## Web Interface

### Problem: "Session data lost" eller manglende resultater mellem sider

**Årsag**: Problemer med Flask's sessionstyring.

**Løsning**:
1. Kontroller at en stærk `SECRET_KEY` er konfigureret i `app.py`
2. Sørg for at `session.modified = True` er sat efter sessionsændringer
3. Tjek for sessionsudløb (standard er 30 minutter)
4. Ryd browsercookies og prøv igen

### Problem: Langsom respons på webgrænsefladen

**Årsag**: Lange AI-generationstider eller manglende caching.

**Løsning**:
1. Implementer mere aggressive caching-strategier
2. Reducer mængden af data i sessionen
3. Brug mere effektive AI-modeller for hurtigere respons
4. Tilføj bedre progress-indikatorer til UI

### Problem: "Internal Server Error" (500)

**Årsag**: Ubehandlede undtagelser i Flask-applikationen.

**Løsning**:
1. Tjek loggerne i `vinkeljernet_web.log` for detaljerede fejlmeddelelser
2. Kør app i debug-tilstand: `app.run(debug=True)` for mere information
3. Tilføj flere try/except-blokke omkring problemområderne

## Command Line Interface

### Problem: "Command not found" ved brug af CLI

**Årsag**: Python-sti eller virtuel miljø-problemer.

**Løsning**:
1. Kontroller at du er i det korrekte directory
2. Sørg for at det virtuelle miljø er aktiveret:
   ```bash
   source venv/bin/activate  # Unix
   venv\Scripts\activate     # Windows
   ```
3. Brug fulde stier: `python /path/to/main.py`

### Problem: CLI afbrydes eller fryser

**Årsag**: Problemer med asynkron kode eller manglende fejlhåndtering.

**Løsning**:
1. Tilføj timeout til API-kald
2. Kør med `--debug` flag for mere detaljeret output
3. Kontroller at circuit breakers fungerer korrekt
4. Brug `CTRL+C` til at afbryde, og prøv igen med et mere specifikt emne

## Cache Problemer

### Problem: Stale data eller gamle resultater

**Årsag**: Caching fungerer ikke korrekt eller returnerer forældede data.

**Løsning**:
1. Brug `--clear-cache` flag for at rydde cache
2. Brug `--bypass-cache` flag for at tvinge nye API-kald
3. Kontroller cache-opsætningen i `cache_manager.py`
4. Tjek cachefilerne i `~/.cache/vinkeljernet/` (standard lokation)

### Problem: Cache tager for meget diskplads

**Årsag**: For mange cachede API-kald over tid.

**Løsning**:
1. Ryd cache manuelt: `python -c "from cache_manager import clear_cache; clear_cache()"`
2. Implementer automatisk cache-udløb eller rotation
3. Begrænse cache-størrelse i konfigurationen

## Afhængighedsproblemer

### Problem: "Module not found" fejl

**Årsag**: Manglende Python-pakker eller forkert installationsmiljø.

**Løsning**:
1. Installer alle afhængigheder: `pip install -r requirements.txt`
2. Kontroller at du bruger det korrekte virtuelle miljø
3. Opdater pip: `pip install --upgrade pip`
4. Kontroller at pakkeversioner er kompatible

### Problem: "Incompatible version" fejl

**Årsag**: Pakkeversioner matcher ikke kravene i Vinkeljernet.

**Løsning**:
1. Følg nøjagtigt de versionskrav, der er angivet i `requirements.txt`
2. Opret et nyt virtuelt miljø med præcise versionskrav:
   ```bash
   python -m venv fresh_venv
   source fresh_venv/bin/activate
   pip install -r requirements.txt
   ```
3. Brug Docker for at isolere afhængigheder (hvis tilgængeligt)

## Logging og Debugging

### Problem: Manglende fejldetaljer

**Årsag**: Utilstrækkelig logning eller forkerte logningsniveauer.

**Løsning**:
1. Kør med `--debug` flag for mere detaljeret output
2. Kontroller logfilerne i projektmappen (`vinkeljernet.log` og `vinkeljernet_web.log`)
3. Forøg logningsniveauet i `main.py` og `app.py`

### Problem: For mange logningsbeskeder

**Årsag**: For detaljeret logning, der gør det svært at finde relevante informationer.

**Løsning**:
1. Juster logningsniveauer (INFO, WARNING, ERROR)
2. Filtrer logbeskeder efter relevans
3. Implementer struktureret logning med forskellige filer til forskellige komponenter

## Andre Almindelige Problemer

### Problem: Fejl ved import eller eksport af data

**Årsag**: Problemer med filformater eller sti-håndtering.

**Løsning**:
1. Kontroller at filudfald er korrekt formatteret (JSON, Markdown, HTML)
2. Brug absolutte stier til filer
3. Kontroller skriverettigheder i outputmappen

### Problem: Dårligt formateret output i terminal

**Årsag**: Problemer med Rich-formateringen eller terminalkodning.

**Løsning**:
1. Kontroller at din terminal understøtter Unicode og farver
2. Brug `--output` flag til at gemme resultater i en fil i stedet
3. Juster terminalvinduets bredde ved visning af tabeller