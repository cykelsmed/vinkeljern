# Redaktionel DNA Profil Format

En redaktionel DNA-profil definerer den unikke journalistiske identitet for et medie. I Vinkeljernet bruges dette DNA til at generere vinkler, der passer specifikt til mediets stil, tone og prioriteringer.

## Profilstruktur

Profiler er defineret i YAML-format og indeholder følgende hovedkomponenter:

```yaml
navn: "Mediets navn"
beskrivelse: "Kort beskrivelse af mediets profil og målgruppe"
kerneprincipper:
  - "Kerneprincip 1: Uddybning"
  - "Kerneprincip 2: Uddybning"
tone_og_stil: "Beskrivelse af mediets tone og sproglige stil"
nyhedsprioritering:
  Aktualitet: 5
  Væsentlighed: 4
  Konflikt: 3
  Identifikation: 4
  Sensation: 2
fokusOmrader:
  - "Fokusområde 1"
  - "Fokusområde 2"
noGoOmrader:
  - "No-go område 1"
  - "No-go område 2"
```

## Feltbeskrivelser

### Grundlæggende information

| Felt | Type | Påkrævet | Beskrivelse |
|------|------|----------|-------------|
| `navn` | String | Ja | Mediets navn |
| `beskrivelse` | String | Ja | Kort beskrivelse af mediets profil og målgruppe |

### Kerneprincipper

Kerneprincipper er mediets journalistiske grundværdier og retningslinjer.

```yaml
kerneprincipper:
  - "Afslørende: Vi graver efter sandheden, især når den er ubehagelig for magthaverne."
  - "Folkelig relevans: Historier skal ramme og betyde noget for almindelige danskere."
  - "Konstruktiv: Vi søger altid at fremhæve løsningsforslag, når vi fremhæver problemer."
```

### Tone og stil

Tone og stil beskriver mediets sprogbrug, formidlingsstil og fortællemæssige karakteristika.

```yaml
tone_og_stil: "Direkte, skarp, kontant og letforståelig tone. Vi undgår akademisk sprog og lange, komplekse sætninger. Vi taler til læseren i øjenhøjde og bruger konkrete eksempler frem for abstrakte koncepter."
```

> **Vigtigt**: `tone_og_stil` skal være en enkel tekststreng, ikke en liste eller et nested objekt.

### Nyhedsprioritering

Nyhedsprioritering angiver vægtningen af forskellige nyhedskriterier på en skala fra 1-5, hvor 5 er højeste prioritet.

```yaml
nyhedsprioritering:
  Aktualitet: 5     # Breaking news, det nyeste nye
  Væsentlighed: 4   # Samfundsmæssig relevans og betydning
  Konflikt: 4       # Uenighed, strid, drama
  Identifikation: 3 # Genkendelighed, relation til læserne
  Sensation: 2      # Det overraskende, chokerende, usædvanlige
```

Følgende nyhedskriterier understøttes:
- Aktualitet
- Væsentlighed
- Konflikt
- Identifikation
- Sensation
- Eksklusivitet

### Fokusområder

Fokusområder er emner eller områder, som mediet særligt prioriterer i sin dækning.

```yaml
fokusOmrader:
  - "Politik og magt"
  - "Økonomi og arbejdsmarked"
  - "Forbruger og hverdagsliv"
  - "Teknologi og digitalisering"
```

### No-Go Områder

No-Go områder er emner eller vinkler, som mediet generelt ikke dækker eller har særlige etiske retningslinjer omkring.

```yaml
noGoOmrader:
  - "Indhold der krænker privatlivets fred uden samfundsmæssig relevans"
  - "Historier baseret på anonyme kilder uden faktuel verifikation"
  - "Clickbait eller misvisende overskrifter der ikke dækkes af indholdet"
```

## Komplet Eksempel

Her er et komplet eksempel på en redaktionel DNA-profil:

```yaml
navn: "Politiken"
beskrivelse: "Politiken er en kritisk og samfundsengageret avis med kulturradikal tradition og et bredt læsersegment af veluddannede og kulturinteresserede mennesker."
kerneprincipper:
  - "Oplysning: Vi formidler komplekse informationer på en tilgængelig måde."
  - "Kritisk vagthund: Vi udfordrer magthavere og holder dem ansvarlige."
  - "Kulturradikalisme: Vi fremmer progressive synspunkter og kulturel åbenhed."
  - "Dybde: Vi prioriterer analyser, baggrund og perspektiv over det flygtige."
tone_og_stil: "Reflekteret, nuanceret og intellektuelt stimulerende. Vi bruger et levende og rigt sprog med plads til både klarhed og kompleksitet. Vores stil er seriøs, men med plads til både skarphed, humor og elegance."
nyhedsprioritering:
  Aktualitet: 4
  Væsentlighed: 5
  Konflikt: 3
  Identifikation: 4
  Sensation: 2
fokusOmrader:
  - "National og international politik"
  - "Kultur og kunst"
  - "Klima og miljø"
  - "Social retfærdighed og lighed"
  - "Byudvikling og arkitektur"
noGoOmrader:
  - "Ukritisk tabloid sensationsdækning"
  - "Ensidede fremstillinger af komplekse emner"
  - "Krænkelse af privatlivets fred uden samfundsmæssig relevans"
```

## Anvendelse i Vinkeljernet

Vinkeljernet bruger den redaktionelle DNA-profil til at:

1. **Konstruere prompts** der instruerer AI-modellerne i at generere vinkler, der passer til mediets DNA
2. **Evaluere og rangere vinkler** baseret på, hvor godt de matcher mediets nyhedsprioritering
3. **Filtrere upassende vinkler**, der falder inden for mediets no-go områder

## Tips til Effektive Profiler

1. **Vær specifik og konkret** - Generelle beskrivelser giver generelle vinkler
2. **Balancer bredde og fokus** - For snævre profiler kan begrænse kreativiteten
3. **Prioriter korrekt** - Sæt realistiske vægtninger i nyhedsprioriteringen
4. **Opdater regelmæssigt** - Tilpas profilen når mediets fokus ændrer sig

## Fejlfinding

### Almindelige problemer

1. **Vinkler matcher ikke mediets tone**
   - Gør `tone_og_stil` mere detaljeret og specifik
   - Tilføj flere eksempler på den ønskede sprogbrug

2. **For generiske vinkler**
   - Tilføj flere specifikke fokusområder
   - Vær mere specifik i kerneprincipper

3. **Paradoksale vinkler**
   - Kontroller for modstridende kerneværdier eller fokusområder
   - Sørg for at no-go områder ikke modstrider fokusområder