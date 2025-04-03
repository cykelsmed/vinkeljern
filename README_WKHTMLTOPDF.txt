
# WKHTMLTOPDF INSTALLATION FOR VINKELJERNET

For at kunne generere PDF-filer skal wkhtmltopdf være installeret på systemet.
Vinkeljernet bruger pdfkit Python-biblioteket, som er en wrapper omkring wkhtmltopdf-værktøjet.

## Installation

### macOS:
```
brew install wkhtmltopdf
```

### Ubuntu/Debian:
```
sudo apt-get update
sudo apt-get install wkhtmltopdf
```

### CentOS/RHEL:
```
sudo yum install wkhtmltopdf
```

### Windows:
1. Download installationsprogrammet fra https://wkhtmltopdf.org/downloads.html
2. Kør installationsprogrammet og følg vejledningen
3. Sørg for at tilføje installationsstien til systemets PATH-miljøvariabel

## Fejlfinding

Hvis du oplever problemer med PDF-generering:

1. Kontroller at wkhtmltopdf er korrekt installeret ved at køre følgende kommando i terminalen:
   ```
   wkhtmltopdf --version
   ```

2. Hvis wkhtmltopdf er installeret, men PDF-generering stadig fejler, kan det være nødvendigt at angive den fulde sti til wkhtmltopdf-programmet i app.py:
   
   Find disse linjer i app.py:
   ```python
   # Configure pdfkit
   config = None
   if wkhtmltopdf_path:
       config = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)
   ```
   
   Og rediger dem til at angive den fulde sti:
   ```python
   # Configure pdfkit with explicit path
   config = pdfkit.configuration(wkhtmltopdf='/sti/til/wkhtmltopdf')
   ```

3. Erstat '/sti/til/wkhtmltopdf' med den faktiske sti til wkhtmltopdf på dit system:
   - På Windows: typisk 'C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe'
   - På macOS (med Homebrew): typisk '/usr/local/bin/wkhtmltopdf'
   - På Linux: typisk '/usr/bin/wkhtmltopdf'

## Bemærk

Hvis du ikke installerer wkhtmltopdf, vil PDF-generering fejle, men resten af Vinkeljernet-applikationen vil stadig fungere. Du vil stadig kunne bruge andre formater som tekstdownload.