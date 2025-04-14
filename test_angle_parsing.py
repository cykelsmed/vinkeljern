import json_parser
import prompt_engineering
import json
import sys

# Test data - sample LLM response with multiple angles
test_data = """[
  {
    "overskrift": "Millioninvestering i fixerum deler vandene politisk",
    "beskrivelse": "Københavns Kommune bruger over 30 millioner kr. årligt på fixerummet H17. Det skaber politisk debat om prioriteringen af ressourcer.",
    "begrundelse": "Vinklen matcher Berlingskes fokus på politik og samfundsøkonomi med en neutral tilgang til det kontroversielle emne.",
    "nyhedskriterier": ["væsentlighed", "aktualitet", "konflikt"],
    "startSpørgsmål": ["Hvordan retfærdiggør kommunen den store økonomiske investering?", "Hvilke alternative løsninger foreslår kritikerne?", "Hvordan måler man effekten af fixerummet i forhold til prisen?"]
  },
  {
    "overskrift": "Stofbrugere: Fixerummet redder liv, men flere lokale rum ønskes",
    "beskrivelse": "Brugere af fixerummet H17 fremhæver dets livreddende funktioner, men efterlyser mindre, lokale fixerum i andre bydele for at undgå centralisering.",
    "begrundelse": "Vinklen giver stemme til de berørte og belyser både positive effekter og forbedringsmuligheder, hvilket passer til Berlingskes nuancerede dækning.",
    "nyhedskriterier": ["identifikation", "væsentlighed"],
    "startSpørgsmål": ["Hvilke konkrete fordele oplever I ved at bruge fixerummet?", "Hvorfor er lokale fixerum bedre end ét stort centralt?", "Hvilke risici er der ved den nuværende centralisering?"]
  },
  {
    "overskrift": "Sundhedspersonale: Fixerum reducerer smitterisiko og overdoser markant",
    "beskrivelse": "Læger og sygeplejersker med tilknytning til H17 dokumenterer et fald i HIV/hepatitis-smitte og overdoser blandt brugerne siden åbningen i 2016.",
    "begrundelse": "Vinklen fokuserer på faktabaseret, sundhedsfaglig dokumentation, hvilket stemmer overens med Berlingskes vægt på troværdighed og grundighed.",
    "nyhedskriterier": ["væsentlighed", "aktualitet"],
    "startSpørgsmål": ["Hvilke sundhedsmæssige forbedringer kan I dokumentere?", "Hvordan håndterer I overdoser i fixerummet?", "Hvad koster de sundhedsmæssige gevinster sammenlignet med udgifterne til fixerummet?"]
  }
]"""

print("Testing json_parser module...")
json_parser_angles = json_parser.parse_angles_from_llm_response(test_data)
print(f"json_parser found {len(json_parser_angles)} angles")
for i, angle in enumerate(json_parser_angles):
    print(f"  {i+1}: {angle.get(\"overskrift\", \"No headline\")}")

print("
Testing prompt_engineering module...")
prompt_engineering_angles = prompt_engineering.parse_angles_from_response(test_data)
print(f"prompt_engineering found {len(prompt_engineering_angles)} angles")
for i, angle in enumerate(prompt_engineering_angles):
    print(f"  {i+1}: {angle.get(\"overskrift\", \"No headline\")}")

print("
Done")
