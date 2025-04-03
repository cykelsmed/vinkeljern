"""
Flask web interface for Vinkeljernet.

This module provides a web interface for the Vinkeljernet application,
allowing users to generate news angles through a web browser.
"""
import os
import json
import asyncio
import pdfkit
from pathlib import Path
from datetime import datetime
from io import BytesIO
from flask import Flask, render_template, request, redirect, url_for, session, send_file, flash
from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, SubmitField
from wtforms.validators import DataRequired

# Import from Vinkeljernet
from config_loader import load_and_validate_profile
from api_clients import fetch_topic_information
from prompt_engineering import construct_angle_prompt, parse_angles_from_response
from angle_processor import filter_and_rank_angles
from models import RedaktionelDNA
from config import ANTHROPIC_API_KEY
import requests

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-for-vinkeljernet')
app.config['PROFILE_DIR'] = 'config'

# Form for news angle generation
class AngleGenerationForm(FlaskForm):
    topic = StringField('Nyhedsemne', validators=[DataRequired()])
    profile = SelectField('Profil', validators=[DataRequired()])
    submit = SubmitField('Generer vinkler')

def get_available_profiles():
    """Get list of available profile files."""
    profile_dir = Path(app.config['PROFILE_DIR'])
    if not profile_dir.exists() or not profile_dir.is_dir():
        return []
    
    profiles = list(profile_dir.glob("*.yaml"))
    return [str(p) for p in profiles]

def get_profile_choices():
    """Get profile choices for the select field."""
    profiles = get_available_profiles()
    return [(p, Path(p).stem) for p in profiles]

@app.route('/', methods=['GET', 'POST'])
def index():
    """Home page with form for angle generation."""
    form = AngleGenerationForm()
    
    # Populate profile choices
    form.profile.choices = get_profile_choices()
    
    if form.validate_on_submit():
        # Store form data in session
        session['topic'] = form.topic.data
        session['profile'] = form.profile.data
        
        # Redirect to results page
        return redirect(url_for('generate'))
    
    return render_template('index.html', form=form)

@app.route('/generate')
def generate():
    """Generate angles and show results."""
    # Get data from session
    topic = session.get('topic')
    profile_path = session.get('profile')
    
    if not topic or not profile_path:
        flash('Manglende data. Udfyld formularen igen.')
        return redirect(url_for('index'))
    
    try:
        # Load the profile
        profile = load_and_validate_profile(Path(profile_path))
        
        # Process the request
        ranked_angles = process_generation_request(topic, profile)
        
        if not ranked_angles:
            flash('Ingen vinkler kunne genereres. Prøv et andet emne eller en anden profil.')
            return redirect(url_for('index'))
        
        # Store results in session for download options
        session['results'] = {
            'topic': topic,
            'profile_name': Path(profile_path).stem,
            'angles': ranked_angles,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return render_template(
            'results.html',
            topic=topic,
            profile_name=Path(profile_path).stem,
            angles=ranked_angles
        )
        
    except Exception as e:
        flash(f'Fejl under generering: {str(e)}')
        return redirect(url_for('index'))

@app.route('/download/<format>')
def download(format):
    """Download results in different formats."""
    results = session.get('results')
    
    if not results:
        flash('Ingen resultater at downloade. Generer vinkler først.')
        return redirect(url_for('index'))
    
    topic = results['topic']
    profile_name = results['profile_name']
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"vinkeljernet_{topic}_{profile_name}_{timestamp}"
    
    if format == 'text':
        # Generate text content
        content = generate_text_content(results)
        
        # Return as downloadable file
        buffer = BytesIO(content.encode('utf-8'))
        buffer.seek(0)
        return send_file(
            buffer,
            as_attachment=True,
            download_name=f"{filename}.txt",
            mimetype="text/plain"
        )
        
    elif format == 'pdf':
        # Generate HTML content
        html_content = generate_html_content(results)
        
        try:
            # Convert HTML to PDF
            pdf = pdfkit.from_string(html_content, False)
            
            # Return as downloadable file
            buffer = BytesIO(pdf)
            buffer.seek(0)
            return send_file(
                buffer,
                as_attachment=True,
                download_name=f"{filename}.pdf",
                mimetype="application/pdf"
            )
        except Exception as e:
            flash(f'Fejl ved generering af PDF: {str(e)}')
            return redirect(url_for('results'))
    
    flash(f'Ukendt format: {format}')
    return redirect(url_for('index'))

def generate_text_content(results):
    """Generate plain text content for download."""
    topic = results['topic']
    profile_name = results['profile_name']
    angles = results['angles']
    timestamp = results['timestamp']
    
    content = [
        f"VINKELJERNET - GENEREREDE NYHEDSVINKLER",
        f"Emne: {topic}",
        f"Profil: {profile_name}",
        f"Genereret: {timestamp}",
        "\n" + "="*50 + "\n"
    ]
    
    for i, angle in enumerate(angles, 1):
        content.append(f"VINKEL {i}: {angle.get('overskrift', 'Ingen overskrift')}")
        content.append(f"\nBeskrivelse: {angle.get('beskrivelse', 'Ingen beskrivelse')}")
        content.append(f"\nBegrundelse: {angle.get('begrundelse', 'Ingen begrundelse')}")
        
        criteria = angle.get('nyhedskriterier', [])
        content.append(f"\nNyhedskriterier: {', '.join(criteria)}")
        
        questions = angle.get('startSpørgsmål', [])
        if questions:
            content.append("\nStartspørgsmål:")
            for q in questions:
                content.append(f"- {q}")
        
        content.append("\n" + "-"*50 + "\n")
    
    return "\n".join(content)

def generate_html_content(results):
    """Generate HTML content for PDF conversion."""
    topic = results['topic']
    profile_name = results['profile_name']
    angles = results['angles']
    timestamp = results['timestamp']
    
    html = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset='UTF-8'>",
        "<title>Vinkeljernet Resultater</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 20px; }",
        "h1 { color: #2c3e50; }",
        "h2 { color: #3498db; }",
        ".meta { color: #7f8c8d; margin-bottom: 20px; }",
        ".angle { margin-bottom: 30px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }",
        ".angle h3 { color: #2980b9; margin-top: 0; }",
        ".description { margin-bottom: 10px; }",
        ".rationale { color: #555; font-style: italic; margin-bottom: 10px; }",
        ".criteria { margin-bottom: 10px; }",
        ".questions { margin-top: 15px; }",
        "ul { padding-left: 20px; }",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>Vinkeljernet - Genererede nyhedsvinkler</h1>",
        f"<div class='meta'>",
        f"<p><strong>Emne:</strong> {topic}</p>",
        f"<p><strong>Profil:</strong> {profile_name}</p>",
        f"<p><strong>Genereret:</strong> {timestamp}</p>",
        f"</div>"
    ]
    
    for i, angle in enumerate(angles, 1):
        headline = angle.get('overskrift', 'Ingen overskrift')
        description = angle.get('beskrivelse', 'Ingen beskrivelse')
        rationale = angle.get('begrundelse', 'Ingen begrundelse')
        criteria = angle.get('nyhedskriterier', [])
        questions = angle.get('startSpørgsmål', [])
        
        html.append(f"<div class='angle'>")
        html.append(f"<h3>Vinkel {i}: {headline}</h3>")
        html.append(f"<div class='description'>{description}</div>")
        html.append(f"<div class='rationale'><strong>Begrundelse:</strong> {rationale}</div>")
        html.append(f"<div class='criteria'><strong>Nyhedskriterier:</strong> {', '.join(criteria)}</div>")
        
        if questions:
            html.append("<div class='questions'><strong>Startspørgsmål:</strong>")
            html.append("<ul>")
            for q in questions:
                html.append(f"<li>{q}</li>")
            html.append("</ul>")
            html.append("</div>")
        
        html.append("</div>")
    
    html.append("</body>")
    html.append("</html>")
    
    return "\n".join(html)

def process_generation_request(topic, profile):
    """Process a generation request with synchronous API calls.
    
    This is a synchronous version of the async process_generation_request 
    function from main.py, adapted for Flask.
    """
    try:
        # Convert profile into strings for prompt construction
        principper = "\n".join([f"- {p}" for p in profile.kerneprincipper])
        nyhedskriterier = "\n".join([f"- {k}: {v}" for k, v in profile.nyhedsprioritering.items()])
        fokusomrader = "\n".join([f"- {f}" for f in profile.fokusOmrader])
        nogo_omrader = "\n".join([f"- {n}" for n in profile.noGoOmrader]) if profile.noGoOmrader else "Ingen"
        
        # Get topic information with synchronous API call
        topic_info = get_topic_info_sync(topic)
        
        if not topic_info:
            topic_info = f"Emnet handler om {topic}. Ingen yderligere baggrundsinformation tilgængelig."
        
        # Create the prompt
        prompt = construct_angle_prompt(
            topic,
            topic_info,
            principper,
            profile.tone_og_stil,
            fokusomrader,
            nyhedskriterier,
            nogo_omrader
        )
        
        # Make the Claude API call
        claude_response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01"
            },
            json={
                "model": "claude-3-opus-20240229",
                "max_tokens": 2500,
                "temperature": 0.7,
                "system": "Du er en erfaren journalist med ekspertise i at udvikle kreative og relevante nyhedsvinkler.",
                "messages": [{"role": "user", "content": prompt}],
            }
        )
        
        # Parse Claude response
        if claude_response.status_code != 200:
            raise ValueError(f"Claude API fejl: {claude_response.status_code}")
            
        response_data = claude_response.json()
        response_text = response_data['content'][0]['text']
        
        angles = parse_angles_from_response(response_text)
        
        # Add perplexity information to each angle
        if angles and isinstance(angles, list):
            perplexity_extract = topic_info[:1000] + ("..." if len(topic_info) > 1000 else "")
            
            # Generate source suggestions using Claude
            source_suggestions_prompt = f"""
            Baseret på emnet '{topic}', giv en kort liste med 3-5 relevante og troværdige danske kilder, 
            som en journalist kunne bruge til research. Inkluder officielle hjemmesider, forskningsinstitutioner, 
            eksperter og organisationer. Formater som en simpel punktopstilling med korte beskrivelser på dansk.
            Hold dit svar under 250 ord og fokuser kun på de mest pålidelige kilder.
            """
            
            try:
                # Call the API
                source_response = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "Content-Type": "application/json",
                        "x-api-key": ANTHROPIC_API_KEY,
                        "anthropic-version": "2023-06-01"
                    },
                    json={
                        "model": "claude-3-haiku-20240307",
                        "max_tokens": 500,
                        "temperature": 0.2,
                        "system": "Du er en hjælpsom researchassistent med stort kendskab til troværdige danske kilder. Du svarer altid på dansk.",
                        "messages": [{"role": "user", "content": source_suggestions_prompt}],
                    }
                )
                
                if source_response.status_code == 200:
                    source_data = source_response.json()
                    source_text = source_data['content'][0]['text']
                    
                    # Add both perplexity info and source suggestions to each angle
                    for angle in angles:
                        if isinstance(angle, dict):
                            angle['perplexityInfo'] = perplexity_extract
                            angle['kildeForslagInfo'] = source_text
                else:
                    # If source generation fails, just add perplexity info
                    for angle in angles:
                        if isinstance(angle, dict):
                            angle['perplexityInfo'] = perplexity_extract
            except Exception:
                # If there's an error, just add perplexity info
                for angle in angles:
                    if isinstance(angle, dict):
                        angle['perplexityInfo'] = perplexity_extract
        
        # Filter and rank angles
        if angles:
            ranked_angles = filter_and_rank_angles(angles, profile, 5)
            return ranked_angles
        
        return None
        
    except Exception as e:
        app.logger.error(f"Error during generation: {e}")
        raise

def get_topic_info_sync(topic):
    """Synchronous version of fetch_topic_information."""
    try:
        # Use the Perplexity API directly
        from config import PERPLEXITY_API_KEY
        
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "sonar",
            "messages": [
                {"role": "system", "content": "Du er en professionel journalist, der skal give koncis og faktabaseret information om et aktuelt nyhedsemne. Giv grundige, men velstrukturerede fakta, baggrund og kontekst. Inkluder omhyggeligt datoer, tal og faktuelle detaljer, der er relevante for emnet. Undgå at udelade væsentlig information."},
                {"role": "user", "content": f"Giv mig en grundig, men velstruktureret oversigt over den aktuelle situation vedrørende følgende nyhedsemne: {topic}. Inkluder relevante fakta, baggrund, kontekst og eventuelle nylige udviklinger. Vær præcis og faktabaseret."}
            ],
            "max_tokens": 1000,
            "temperature": 0.2,
            "top_p": 0.9,
            "return_images": False,
            "return_related_questions": False
        }
        
        response = requests.post(
            'https://api.perplexity.ai/chat/completions',
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            data = response.json()
            return data['choices'][0]['message']['content']
        
        return None
        
    except Exception as e:
        app.logger.error(f"Error fetching topic information: {e}")
        return None

if __name__ == '__main__':
    # Create templates folder if it doesn't exist
    templates_dir = Path('templates')
    templates_dir.mkdir(exist_ok=True)
    
    # Create template files if they don't exist
    index_template = templates_dir / 'index.html'
    if not index_template.exists():
        with open(index_template, 'w') as f:
            f.write("""
{% extends 'base.html' %}

{% block title %}Vinkeljernet - Journalistisk vinkelgenerator{% endblock %}

{% block content %}
<div class="container mt-5">
    <div class="row justify-content-center">
        <div class="col-md-8">
            <div class="card">
                <div class="card-header bg-primary text-white">
                    <h2 class="mb-0">Vinkeljernet</h2>
                </div>
                <div class="card-body">
                    <p class="lead mb-4">
                        Generer kreative og relevante nyhedsvinkler baseret på redaktionelle DNA-profiler.
                    </p>
                    
                    {% with messages = get_flashed_messages() %}
                        {% if messages %}
                            {% for message in messages %}
                                <div class="alert alert-danger">{{ message }}</div>
                            {% endfor %}
                        {% endif %}
                    {% endwith %}
                    
                    <form method="POST" action="{{ url_for('index') }}">
                        {{ form.hidden_tag() }}
                        
                        <div class="mb-3">
                            {{ form.topic.label(class="form-label") }}
                            {{ form.topic(class="form-control", placeholder="F.eks. klimaforandringer, blockchain, teknologipolitik...") }}
                        </div>
                        
                        <div class="mb-3">
                            {{ form.profile.label(class="form-label") }}
                            {{ form.profile(class="form-select") }}
                        </div>
                        
                        <div class="d-grid gap-2">
                            {{ form.submit(class="btn btn-primary btn-lg") }}
                        </div>
                    </form>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}
            """)
    
    results_template = templates_dir / 'results.html'
    if not results_template.exists():
        with open(results_template, 'w') as f:
            f.write("""
{% extends 'base.html' %}

{% block title %}Vinkler for {{ topic }} | Vinkeljernet{% endblock %}

{% block content %}
<div class="container mt-5">
    <div class="row justify-content-center">
        <div class="col-md-10">
            <h1 class="mb-4">Vinkler for: {{ topic }}</h1>
            <p class="text-muted">Profil: {{ profile_name }}</p>
            
            <div class="mb-4">
                <a href="{{ url_for('index') }}" class="btn btn-outline-secondary me-2">
                    <i class="bi bi-arrow-left"></i> Ny søgning
                </a>
                <a href="{{ url_for('download', format='text') }}" class="btn btn-outline-primary me-2">
                    <i class="bi bi-file-text"></i> Download som tekst
                </a>
                <a href="{{ url_for('download', format='pdf') }}" class="btn btn-outline-danger">
                    <i class="bi bi-file-pdf"></i> Download som PDF
                </a>
            </div>
            
            {% for angle in angles %}
            <div class="card mb-4">
                <div class="card-header bg-primary text-white">
                    <h2 class="h5 mb-0">{{ angle.overskrift }}</h2>
                </div>
                <div class="card-body">
                    <p class="lead">{{ angle.beskrivelse }}</p>
                    
                    <div class="mb-3">
                        <h5>Begrundelse:</h5>
                        <p class="text-muted">{{ angle.begrundelse }}</p>
                    </div>
                    
                    <div class="mb-3">
                        <h5>Nyhedskriterier:</h5>
                        <div class="d-flex flex-wrap">
                            {% for kriterium in angle.nyhedskriterier %}
                                <span class="badge bg-secondary me-2 mb-2">{{ kriterium }}</span>
                            {% endfor %}
                        </div>
                    </div>
                    
                    {% if angle.startSpørgsmål %}
                    <div class="mb-3">
                        <h5>Startspørgsmål:</h5>
                        <ul class="list-group">
                            {% for question in angle.startSpørgsmål %}
                                <li class="list-group-item">{{ question }}</li>
                            {% endfor %}
                        </ul>
                    </div>
                    {% endif %}
                    
                    {% if angle.kildeForslagInfo %}
                    <div class="mb-3">
                        <h5>Kildeforslag:</h5>
                        <div class="card bg-light">
                            <div class="card-body">
                                {{ angle.kildeForslagInfo | safe }}
                            </div>
                        </div>
                    </div>
                    {% endif %}
                </div>
            </div>
            {% endfor %}
        </div>
    </div>
</div>
{% endblock %}
            """)
    
    base_template = templates_dir / 'base.html'
    if not base_template.exists():
        with open(base_template, 'w') as f:
            f.write("""
<!DOCTYPE html>
<html lang="da">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Vinkeljernet{% endblock %}</title>
    
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css">
    
    <style>
        body {
            background-color: #f8f9fa;
        }
        .navbar-brand {
            font-weight: bold;
        }
        footer {
            margin-top: 4rem;
            padding: 1rem 0;
            background-color: #f1f1f1;
        }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <div class="container">
            <a class="navbar-brand" href="{{ url_for('index') }}">
                <i class="bi bi-newspaper"></i> Vinkeljernet
            </a>
        </div>
    </nav>

    <main>
        {% block content %}{% endblock %}
    </main>

    <footer class="text-center text-muted">
        <div class="container">
            <p>Vinkeljernet - Journalistisk vinkelgenerator</p>
        </div>
    </footer>

    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
            """)
    
    # Create wkhtmltopdf check file
    wkhtmltopdf_note = Path('README_WKHTMLTOPDF.txt')
    if not wkhtmltopdf_note.exists():
        with open(wkhtmltopdf_note, 'w') as f:
            f.write("""
For at kunne generere PDF-filer skal wkhtmltopdf være installeret på systemet.

Installation:

- macOS: brew install wkhtmltopdf
- Ubuntu/Debian: sudo apt-get install wkhtmltopdf
- Windows: Download fra https://wkhtmltopdf.org/downloads.html

Hvis du ikke installerer wkhtmltopdf, vil PDF-generering fejle, men resten af applikationen vil stadig fungere.
            """)
    
    app.run(debug=True)