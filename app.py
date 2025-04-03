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
        
        # Get background information if available
        background_info = None
        for angle in ranked_angles:
            if 'perplexityInfo' in angle:
                background_info = angle['perplexityInfo']
                break
        
        # Forbedret session data struktur
        current_time = datetime.now()
        timestamp = current_time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Prepare session data
        session_data = {
            'topic': topic,
            'profile_name': Path(profile_path).stem,
            'angles': ranked_angles,
            'background_info': background_info,
            # Tidsstempler
            'generated_at': timestamp,
            'last_updated': timestamp,
            # Status flags
            'has_editorial_considerations': False,
            'has_detailed_background': False,
            # Default tom værdi for fremtidige data
            'editorial_considerations': '',
            'source_suggestions': ''
        }
        
        # Bevar eksisterende session data hvis det findes
        if 'results' in session:
            # Bevar fuld baggrundsinformation, hvis den findes
            if 'full_background_info' in session['results']:
                session_data['full_background_info'] = session['results']['full_background_info']
                session_data['has_detailed_background'] = True
            
            # Bevar redaktionelle overvejelser, hvis de findes
            if 'editorial_considerations' in session['results']:
                session_data['editorial_considerations'] = session['results']['editorial_considerations']
                session_data['has_editorial_considerations'] = True
                
            # Bevar source suggestions, hvis de findes
            if 'source_suggestions' in session['results']:
                session_data['source_suggestions'] = session['results']['source_suggestions']
        
        # Store results in session for download options and detailed report
        session['results'] = session_data
        session.modified = True
        
        # Calculate age of data for display
        generated_at = session['results'].get('generated_at', current_time)
        try:
            gen_date = datetime.strptime(generated_at, '%Y-%m-%d %H:%M:%S')
            now = datetime.now()
            # Calculate age in days
            data_age = (now - gen_date).total_seconds() / (24 * 3600)
        except (ValueError, TypeError):
            data_age = 0
            
        return render_template(
            'results.html',
            topic=topic,
            profile_name=Path(profile_path).stem,
            angles=ranked_angles,
            background_info=background_info,
            generated_at=generated_at,
            data_age=data_age
        )
        
    except Exception as e:
        flash(f'Fejl under generering: {str(e)}')
        return redirect(url_for('index'))

@app.route('/detailed_report')
def detailed_report():
    """Show detailed report with comprehensive background information."""
    # Get stored results from session
    results = session.get('results', {})
    
    if not results:
        flash('Ingen resultater at vise. Generer vinkler først.')
        return redirect(url_for('index'))
    
    topic = results.get('topic', 'Ukendt emne')
    profile_name = results.get('profile_name', 'Ukendt profil')
    angles = results.get('angles', [])
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Calculate age of data for display
    generated_at = results.get('generated_at', current_time)
    last_updated = results.get('last_updated', generated_at)
    
    # Convert to datetime objects for age calculation
    try:
        gen_date = datetime.strptime(generated_at, '%Y-%m-%d %H:%M:%S')
        update_date = datetime.strptime(last_updated, '%Y-%m-%d %H:%M:%S')
        now = datetime.now()
        
        # Calculate age in days
        data_age = (now - gen_date).total_seconds() / (24 * 3600)
        updated_data_age = (now - update_date).total_seconds() / (24 * 3600)
    except (ValueError, TypeError):
        # Fallback if date parsing fails
        data_age = 0
        updated_data_age = 0
    
    # Check for error states from session
    background_info_error = results.get('background_info_error', '')
    editorial_considerations_error = results.get('editorial_considerations_error', '')
    pdf_error = results.get('pdf_error', '')
    
    # Status flags
    has_background_info_error = bool(background_info_error)
    has_editorial_considerations_error = bool(editorial_considerations_error)
    
    # Check for full background info first, then fall back to summary
    background_info = results.get('full_background_info', results.get('background_info', ''))
    
    # If we don't have any background info in the session, try to get more comprehensive information
    if not background_info and not has_background_info_error:
        try:
            # Get more comprehensive background information
            background_info = get_topic_info_sync(topic, detailed=True)
            
            # Store this detailed info in the session for future use
            if 'results' in session:
                session['results']['full_background_info'] = background_info
                session['results']['has_detailed_background'] = True
                session['results']['last_updated'] = current_time
                session['results']['background_info_error'] = ''
                session.modified = True
        except Exception as e:
            app.logger.error(f"Error fetching detailed background: {e}")
            # Store error info in session
            if 'results' in session:
                session['results']['background_info_error'] = str(e)
                session['results']['has_detailed_background'] = False
                session.modified = True
            background_info = ""
            has_background_info_error = True
            background_info_error = str(e)
    
    # Get source suggestions either from session or extract from angles
    source_suggestions = results.get('source_suggestions', '')
    
    # If we don't have source suggestions in session, extract from angles
    if not source_suggestions:
        for angle in angles:
            if 'kildeForslagInfo' in angle:
                source_suggestions = angle['kildeForslagInfo']
                # Store in session for future use
                if 'results' in session:
                    session['results']['source_suggestions'] = source_suggestions
                    session['results']['last_updated'] = current_time
                    session.modified = True
                break
    
    # Check if we already have editorial considerations in session
    editorial_considerations = results.get('editorial_considerations', '')
    
    # If we don't have editorial considerations and no recorded error, generate them
    if not editorial_considerations and not has_editorial_considerations_error and angles:
        try:
            # Generate editorial considerations
            editorial_considerations = generate_editorial_considerations(topic, profile_name, angles)
            
            # Store the editorial considerations in the session
            if 'results' in session:
                session['results']['editorial_considerations'] = editorial_considerations
                session['results']['has_editorial_considerations'] = True
                session['results']['editorial_considerations_error'] = ''
                session['results']['last_updated'] = current_time
                session.modified = True
        except Exception as e:
            app.logger.error(f"Error generating editorial considerations: {e}")
            # Store error info in session
            if 'results' in session:
                session['results']['editorial_considerations_error'] = str(e)
                session['results']['has_editorial_considerations'] = False
                session.modified = True
            editorial_considerations = ""
            has_editorial_considerations_error = True
            editorial_considerations_error = str(e)
    
    return render_template(
        'detailed_report.html',
        topic=topic,
        profile_name=profile_name,
        background_info=background_info,
        source_suggestions=source_suggestions,
        angles=angles,
        editorial_considerations=editorial_considerations,
        # Timestamps for age calculation
        generated_at=generated_at,
        last_updated=last_updated,
        data_age=data_age,
        updated_data_age=updated_data_age,
        # Error states
        background_info_error=background_info_error,
        has_background_info_error=has_background_info_error,
        editorial_considerations_error=editorial_considerations_error,
        has_editorial_considerations_error=has_editorial_considerations_error,
        pdf_error=pdf_error
    )

@app.route('/regenerate_considerations')
def regenerate_considerations():
    """Regenerate editorial considerations."""
    # Get stored results from session
    results = session.get('results', {})
    
    if not results:
        flash('Ingen resultater at vise. Generer vinkler først.')
        return redirect(url_for('index'))
    
    topic = results.get('topic', 'Ukendt emne')
    profile_name = results.get('profile_name', 'Ukendt profil')
    angles = results.get('angles', [])
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Clear existing editorial considerations
    if 'results' in session:
        session['results']['editorial_considerations'] = ''
        session['results']['editorial_considerations_error'] = ''
        session['results']['has_editorial_considerations'] = False
        session.modified = True
    
    # Only try to regenerate if we have angles
    if angles:
        try:
            # Generate editorial considerations
            editorial_considerations = generate_editorial_considerations(topic, profile_name, angles)
            
            # Store the editorial considerations in the session
            if 'results' in session:
                session['results']['editorial_considerations'] = editorial_considerations
                session['results']['has_editorial_considerations'] = True
                session['results']['editorial_considerations_error'] = ''
                session['results']['last_updated'] = current_time
                session.modified = True
                
            flash('Redaktionelle overvejelser blev regenereret.')
        except Exception as e:
            app.logger.error(f"Error regenerating editorial considerations: {e}")
            # Store error info in session
            if 'results' in session:
                session['results']['editorial_considerations_error'] = str(e)
                session['results']['has_editorial_considerations'] = False
                session.modified = True
                
            flash(f'Kunne ikke regenerere redaktionelle overvejelser: {str(e)}', 'error')
    else:
        flash('Ingen vinkler fundet. Kan ikke generere redaktionelle overvejelser.', 'error')
    
    return redirect(url_for('detailed_report'))

@app.route('/regenerate_background')
def regenerate_background():
    """Regenerate background information."""
    # Get stored results from session
    results = session.get('results', {})
    
    if not results:
        flash('Ingen resultater at vise. Generer vinkler først.')
        return redirect(url_for('index'))
    
    topic = results.get('topic', 'Ukendt emne')
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Clear existing background info
    if 'results' in session:
        session['results']['full_background_info'] = ''
        session['results']['background_info_error'] = ''
        session['results']['has_detailed_background'] = False
        session.modified = True
    
    try:
        # Get more comprehensive background information
        background_info = get_topic_info_sync(topic, detailed=True)
        
        # Store this detailed info in the session for future use
        if 'results' in session:
            session['results']['full_background_info'] = background_info
            session['results']['has_detailed_background'] = True
            session['results']['background_info_error'] = ''
            session['results']['last_updated'] = current_time
            session.modified = True
            
        flash('Baggrundsinformation blev opdateret.')
    except Exception as e:
        app.logger.error(f"Error regenerating background info: {e}")
        # Store error info in session
        if 'results' in session:
            session['results']['background_info_error'] = str(e)
            session['results']['has_detailed_background'] = False
            session.modified = True
            
        flash(f'Kunne ikke opdatere baggrundsinformation: {str(e)}', 'error')
    
    return redirect(url_for('detailed_report'))

@app.route('/download_report')
def download_report():
    """Download the full detailed report as PDF."""
    results = session.get('results', {})
    
    if not results:
        flash('Ingen resultater at downloade. Generer vinkler først.')
        return redirect(url_for('index'))
    
    topic = results.get('topic', 'Ukendt emne')
    profile_name = results.get('profile_name', 'Ukendt profil')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Tilføj tidspunkt for generering i filnavnet
    generation_time = results.get('generated_at', timestamp.replace('_', ' '))
    generation_date = generation_time.split(' ')[0] if ' ' in generation_time else generation_time
    filename = f"vinkeljernet_rapport_{topic}_{profile_name}_{generation_date}_{timestamp}.pdf"
    
    try:
        # Generate HTML for the report
        html_content = generate_report_html(results)
        
        # Check for wkhtmltopdf installation
        wkhtmltopdf_path = None
        try:
            # Try to find wkhtmltopdf in PATH
            import subprocess
            wkhtmltopdf_path = subprocess.check_output(['which', 'wkhtmltopdf'], 
                                                      stderr=subprocess.STDOUT).decode().strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            # If not found, look in common locations
            common_paths = [
                '/usr/local/bin/wkhtmltopdf',
                '/usr/bin/wkhtmltopdf',
                'C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe',
                'C:\\Program Files (x86)\\wkhtmltopdf\\bin\\wkhtmltopdf.exe'
            ]
            for path in common_paths:
                if os.path.exists(path):
                    wkhtmltopdf_path = path
                    break
        
        # Configure pdfkit
        config = None
        if wkhtmltopdf_path:
            config = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)
        
        # PDF generation options
        options = {
            'encoding': 'UTF-8',
            'page-size': 'A4',
            'margin-top': '20mm',
            'margin-right': '20mm',
            'margin-bottom': '20mm',
            'margin-left': '20mm',
            'footer-right': 'Side [page] af [topage]',
            'footer-font-size': '9',
            'title': f'Vinkeljernet Rapport - {topic}',
            'enable-local-file-access': None,  # Allow access to local files
            'quiet': None  # Hide console messages
        }
        
        # Convert HTML to PDF
        pdf = pdfkit.from_string(html_content, False, options=options, configuration=config)
        
        # Return as downloadable file
        buffer = BytesIO(pdf)
        buffer.seek(0)
        return send_file(
            buffer,
            as_attachment=True,
            download_name=filename,
            mimetype="application/pdf",
            headers={'Cache-Control': 'no-cache, no-store, must-revalidate'}
        )
    except Exception as e:
        error_message = str(e)
        app.logger.error(f"Error generating report PDF: {error_message}")
        
        # Store error in session for better display
        if 'results' in session:
            session['results']['pdf_error'] = error_message
            session.modified = True
        
        # More user-friendly error message based on common problems
        if "wkhtmltopdf" in error_message.lower():
            user_message = 'PDF-generering fejlede: wkhtmltopdf er ikke installeret eller kan ikke findes. Se README_WKHTMLTOPDF.txt for instruktioner.'
        elif "connection" in error_message.lower() or "timeout" in error_message.lower():
            user_message = 'PDF-generering fejlede på grund af netværksproblemer. Prøv igen senere.'
        else:
            user_message = f'Fejl ved generering af rapport: {error_message}'
            
        flash(user_message, 'error')
        return redirect(url_for('detailed_report'))

@app.route('/download/<format>')
def download(format):
    """Download results in different formats."""
    results = session.get('results', {})
    
    if not results:
        flash('Ingen resultater at downloade. Generer vinkler først.')
        return redirect(url_for('index'))
    
    topic = results.get('topic', 'Ukendt emne')
    profile_name = results.get('profile_name', 'Ukendt profil')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Inkluder tidspunkt for generering i filnavnet
    generation_time = results.get('generated_at', timestamp.replace('_', ' '))
    generation_date = generation_time.split(' ')[0] if ' ' in generation_time else generation_time
    filename = f"vinkeljernet_{topic}_{profile_name}_{generation_date}"
    
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
            # Check for wkhtmltopdf installation
            wkhtmltopdf_path = None
            try:
                # Try to find wkhtmltopdf in PATH
                import subprocess
                wkhtmltopdf_path = subprocess.check_output(['which', 'wkhtmltopdf'], 
                                                           stderr=subprocess.STDOUT).decode().strip()
            except (subprocess.CalledProcessError, FileNotFoundError):
                # If not found, look in common locations
                common_paths = [
                    '/usr/local/bin/wkhtmltopdf',
                    '/usr/bin/wkhtmltopdf',
                    'C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe',
                    'C:\\Program Files (x86)\\wkhtmltopdf\\bin\\wkhtmltopdf.exe'
                ]
                for path in common_paths:
                    if os.path.exists(path):
                        wkhtmltopdf_path = path
                        break
            
            # Configure pdfkit
            config = None
            if wkhtmltopdf_path:
                config = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)
            
            # Set PDF options
            options = {
                'encoding': 'UTF-8',
                'page-size': 'A4',
                'margin-top': '20mm',
                'margin-right': '20mm',
                'margin-bottom': '20mm',
                'margin-left': '20mm',
                'footer-right': 'Side [page] af [topage]',
                'footer-font-size': '9',
                'title': f'Vinkeljernet - {topic}',
                'enable-local-file-access': None,
                'quiet': None
            }
            
            # Convert HTML to PDF
            pdf = pdfkit.from_string(html_content, False, options=options, configuration=config)
            
            # Return as downloadable file
            buffer = BytesIO(pdf)
            buffer.seek(0)
            return send_file(
                buffer,
                as_attachment=True,
                download_name=f"{filename}.pdf",
                mimetype="application/pdf",
                headers={'Cache-Control': 'no-cache, no-store, must-revalidate'}
            )
        except Exception as e:
            error_message = str(e)
            app.logger.error(f"Error generating PDF: {error_message}")
            
            # Store error in session for better display
            if 'results' in session:
                session['results']['pdf_error'] = error_message
                session.modified = True
            
            # More user-friendly error message
            if "wkhtmltopdf" in error_message.lower():
                user_message = 'PDF-generering fejlede: wkhtmltopdf er ikke installeret eller kan ikke findes. Se README_WKHTMLTOPDF.txt for instruktioner.'
            elif "connection" in error_message.lower() or "timeout" in error_message.lower():
                user_message = 'PDF-generering fejlede på grund af netværksproblemer. Prøv igen senere.'
            else:
                user_message = f'Fejl ved generering af PDF: {error_message}'
                
            flash(user_message, 'error')
            return redirect(url_for('detailed_report'))
    
    flash(f'Ukendt format: {format}')
    return redirect(url_for('index'))

def generate_text_content(results):
    """Generate plain text content for download."""
    topic = results.get('topic', 'Ukendt emne')
    profile_name = results.get('profile_name', 'Ukendt profil')
    angles = results.get('angles', [])
    
    # Tidsstempler
    generated_at = results.get('generated_at', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    last_updated = results.get('last_updated', generated_at)
    timestamp = generated_at
    
    # Indhold
    background_info = results.get('full_background_info', results.get('background_info', ''))
    editorial_considerations = results.get('editorial_considerations', '')
    source_suggestions = results.get('source_suggestions', '')
    
    content = [
        f"VINKELJERNET - GENEREREDE NYHEDSVINKLER",
        f"Emne: {topic}",
        f"Profil: {profile_name}",
        f"Genereret: {generated_at}",
    ]
    
    # Tilføj information om opdatering, hvis relevant
    if generated_at != last_updated:
        content.append(f"Sidst opdateret: {last_updated}")
    
    content.append("\n" + "="*50 + "\n")
    
    # Add background info if available
    if background_info:
        content.append("BAGGRUNDSINFORMATION")
        content.append("-"*50)
        content.append(background_info)
        content.append("\n" + "="*50 + "\n")
    
    # Add editorial considerations if available
    if editorial_considerations:
        content.append("REDAKTIONELLE OVERVEJELSER")
        content.append("-"*50)
        content.append(editorial_considerations)
        content.append("\n" + "="*50 + "\n")
        
    # Add source suggestions if available
    if source_suggestions:
        content.append("RELEVANTE KILDER OG RESSOURCER")
        content.append("-"*50)
        content.append(source_suggestions)
        content.append("\n" + "="*50 + "\n")
    
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
    background_info = results.get('background_info', '')
    editorial_considerations = results.get('editorial_considerations', '')
    
    html = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset='UTF-8'>",
        "<title>Vinkeljernet Resultater</title>",
        "<style>",
        "@page {",
        "    margin: 2cm;",
        "    @bottom-right {",
        "        content: 'Side ' counter(page) ' af ' counter(pages);",
        "        font-size: 9pt;",
        "    }",
        "}",
        "body {",
        "    font-family: 'Helvetica', 'Arial', sans-serif;",
        "    line-height: 1.6;",
        "    color: #333;",
        "    margin: 0;",
        "    padding: 0;",
        "    font-size: 10pt;",
        "}",
        ".container {",
        "    width: 100%;",
        "    max-width: 800px;",
        "    margin: 0 auto;",
        "    padding: 20px;",
        "}",
        "h1 {",
        "    color: #2c3e50;",
        "    border-bottom: 2px solid #3498db;",
        "    padding-bottom: 10px;",
        "    font-size: 18pt;",
        "    margin-top: 0;",
        "    margin-bottom: 20px;",
        "}",
        "h2 {",
        "    color: #3498db;",
        "    margin-top: 30px;",
        "    font-size: 16pt;",
        "    page-break-after: avoid;",
        "}",
        "h3 {",
        "    color: #2980b9;",
        "    font-size: 14pt;",
        "    page-break-after: avoid;",
        "}",
        ".meta {",
        "    color: #7f8c8d;",
        "    margin-bottom: 30px;",
        "    font-size: 9pt;",
        "    border: 1px solid #eee;",
        "    padding: 10px;",
        "    background-color: #f9f9f9;",
        "}",
        ".angle {",
        "    margin-bottom: 20px;",
        "    padding: 15px;",
        "    background-color: #f8f9fa;",
        "    border-left: 4px solid #2ecc71;",
        "    border-radius: 5px;",
        "    page-break-inside: avoid;",
        "}",
        ".angle h3 {",
        "    margin-top: 0;",
        "}",
        ".description {",
        "    margin-bottom: 10px;",
        "}",
        ".rationale {",
        "    color: #555;",
        "    font-style: italic;",
        "    margin-bottom: 10px;",
        "}",
        ".criteria {",
        "    margin-bottom: 10px;",
        "}",
        ".questions {",
        "    margin-top: 15px;",
        "}",
        ".section {",
        "    margin-top: 30px;",
        "    margin-bottom: 30px;",
        "    padding: 15px;",
        "    border: 1px solid #ddd;",
        "    border-radius: 5px;",
        "    background-color: #f9f9f9;",
        "    page-break-inside: avoid;",
        "}",
        "ul {",
        "    padding-left: 20px;",
        "}",
        "li {",
        "    margin-bottom: 5px;",
        "}",
        ".badge {",
        "    display: inline-block;",
        "    padding: 5px 8px;",
        "    font-size: 8pt;",
        "    font-weight: bold;",
        "    background-color: #95a5a6;",
        "    color: white;",
        "    border-radius: 3px;",
        "    margin-right: 5px;",
        "}",
        "footer {",
        "    font-size: 8pt;",
        "    text-align: center;",
        "    color: #7f8c8d;",
        "    margin-top: 30px;",
        "    border-top: 1px solid #eee;",
        "    padding-top: 10px;",
        "}",
        "</style>",
        "</head>",
        "<body>",
        "<div class='container'>",
        f"<h1>Vinkeljernet - Genererede nyhedsvinkler</h1>",
        f"<div class='meta'>",
        f"<p><strong>Emne:</strong> {topic}</p>",
        f"<p><strong>Profil:</strong> {profile_name}</p>",
        f"<p><strong>Genereret:</strong> {timestamp}</p>",
        f"</div>"
    ]
    
    # Add background information if available
    if background_info:
        html.append("<div class='section'>")
        html.append("<h2>Baggrundsinformation</h2>")
        html.append(f"<div>{background_info.replace(chr(10), '<br>')}</div>")
        html.append("</div>")
    
    # Add editorial considerations if available
    if editorial_considerations:
        html.append("<div class='section'>")
        html.append("<h2>Redaktionelle overvejelser</h2>")
        html.append(f"<div>{editorial_considerations.replace(chr(10), '<br>')}</div>")
        html.append("</div>")
    
    # Add page break before angles
    html.append("<div class='page-break'></div>")
    
    # Add section header for angles
    html.append("<h2>Anbefalede vinkler</h2>")
    
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
        
        html.append("<div class='criteria'><strong>Nyhedskriterier:</strong> ")
        for criterion in criteria:
            html.append(f"<span class='badge'>{criterion}</span>")
        html.append("</div>")
        
        if questions:
            html.append("<div class='questions'><strong>Startspørgsmål:</strong>")
            html.append("<ul>")
            for q in questions:
                html.append(f"<li>{q}</li>")
            html.append("</ul>")
            html.append("</div>")
        
        html.append("</div>")
    
    # Add footer
    html.append("<footer>")
    html.append(f"<p>Genereret med Vinkeljernet {timestamp}</p>")
    html.append("</footer>")
    
    html.append("</div>") # Close container
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
            
        # Store the full background information in the session
        if 'results' in session:
            # Opdater eksisterende session med ny baggrundsinformation
            session['results']['full_background_info'] = topic_info
            session['results']['has_detailed_background'] = True
            session['results']['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            session.modified = True
        else:
            # Initialize results dict if it doesn't exist
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            profile_name = Path(profile_path).stem if 'profile_path' in locals() else ''
            
            session['results'] = {
                'topic': topic,
                'profile_name': profile_name,
                'full_background_info': topic_info,
                'background_info': topic_info[:1000] + ("..." if len(topic_info) > 1000 else ""),
                # Tidsstempler
                'generated_at': current_time,
                'last_updated': current_time,
                # Status flags
                'has_editorial_considerations': False,
                'has_detailed_background': True,
                # Default tom værdi for fremtidige data
                'editorial_considerations': '',
                'source_suggestions': '',
                'angles': []
            }
            session.modified = True
        
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

def get_topic_info_sync(topic, detailed=False):
    """Synchronous version of fetch_topic_information.
    
    Args:
        topic: The news topic to gather information about
        detailed: If True, get more comprehensive information
    """
    try:
        # Use the Perplexity API directly
        from config import PERPLEXITY_API_KEY
        
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json"
        }
        
        if detailed:
            # More comprehensive prompt for detailed report with specific structure
            user_prompt = f"""
            Giv en grundig og velstruktureret analyse af emnet '{topic}' med følgende sektioner:
            
            # OVERSIGT
            En kort 3-5 linjers opsummering af emnet, der dækker det mest centrale.
            
            # BAGGRUND
            Relevant historisk kontekst og udvikling indtil nu. Inkluder vigtige begivenheder og milepæle med præcise datoer.
            
            # AKTUEL STATUS
            Den nuværende situation med fokus på de seneste udviklinger. Beskriv præcist hvad der sker lige nu og hvorfor det er vigtigt.
            
            # NØGLETAL
            Konkrete statistikker, data og fakta relateret til emnet. Inkluder tal, procenter og, hvis muligt, kilder til informationen.
            
            # PERSPEKTIVER
            De forskellige synspunkter og holdninger til emnet fra forskellige aktører og interessenter. Præsenter de forskellige sider objektivt.
            
            # RELEVANS FOR DANMARK
            Hvordan emnet specifikt relaterer til eller påvirker Danmark og danskerne. Inkluder lokale eksempler når det er relevant.
            
            # FREMTIDSUDSIGTER
            Forventede eller mulige fremtidige udviklinger og tendenser baseret på de aktuelle fakta.
            
            # KILDER
            En liste over 3-5 pålidelige danske kilder, hvor man kan finde yderligere information om emnet.
            
            Formatér svaret med tydelige overskrifter for hver sektion. Sørg for at information er så faktabaseret og objektiv som muligt.
            """
            max_tokens = 1500
        else:
            # Forbedret standardprompt med mere struktureret information
            user_prompt = f"""
            Giv en koncis og velstruktureret oversigt over følgende nyhedsemne: '{topic}'.
            
            Din oversigt skal indeholde:
            
            # OVERSIGT
            En kort 2-3 linjers sammenfatning af, hvad emnet handler om.
            
            # AKTUEL STATUS
            Den nuværende situation og hvorfor emnet er relevant lige nu.
            
            # NØGLETAL
            2-3 vigtige fakta, statistikker eller tal relateret til emnet.
            
            # PERSPEKTIVER
            Kort opsummering af forskellige synspunkter på emnet.
            
            Hold svaret faktuelt og præcist med vigtige datoer og konkrete detaljer.
            """
            max_tokens = 1000

        # Forbedret system prompt til at sikre struktureret, faktabaseret indhold
        system_prompt = """Du er en erfaren dansk journalist med ekspertise i at fremstille komplekse emner på en struktureret og faktabaseret måde.

Din opgave er at give pålidelig og velstruktureret information om aktuelle nyhedsemner. Følg disse retningslinjer:

1. Organiser dit svar efter de angivne sektioner med PRÆCISE overskrifter
2. Prioriter fakta og konkrete detaljer frem for generelle beskrivelser
3. Inkluder specifik information: præcise tal, datoer, navne og steder
4. Bevar objektivitet ved at præsentere forskellige perspektiver neutralt
5. Fokuser på den danske kontekst, når det er relevant
6. Nævn kilder til information hvor muligt, især for statistik og citater
7. Vær koncis og præcis - prioriter kvalitet over kvantitet

Undgå vaghed, generalisering og personlige holdninger. Brug et klart og professionelt sprog."""

        payload = {
            "model": "sonar",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2,  # Lavt for at sikre fakta-fokus
            "top_p": 0.85,       # Justeret for at reducere variabilitet
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

def generate_editorial_considerations(topic, profile_name, angles):
    """
    Generate editorial considerations for the given angles using Claude API.
    
    Args:
        topic: The news topic
        profile_name: Name of the editorial profile used
        angles: List of generated angles
        
    Returns:
        Formatted string with editorial considerations or error message
    """
    try:
        from config import ANTHROPIC_API_KEY
        
        # Format angles for the prompt
        formatted_angles = ""
        for i, angle in enumerate(angles, 1):
            headline = angle.get('overskrift', 'Ingen overskrift')
            description = angle.get('beskrivelse', 'Ingen beskrivelse')
            criteria = angle.get('nyhedskriterier', [])
            criteria_str = ", ".join(criteria) if criteria else "Ingen angivne"
            
            formatted_angles += f"Vinkel {i}: {headline}\n"
            formatted_angles += f"Beskrivelse: {description}\n"
            formatted_angles += f"Nyhedskriterier: {criteria_str}\n\n"
        
        # Construct the prompt
        prompt = f"""
        Som erfaren nyhedsredaktør, foretag en grundig redaktionel analyse af følgende vinkelforslag 
        til emnet "{topic}" med henblik på "{profile_name}" profilen:
        
        {formatted_angles}
        
        Giv en saglig og konstruktiv analyse der omfatter:
        
        1. JOURNALISTISKE STYRKER: Hvilke journalistiske styrker har disse vinkler samlet set? 
           Hvilke vinkler er særligt stærke og hvorfor?
        
        2. SVAGHEDER/BEGRÆNSNINGER: Hvor er der potentielle journalistiske svagheder eller begrænsninger i de foreslåede vinkler?
           Er der særlige forbehold eller udfordringer en redaktør bør være opmærksom på?
        
        3. REDAKTIONELLE MULIGHEDER: Hvilke særlige redaktionelle muligheder åbner disse vinkler for?
           Hvilke journalistiske formater ville passe godt (reportage, interview, analyse, etc.)?
        
        4. RESSOURCEBEHOV: Hvilke redaktionelle ressourcer og kompetencer ville kræves for at realisere disse vinkler effektivt?
           Anslå omfang af research, tid, specialviden, og evt. særlige produktionskrav.
        
        5. PRIORITERING: Giv en anbefaling om hvilke 2-3 vinkler der bør prioriteres højest fra et redaktionelt perspektiv og begrund dette.
        
        Vær specifik og konkret med direkte reference til de angivne vinkler. 
        Skriv på dansk og formatér svaret i klart adskilte sektioner med overskrifter.
        """
        
        # Make the API call to Claude
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01"
            },
            json={
                "model": "claude-3-haiku-20240307",  # Using a smaller, faster model for this analysis
                "max_tokens": 1500,
                "temperature": 0.2,  # Lower temperature for more focused analysis
                "system": "Du er en erfaren redaktør på et dansk nyhedsmedie med stor ekspertise i journalistiske vinkler og redaktionelle prioriteringer.",
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30  # Set a reasonable timeout
        )
        
        # Check for successful response
        if response.status_code != 200:
            app.logger.error(f"Claude API error: {response.status_code}: {response.text}")
            return "Kunne ikke generere redaktionelle overvejelser. Der opstod en fejl ved kontakt til AI-systemet."
        
        # Extract and return the content
        response_data = response.json()
        editorial_text = response_data['content'][0]['text']
        return editorial_text
        
    except Exception as e:
        app.logger.error(f"Error generating editorial considerations: {e}")
        return f"Kunne ikke generere redaktionelle overvejelser: {str(e)}"

def generate_report_html(results):
    """
    Generate HTML for the full detailed report PDF.
    
    Args:
        results: Dictionary containing all necessary data from session
        
    Returns:
        HTML string for the report
    """
    try:
        from flask import render_template
        
        # Extract data from results with forbedrede standardværdier
        topic = results.get('topic', 'Ukendt emne')
        profile_name = results.get('profile_name', 'Ukendt profil')
        angles = results.get('angles', [])
        
        # Prioritize full background info over summary
        background_info = results.get('full_background_info', results.get('background_info', ''))
        
        # Hent redaktionelle overvejelser og kilde forslag fra session
        editorial_considerations = results.get('editorial_considerations', '')
        source_suggestions = results.get('source_suggestions', '')
        
        # Information om tidspunkter
        generated_at = results.get('generated_at', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        last_updated = results.get('last_updated', generated_at)
        timestamp = generated_at
        
        # Metadata til visning på rapporten
        metadata = {
            'generated_at': generated_at,
            'last_updated': last_updated,
            'has_detailed_background': results.get('has_detailed_background', False),
            'has_editorial_considerations': results.get('has_editorial_considerations', False)
        }
        
        # Hvis source suggestions ikke findes i session, forsøg at hente fra angles
        if not source_suggestions:
            for angle in angles:
                if 'kildeForslagInfo' in angle:
                    source_suggestions = angle['kildeForslagInfo']
                    break
        
        # Use the dedicated PDF template
        html_content = render_template(
            'report_pdf.html', 
            topic=topic, 
            profile_name=profile_name,
            angles=angles,
            background_info=background_info,
            editorial_considerations=editorial_considerations,
            source_suggestions=source_suggestions,
            timestamp=timestamp,
            metadata=metadata,
            generated_at=generated_at,
            last_updated=last_updated
        )
        
        return html_content
        
    except Exception as e:
        app.logger.error(f"Error generating report HTML: {e}")
        # Return a simple error HTML
        return f"""
        <!DOCTYPE html>
        <html>
        <head><title>Fejl</title></head>
        <body>
            <h1>Der opstod en fejl ved generering af rapporten</h1>
            <p>{str(e)}</p>
        </body>
        </html>
        """

@app.route('/clear_pdf_error', methods=['POST'])
def clear_pdf_error():
    """Clear PDF error from session."""
    if 'results' in session and 'pdf_error' in session['results']:
        session['results']['pdf_error'] = ''
        session.modified = True
    return '', 204  # No content response

@app.template_filter('nl2br')
def nl2br(value):
    """Convert newlines to <br> tags for proper display in HTML."""
    if not value:
        return ""
    
    # Replace newlines with <br> tags
    value = value.replace('\n', '<br>')
    value = value.replace('\r\n', '<br>')
    return value

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