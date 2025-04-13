"""
Flask web interface for Vinkeljernet.

This module provides a secure web interface for the Vinkeljernet application,
allowing users to generate news angles through a web browser.
"""
import os
import json
import asyncio
import pdfkit
import time
import logging
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from io import BytesIO
from functools import wraps
from flask import (
    Flask, render_template, request, redirect, url_for, session, 
    send_file, flash, abort, jsonify, current_app, g
)
from flask_wtf import FlaskForm, CSRFProtect
from wtforms import StringField, SelectField, SubmitField
from wtforms.validators import DataRequired, Length, ValidationError, Regexp
import hmac
import secrets
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.exceptions import HTTPException

# Import from Vinkeljernet
from config_loader import load_and_validate_profile, get_available_profiles
from api_clients_wrapper import fetch_topic_information, generate_angles
from prompt_engineering import construct_angle_prompt, parse_angles_from_response
from angle_processor import filter_and_rank_angles
from models import RedaktionelDNA
import requests

# Import security-related modules
from security import (
    sanitize_input, validate_url, validate_news_criteria, 
    rate_limit, setup_security_middleware, audit_log, 
    InvalidInputException, secure_filename
)

# Import from configuration
from config import (
    APP_ENV, SECRET_KEY, COOKIE_SECURE, COOKIE_HTTPONLY, 
    COOKIE_SAMESITE, SESSION_LIFETIME, MAX_CONTENT_LENGTH,
    RATE_LIMIT_DEFAULT, RATE_LIMIT_API, ANTHROPIC_API_KEY
)

# Configure logging
logger = logging.getLogger("vinkeljernet.app")

# Define DEBUG mode based on environment
DEBUG = APP_ENV == "development"  # True if in development mode, False otherwise

# Initialize Flask app with security settings
app = Flask(__name__)
app.config.from_mapping(
    # Core settings
    SECRET_KEY=SECRET_KEY,
    PROFILE_DIR='config',
    MAX_CONTENT_LENGTH=MAX_CONTENT_LENGTH,
    
    # Session security
    SESSION_COOKIE_SECURE=COOKIE_SECURE,
    SESSION_COOKIE_HTTPONLY=COOKIE_HTTPONLY,
    SESSION_COOKIE_SAMESITE=COOKIE_SAMESITE,
    PERMANENT_SESSION_LIFETIME=timedelta(seconds=SESSION_LIFETIME),
    SESSION_TYPE='filesystem',
    
    # CSRF protection
    WTF_CSRF_TIME_LIMIT=SESSION_LIFETIME,
    WTF_CSRF_SSL_STRICT=True,
    WTF_CSRF_CHECK_DEFAULT=True,
    
    # Prevent browser caching of sensitive pages
    SEND_FILE_MAX_AGE_DEFAULT=0,
)

# Fix for proxied environments
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

# Add CSRF protection
csrf = CSRFProtect(app)

# Set up security middleware
setup_security_middleware(app)

# Configure comprehensive logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Set up file handler for app logs
file_handler = logging.FileHandler(os.path.join(log_dir, 'vinkeljernet_web.log'))
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
file_handler.setLevel(logging.INFO)
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

# Set up security audit log
audit_handler = logging.FileHandler(os.path.join(log_dir, 'security_audit.log'))
audit_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levellevel)s - %(message)s'
))
audit_handler.setLevel(logging.INFO)
security_logger = logging.getLogger('vinkeljernet.security')
security_logger.addHandler(audit_handler)
security_logger.setLevel(logging.INFO)

# Log app startup
app.logger.info("Vinkeljernet web application starting")

#
# Security and Rate Limiting
#

# Add request security logging and form validation

@app.before_request
def log_request_info():
    """Log basic information about each request for security auditing."""
    # Skip for static files
    if request.path.startswith('/static/'):
        return
        
    # Generate unique request ID and store in g
    g.request_id = str(uuid.uuid4())
    
    # Log request details
    logger.info(
        f"Request {g.request_id}: {request.method} {request.path} - "
        f"IP: {request.remote_addr}, User-Agent: {request.user_agent}"
    )

@app.after_request
def add_security_headers(response):
    """Add security headers to all responses."""
    # Add security headers
    response.headers['X-Request-ID'] = getattr(g, 'request_id', 'unknown')
    
    # Log response status
    logger.info(f"Response {getattr(g, 'request_id', 'unknown')}: {response.status_code}")
    
    return response

def validate_profile_path(form, field):
    """Custom validator for profile paths."""
    path = field.data
    if path is None or not path:
        raise ValidationError("Profile path cannot be empty")
        
    # Ensure path is within the allowed directory
    profile_dir = Path(app.config['PROFILE_DIR']).resolve()
    try:
        path_obj = Path(path).resolve()
        if not str(path_obj).startswith(str(profile_dir)):
            raise ValidationError("Invalid profile path")
    except (ValueError, TypeError):
        raise ValidationError("Invalid profile path format")
        
    # Verify file exists
    if not path_obj.exists() or not path_obj.is_file():
        raise ValidationError("Profile file does not exist")
        
    # Check file extension
    if not path_obj.suffix.lower() in ('.yaml', '.yml'):
        raise ValidationError("Profile must be a YAML file")

def validate_topic(form, field):
    """
    Validate topic field for security and input constraints.
    
    Blocks potentially dangerous input patterns.
    """
    topic = field.data
    
    # Empty check
    if not topic:
        raise ValidationError("Topic cannot be empty")
        
    # Length check
    if len(topic) < 3:
        raise ValidationError("Topic must be at least 3 characters")
    if len(topic) > 100:
        raise ValidationError("Topic must be under 100 characters")
    
    # Disallow HTML tags
    if '<' in topic and '>' in topic:
        raise ValidationError("HTML tags are not allowed in topics")
        
    # Disallow potential script injection patterns
    dangerous_patterns = [
        'javascript:',
        'data:',
        'vbscript:',
        'document.cookie',
        'eval(',
        'onload=',
        'onerror=',
        'script',
        '<img',
        '${',
        '#{',
        ';',
        '||',
        '&&',
        '../',
        '..\\'
    ]
    
    for pattern in dangerous_patterns:
        if pattern.lower() in topic.lower():
            raise ValidationError("Topic contains disallowed pattern")
    
    # Validate character set
    try:
        sanitize_input(topic, max_length=100)
    except InvalidInputException as e:
        raise ValidationError(str(e))

# Enhanced request tracking and permissions
class RequestTracker:
    """Track request metrics and enforce rate limits."""
    
    def __init__(self):
        self.requests = {}
        self.blocked_ips = {}
        
    def track_request(self, ip, path):
        """Track a request from a specific IP address."""
        current_time = time.time()
        
        # Initialize tracking for this IP if it doesn't exist
        if ip not in self.requests:
            self.requests[ip] = {
                'total_count': 0,
                'paths': {},
                'first_seen': current_time,
                'last_seen': current_time
            }
        
        # Update request counts
        self.requests[ip]['total_count'] += 1
        self.requests[ip]['last_seen'] = current_time
        
        # Track by path
        if path not in self.requests[ip]['paths']:
            self.requests[ip]['paths'][path] = {'count': 0, 'last': current_time}
        self.requests[ip]['paths'][path]['count'] += 1
        self.requests[ip]['paths'][path]['last'] = current_time
        
        # Log suspicious behavior (rapid requests)
        if self.requests[ip]['total_count'] > 100 and \
           (current_time - self.requests[ip]['first_seen']) < 60:
            logger.warning(f"Suspicious activity from IP {ip}: {self.requests[ip]['total_count']} requests in under a minute")
            
        return self.requests[ip]['total_count']
        
    def check_blocked(self, ip):
        """Check if an IP is currently blocked."""
        current_time = time.time()
        if ip in self.blocked_ips and self.blocked_ips[ip] > current_time:
            block_remaining = int(self.blocked_ips[ip] - current_time)
            return True, block_remaining
        return False, 0
        
    def block_ip(self, ip, duration=300):
        """Block an IP for a specified duration (in seconds)."""
        self.blocked_ips[ip] = time.time() + duration
        logger.warning(f"Blocking IP {ip} for {duration} seconds")
        
    def cleanup(self):
        """Clean up old tracking data."""
        current_time = time.time()
        # Remove data over 1 hour old
        cutoff = current_time - 3600
        
        # Clean up requests
        for ip in list(self.requests.keys()):
            if self.requests[ip]['last_seen'] < cutoff:
                del self.requests[ip]
                
        # Clean up blocked IPs
        for ip in list(self.blocked_ips.keys()):
            if self.blocked_ips[ip] < current_time:
                del self.blocked_ips[ip]

# Create global tracker instance
request_tracker = RequestTracker()

def enhanced_rate_limit(limit=RATE_LIMIT_DEFAULT, window=60, block_after=5):
    """
    Enhanced rate limiting decorator with IP blocking for abuse prevention.
    
    Args:
        limit: Maximum number of requests allowed in the window
        window: Time window in seconds
        block_after: Number of consecutive violations before temporary block
    """
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            # Get client identifier (IP address)
            client_ip = request.remote_addr or 'unknown'
            
            # Get request path
            request_path = request.path
            
            # Check if IP is blocked
            blocked, remaining = request_tracker.check_blocked(client_ip)
            if blocked:
                logger.warning(f"Blocked request from {client_ip} to {request_path}")
                response = jsonify({
                    "error": "Too many requests",
                    "message": f"Your IP has been temporarily blocked. Try again in {remaining} seconds.",
                    "status_code": 429
                })
                response.status_code = 429
                return response
            
            # Track this request
            request_count = request_tracker.track_request(client_ip, request_path)
            
            # Add client info to request context for logging
            g.client_ip = client_ip
            g.request_count = request_count
            
            # Rate limit check (simplified but more robust than before)
            path_key = f"{client_ip}:{request_path}"
            key = f"rate_limit:{path_key}"
            
            if hasattr(g, 'rate_limit_data'):
                rate_data = g.rate_limit_data
            else:
                g.rate_limit_data = {}
                rate_data = g.rate_limit_data
                
            current_time = time.time()
            
            # Check/initialize rate limit data
            if key not in rate_data:
                rate_data[key] = {
                    'count': 0,
                    'reset_at': current_time + window,
                    'violations': 0
                }
            
            # Reset if window has expired
            if current_time > rate_data[key]['reset_at']:
                rate_data[key] = {
                    'count': 0,
                    'reset_at': current_time + window,
                    'violations': 0
                }
            
            # Increment and check
            rate_data[key]['count'] += 1
            
            # Check if over limit
            if rate_data[key]['count'] > limit:
                rate_data[key]['violations'] += 1
                
                # Check if we should block this IP
                if rate_data[key]['violations'] >= block_after:
                    # Calculate block duration with exponential backoff
                    block_duration = min(300 * (2 ** (rate_data[key]['violations'] - block_after)), 86400)
                    request_tracker.block_ip(client_ip, block_duration)
                    
                    # Audit log the blocking event
                    audit_log(
                        action="IP_BLOCKED",
                        details={
                            'ip': client_ip,
                            'path': request_path,
                            'violations': rate_data[key]['violations'],
                            'duration': block_duration
                        }
                    )
                    
                    # Return block response
                    response = jsonify({
                        "error": "Too many requests",
                        "message": f"Your IP has been temporarily blocked due to excessive requests. Try again later.",
                        "status_code": 429
                    })
                    response.status_code = 429
                    return response
                
                # Return rate limit response
                retry_after = int(rate_data[key]['reset_at'] - current_time)
                response = jsonify({
                    "error": "Rate limit exceeded",
                    "retry_after": retry_after,
                    "status_code": 429
                })
                response.headers['Retry-After'] = str(retry_after)
                response.status_code = 429
                return response
            
            # Execute the function
            return f(*args, **kwargs)
        return wrapped
    return decorator

# Scheduled cleanup
def initialize():
    """Set up scheduled cleanup of tracking data."""
    def schedule_cleanup():
        while True:
            time.sleep(300)  # Run every 5 minutes
            request_tracker.cleanup()
            
    import threading
    cleanup_thread = threading.Thread(target=schedule_cleanup)
    cleanup_thread.daemon = True
    cleanup_thread.start()

with app.app_context():
    initialize()

#
# Form Classes
#

class SecurityForm(FlaskForm):
    """Base form with enhanced security features"""
    
    def __init__(self, *args, **kwargs):
        super(SecurityForm, self).__init__(*args, **kwargs)
        # Store form submission time for time-based attacks detection
        self.submission_time = time.time()
        
    def validate(self, extra_validators=None):
        """Enhanced validation with security checks"""
        # First run the standard validation
        if not super(SecurityForm, self).validate(extra_validators=extra_validators):
            return False
            
        # Add additional security checks
        
        # Check for too-fast form submissions (potential bot)
        elapsed = time.time() - self.submission_time
        if elapsed < 0.5:  # Less than 0.5 seconds
            logger.warning(f"Suspiciously fast form submission: {elapsed:.2f}s")
            self._errors = {"_form": ["Please resubmit your form."]}
            audit_log(
                action="FAST_SUBMISSION",
                details={'time': elapsed, 'ip': request.remote_addr}
            )
            return False
            
        return True

class AngleGenerationForm(SecurityForm):
    """Form for news angle generation with enhanced security validation"""
    topic = StringField('Nyhedsemne', validators=[
        DataRequired(message="Emnet skal udfyldes"),
        Length(min=3, max=100, message="Emnet skal være mellem 3 og 100 tegn"),
        # Add regex validation for safer input patterns
        Regexp(r'^[a-zA-Z0-9æøåÆØÅ\s.,;:!?\(\)\[\]{}\-\'\"]+$', 
               message="Emnet indeholder ugyldige tegn")
    ])
    profile = SelectField('Profil', validators=[
        DataRequired(message="Vælg venligst en profil"),
        validate_profile_path  # Custom validator for path safety
    ])
    submit = SubmitField('Generer vinkler')
    
    # Add CSRF token validation
    def validate_csrf_token(self, field):
        """Validate CSRF token with proper error handling"""
        if not field.data:
            audit_log(
                action="CSRF_MISSING",
                details={'form': 'AngleGenerationForm', 'ip': request.remote_addr}
            )
            raise ValidationError("CSRF beskyttelsestoken mangler")

    # Use our enhanced topic validator
    validate_topic = validate_topic

#
# Helper Functions
#

def get_available_profiles():
    """
    Get list of available profile files with path traversal protection.
    
    Returns:
        List of profile paths that are safely within the profile directory
    """
    # Get the configured profile directory 
    profile_dir = Path(app.config['PROFILE_DIR']).resolve()
    if not profile_dir.exists() or not profile_dir.is_dir():
        logger.warning(f"Profile directory {profile_dir} not found")
        return []
    
    # Find all YAML files
    yaml_files = list(profile_dir.glob("*.yaml")) + list(profile_dir.glob("*.yml"))
    
    # Make sure all paths are within the profile directory (protect against symlink attacks)
    safe_profiles = []
    for path in yaml_files:
        try:
            # Resolve to absolute path, following symlinks
            resolved_path = path.resolve()
            # Check that the resolved path is still within the profile directory
            if str(resolved_path).startswith(str(profile_dir)):
                safe_profiles.append(str(path))
            else:
                logger.warning(f"Skipping profile {path}: Path traversal attempt detected")
                # Audit log for security monitoring
                audit_log(
                    action="SECURITY_VIOLATION",
                    details={
                        'type': 'path_traversal', 
                        'path': str(path),
                        'resolved_path': str(resolved_path)
                    }
                )
        except (ValueError, RuntimeError) as e:
            logger.error(f"Error resolving path {path}: {e}")
    
    return safe_profiles

def get_profile_choices():
    """Get profile choices for the select field."""
    profiles = get_available_profiles()
    return [(p, Path(p).stem) for p in profiles]

def get_topic_info_sync(topic, detailed=False):
    """
    Synchronous version of fetch_topic_information.
    
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
        
        app.logger.warning(f"Perplexity API error: {response.status_code} - {response.text}")
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

async def process_generation_request_async(topic, profile):
    """
    Process a generation request using optimized asynchronous API calls.
    
    Args:
        topic: News topic to generate angles for
        profile: Editorial DNA profile to use
        
    Returns:
        List[Dict]: Generated angles
    """
    try:
        # Security: Validate input parameters first
        sanitize_input(topic, max_length=200)
        
        # Use the optimized implementation from ai_providers_optimized
        from ai_providers_optimized import process_generation_request_parallel
        
        # Define a progress tracker if needed
        progress = {"value": 0}
        
        async def progress_callback(value):
            progress["value"] = value
        
        # Call the optimized parallel implementation
        return await process_generation_request_parallel(
            topic=topic,
            profile=profile,
            progress_callback=progress_callback,
            use_fast_models=True,  # Use faster models for better performance
            timeout=45  # Reasonable timeout to prevent excessive waiting
        )
        
    except Exception as e:
        app.logger.error(f"Error in optimized angle generation: {e}")
        # Fallback to the original implementation if available
        try:
            # Import the original implementation
            from api_clients_wrapper import process_generation_request as original_process
            
            # Run the original in the event loop
            return await original_process(topic, profile)
        except Exception as fallback_error:
            app.logger.error(f"Even fallback angle generation failed: {fallback_error}")
            raise ValueError(f"Failed to generate angles: {e}")

def process_generation_request(topic, profile):
    """
    Synchronous wrapper for the asynchronous process_generation_request function.
    
    This maintains backward compatibility with the existing API while using
    the optimized async implementation internally.
    """
    # Create a new event loop if needed
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # No event loop exists in current thread, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    try:
        # Run the async function in the event loop
        return loop.run_until_complete(process_generation_request_async(topic, profile))
    except Exception as e:
        app.logger.error(f"Error in process_generation_request: {e}")
        # Re-raise for proper error handling upstream
        raise

# Original implementation kept for reference
def _original_process_generation_request(topic, profile):
    """
    Process a generation request with synchronous API calls.
    
    This is a synchronous version of the async process_generation_request 
    function from main.py, adapted for Flask.
    """
    try:
        # Security: Validate input parameters first
        try:
            sanitize_input(topic, max_length=200)
            
            # Check profile has required fields (defensive validation)
            required_profile_fields = ['kerneprincipper', 'nyhedsprioritering', 'fokusOmrader', 'tone_og_stil']
            for field in required_profile_fields:
                if not hasattr(profile, field):
                    raise ValueError(f"Profile is missing required field: {field}")
        except (InvalidInputException, ValueError) as e:
            logger.error(f"Input validation failed: {e}")
            audit_log(
                action="VALIDATION_ERROR",
                details={
                    'error': str(e), 
                    'topic': topic,
                    'ip': request.remote_addr
                }
            )
            raise
            
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
            profile_name = ''
            if hasattr(profile, 'navn'):
                profile_name = profile.navn
            
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
        
        # Validate the prompt for security
        from security import validate_api_prompt
        try:
            validate_api_prompt(prompt)
        except InvalidInputException as e:
            logger.error(f"API prompt validation failed: {e}")
            audit_log(
                action="PROMPT_VALIDATION_ERROR",
                details={
                    'error': str(e), 
                    'topic': topic,
                    'ip': request.remote_addr
                }
            )
            raise ValueError(f"API prompt er ikke gyldig: {e}")
        
        # Make the Claude API call with enhanced security
        api_url = "https://api.anthropic.com/v1/messages"
        
        # Securely construct headers with appropriate authentication
        headers = {
            "Content-Type": "application/json",
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01"
        }
        
        # Generate unique request ID for tracking
        request_id = str(uuid.uuid4())
        
        # Add request ID to headers for API call tracing
        headers["X-Request-ID"] = request_id
        
        # Log the API request (with key details masked)
        logger.info(f"Making API request to Anthropic (ID: {request_id}, IP: {request.remote_addr})")
        
        # Create payload with secure defaults
        payload = {
            "model": "claude-3-opus-20240229",
            "max_tokens": 2500,
            "temperature": 0.7,
            "system": "Du er en erfaren journalist med ekspertise i at udvikle kreative og relevante nyhedsvinkler.",
            "messages": [{"role": "user", "content": prompt}],
        }
        
        # Audit log of the API call (without the full prompt for security)
        audit_log(
            action="API_CALL",
            details={
                'provider': 'anthropic',
                'model': payload["model"],
                'request_id': request_id,
                'topic': topic,
                'ip': request.remote_addr
            }
        )
        
        # Make the API call with a timeout for security
        try:
            claude_response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=60  # 60 second timeout to prevent hanging
            )
        except requests.exceptions.Timeout:
            logger.error(f"API request to Anthropic timed out (ID: {request_id})")
            audit_log(
                action="API_TIMEOUT",
                details={
                    'provider': 'anthropic',
                    'request_id': request_id,
                    'ip': request.remote_addr
                }
            )
            raise ValueError("API forespørgsel udløb - prøv venligst igen senere")
        
        # Parse Claude response with enhanced error handling
        if claude_response.status_code != 200:
            # Log API error details
            error_message = f"Claude API error: {claude_response.status_code}"
            error_content = ""
            
            try:
                error_content = claude_response.text
                error_data = claude_response.json()
                if 'error' in error_data:
                    error_message = f"{error_message} - {error_data.get('error', {}).get('message', 'Unknown error')}"
            except Exception:
                error_message = f"{error_message} - Could not parse error response"
            
            # Audit log the error
            audit_log(
                action="API_ERROR",
                details={
                    'provider': 'anthropic',
                    'request_id': request_id,
                    'status_code': claude_response.status_code,
                    'error': error_content[:500],  # Limit error content length
                    'ip': request.remote_addr
                }
            )
            
            app.logger.error(f"{error_message} (ID: {request_id})")
            raise ValueError(f"API fejl: {error_message}")
            
        # Process successful response
        try:
            response_data = claude_response.json()
            response_text = response_data['content'][0]['text']
            
            # Log success
            logger.info(f"Received successful API response (ID: {request_id}, length: {len(response_text)})")
            
            # Parse the response into angles
            angles = parse_angles_from_response(response_text)
            
            # Additional validation on response structure
            if not angles or not isinstance(angles, list):
                raise ValueError("API svar havde ikke den forventede struktur")
                
        except (KeyError, ValueError, TypeError) as e:
            # Handle parsing errors
            logger.error(f"Error parsing API response: {e} (ID: {request_id})")
            audit_log(
                action="RESPONSE_PARSE_ERROR",
                details={
                    'provider': 'anthropic',
                    'request_id': request_id,
                    'error': str(e),
                    'ip': request.remote_addr
                }
            )
            raise ValueError(f"Kunne ikke fortolke API svar: {e}")
        
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
            except Exception as e:
                app.logger.error(f"Error generating source suggestions: {e}")
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

def generate_report_html(results):
    """
    Generate HTML for the full detailed report PDF.
    
    Args:
        results: Dictionary containing all necessary data from session
        
    Returns:
        HTML string for the report
    """
    try:
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

def generate_html_content(results):
    """Generate HTML content for PDF conversion."""
    topic = results.get('topic', 'Ukendt emne')
    profile_name = results.get('profile_name', 'Ukendt profil')
    angles = results.get('angles', [])
    timestamp = results.get('generated_at', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
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

def generate_text_content(results):
    """Generate plain text content for download."""
    topic = results.get('topic', 'Ukendt emne')
    profile_name = results.get('profile_name', 'Ukendt profil')
    angles = results.get('angles', [])
    
    # Tidsstempler
    generated_at = results.get('generated_at', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    last_updated = results.get('last_updated', generated_at)
    
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

def find_wkhtmltopdf_path():
    """Find path to wkhtmltopdf executable"""
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
    return wkhtmltopdf_path

def get_pdf_config():
    """Get PDF configuration options"""
    wkhtmltopdf_path = find_wkhtmltopdf_path()
    config = None
    if wkhtmltopdf_path:
        config = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)
    return config

def get_pdf_options():
    """Get standard PDF generation options"""
    return {
        'encoding': 'UTF-8',
        'page-size': 'A4',
        'margin-top': '20mm',
        'margin-right': '20mm',
        'margin-bottom': '20mm',
        'margin-left': '20mm',
        'footer-right': 'Side [page] af [topage]',
        'footer-font-size': '9',
        'enable-local-file-access': None,
        'quiet': None
    }

#
# Routes
#

@app.route('/', methods=['GET', 'POST'])
@enhanced_rate_limit(limit=20, window=60, block_after=10)  # 20 requests per minute with blocking
def index():
    """Home page with form for angle generation."""
    # Audit log for security monitoring
    audit_log(action="PAGE_VIEW", details={'page': 'index', 'ip': request.remote_addr})
    form = AngleGenerationForm()
    
    # Populate profile choices
    form.profile.choices = get_profile_choices()
    
    if form.validate_on_submit():
        # Store form data in session
        session['topic'] = form.topic.data
        session['profile'] = form.profile.data
        
        # Security audit logging for form submission
        audit_log(
            action="FORM_SUBMIT", 
            details={
                'form': 'AngleGenerationForm',
                'topic': form.topic.data,
                'profile': Path(form.profile.data).stem,
                'ip': request.remote_addr
            }
        )
        
        # Redirect to results page
        return redirect(url_for('generate'))
    
    return render_template('index.html', form=form)

@app.route('/generate')
@enhanced_rate_limit(limit=5, window=60, block_after=5)  # 5 requests per minute for generation
def generate():
    """Generate angles and show results."""
    # Audit log for security monitoring
    audit_log(action="PAGE_VIEW", details={'page': 'generate', 'ip': request.remote_addr})
    
    # Get data from session
    topic = session.get('topic')
    profile_path = session.get('profile')
    
    if not topic or not profile_path:
        flash('Manglende data. Udfyld formularen igen.', 'error')
        return redirect(url_for('index'))
    
    try:
        # Load the profile
        profile = load_and_validate_profile(Path(profile_path))
        
        # Process the request
        ranked_angles = process_generation_request(topic, profile)
        
        if not ranked_angles:
            flash('Ingen vinkler kunne genereres. Prøv et andet emne eller en anden profil.', 'error')
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
            'profile_path': profile_path,
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
        app.logger.error(f"Error during generation: {str(e)}")
        flash(f'Fejl under generering: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/detailed_report')
@enhanced_rate_limit(limit=10, window=60, block_after=8)
def detailed_report():
    """Show detailed report with comprehensive background information."""
    # Audit log for security monitoring
    audit_log(action="PAGE_VIEW", details={'page': 'detailed_report', 'ip': request.remote_addr})
    
    # Get stored results from session
    results = session.get('results', {})
    
    if not results:
        flash('Ingen resultater at vise. Generer vinkler først.', 'error')
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

@app.route('/regenerate_considerations', methods=['POST'])
@enhanced_rate_limit(limit=5, window=60, block_after=5)
def regenerate_considerations():
    """Regenerate editorial considerations."""
    # Get stored results from session
    results = session.get('results', {})
    
    if not results:
        flash('Ingen resultater at vise. Generer vinkler først.', 'error')
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
                
            flash('Redaktionelle overvejelser blev regenereret.', 'success')
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

@app.route('/regenerate_background', methods=['POST'])
@enhanced_rate_limit(limit=5, window=60, block_after=5)
def regenerate_background():
    """Regenerate background information."""
    # Get stored results from session
    results = session.get('results', {})
    
    if not results:
        flash('Ingen resultater at vise. Generer vinkler først.', 'error')
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
            
        flash('Baggrundsinformation blev opdateret.', 'success')
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
@enhanced_rate_limit(limit=5, window=300, block_after=5)  # 5 requests per 5 minutes
def download_report():
    """Download the full detailed report as PDF."""
    # Audit log for security monitoring
    audit_log(action="DOWNLOAD", details={'type': 'detailed_report_pdf', 'ip': request.remote_addr})
    
    results = session.get('results', {})
    
    if not results:
        flash('Ingen resultater at downloade. Generer vinkler først.', 'error')
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
        
        # Configure pdfkit
        config = get_pdf_config()
        options = get_pdf_options()
        options['title'] = f'Vinkeljernet Rapport - {topic}'
        
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
            max_age=0,
            conditional=True
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
@enhanced_rate_limit(limit=10, window=300, block_after=6)  # 10 requests per 5 minutes
def download(format):
    """Download results in different formats."""
    # Audit log for security monitoring
    audit_log(action="DOWNLOAD", details={'type': format, 'ip': request.remote_addr})
    
    results = session.get('results', {})
    
    if not results:
        flash('Ingen resultater at downloade. Generer vinkler først.', 'error')
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
            # Configure pdfkit
            config = get_pdf_config()
            options = get_pdf_options()
            options['title'] = f'Vinkeljernet - {topic}'
            
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
                max_age=0,
                conditional=True
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
    
    # Invalid format
    flash(f'Ukendt format: {format}', 'error')
    return redirect(url_for('index'))

@app.route('/clear_pdf_error', methods=['POST'])
def clear_pdf_error():
    """Clear PDF error from session."""
    if 'results' in session and 'pdf_error' in session['results']:
        session['results']['pdf_error'] = ''
        session.modified = True
    return '', 204  # No content response

@app.route('/api/health')
def health_check():
    """API endpoint for health check."""
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "service": "vinkeljernet"
    })

#
# Error Handlers
#

@app.errorhandler(404)
def page_not_found(e):
    request_id = getattr(g, 'request_id', 'unknown')
    
    # Log the error
    logger.warning(f"404 Not Found: {request.path} (Request ID: {request_id})")
    
    # Audit log for security monitoring
    audit_log(
        action="ERROR",
        details={
            'error_code': 404, 
            'path': request.path,
            'ip': request.remote_addr,
            'request_id': request_id
        }
    )
    
    return render_template('error.html', 
                          error_code=404,
                          error_message="Siden blev ikke fundet",
                          request_id=request_id), 404

@app.errorhandler(403)
def forbidden(e):
    request_id = getattr(g, 'request_id', 'unknown')
    
    # Log the error
    logger.warning(f"403 Forbidden: {request.path} (Request ID: {request_id})")
    
    # Audit log for security monitoring
    audit_log(
        action="SECURITY_ERROR",
        details={
            'error_code': 403, 
            'path': request.path,
            'ip': request.remote_addr,
            'request_id': request_id
        }
    )
    
    return render_template('error.html',
                          error_code=403,
                          error_message="Adgang nægtet",
                          request_id=request_id), 403

@app.errorhandler(401)
def unauthorized(e):
    request_id = getattr(g, 'request_id', 'unknown')
    
    # Log the error
    logger.warning(f"401 Unauthorized: {request.path} (Request ID: {request_id})")
    
    # Audit log for security monitoring
    audit_log(
        action="SECURITY_ERROR",
        details={
            'error_code': 401, 
            'path': request.path,
            'ip': request.remote_addr,
            'request_id': request_id
        }
    )
    
    return render_template('error.html',
                          error_code=401,
                          error_message="Godkendelse påkrævet",
                          request_id=request_id), 401

@app.errorhandler(500)
def server_error(e):
    request_id = getattr(g, 'request_id', 'unknown')
    
    # Log the detailed error
    logger.error(f"500 Server Error: {str(e)} (Request ID: {request_id})")
    
    # Audit log for security monitoring
    audit_log(
        action="SERVER_ERROR",
        details={
            'error_code': 500, 
            'path': request.path,
            'ip': request.remote_addr,
            'error': str(e),
            'request_id': request_id
        }
    )
    
    # Only include detailed error information in development mode
    error_details = str(e) if DEBUG else None
    
    return render_template('error.html',
                          error_code=500,
                          error_message="Der opstod en fejl på serveren",
                          error_details=error_details,
                          request_id=request_id), 500

@app.errorhandler(429)
def too_many_requests(e):
    request_id = getattr(g, 'request_id', 'unknown')
    
    # Log the error
    logger.warning(f"429 Too Many Requests: {request.path} (Request ID: {request_id})")
    
    # Audit log for security monitoring
    audit_log(
        action="RATE_LIMIT_EXCEEDED",
        details={
            'error_code': 429, 
            'path': request.path,
            'ip': request.remote_addr,
            'request_id': request_id
        }
    )
    
    return render_template('error.html',
                          error_code=429,
                          error_message="For mange forespørgsler. Prøv igen senere.",
                          request_id=request_id), 429
                          
@app.errorhandler(400)
def bad_request(e):
    request_id = getattr(g, 'request_id', 'unknown')
    
    # Log the error
    logger.warning(f"400 Bad Request: {request.path} (Request ID: {request_id})")
    
    # Audit log for security monitoring
    audit_log(
        action="BAD_REQUEST",
        details={
            'error_code': 400, 
            'path': request.path,
            'ip': request.remote_addr,
            'request_id': request_id
        }
    )
    
    return render_template('error.html',
                          error_code=400,
                          error_message="Ugyldig forespørgsel",
                          request_id=request_id), 400

@app.errorhandler(Exception)
def handle_exception(e):
    # Pass through HTTP errors (they'll be handled by the specific error handlers)
    if isinstance(e, HTTPException):
        return e
        
    request_id = getattr(g, 'request_id', 'unknown')
    
    # Log the detailed error
    logger.exception(f"Unhandled Exception: {str(e)} (Request ID: {request_id})")
    
    # Audit log for security monitoring
    audit_log(
        action="UNHANDLED_EXCEPTION",
        details={
            'error_type': e.__class__.__name__, 
            'path': request.path,
            'ip': request.remote_addr,
            'error': str(e),
            'request_id': request_id
        }
    )
    
    # Only include detailed error information in development mode
    error_details = str(e) if DEBUG else None
    
    return render_template('error.html',
                          error_code=500,
                          error_message="Der opstod en uventet fejl på serveren",
                          error_details=error_details,
                          request_id=request_id), 500

#
# Template Filters
#

@app.template_filter('nl2br')
def nl2br(value):
    """Convert newlines to <br> tags for proper display in HTML."""
    if not value:
        return ""
    
    # Replace newlines with <br> tags
    value = value.replace('\n', '<br>')
    value = value.replace('\r\n', '<br>')
    return value

#
# Main Entry Point
#

if __name__ == '__main__':
    # Create templates folder if it doesn't exist
    templates_dir = Path('templates')
    templates_dir.mkdir(exist_ok=True)
    
    # Create error template file if it doesn't exist
    error_template = templates_dir / 'error.html'
    if not error_template.exists():
        with open(error_template, 'w') as f:
            f.write("""
{% extends 'base.html' %}

{% block title %}Fejl {{ error_code }} | Vinkeljernet{% endblock %}

{% block content %}
<div class="container mt-5">
    <div class="row justify-content-center">
        <div class="col-md-8">
            <div class="card border-danger">
                <div class="card-header bg-danger text-white">
                    <h2 class="mb-0">Fejl {{ error_code }}</h2>
                </div>
                <div class="card-body">
                    <p class="lead mb-4">{{ error_message }}</p>
                    
                    <div class="text-center mt-4">
                        <a href="{{ url_for('index') }}" class="btn btn-outline-primary">
                            <i class="bi bi-house-door"></i> Gå til forsiden
                        </a>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}
""")
    
    # Run the Flask application
    app.run(debug=True, host='0.0.0.0', port=5001)