"""
Security module for Vinkeljernet.

This module provides security utilities for the Vinkeljernet application,
including secure API key handling, input sanitization, and protection
against common vulnerabilities.
"""

import os
import re
import logging
import secrets
import hashlib
import hmac
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from functools import wraps
from urllib.parse import urlparse

from flask import request, session, abort, current_app, g, Response

# Configure logging
logger = logging.getLogger("vinkeljernet.security")

# Regular expressions for input validation
SAFE_STRING_REGEX = re.compile(r'^[a-zA-Z0-9æøåÆØÅ\s.,;:!?()[\]{}@#%&*_+\-=|~^$"\']+$')
URL_REGEX = re.compile(r'^https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+')
EMAIL_REGEX = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

# Lists of allowed values for certain fields
ALLOWED_NEWS_CRITERIA = {
    'aktualitet', 'væsentlighed', 'identifikation', 'sensation', 'konflikt', 
    'tidsaktualitet', 'relevans', 'nærhed', 'usædvanlighed', 'udvikling', 
    'konsekvens', 'nyhedsværdi'
}

# Security constants
TOKEN_BYTES = 32  # 256 bits
SESSION_DURATION = 30 * 60  # 30 minutes in seconds
RATE_LIMIT_WINDOW = 60  # 1 minute
RATE_LIMIT_MAX_REQUESTS = 10
MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5MB

# IP address tracking for rate limiting
ip_tracking = {}

class IPBlockedException(Exception):
    """Exception raised when an IP is blocked for too many failed attempts."""
    pass

class InvalidInputException(Exception):
    """Exception raised when input validation fails."""
    pass

def sanitize_input(input_str: str, max_length: int = 100) -> str:
    """
    Sanitize user input by removing potentially dangerous characters.
    
    Args:
        input_str: The string to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized string
        
    Raises:
        InvalidInputException: If the input contains unsafe characters or is too long
    """
    if not input_str:
        return ""
        
    # Check length
    if len(input_str) > max_length:
        raise InvalidInputException(f"Input too long (max {max_length} characters)")
    
    # Remove potentially dangerous characters
    if not SAFE_STRING_REGEX.match(input_str):
        unsafe_chars = set(input_str) - set(SAFE_STRING_REGEX.pattern[1:-1])
        raise InvalidInputException(f"Input contains unsafe characters: {', '.join(unsafe_chars)}")
        
    return input_str

def validate_url(url: str) -> str:
    """
    Validate a URL to ensure it's safe.
    
    Args:
        url: The URL to validate
        
    Returns:
        The validated URL
        
    Raises:
        InvalidInputException: If the URL is invalid or uses a non-HTTP/HTTPS scheme
    """
    if not url:
        raise InvalidInputException("URL cannot be empty")
        
    # Check if URL matches pattern
    if not URL_REGEX.match(url):
        raise InvalidInputException("Invalid URL format")
    
    # Ensure only HTTP/HTTPS protocols
    parsed = urlparse(url)
    if parsed.scheme not in ('http', 'https'):
        raise InvalidInputException("URL scheme must be HTTP or HTTPS")
        
    return url

def validate_news_criteria(criteria: List[str]) -> List[str]:
    """
    Validate a list of news criteria.
    
    Args:
        criteria: List of news criteria to validate
        
    Returns:
        The validated list of news criteria
        
    Raises:
        InvalidInputException: If any criterion is invalid
    """
    if not criteria:
        raise InvalidInputException("Must provide at least one news criterion")
        
    # Normalize criteria to lowercase for comparison
    normalized = [c.lower() for c in criteria]
    
    # Check for invalid criteria
    invalid = set(normalized) - ALLOWED_NEWS_CRITERIA
    if invalid:
        raise InvalidInputException(f"Invalid news criteria: {', '.join(invalid)}")
        
    return criteria
    
def validate_api_prompt(prompt: str, max_length: int = 5000) -> str:
    """
    Validate and sanitize an API prompt to prevent prompt injection.
    
    Args:
        prompt: The prompt to validate
        max_length: Maximum allowed length
        
    Returns:
        The validated prompt
        
    Raises:
        InvalidInputException: If the prompt is invalid or too long
    """
    if not prompt:
        raise InvalidInputException("Prompt cannot be empty")
        
    # Check length
    if len(prompt) > max_length:
        raise InvalidInputException(f"Prompt too long (max {max_length} characters)")
    
    # Block specific harmful patterns
    dangerous_patterns = [
        # Control characters that could confuse LLMs
        '\u0000', '\u001F', '\u007F',
        # Patterns that might trigger prompt injection
        "ignore previous instructions",
        "disregard previous instructions",
        "forget your instructions"
    ]
    
    for pattern in dangerous_patterns:
        if pattern.lower() in prompt.lower():
            raise InvalidInputException(f"Prompt contains disallowed pattern: {pattern}")
            
    return prompt

def secure_filename(filename: str) -> str:
    """
    Ensure a filename is secure by removing potentially dangerous characters.
    
    Args:
        filename: The filename to sanitize
        
    Returns:
        A secure version of the filename
    """
    # Extract extension
    parts = filename.rsplit('.', 1)
    if len(parts) > 1:
        base, ext = parts
        ext = '.' + ext
    else:
        base = parts[0]
        ext = ''
    
    # Replace any unsafe characters with underscores
    base = re.sub(r'[^a-zA-Z0-9æøåÆØÅ_-]', '_', base)
    
    # Ensure the extension is safe
    ext = re.sub(r'[^a-zA-Z0-9.]', '', ext)
    
    # Limit length
    if len(base) > 100:
        base = base[:100]
        
    return base + ext

def generate_secure_token() -> str:
    """
    Generate a cryptographically secure random token.
    
    Returns:
        A secure token as a hex string
    """
    return secrets.token_hex(TOKEN_BYTES)

def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """
    Hash a password using a secure method (Argon2id if available, else PBKDF2).
    
    Args:
        password: The password to hash
        salt: Optional salt, generated if not provided
        
    Returns:
        Tuple of (hashed_password, salt)
    """
    if salt is None:
        salt = secrets.token_hex(16)
        
    try:
        # Try to use Argon2id (needs argon2-cffi package)
        import argon2
        ph = argon2.PasswordHasher()
        hashed = ph.hash(password + salt)
        return hashed, salt
    except ImportError:
        # Fall back to PBKDF2
        dk = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 150000)
        return dk.hex(), salt

def verify_password(password: str, hashed_password: str, salt: str) -> bool:
    """
    Verify a password against its hashed value.
    
    Args:
        password: The password to verify
        hashed_password: The stored hash
        salt: The salt used for hashing
        
    Returns:
        True if the password matches, False otherwise
    """
    try:
        # Try to use Argon2id
        import argon2
        ph = argon2.PasswordHasher()
        try:
            ph.verify(hashed_password, password + salt)
            return True
        except argon2.exceptions.VerifyMismatchError:
            return False
    except ImportError:
        # Fall back to PBKDF2
        dk = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 150000)
        return hmac.compare_digest(dk.hex(), hashed_password)

def secure_api_keys() -> Dict[str, str]:
    """
    Securely load API keys from environment variables or a dedicated secure storage.
    
    Returns:
        Dictionary of API keys with their names as keys
    """
    keys = {}
    
    # Load from environment variables
    for key_name in ['OPENAI_API_KEY', 'PERPLEXITY_API_KEY', 'ANTHROPIC_API_KEY']:
        key_value = os.environ.get(key_name)
        if key_value:
            # In a real application, you might want to encrypt these in memory
            keys[key_name] = key_value
            
    # If no keys were found, log a warning
    if not keys:
        logger.warning("No API keys found in environment variables")
        
    return keys

def mask_api_key(key: str) -> str:
    """
    Mask an API key for display or logging purposes.
    
    Args:
        key: The API key to mask
        
    Returns:
        A masked version of the key, showing only first 4 and last 4 characters
    """
    if not key or len(key) < 12:
        return "[INVALID KEY]"
        
    return key[:4] + '*' * (len(key) - 8) + key[-4:]

def rate_limit(max_requests: int = RATE_LIMIT_MAX_REQUESTS, 
               window: int = RATE_LIMIT_WINDOW,
               block_threshold: int = 50):
    """
    Flask decorator to implement rate limiting.
    
    Args:
        max_requests: Maximum number of requests allowed in the window
        window: Time window in seconds
        block_threshold: Number of consecutive limit violations before blocking
        
    Returns:
        Decorator function
    """
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            current_time = time.time()
            client_ip = request.remote_addr
            
            # Initialize tracking data for this IP if it doesn't exist
            if client_ip not in ip_tracking:
                ip_tracking[client_ip] = {
                    'requests': [],
                    'violations': 0,
                    'blocked_until': 0
                }
                
            # Check if IP is currently blocked
            if ip_tracking[client_ip]['blocked_until'] > current_time:
                logger.warning(f"Blocked request from {client_ip}")
                abort(403, "IP address temporarily blocked due to rate limit violations")
                
            # Clean up old requests
            ip_tracking[client_ip]['requests'] = [
                timestamp for timestamp in ip_tracking[client_ip]['requests']
                if current_time - timestamp < window
            ]
            
            # Count requests in the current window
            request_count = len(ip_tracking[client_ip]['requests'])
            
            # If over the limit, increment violations and possibly block
            if request_count >= max_requests:
                ip_tracking[client_ip]['violations'] += 1
                
                # Calculate exponential backoff for repeated violations
                if ip_tracking[client_ip]['violations'] >= block_threshold:
                    block_time = min(2 ** (ip_tracking[client_ip]['violations'] - block_threshold + 2), 86400)  # Max 1 day
                    ip_tracking[client_ip]['blocked_until'] = current_time + block_time
                    logger.warning(f"IP {client_ip} blocked for {block_time} seconds")
                    abort(403, "IP address temporarily blocked due to rate limit violations")
                
                headers = {
                    'Retry-After': str(window),
                    'X-RateLimit-Limit': str(max_requests),
                    'X-RateLimit-Remaining': '0',
                    'X-RateLimit-Reset': str(int(current_time + window))
                }
                
                response = current_app.response_class(
                    response=f"Rate limit exceeded. Try again in {window} seconds.",
                    status=429,
                    headers=headers
                )
                return response
                
            # Otherwise, record this request and reset violations if any
            ip_tracking[client_ip]['requests'].append(current_time)
            if request_count == 0:  # Reset violations when under the limit
                ip_tracking[client_ip]['violations'] = 0
                
            # Add rate limit headers to the response
            def add_headers(response):
                response.headers['X-RateLimit-Limit'] = str(max_requests)
                response.headers['X-RateLimit-Remaining'] = str(max_requests - len(ip_tracking[client_ip]['requests']))
                response.headers['X-RateLimit-Reset'] = str(int(current_time + window))
                return response
                
            # Save hook to add headers after the view is executed
            g.add_rate_limit_headers = add_headers
            
            return f(*args, **kwargs)
        return wrapped
    return decorator

def setup_security_middleware(app):
    """
    Set up security middleware for a Flask application.
    
    Args:
        app: The Flask application
    """
    # Set secure headers for all responses
    @app.after_request
    def add_security_headers(response):
        # Add security headers
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'SAMEORIGIN'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' https://cdn.jsdelivr.net; style-src 'self' https://cdn.jsdelivr.net; img-src 'self' data:; font-src 'self' https://cdn.jsdelivr.net; connect-src 'self'"
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        response.headers['Permissions-Policy'] = 'camera=(), microphone=(), geolocation=()'
        
        # Add rate limit headers if available
        if hasattr(g, 'add_rate_limit_headers'):
            response = g.add_rate_limit_headers(response)
            
        return response
        
    # Set maximum content length for request data
    app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

def audit_log(action: str, user: str = "anonymous", success: bool = True, 
             details: Optional[Dict[str, Any]] = None) -> None:
    """
    Log an audit event.
    
    Args:
        action: The action being performed
        user: The user performing the action
        success: Whether the action was successful
        details: Additional details about the action
    """
    if details is None:
        details = {}
    
    # Include client IP if available
    try:
        if request:
            details['ip_address'] = request.remote_addr
            details['user_agent'] = request.user_agent.string
    except:
        pass
    
    logger.info(
        f"AUDIT: {action} | User: {user} | Success: {success} | " +
        f"Details: {details}"
    )

def validate_yaml_content(content: str, max_size: int = 100 * 1024) -> bool:
    """
    Validate YAML content to ensure it's safe.
    
    Args:
        content: The YAML content to validate
        max_size: Maximum size in bytes
        
    Returns:
        True if the content is valid, False otherwise
    """
    import yaml
    
    # Check size
    if len(content) > max_size:
        logger.warning(f"YAML content too large: {len(content)} bytes")
        return False
        
    # Check for dangerous patterns
    dangerous_patterns = [
        r'!!python/object',  # Python object instantiation
        r'!!python/name',    # Python name reference
        r'!![a-z]+/[a-z]+',  # Any tag directive
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, content):
            logger.warning(f"Dangerous pattern {pattern} found in YAML content")
            return False
    
    # Attempt to parse the YAML safely
    try:
        yaml.safe_load(content)
        return True
    except yaml.YAMLError as e:
        logger.warning(f"Error parsing YAML: {e}")
        return False

def is_authenticated() -> bool:
    """
    Check if the current request is authenticated.
    
    Returns:
        True if authenticated, False otherwise
    """
    # For now, just check if a user is in the session
    return 'user' in session and session['user'] is not None

def require_auth(f):
    """
    Decorator to require authentication for a route.
    
    Args:
        f: The route function to decorate
        
    Returns:
        Decorated function that will abort with 401 if not authenticated
    """
    @wraps(f)
    def wrapped(*args, **kwargs):
        if not is_authenticated():
            return abort(401, "Authentication required")
        return f(*args, **kwargs)
    return wrapped

def init_security_checks():
    """
    Initialize security checks for the application.
    
    This should be called during application startup to verify that
    the environment is properly secured.
    """
    # Check if we're running in a debug environment
    if os.environ.get("FLASK_ENV") == "development" or os.environ.get("DEBUG") == "1":
        logger.warning("Running in development/debug mode - security features may be limited")
        
    # Check if secure API keys are available
    keys = secure_api_keys()
    if not keys:
        logger.critical("No API keys available - application functionality will be limited")
        
    # Check for common security misconfigurations
    if not os.environ.get("SECRET_KEY"):
        logger.critical("SECRET_KEY not set in environment - sessions will not be secure")
        
    # Check for HTTPS capability
    if not os.environ.get("HTTPS") and os.environ.get("FLASK_ENV") == "production":
        logger.warning("HTTPS not explicitly enabled in production environment")
        
    # Log startup security information
    logger.info("Security module initialized")