"""
Configuration module for Vinkeljernet project.

This module loads API keys from a .env file and makes them available
for use throughout the application. It performs validation to ensure
required keys are present in the environment.
"""

import os
import sys
import logging
from typing import Dict, Any
from dotenv import load_dotenv
from json_parser import robust_json_parse, parse_angles_from_response

# Configure logging
logger = logging.getLogger("vinkeljernet.config")

# Load environment variables from .env file
load_dotenv()

# Application environment
APP_ENV = os.getenv("FLASK_ENV", "development")
DEBUG = APP_ENV == "development" or os.getenv("DEBUG", "0").lower() in ("1", "true", "yes")

# Security settings
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY and APP_ENV == "production":
    logger.critical("SECRET_KEY not set in production environment!")
    if not DEBUG:
        raise ValueError("SECRET_KEY must be set in production")
elif not SECRET_KEY:
    # Generate a random key for development
    import secrets
    SECRET_KEY = secrets.token_hex(32)
    logger.warning("Using randomly generated SECRET_KEY for development")

# Secure cookie settings
COOKIE_SECURE = APP_ENV == "production"
COOKIE_HTTPONLY = True
COOKIE_SAMESITE = 'Lax'
SESSION_LIFETIME = int(os.getenv("SESSION_LIFETIME", "1800"))  # 30 minutes in seconds

# Rate limiting configuration
RATE_LIMIT_DEFAULT = int(os.getenv("RATE_LIMIT_DEFAULT", "30"))  # requests per minute
RATE_LIMIT_API = int(os.getenv("RATE_LIMIT_API", "10"))  # requests per minute

# Content size limits
MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", "5242880"))  # 5MB

# API Keys (using secure loading method)
def load_api_keys() -> Dict[str, str]:
    """
    Securely load API keys with validation.
    
    Returns:
        Dict of API key names and values
    
    Raises:
        ValueError if required keys are missing in production
    """
    # Get API keys from environment variables
    keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "PERPLEXITY_API_KEY": os.getenv("PERPLEXITY_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
    }
    
    # Validate required keys for non-debug environments
    if not DEBUG:
        for key_name, key_value in keys.items():
            if not key_value:
                error_msg = f"{key_name} not found in environment variables. Add it to your .env file."
                logger.critical(error_msg)
                raise ValueError(error_msg)
                
    # Log loaded keys (kun generel besked eller sidste 4 tegn)
    for key_name, key_value in keys.items():
        if key_value:
            logger.info(f"{key_name} API Key loaded successfully")
        else:
            logger.warning(f"{key_name} not found")
            
    return keys

# Load API keys
api_keys = load_api_keys()

# Extract individual keys for backwards compatibility
OPENAI_API_KEY = api_keys.get("OPENAI_API_KEY")
PERPLEXITY_API_KEY = api_keys.get("PERPLEXITY_API_KEY")
ANTHROPIC_API_KEY = api_keys.get("ANTHROPIC_API_KEY")

# API Timeout configurations (in seconds)
PERPLEXITY_TIMEOUT = int(os.getenv("PERPLEXITY_TIMEOUT", "60"))
ANTHROPIC_TIMEOUT = int(os.getenv("ANTHROPIC_TIMEOUT", "90"))
OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", "60"))

# Performance optimization settings
USE_STREAMING = os.getenv("USE_STREAMING", "False").lower() in ("true", "1", "yes")
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "5"))
CACHE_SIZE_MB = int(os.getenv("CACHE_SIZE_MB", "500"))

# Caching configuration
DEFAULT_CACHE_TTL = int(os.getenv("DEFAULT_CACHE_TTL", "3600"))  # Default 1 hour
MEMORY_CACHE_SIZE = int(os.getenv("MEMORY_CACHE_SIZE", "100"))  # Items in memory

# File storage locations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILE_DIR = os.path.join(BASE_DIR, "config")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Create log directory if it doesn't exist
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR, exist_ok=True)

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.path.join(LOG_DIR, "vinkeljernet.log")

# Application URL (for use in absolute URLs)
APP_URL = os.getenv("APP_URL", "http://localhost:5000")

# CORS settings
CORS_ENABLED = os.getenv("CORS_ENABLED", "False").lower() in ("true", "1", "yes")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
