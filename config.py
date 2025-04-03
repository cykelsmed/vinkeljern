"""
Configuration module for Vinkeljernet project.

This module loads API keys from a .env file and makes them available
for use throughout the application. It performs validation to ensure
required keys are present in the environment.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
PERPLEXITY_API_KEY: str = os.getenv("PERPLEXITY_API_KEY")
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY")

# Validate that required API keys are present
if not ANTHROPIC_API_KEY:
    error_msg = "ANTHROPIC_API_KEY not found in environment variables. Please add it to your .env file."
    print(error_msg)
    raise ValueError(error_msg)

if not PERPLEXITY_API_KEY:
    error_msg = "PERPLEXITY_API_KEY not found in environment variables. Please add it to your .env file."
    print(error_msg)
    raise ValueError(error_msg)