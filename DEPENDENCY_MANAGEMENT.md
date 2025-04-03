# Dependency Management Guide

This document outlines how dependencies are managed in the Vinkeljernet project and provides instructions for installation and development.

## Overview

Vinkeljernet uses a structured approach to dependency management:

1. **requirements.txt** - Contains all runtime dependencies with pinned versions
2. **requirements-dev.txt** - Contains additional development dependencies
3. **setup.py** - Provides packaging configuration for installation
4. **pyproject.toml** - Contains build system requirements and tool configurations

## Installation Options

### Basic Installation

For users who just want to use Vinkeljernet:

```bash
# Install from local directory
pip install .

# Or install in development mode
pip install -e .
```

### Development Installation

For developers who want to contribute to Vinkeljernet:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Or install everything from requirements-dev.txt
pip install -r requirements-dev.txt
```

### Documentation Building

```bash
pip install -e ".[docs]"
```

## Virtual Environment Setup

It's recommended to use a virtual environment:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt  # For basic usage
# OR
pip install -r requirements-dev.txt  # For development
```

## Dependency Structure

### Core Dependencies

- **langchain** - AI orchestration framework
- **openai** and **anthropic** - LLM API clients
- **pydantic** - Data validation
- **PyYAML** - Configuration file parsing
- **requests** and **aiohttp** - HTTP clients
- **python-dotenv** - Environment variable management

### Web Interface

- **Flask** - Web framework
- **Flask-WTF** - Form handling
- **pdfkit** - PDF generation

### CLI and Formatting

- **rich** - Terminal formatting
- **prompt_toolkit** - Interactive CLI

### Testing and Development

- **pytest** - Testing framework
- **black**, **flake8**, **isort** - Code formatting and linting
- **mypy** - Type checking

## Adding New Dependencies

When adding new dependencies:

1. Add them to `requirements.txt` with specific version pins
2. If they're development-only dependencies, add them to `requirements-dev.txt`
3. Update `setup.py` as needed
4. Run `pip install -e .` to update your local installation

## Version Management

All dependencies should have pinned versions to ensure reproducible builds. When updating dependencies:

1. Test thoroughly to ensure compatibility
2. Update version numbers in the appropriate requirements file
3. Document significant dependency changes in commit messages

## Notes on External Dependencies

### PDF Generation

For PDF generation functionality, `wkhtmltopdf` must be installed on the system:

- **macOS**: `brew install wkhtmltopdf`
- **Ubuntu/Debian**: `sudo apt-get install wkhtmltopdf`
- **Windows**: Download from https://wkhtmltopdf.org/downloads.html

See `README_WKHTMLTOPDF.txt` for more details.