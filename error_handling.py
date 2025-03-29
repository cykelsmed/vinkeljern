"""
Error handling module for Vinkeljernet project.

This module provides decorators and utilities for handling errors gracefully,
providing user-friendly messages, and supporting debugging.
"""

import sys
import traceback
import functools
import inspect
import logging
from typing import Callable, Any, TypeVar, Optional, Dict, List, Union
from rich import print as rprint
from rich.panel import Panel
from rich.console import Console
from rich.traceback import Traceback

# Type variable for generic function decorator
F = TypeVar('F', bound=Callable[..., Any])

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("vinkeljernet.log"), logging.StreamHandler()]
)
logger = logging.getLogger("vinkeljernet")

# Console for fancy output
console = Console()

class APIKeyMissingError(Exception):
    """Raised when a required API key is missing."""
    pass

class SSLVerificationError(Exception):
    """Raised when there's an issue with SSL verification."""
    pass

class APIConnectionError(Exception):
    """Raised when connection to an API fails."""
    pass

class APIResponseError(Exception):
    """Raised when an API returns an error response."""
    def __init__(self, message: str, status_code: Optional[int] = None, response_body: Optional[str] = None):
        self.status_code = status_code
        self.response_body = response_body
        super().__init__(message)

class DataParsingError(Exception):
    """Raised when parsing data (like JSON) fails."""
    pass

def safe_execute(fallback_return: Any = None, log_level: int = logging.ERROR) -> Callable[[F], F]:
    """
    Decorator that catches exceptions and returns fallback value if something fails.
    
    Args:
        fallback_return: Value to return if function fails
        log_level: Logging level for errors
        
    Returns:
        Decorated function that handles errors gracefully
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log the error
                func_name = getattr(func, "__name__", "unknown")
                logger.log(log_level, f"Error in {func_name}: {str(e)}", exc_info=True)
                
                # Return fallback
                return fallback_return
        return wrapper  # type: ignore
    return decorator

def safe_execute_async(fallback_return: Any = None, log_level: int = logging.ERROR) -> Callable[[F], F]:
    """
    Decorator that catches exceptions in async functions and returns fallback value if something fails.
    
    Args:
        fallback_return: Value to return if function fails
        log_level: Logging level for errors
        
    Returns:
        Decorated async function that handles errors gracefully
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Log the error
                func_name = getattr(func, "__name__", "unknown")
                logger.log(log_level, f"Error in async {func_name}: {str(e)}", exc_info=True)
                
                # Return fallback
                return fallback_return
        return wrapper  # type: ignore
    return decorator

def user_friendly_error(
    show_traceback: bool = False,
    exit_on_error: bool = False,
    error_code: int = 1
) -> Callable[[F], F]:
    """
    Decorator that catches exceptions and displays user-friendly error messages.
    
    Args:
        show_traceback: Whether to display a detailed traceback
        exit_on_error: Whether to exit the program on error
        error_code: Exit code to use if exiting
        
    Returns:
        Decorated function that displays user-friendly error messages
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except APIKeyMissingError as e:
                display_api_key_error(str(e))
            except SSLVerificationError as e:
                display_ssl_error(str(e))
            except APIConnectionError as e:
                display_connection_error(str(e))
            except APIResponseError as e:
                display_api_response_error(str(e), e.status_code, e.response_body)
            except DataParsingError as e:
                display_parsing_error(str(e))
            except Exception as e:
                display_generic_error(str(e), type(e).__name__)
            
            # Show traceback if requested
            if show_traceback:
                console.print(Traceback())
            
            # Exit if requested
            if exit_on_error:
                sys.exit(error_code)
            
            # Return None if we didn't exit
            return None
        return wrapper  # type: ignore
    return decorator

def display_api_key_error(message: str) -> None:
    """Display an error about missing API keys with helpful instructions."""
    rprint(Panel.fit(
        f"[bold red]API Nøgle Fejl:[/bold red] {message}\n\n"
        "[yellow]Tips til løsning:[/yellow]\n"
        "1. Tjek at du har oprettet en .env fil i projektmappen\n"
        "2. Sørg for at .env filen indeholder de nødvendige API nøgler\n"
        "3. Kontroller formattet: OPENAI_API_KEY=din_nøgle_her\n"
        "4. Brug 'python -m dotenv check' til at verificere din .env fil",
        title="API Nøgle Mangler",
        border_style="red"
    ))

def display_ssl_error(message: str) -> None:
    """Display an error about SSL verification with helpful instructions."""
    rprint(Panel.fit(
        f"[bold red]SSL Verifikation Fejl:[/bold red] {message}\n\n"
        "[yellow]Tips til løsning:[/yellow]\n"
        "1. Sørg for at dine CA-certifikater er opdaterede\n"
        "2. Hvis du er på et virksomhedsnetværk, kan der være en firewall i vejen\n"
        "3. Prøv at køre programmet med --dev-mode flaget (kun i udviklingsmiljø!)\n"
        "   Eksempel: python main.py --emne \"Dit emne\" --profil \"din_profil.yaml\" --dev-mode",
        title="SSL Certifikat Problem",
        border_style="red"
    ))

def display_connection_error(message: str) -> None:
    """Display an error about connection issues with helpful instructions."""
    rprint(Panel.fit(
        f"[bold red]Forbindelsesfejl:[/bold red] {message}\n\n"
        "[yellow]Tips til løsning:[/yellow]\n"
        "1. Tjek din internetforbindelse\n"
        "2. Kontroller at API'et ikke er nede (se statussider)\n"
        "3. Hvis du bruger VPN, prøv at slå det fra\n"
        "4. Kontroller at du ikke har nået API'ets rate limits",
        title="Kan ikke forbinde til API",
        border_style="red"
    ))

def display_api_response_error(message: str, status_code: Optional[int] = None, response_body: Optional[str] = None) -> None:
    """Display an error about API response issues with helpful instructions."""
    status_info = f"Status kode: {status_code}" if status_code else ""
    
    detail_text = ""
    if response_body and len(response_body) < 500:  # Only show if not too large
        detail_text = f"\n\n[dim]API Svar:[/dim]\n[dim]{response_body}[/dim]"
    
    rprint(Panel.fit(
        f"[bold red]API Svar Fejl:[/bold red] {message} {status_info}\n\n"
        "[yellow]Tips til løsning:[/yellow]\n"
        "1. Kontroller at din API nøgle har de rette tilladelser\n"
        "2. Tjek at du ikke har overskredet dit API forbrug\n"
        "3. Bekræft at API'et understøtter de parametre, du sender"
        f"{detail_text}",
        title="API returnerede en fejl",
        border_style="red"
    ))

def display_parsing_error(message: str) -> None:
    """Display an error about data parsing issues."""
    rprint(Panel.fit(
        f"[bold red]Data Parsing Fejl:[/bold red] {message}\n\n"
        "[yellow]Tips til løsning:[/yellow]\n"
        "1. Tjek at API'et returnerede gyldigt JSON\n"
        "2. Kontroller om API'ets svarformat er ændret\n"
        "3. Prøv at køre programmet igen (der kan være et midlertidigt problem)",
        title="Kunne ikke behandle data",
        border_style="red"
    ))

def display_generic_error(message: str, error_type: str) -> None:
    """Display a generic error message."""
    rprint(Panel.fit(
        f"[bold red]Fejl ({error_type}):[/bold red] {message}\n\n"
        "[yellow]Tips til løsning:[/yellow]\n"
        "1. Tjek programmets log fil for detaljer\n"
        "2. Kontroller at alle påkrævede filer eksisterer\n"
        "3. Bekræft at du bruger en understøttet Python version (3.9+)",
        title="Der opstod en fejl",
        border_style="red"
    ))

def log_info(message: str, *args: Any, **kwargs: Any) -> None:
    """Log an informational message."""
    logger.info(message, *args, **kwargs)

def log_warning(message: str, *args: Any, **kwargs: Any) -> None:
    """Log a warning message."""
    logger.warning(message, *args, **kwargs)

def log_error(message: str, *args: Any, **kwargs: Any) -> None:
    """Log an error message."""
    logger.error(message, *args, **kwargs)

def handle_api_error(
    status_code: int, 
    response_body: str, 
    api_name: str
) -> Dict[str, str]:
    """
    Handle API errors based on status code and return user-friendly messages.
    
    Args:
        status_code: HTTP status code
        response_body: Response body from the API
        api_name: Name of the API (e.g., "OpenAI", "Perplexity")
        
    Returns:
        Dict with error message and suggested action
    """
    # Standard error messages by status code
    error_map = {
        400: {"message": "Ugyldig forespørgsel", "action": "Tjek parametre og formattering"},
        401: {"message": "Uautoriseret", "action": "Kontroller din API nøgle"},
        403: {"message": "Forbudt", "action": "Din API nøgle har ikke tilladelse til denne handling"},
        404: {"message": "Ressource ikke fundet", "action": "Tjek at URL/endpoint er korrekt"},
        429: {"message": "For mange forespørgsler", "action": "Vent lidt og prøv igen senere"},
        500: {"message": "Intern server fejl", "action": f"{api_name} API'et har tekniske problemer"},
        502: {"message": "Bad gateway", "action": "Netværksproblem eller API'et er nede"},
        503: {"message": "Service utilgængelig", "action": f"{api_name} API'et er midlertidigt nede"},
        504: {"message": "Gateway timeout", "action": "Serveren tog for lang tid om at svare"}
    }
    
    # Default for unknown status codes
    default = {"message": f"Ukendt fejl ({status_code})", "action": "Tjek logs for detaljer"}
    
    return error_map.get(status_code, default)

def retry_with_backoff(
    max_retries: int = 3,
    initial_backoff: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Union[List[Exception], Exception] = Exception
) -> Callable[[F], F]:
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_backoff: Initial backoff time in seconds
        backoff_factor: Factor to multiply backoff time by after each failure
        exceptions: Exception or list of exceptions to catch and retry on
        
    Returns:
        Decorated function that implements retry logic
    """
    import time
    import random
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Convert single exception to list
            exception_types = exceptions if isinstance(exceptions, list) else [exceptions]
            
            retries = 0
            current_backoff = initial_backoff
            
            while True:
                try:
                    return func(*args, **kwargs)
                except tuple(exception_types) as e:
                    retries += 1
                    if retries > max_retries:
                        # Log the final error before re-raising
                        logger.error(f"Failed after {max_retries} retries: {str(e)}")
                        raise
                    
                    # Add jitter to avoid thundering herd problem
                    jitter = random.uniform(0.8, 1.2)
                    wait_time = current_backoff * jitter
                    
                    logger.warning(f"Retry {retries}/{max_retries} after error: {str(e)}. "
                                   f"Waiting {wait_time:.2f}s")
                    
                    time.sleep(wait_time)
                    current_backoff *= backoff_factor
        
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            import asyncio
            
            # Convert single exception to list
            exception_types = exceptions if isinstance(exceptions, list) else [exceptions]
            
            retries = 0
            current_backoff = initial_backoff
            
            while True:
                try:
                    return await func(*args, **kwargs)
                except tuple(exception_types) as e:
                    retries += 1
                    if retries > max_retries:
                        # Log the final error before re-raising
                        logger.error(f"Failed after {max_retries} retries: {str(e)}")
                        raise
                    
                    # Add jitter to avoid thundering herd problem
                    jitter = random.uniform(0.8, 1.2)
                    wait_time = current_backoff * jitter
                    
                    logger.warning(f"Retry {retries}/{max_retries} after error: {str(e)}. "
                                   f"Waiting {wait_time:.2f}s")
                    
                    await asyncio.sleep(wait_time)
                    current_backoff *= backoff_factor
        
        # Choose the appropriate wrapper based on whether the function is async
        if inspect.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return wrapper  # type: ignore
    
    return decorator