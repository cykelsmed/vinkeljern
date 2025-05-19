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
import time
import random
import asyncio
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Any, TypeVar, Optional, Dict, List, Union, Set, Tuple, Awaitable

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

class CircuitState(Enum):
    """Possible states for a circuit breaker"""
    CLOSED = 'closed'      # Normal operation, requests are allowed
    OPEN = 'open'          # Circuit is open, requests are blocked
    HALF_OPEN = 'half_open'  # Testing if service is back to normal

class CircuitBreaker:
    """
    Implementerer Circuit Breaker designmønster for at forhindre gentagne kald til 
    en fejlende tjeneste og for at give den tid til at genoprette sig.
    """
    
    # Klasseattributter til at holde styr på alle circuit breakers
    _breakers: Dict[str, 'CircuitBreaker'] = {}
    
    def __init__(
        self, 
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
        half_open_max_calls: int = 3,
        reset_timeout: int = 300,
        excluded_exceptions: Optional[Set[type]] = None
    ):
        """
        Initialiserer en CircuitBreaker.
        
        Args:
            name: Unikt navn til denne circuit breaker
            failure_threshold: Antal fejl før circuit åbnes
            recovery_timeout: Sekunder at vente før test af recovery (half-open state)
            half_open_max_calls: Maksimalt antal kald tilladt i half-open state
            reset_timeout: Sekunder før circuit automatisk lukkes igen
            excluded_exceptions: Exceptions der ikke tæller som fejl
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.reset_timeout = reset_timeout
        self.excluded_exceptions = excluded_exceptions or set()
        
        # State management
        self.state = CircuitState.CLOSED
        self.failures = 0
        self.last_failure_time = None
        self.last_state_change_time = datetime.now()
        self.half_open_calls = 0
        self.successful_calls = 0
        self.total_calls = 0
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        # Register this circuit breaker
        CircuitBreaker._breakers[name] = self
        
    @property
    def is_closed(self) -> bool:
        """Er circuit'en lukket (normal operation)?"""
        self._check_auto_reset()
        return self.state == CircuitState.CLOSED
    
    @property
    def is_open(self) -> bool:
        """Er circuit'en åben (blokerer requests)?"""
        self._check_auto_reset()
        return self.state == CircuitState.OPEN
    
    @property
    def is_half_open(self) -> bool:
        """Er circuit'en i half-open state (tester recovery)?"""
        self._check_auto_reset()
        return self.state == CircuitState.HALF_OPEN
        
    @property
    def failure_rate(self) -> float:
        """Fejlrate som procent af totale kald"""
        if self.total_calls == 0:
            return 0.0
        return (self.failures / self.total_calls) * 100
    
    def _check_auto_reset(self) -> None:
        """Check om circuit skal automatisk resets baseret på timeout"""
        now = datetime.now()
        
        # Hvis vi er i OPEN state og recovery timeout er gået
        if (self.state == CircuitState.OPEN and 
            self.last_state_change_time and 
            now - self.last_state_change_time > timedelta(seconds=self.recovery_timeout)):
            # Skift til HALF_OPEN for at teste om servicen er tilbage
            self.state = CircuitState.HALF_OPEN
            self.half_open_calls = 0
            self.last_state_change_time = now
            logger.info(f"Circuit '{self.name}' changed from OPEN to HALF_OPEN after recovery timeout")
            
        # Hvis vi har været i fejltilstand i meget lang tid, tving et reset
        if (self.state != CircuitState.CLOSED and 
            self.last_state_change_time and 
            now - self.last_state_change_time > timedelta(seconds=self.reset_timeout)):
            self.reset()
            logger.info(f"Circuit '{self.name}' force reset after {self.reset_timeout} seconds")
    
    async def record_success(self) -> None:
        """Registrer en succesfuld operation"""
        async with self._lock:
            self.successful_calls += 1
            self.total_calls += 1
            
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_calls += 1
                
                # Hvis vi har nok succesfulde kald i half-open state, luk circuit'en igen
                if self.half_open_calls >= self.half_open_max_calls:
                    self.state = CircuitState.CLOSED
                    self.failures = 0
                    self.half_open_calls = 0
                    self.last_state_change_time = datetime.now()
                    logger.info(f"Circuit '{self.name}' changed from HALF_OPEN to CLOSED after {self.half_open_max_calls} successful calls")
    
    async def record_failure(self, exception: Exception = None) -> None:
        """Registrer en fejlet operation"""
        # Tjek om denne exception er ekskluderet fra circuit breaking
        if exception and type(exception) in self.excluded_exceptions:
            logger.debug(f"Exception {type(exception).__name__} is excluded from circuit breaking")
            return
            
        async with self._lock:
            self.failures += 1
            self.total_calls += 1
            self.last_failure_time = datetime.now()
            
            # Hvis vi er i closed state og har nået failure threshold, åbn circuit'en
            if self.state == CircuitState.CLOSED and self.failures >= self.failure_threshold:
                self.state = CircuitState.OPEN
                self.last_state_change_time = datetime.now()
                logger.warning(f"Circuit '{self.name}' changed from CLOSED to OPEN after {self.failures} failures")
            
            # Hvis vi er i half-open state, enhver fejl åbner circuit'en igen
            elif self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                self.last_state_change_time = datetime.now()
                logger.warning(f"Circuit '{self.name}' changed from HALF_OPEN to OPEN after a failure")
    
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Executes the given function if the circuit allows it.
        
        Args:
            func: The function to execute
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the function
            
        Raises:
            CircuitOpenError: If the circuit is open
            Original exception: If the function fails and circuit does not open
        """
        self._check_auto_reset()
        
        # Hvis circuit'en er åben, tillad ikke kald
        if self.state == CircuitState.OPEN:
            wait_time = self.recovery_timeout - (datetime.now() - self.last_state_change_time).total_seconds()
            if wait_time > 0:
                logger.warning(f"Circuit '{self.name}' is OPEN. Retry after {wait_time:.1f}s")
                raise CircuitOpenError(
                    f"Circuit '{self.name}' is open. Try again in {wait_time:.1f} seconds.", 
                    wait_time=wait_time
                )
            
            # Hvis recovery timeout er gået, tillad ét kald i half-open state
            self.state = CircuitState.HALF_OPEN
            self.half_open_calls = 0
            self.last_state_change_time = datetime.now()
            logger.info(f"Circuit '{self.name}' changed from OPEN to HALF_OPEN for recovery testing")
        
        # Execute the function
        try:
            if inspect.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
                
            # Record success
            await self.record_success()
            return result
            
        except Exception as e:
            # Record failure
            await self.record_failure(e)
            raise
    
    def reset(self) -> None:
        """Reset circuit breaker to closed state"""
        self.state = CircuitState.CLOSED
        self.failures = 0
        self.half_open_calls = 0
        self.last_state_change_time = datetime.now()
        logger.info(f"Circuit '{self.name}' manually reset to CLOSED state")
    
    def get_stats(self) -> Dict[str, Any]:
        """Få statistik om denne circuit breaker"""
        self._check_auto_reset()  # Ensure state is current
        
        return {
            "name": self.name,
            "state": self.state.value,
            "failures": self.failures,
            "failure_threshold": self.failure_threshold,
            "failure_rate": f"{self.failure_rate:.1f}%",
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "last_failure": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "last_state_change": self.last_state_change_time.isoformat(),
            "time_in_current_state": (datetime.now() - self.last_state_change_time).total_seconds()
        }
    
    @classmethod
    def get(cls, name: str) -> Optional['CircuitBreaker']:
        """Get a circuit breaker by name"""
        return cls._breakers.get(name)
        
    @classmethod
    def get_all_stats(cls) -> Dict[str, Dict[str, Any]]:
        """Get stats for all circuit breakers"""
        return {name: breaker.get_stats() for name, breaker in cls._breakers.items()}
        
    @classmethod
    def reset_all(cls) -> None:
        """Reset all circuit breakers"""
        for breaker in cls._breakers.values():
            breaker.reset()
        logger.info("All circuit breakers reset")

class CircuitOpenError(Exception):
    """Raised when a circuit breaker is open"""
    def __init__(self, message: str, wait_time: float = None):
        self.wait_time = wait_time
        super().__init__(message)

def with_circuit_breaker(circuit_name: str, **circuit_kwargs) -> Callable[[F], F]:
    """
    Decorator to wrap a function with a circuit breaker.
    
    Args:
        circuit_name: Name for the circuit breaker
        **circuit_kwargs: Additional arguments for CircuitBreaker constructor
        
    Returns:
        Decorated function that respects circuit breaker state
    """
    def decorator(func: F) -> F:
        # Ensure the circuit breaker exists
        if circuit_name not in CircuitBreaker._breakers:
            CircuitBreaker(circuit_name, **circuit_kwargs)
        
        circuit = CircuitBreaker.get(circuit_name)
        
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            return await circuit.execute(func, *args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Use asyncio.run for the sync case
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(circuit.execute(func, *args, **kwargs))
            finally:
                loop.close()
        
        if inspect.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore
    
    return decorator

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
    Decorator for async functions to handle errors gracefully and return a fallback value.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                # Filtrer kwargs så kun accepterede argumenter sendes videre
                sig = inspect.signature(func)
                accepted = set(sig.parameters.keys())
                filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted}
                return await func(*args, **filtered_kwargs)
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

def format_json_parsing_error(error_info: dict) -> str:
    """
    Formaterer JSON-parsing fejl til en brugervenlig men detaljeret fejlmeddelelse.
    
    Args:
        error_info: Dictionary med fejlinformation fra enhanced_safe_parse_json
        
    Returns:
        str: Formateret fejlmeddelelse
    """
    message_parts = []
    message_parts.append("### JSON Parsing Fejl")
    message_parts.append("")
    
    # Tilføj grundlæggende information
    if "error" in error_info:
        message_parts.append(f"**Fejl:** {error_info['error']}")
    
    # Vis forsøgte parsing-metoder
    if "parsing_attempts" in error_info:
        message_parts.append("")
        message_parts.append("**Forsøgte parsing-metoder:**")
        for attempt in error_info['parsing_attempts']:
            method_name = {
                "direct_json_parse": "Direkte JSON parsing",
                "robust_json_parse": "Robust JSON parsing",
                "regex_extraction": "Regex udtrækning",
                "json_repair": "JSON reparation",
                "ai_repair": "AI-baseret reparation"
            }.get(attempt, attempt)
            message_parts.append(f"- {method_name}")
    
    # Vis detaljer om fejl
    if "errors" in error_info and error_info["errors"]:
        message_parts.append("")
        message_parts.append("**Detaljer:**")
        for i, error in enumerate(error_info["errors"][:5]):  # Vis kun de første 5 fejl
            message_parts.append(f"- {error}")
        
        if len(error_info["errors"]) > 5:
            message_parts.append(f"- ... og {len(error_info['errors'])-5} flere fejl")
    
    # Vis tekst forpreview hvis tilgængelig
    if "original_text_preview" in error_info and error_info["original_text_preview"]:
        message_parts.append("")
        message_parts.append("**Tekst preview:**")
        message_parts.append("```")
        preview = error_info["original_text_preview"]
        if len(preview) > 200:
            preview = preview[:197] + "..."
        message_parts.append(preview)
        message_parts.append("```")
    
    # Vis tips til at løse problemet
    message_parts.append("")
    message_parts.append("**Tips til fejlsøgning:**")
    message_parts.append("- Tjek at AI-modellen genererer korrekt formateret JSON")
    message_parts.append("- Prøv at bruge en mindre temperatur-værdi (0.2-0.4) for mere forudsigelige svar")
    message_parts.append("- Inkluder eksplicitte instruktioner om JSON-formatering i prompten")
    message_parts.append("- Prøv at generere vinkler igen, nogle gange kan en ny generering fungere bedre")
    message_parts.append("- Overvej at bruge fallback-vinkler hvis fejlen fortsætter")
    
    return "\n".join(message_parts)

class DetailedAngleError(Exception):
    """
    En detaljeret fejl for vinkelgenerering med information til debugging
    og brugervenlige fejlmeddelelser.
    """
    
    def __init__(
        self, 
        message: str, 
        error_type: str = "general", 
        debug_info: dict = None, 
        user_friendly_message: str = None,
        suggestions: list = None
    ):
        """
        Initialiserer en DetailedAngleError.
        
        Args:
            message: Teknisk fejlbesked
            error_type: Type af fejl (json_parsing, api_error, validation, osv.)
            debug_info: Ekstra debugging information
            user_friendly_message: Brugervenlig fejlbesked
            suggestions: Liste med forslag til at løse problemet
        """
        super().__init__(message)
        self.error_type = error_type
        self.debug_info = debug_info or {}
        self.user_friendly_message = user_friendly_message or "Der opstod en fejl under generering af vinkler"
        self.suggestions = suggestions or []
        self.timestamp = __import__('datetime').datetime.now().isoformat()
        
    def to_dict(self) -> dict:
        """Konverterer fejlen til en dictionary."""
        return {
            "error": True,
            "error_type": self.error_type,
            "message": str(self),
            "user_message": self.user_friendly_message,
            "debug_info": self.debug_info,
            "suggestions": self.suggestions,
            "timestamp": self.timestamp
        }
        
    def to_user_message(self) -> str:
        """Genererer en brugervenlig fejlmeddelelse."""
        parts = [f"# {self.user_friendly_message}"]
        
        if self.error_type == "json_parsing":
            parts.append("\nDer var problemer med at fortolke JSON-formatet fra AI-svaret.")
        elif self.error_type == "api_error":
            parts.append("\nDer var problemer med at kommunikere med AI-tjenesten.")
        elif self.error_type == "validation":
            parts.append("\nDe genererede vinkler overholdt ikke det forventede format.")
        
        if self.suggestions:
            parts.append("\n## Forslag til løsning")
            for suggestion in self.suggestions:
                parts.append(f"- {suggestion}")
        
        # Inkluder debugging info hvis relevant
        if self.error_type == "json_parsing" and self.debug_info:
            parts.append("\n## Tekniske detaljer")
            
            # Vis forsøgte parsing-metoder
            if "parsing_attempts" in self.debug_info:
                parts.append("\n**Forsøgte parsing-metoder:**")
                for attempt in self.debug_info['parsing_attempts']:
                    method_name = {
                        "direct_json_parse": "Direkte JSON parsing",
                        "robust_json_parse": "Robust JSON parsing",
                        "regex_extraction": "Regex udtrækning",
                        "json_repair": "JSON reparation",
                        "ai_repair": "AI-baseret reparation"
                    }.get(attempt, attempt)
                    parts.append(f"- {method_name}")
        
        return "\n".join(parts)
    
def generate_fallback_response(error_info: dict = None) -> dict:
    """
    Genererer et fallback-svar når vinkelgenerering fejler,
    med diagnostiske informationer og en positiv brugeroplevelse.
    
    Args:
        error_info: Ekstra fejlinformation
        
    Returns:
        dict: Fallback-response med fejlinformation og vinkelalternativer
    """
    import time
    import json
    import logging
    
    # Opret timestamp for fejlreference
    error_id = f"ERR-{int(time.time())}"
    
    # Log den detaljerede fejl
    if error_info:
        logging.error(f"Vinkelgenerering fejlede [{error_id}]: {json.dumps(error_info, default=str)}")
    
    # Generer et struktureret fejlsvar med constructive fejlmeddelelser
    response = {
        "error": True,
        "error_id": error_id,
        "message": "Der opstod en fejl under generering af vinkler",
        "status": "fallback_activated",
        "user_message": (
            "Vi kunne ikke generere specifikke vinkler denne gang. "
            "Vi har i stedet forberedt nogle generelle vinkler, du kan bruge som udgangspunkt."
        ),
        "suggestions": [
            "Prøv at generere vinkler igen",
            "Brug en mere specifik beskrivelse af emnet",
            "Prøv en anden redaktionel profil"
        ],
        "fallback_angles": [
            {
                "overskrift": "Aktuel status på emnet",
                "beskrivelse": "En opdateret analyse af den nuværende situation og udvikling.",
                "begrundelse": "Aktualitet er et vigtigt nyhedskriterie, og denne vinkel giver læserne et overblik over den seneste udvikling.",
                "nyhedskriterier": ["aktualitet", "væsentlighed"],
                "startSpørgsmål": ["Hvad er den seneste udvikling i sagen?", "Hvordan ser situationen ud lige nu?"]
            },
            {
                "overskrift": "Eksperter vurderer konsekvenserne",
                "beskrivelse": "Forskellige eksperters vurdering af mulige konsekvenser på kort og lang sigt.",
                "begrundelse": "Ekspertvurderinger giver dybde og troværdighed til historien, samt hjælper med at forstå perspektiverne.",
                "nyhedskriterier": ["væsentlighed", "aktualitet"],
                "startSpørgsmål": ["Hvilke konsekvenser forudser eksperterne?", "Hvordan kan dette påvirke samfundet?"]
            },
            {
                "overskrift": "Det skal du vide som borger",
                "beskrivelse": "Praktisk guide til hvordan borgere forholder sig til situationen og hvilke handlemuligheder de har.",
                "begrundelse": "Denne servicejournalistik hjælper læserne med at forstå, hvordan emnet påvirker deres hverdag direkte.",
                "nyhedskriterier": ["identifikation", "væsentlighed"],
                "startSpørgsmål": ["Hvordan påvirker dette den almindelige borger?", "Hvilke handlemuligheder har man som borger?"]
            }
        ]
    }
    
    # Tilføj fejldetaljer hvis tilgængelige
    if error_info:
        response["technical_details"] = {
            "error_type": error_info.get("error_type", "unknown"),
            "debug_info": error_info.get("debug_info", {}),
            "timestamp": error_info.get("timestamp", time.time())
        }
    
    return response

class ServiceDegradedException(Exception):
    """Raised when a service is degraded but may still provide limited functionality."""
    def __init__(self, message: str, service_name: str, severity: str = "medium"):
        self.service_name = service_name
        self.severity = severity  # "low", "medium", "high"
        self.timestamp = datetime.now().isoformat()
        super().__init__(message)

class FaultTolerantCache:
    """
    Fejltolerant cache system der kan bruges til at gemme og hente resultater
    af API-kald, med støtte til graceful degradation og lokalt fallback.
    
    Denne klasse arbejder sammen med CircuitBreaker for at forbedre
    systemets robusthed under API-fejl og netværksproblemer.
    """
    
    def __init__(
        self, 
        name: str,
        storage_dir: str = "./.cache",
        ttl: int = 3600,  # 1 time
        emergency_ttl: int = 86400 * 7,  # 7 dage for fallback
        allow_stale: bool = True,
    ):
        """
        Initialiserer en fejltolerant cache.
        
        Args:
            name: Cache kategori navn
            storage_dir: Mappe til cached filer
            ttl: Time-to-live i sekunder for normale cache entries
            emergency_ttl: Længere TTL for nødstilstand (ved fejl)
            allow_stale: Om forældet cache må bruges ved fejl
        """
        import os
        import json
        import hashlib
        
        self.name = name
        self.storage_dir = os.path.join(storage_dir, name)
        self.ttl = ttl
        self.emergency_ttl = emergency_ttl
        self.allow_stale = allow_stale
        self.degraded_mode = False
        self.stats = {
            "hits": 0,
            "misses": 0,
            "emergency_hits": 0,
            "stale_hits": 0,
            "stores": 0,
            "errors": 0,
        }
        
        # Opret cache-mappen hvis den ikke findes
        if not os.path.exists(self.storage_dir):
            try:
                os.makedirs(self.storage_dir)
                logger.info(f"Created cache directory: {self.storage_dir}")
            except Exception as e:
                logger.error(f"Failed to create cache directory: {str(e)}")
                # Fortsæt selvom vi ikke kan oprette mappen - vi kan stadig fungere uden cache
    
    def _get_key_path(self, key: str) -> str:
        """Konverterer en cache-nøgle til en filsti"""
        import os
        import hashlib
        
        # Hash nøglen for at sikre et gyldigt filnavn
        hashed_key = hashlib.md5(key.encode('utf-8')).hexdigest()
        return os.path.join(self.storage_dir, f"{hashed_key}.json")
    
    def _is_valid(self, metadata: dict) -> bool:
        """Tjekker om en cache entry stadig er gyldig baseret på TTL"""
        import time
        
        now = time.time()
        created = metadata.get("created", 0)
        ttl = metadata.get("ttl", self.ttl)
        
        return now - created < ttl
    
    def _is_emergency_valid(self, metadata: dict) -> bool:
        """Tjekker om en cache entry er gyldig i nødstilstand (længere TTL)"""
        import time
        
        now = time.time()
        created = metadata.get("created", 0)
        emergency_ttl = metadata.get("emergency_ttl", self.emergency_ttl)
        
        return now - created < emergency_ttl
    
    def get(self, key: str, default: Any = None, emergency: bool = False) -> Tuple[Any, dict]:
        """
        Henter en værdi fra cachen.
        
        Args:
            key: Cache nøgle
            default: Default værdi hvis nøglen ikke findes
            emergency: Om vi er i nødstilfælde (tillad forældet cache)
            
        Returns:
            Tuple af (cache_value, metadata)
        """
        import os
        import json
        import time
        
        cache_path = self._get_key_path(key)
        
        try:
            if not os.path.exists(cache_path):
                self.stats["misses"] += 1
                return default, {"exists": False}
            
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            metadata = cache_data.get("metadata", {})
            value = cache_data.get("value", default)
            
            # Tjek om cachen er gyldig
            if emergency or self.degraded_mode:
                # I nødstilfælde bruger vi forældet cache hvis tilladt
                if self._is_valid(metadata):
                    self.stats["hits"] += 1
                    return value, metadata
                elif self._is_emergency_valid(metadata):
                    self.stats["emergency_hits"] += 1
                    metadata["stale"] = True
                    return value, metadata
                else:
                    # Selv for nødstilfælde er cachen for gammel
                    self.stats["misses"] += 1
                    return default, {"exists": False, "too_old": True}
            elif self._is_valid(metadata):
                self.stats["hits"] += 1
                return value, metadata
            else:
                self.stats["misses"] += 1
                return default, {"exists": True, "expired": True}
                
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Error reading from cache ({key}): {str(e)}")
            return default, {"error": str(e)}
    
    def set(self, key: str, value: Any, metadata: dict = None) -> bool:
        """
        Gemmer en værdi i cachen.
        
        Args:
            key: Cache nøgle
            value: Værdi at gemme
            metadata: Ekstra metadata
            
        Returns:
            bool: Om operationen lykkedes
        """
        import os
        import json
        import time
        
        cache_path = self._get_key_path(key)
        
        try:
            # Forbered metadata
            meta = metadata or {}
            meta.update({
                "created": time.time(),
                "ttl": meta.get("ttl", self.ttl),
                "emergency_ttl": meta.get("emergency_ttl", self.emergency_ttl),
                "key": key
            })
            
            # Opret cache data struktur
            cache_data = {
                "value": value,
                "metadata": meta
            }
            
            # Skriv til cache fil
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, default=str)
                
            self.stats["stores"] += 1
            return True
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Error writing to cache ({key}): {str(e)}")
            return False
    
    def invalidate(self, key: str) -> bool:
        """Ugyldiggør en specifik cache entry."""
        import os
        
        try:
            cache_path = self._get_key_path(key)
            if os.path.exists(cache_path):
                os.remove(cache_path)
                return True
            return False
        except Exception as e:
            logger.error(f"Error invalidating cache ({key}): {str(e)}")
            return False
    
    def clear(self) -> int:
        """
        Rydder hele cachen.
        
        Returns:
            int: Antal slettede cache filer
        """
        import os
        import glob
        
        try:
            pattern = os.path.join(self.storage_dir, "*.json")
            cache_files = glob.glob(pattern)
            
            for file_path in cache_files:
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.error(f"Error removing cache file {file_path}: {str(e)}")
            
            return len(cache_files)
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            return 0
    
    def get_stats(self) -> dict:
        """Få cache statistik"""
        import os
        import glob
        
        try:
            pattern = os.path.join(self.storage_dir, "*.json")
            cache_files = glob.glob(pattern)
            
            # Tilføj cache fil information
            self.stats.update({
                "file_count": len(cache_files),
                "degraded_mode": self.degraded_mode,
            })
            
            # Beregn hit rate
            total_requests = self.stats["hits"] + self.stats["misses"] + self.stats["emergency_hits"] + self.stats["stale_hits"]
            hit_rate = 0.0
            if total_requests > 0:
                hit_rate = (self.stats["hits"] + self.stats["emergency_hits"] + self.stats["stale_hits"]) / total_requests * 100
                
            self.stats["hit_rate"] = f"{hit_rate:.1f}%"
            return self.stats
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
            return self.stats
    
    def set_degraded_mode(self, enabled: bool) -> None:
        """Sætter degraded mode (brug forældet cache hvis nødvendigt)"""
        self.degraded_mode = enabled
        logger.warning(f"Cache '{self.name}' degraded mode: {enabled}")

class FaultTolerantService:
    """
    En fejltolerant service klasse der kombinerer CircuitBreaker og FaultTolerantCache
    for at skabe robuste API-kald med graceful degradation.
    """
    
    def __init__(
        self,
        name: str,
        cache_ttl: int = 3600,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        cache_dir: str = "./.cache",
    ):
        """
        Initialiserer en fejltolerant service.
        
        Args:
            name: Service navn (bruges til cache og circuit breaker)
            cache_ttl: Cache time-to-live i sekunder
            failure_threshold: Antal fejl før circuit åbnes
            recovery_timeout: Sekunder før circuit prøver at lukke igen
            cache_dir: Mappe til cache
        """
        self.name = name
        
        # Opret circuit breaker
        if name not in CircuitBreaker._breakers:
            self.circuit = CircuitBreaker(
                name=name, 
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout
            )
        else:
            self.circuit = CircuitBreaker.get(name)
        
        # Opret cache
        self.cache = FaultTolerantCache(
            name=name,
            storage_dir=cache_dir,
            ttl=cache_ttl
        )
        
        # Service status
        self.healthy = True
        self.last_success = datetime.now()
        self.last_error = None
        self.last_error_message = None
        
        logger.info(f"Initialized fault-tolerant service: {name}")
    
    async def call(
        self,
        func: Callable,
        cache_key: str = None,
        use_cache: bool = True,
        store_cache: bool = True,
        fallback: Any = None,
        metadata: dict = None,
        *args,
        **kwargs
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Udfører et API kald med circuit breaking og caching.
        
        Args:
            func: Funktionen der skal udføres
            cache_key: Nøgle til caching (None deaktiverer caching)
            use_cache: Om cache skal bruges
            store_cache: Om resultatet skal caches
            fallback: Fallback værdi hvis kald fejler
            metadata: Ekstra metadata til caching
            *args, **kwargs: Argumenter til funktionen
            
        Returns:
            Tuple af (response, metadata)
        """
        response = None
        result_metadata = {
            "source": None,
            "cached": False,
            "stale": False,
            "degraded": False,
            "error": None,
        }
        
        # Check om vi bruger cache
        if use_cache and cache_key:
            # Prøv at hente fra cache
            cached_response, cache_meta = self.cache.get(
                key=cache_key, 
                emergency=not self.circuit.is_closed
            )
            
            if "exists" in cache_meta and cache_meta.get("exists", False) != False:
                response = cached_response
                result_metadata["source"] = "cache"
                result_metadata["cached"] = True
                result_metadata["cache_metadata"] = cache_meta
                result_metadata["stale"] = cache_meta.get("stale", False)
                
                # Hvis vi er i circuit open, returner cachen selvom den er forældet
                if self.circuit.is_open and "stale" in cache_meta:
                    logger.warning(f"Service {self.name} is degraded, returning stale cached result")
                    result_metadata["degraded"] = True
                    return response, result_metadata
        
        # Hvis vi ikke har et cached resultat, prøv at kalde API'et
        if response is None:
            try:
                # Brug circuit breaker til at udføre funktionen
                response = await self.circuit.execute(func, *args, **kwargs)
                
                # Opdater status
                self.healthy = True
                self.last_success = datetime.now()
                result_metadata["source"] = "api"
                
                # Cache resultatet hvis nødvendigt
                if store_cache and cache_key:
                    self.cache.set(cache_key, response, metadata)
                
            except CircuitOpenError as e:
                # Circuit er åben, brug fallback
                logger.warning(f"Circuit {self.name} is open: {str(e)}")
                response = fallback
                result_metadata["source"] = "fallback"
                result_metadata["error"] = f"Circuit open: {str(e)}"
                result_metadata["degraded"] = True
                
            except Exception as e:
                # Andre fejl
                self.healthy = False
                self.last_error = datetime.now()
                self.last_error_message = str(e)
                
                logger.error(f"Error in service {self.name}: {str(e)}")
                response = fallback
                result_metadata["source"] = "fallback"
                result_metadata["error"] = str(e)
                
                # Hvis cachen er i degraded mode, prøv igen med emergency flag
                if cache_key:
                    cached_response, cache_meta = self.cache.get(
                        key=cache_key, 
                        emergency=True
                    )
                    
                    if cached_response is not None:
                        response = cached_response
                        result_metadata["source"] = "emergency_cache"
                        result_metadata["cached"] = True
                        result_metadata["degraded"] = True
                        result_metadata["stale"] = True
        
        return response, result_metadata
    
    def get_status(self) -> Dict[str, Any]:
        """Få status for denne service"""
        return {
            "name": self.name,
            "healthy": self.healthy,
            "circuit": self.circuit.state.value,
            "last_success": self.last_success.isoformat(),
            "last_error": self.last_error.isoformat() if self.last_error else None,
            "last_error_message": self.last_error_message,
            "cache_stats": self.cache.get_stats(),
            "circuit_stats": self.circuit.get_stats()
        }
    
    def reset(self) -> None:
        """Reset service status og circuit breaker"""
        self.circuit.reset()
        self.healthy = True
        self.last_error = None
        self.last_error_message = None
        logger.info(f"Service {self.name} reset")

# Registry to track all fault-tolerant services
_fault_tolerant_services: Dict[str, FaultTolerantService] = {}

def get_service(name: str) -> FaultTolerantService:
    """Få en fault tolerant service ved navn"""
    if name not in _fault_tolerant_services:
        _fault_tolerant_services[name] = FaultTolerantService(name)
    return _fault_tolerant_services[name]

def get_all_services_status() -> Dict[str, Dict[str, Any]]:
    """Få status for alle services"""
    return {name: service.get_status() for name, service in _fault_tolerant_services.items()}

class FaultTolerantAngleGenerator:
    """
    En fejltolerant vinkelgenerator der sikrer at brugeren altid får et brugbart resultat,
    selv hvis API-kald fejler eller andre komponenter ikke fungerer som forventet.
    
    Denne klasse implementerer strategi #1 fra de ønskede punkter - at sikre at selv 
    hvis en komponent fejler, får brugeren stadig et brugbart resultat.
    """
    
    def __init__(
        self,
        service_name: str = "angle_generator",
        cache_ttl: int = 3600 * 24,  # 24 timer
        failure_threshold: int = 3,
        recovery_timeout: int = 60,
        min_acceptable_angles: int = 3,
        cache_dir: str = "./.cache/angles",
        use_generic_fallbacks: bool = True
    ):
        """
        Initialiserer en fejltolerant vinkelgenerator.
        
        Args:
            service_name: Navn til service registrering
            cache_ttl: Cache levetid i sekunder
            failure_threshold: Antal fejl før circuit breaker åbnes
            recovery_timeout: Tid før circuit breaker tester recovery
            min_acceptable_angles: Minimum antal vinkler før resultatet anses som brugbart
            cache_dir: Mappe til cache
            use_generic_fallbacks: Om der skal bruges generiske fallback vinkler ved fejl
        """
        self.service_name = service_name
        self.min_acceptable_angles = min_acceptable_angles
        self.use_generic_fallbacks = use_generic_fallbacks
        
        # Opret fault tolerant service
        self.service = FaultTolerantService(
            name=service_name,
            cache_ttl=cache_ttl,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            cache_dir=cache_dir
        )
        
        # Stats
        self.successful_generations = 0
        self.partial_successful_generations = 0
        self.failed_generations = 0
        self.fallback_used_count = 0
        
        logger.info(f"Initialized fault-tolerant angle generator: {service_name}")
    
    def _create_cache_key(self, topic: str, profile_id: str) -> str:
        """Genererer en unik cache nøgle baseret på emne og profil"""
        return f"angle_{profile_id}_{topic.lower().replace(' ', '_')}"
    
    def _create_simple_fallback_angles(self, topic: str) -> List[Dict[str, Any]]:
        """Genererer enkle fallback vinkler når alt andet fejler."""
        # Listen af vinkler hentes fra generate_fallback_response
        fallback = generate_fallback_response()
        
        # Tilpas overskrifter med emnet
        angles = fallback.get("fallback_angles", [])
        for angle in angles:
            if "overskrift" in angle:
                angle["overskrift"] = angle["overskrift"].replace("emnet", topic)
            if "beskrivelse" in angle:
                angle["beskrivelse"] = angle["beskrivelse"].replace("emnet", topic)
            # Tilføj en visuel indikation af at dette er en fallback vinkel
            angle["_fallback"] = True
                
        return angles
    
    async def generate_angles(
        self,
        topic: str,
        profile: Any,  # RedaktionelDNA
        generate_func: Callable,
        bypass_cache: bool = False,
        *args,
        **kwargs
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Genererer vinkler med fejltolerance, fallbacks, og caching.
        
        Args:
            topic: Emnet at generere vinkler for
            profile: Redaktionel DNA profil
            generate_func: Funktionen der genererer vinkler
            bypass_cache: Om cache skal ignoreres
            args, kwargs: Ekstra argumenter til generate_func
            
        Returns:
            Tuple of (angles, metadata)
        """
        # Opret metadata til resultatet
        result_metadata = {
            "source": None,
            "cached": False,
            "degraded": False,
            "partial_result": False,
            "fallback_used": False,
            "error": None,
            "angle_count": 0,
            "requested_topic": topic
        }
        
        # Forbered cache nøgle
        profile_id = getattr(profile, "id", None) or getattr(profile, "navn", "unknown_profile")
        cache_key = None if bypass_cache else self._create_cache_key(topic, profile_id)
        
        try:
            # Forbered parametre til generate_func
            kwargs["topic"] = topic
            kwargs["profile"] = profile
            
            # Definér fallback værdi
            fallback_angles = []
            if self.use_generic_fallbacks:
                fallback_angles = self._create_simple_fallback_angles(topic)
                
            # Kald servicen med circuit breaker og cache
            angles, call_metadata = await self.service.call(
                func=generate_func,
                cache_key=cache_key,
                use_cache=not bypass_cache,
                store_cache=True,
                fallback=fallback_angles,
                metadata={"topic": topic, "profile_id": profile_id},
                *args,
                **kwargs
            )
            
            # Opdater metadata fra service kaldet
            result_metadata.update(call_metadata)
            
            # Validér resultatet - sikre at vi har vinkler og at de er i det rigtige format
            if not angles or not isinstance(angles, list):
                logger.warning(f"Generate angles returned invalid result: {type(angles)}")
                
                if fallback_angles:
                    angles = fallback_angles
                    result_metadata["fallback_used"] = True
                    result_metadata["source"] = "internal_fallback"
                    self.fallback_used_count += 1
                else:
                    angles = []
            
            # Tjek om vi har nok vinkler
            angle_count = len(angles)
            result_metadata["angle_count"] = angle_count
            
            if angle_count < self.min_acceptable_angles:
                # Ikke nok vinkler - tilføj fallback vinkler for at nå op på minimum
                if fallback_angles:
                    # Tilføj kun de fallback vinkler vi mangler
                    missing_count = self.min_acceptable_angles - angle_count
                    if missing_count > 0:
                        angles.extend(fallback_angles[:missing_count])
                        result_metadata["partial_result"] = True
                        result_metadata["fallback_used"] = True
                        result_metadata["source"] = result_metadata.get("source", "api") + "_with_fallback"
                        self.partial_successful_generations += 1
                    else:
                        # Vi har ikke brug for fallbacks
                        self.successful_generations += 1
                else:
                    # Ingen fallbacks tilgængelige - returner det vi har
                    if angle_count > 0:
                        result_metadata["partial_result"] = True
                        self.partial_successful_generations += 1
                    else:
                        self.failed_generations += 1
            else:
                # Tilstrækkeligt antal vinkler
                self.successful_generations += 1
            
            return angles, result_metadata
            
        except Exception as e:
            logger.error(f"Error in FaultTolerantAngleGenerator.generate_angles: {str(e)}")
            self.failed_generations += 1
            
            # Returner fallback vinkler ved fejl
            if fallback_angles:
                result_metadata["fallback_used"] = True
                result_metadata["error"] = str(e)
                result_metadata["source"] = "error_fallback"
                result_metadata["degraded"] = True
                result_metadata["angle_count"] = len(fallback_angles)
                self.fallback_used_count += 1
                return fallback_angles, result_metadata
            else:
                # Hvis vi ikke har fallbacks, rajs fejlen videre
                raise
                
    def get_stats(self) -> Dict[str, Any]:
        """Få statistik om vinkelgenerator"""
        total_generations = (
            self.successful_generations + 
            self.partial_successful_generations + 
            self.failed_generations
        )
        
        stats = {
            "service_name": self.service_name,
            "total_generations": total_generations,
            "successful_generations": self.successful_generations,
            "partial_successful_generations": self.partial_successful_generations,
            "failed_generations": self.failed_generations,
            "fallback_used_count": self.fallback_used_count,
        }
        
        # Beregn success rate
        if total_generations > 0:
            full_success_rate = (self.successful_generations / total_generations) * 100
            any_success_rate = ((self.successful_generations + self.partial_successful_generations) 
                              / total_generations) * 100
                              
            stats["full_success_rate"] = f"{full_success_rate:.1f}%"
            stats["any_success_rate"] = f"{any_success_rate:.1f}%"
        
        # Tilføj service stats
        stats["service_status"] = self.service.get_status()
        
        return stats

class FaultTolerantComponent:
    """
    Base klasse for fejltolerante komponenter i Vinkeljernet.
    
    Denne klasse håndterer:
    1. Graceful degradation ved komponent fejl
    2. Lokal caching af resultater
    3. Circuit breaking ved gentagne fejl
    4. Automatisk fallback til simplere implementeringer
    """
    
    def __init__(
        self,
        name: str,
        cache_ttl: int = 3600,  # 1 time
        failure_threshold: int = 3,
        recovery_timeout: int = 60,
        cache_dir: str = "./.cache"
    ):
        """
        Initialiserer en fejltolerant komponent.
        
        Args:
            name: Komponentens navn
            cache_ttl: Cache levetid i sekunder
            failure_threshold: Antal fejl før circuit breaker åbnes
            recovery_timeout: Tid før circuit breaker tester recovery
            cache_dir: Mappe til cache
        """
        self.name = name
        
        # Opret fault tolerant service
        self.service = FaultTolerantService(
            name=name,
            cache_ttl=cache_ttl,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            cache_dir=cache_dir
        )
        
        # Stats
        self.successful_calls = 0
        self.failed_calls = 0
        self.degraded_calls = 0
        self.start_time = datetime.now()
        
        logger.info(f"Initialized fault-tolerant component: {name}")
    
    async def call_method(
        self,
        primary_func: Callable,
        fallback_func: Optional[Callable] = None,
        cache_key: Optional[str] = None,
        use_cache: bool = True,
        *args,
        **kwargs
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Kalder en metode med fejlhåndtering og graceful degradation.
        
        Args:
            primary_func: Den primære funktion at kalde
            fallback_func: En simplere/mere robust fallback funktion
            cache_key: Cache nøgle hvis caching ønskes
            use_cache: Om cache skal bruges
            args, kwargs: Argumenter til funktionen
            
        Returns:
            Tuple af (result, metadata)
        """
        result_metadata = {
            "component": self.name,
            "timestamp": datetime.now().isoformat(),
            "degraded": False,
            "source": None,
            "error": None
        }
        
        try:
            # Prøv at kalde den primære funktion med circuit breaker og cache
            result, call_metadata = await self.service.call(
                func=primary_func,
                cache_key=cache_key,
                use_cache=use_cache,
                *args,
                **kwargs
            )
            
            # Opdater metadata
            result_metadata.update(call_metadata)
            
            # Tjek om det lykkedes
            if result is not None:
                self.successful_calls += 1
                return result, result_metadata
                
        except Exception as e:
            logger.error(f"Error in {self.name} primary function: {str(e)}")
            result_metadata["error"] = str(e)
        
        # Hvis vi kommer hertil, fejlede den primære funktion
        # Prøv fallback funktionen, hvis en er angivet
        if fallback_func:
            try:
                result_metadata["degraded"] = True
                result_metadata["source"] = "fallback"
                
                # Kald fallback funktion
                fallback_result = await fallback_func(*args, **kwargs) if inspect.iscoroutinefunction(fallback_func) else fallback_func(*args, **kwargs)
                
                if fallback_result is not None:
                    self.degraded_calls += 1
                    return fallback_result, result_metadata
                    
            except Exception as e:
                logger.error(f"Error in {self.name} fallback function: {str(e)}")
                result_metadata["fallback_error"] = str(e)
        
        # Både primær og fallback fejlede
        self.failed_calls += 1
        return None, result_metadata
    
    def get_stats(self) -> Dict[str, Any]:
        """Get component statistics"""
        total_calls = self.successful_calls + self.failed_calls + self.degraded_calls
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        stats = {
            "name": self.name,
            "total_calls": total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "degraded_calls": self.degraded_calls,
            "uptime_seconds": uptime
        }
        
        # Calculate success rate
        if total_calls > 0:
            success_rate = (self.successful_calls / total_calls) * 100
            availability_rate = ((self.successful_calls + self.degraded_calls) / total_calls) * 100
            
            stats["success_rate"] = f"{success_rate:.1f}%"
            stats["availability_rate"] = f"{availability_rate:.1f}%"
        
        # Add service stats
        stats["service_status"] = self.service.get_status()
        
        return stats

class VinkeljernetAppStatus:
    """
    Klasse til at overvåge og rapportere om Vinkeljernets samlede systemtilstand.
    Implementerer en monitoring facade der aggregerer information fra alle fejltolerante
    komponenter og services.
    """
    
    def __init__(self):
        """Initialiserer applikationsstatus monitor"""
        self.start_time = datetime.now()
        self.components = {}  # Registered fault-tolerant components
        self.last_check = None
        self.overall_health = "healthy"  # healthy, degraded, unhealthy
        self.health_checks = []
        self.degraded_services = set()
        
    def register_component(self, component_name: str, component: Any) -> None:
        """Registrerer en komponent til monitoring"""
        self.components[component_name] = component
        
    def register_health_check(self, check_func: Callable) -> None:
        """Registrerer en funktion til health checks"""
        self.health_checks.append(check_func)
    
    def run_health_check(self) -> Dict[str, Any]:
        """
        Udfører et health check på hele systemet
        
        Returns:
            Dict med status information
        """
        self.last_check = datetime.now()
        
        # Start med frisk status
        status = {
            "timestamp": self.last_check.isoformat(),
            "uptime_seconds": (self.last_check - self.start_time).total_seconds(),
            "overall_health": "healthy",
            "components": {},
            "services": {},
            "circuit_breakers": {},
            "issues": [],
        }
        
        # Tjek alle komponenter
        for name, component in self.components.items():
            if hasattr(component, 'get_stats'):
                comp_stats = component.get_stats()
                status["components"][name] = comp_stats
        
        # Tjek alle fault tolerant services
        services_status = get_all_services_status()
        status["services"] = services_status
        
        # Tjek alle circuit breakers
        circuit_stats = CircuitBreaker.get_all_stats()
        status["circuit_breakers"] = circuit_stats
        
        # Check for open circuits
        open_circuits = [
            name for name, stats in circuit_stats.items() 
            if stats["state"] != "closed"
        ]
        
        if open_circuits:
            status["issues"].append({
                "type": "circuit_breaker",
                "severity": "warning",
                "message": f"Open circuit breakers: {', '.join(open_circuits)}"
            })
        
        # Find degraded services
        self.degraded_services = set()
        for name, service_stats in services_status.items():
            if not service_stats.get("healthy", True):
                self.degraded_services.add(name)
                status["issues"].append({
                    "type": "service",
                    "severity": "warning",
                    "message": f"Service {name} is unhealthy: {service_stats.get('last_error_message', 'Unknown error')}"
                })
        
        # Kør registrerede health checks
        for check_func in self.health_checks:
            try:
                check_result = check_func()
                if check_result and not check_result.get("healthy", True):
                    status["issues"].append({
                        "type": "health_check",
                        "severity": check_result.get("severity", "warning"),
                        "message": check_result.get("message", "Unknown health check issue"),
                        "component": check_result.get("component", "unknown")
                    })
            except Exception as e:
                logger.error(f"Error in health check function: {str(e)}")
                status["issues"].append({
                    "type": "health_check_error",
                    "severity": "error",
                    "message": f"Health check failed: {str(e)}"
                })
        
        # Determine overall health based on issues
        critical_issues = [i for i in status["issues"] if i.get("severity") == "critical"]
        errors = [i for i in status["issues"] if i.get("severity") == "error"]
        warnings = [i for i in status["issues"] if i.get("severity") == "warning"]
        
        if critical_issues:
            status["overall_health"] = "unhealthy"
        elif errors:
            status["overall_health"] = "degraded"
        elif warnings:
            status["overall_health"] = "degraded"
        else:
            status["overall_health"] = "healthy"
            
        self.overall_health = status["overall_health"]
        
        return status
    
    def is_healthy(self) -> bool:
        """Er systemet fuldt funktionelt?"""
        return self.overall_health == "healthy"
    
    def is_degraded(self) -> bool:
        """Er systemet i degraderet mode?"""
        return self.overall_health == "degraded"
    
    def is_unhealthy(self) -> bool:
        """Er systemet ikke-funktionelt?"""
        return self.overall_health == "unhealthy"
    
    def get_degraded_services(self) -> Set[str]:
        """Returner navne på services i degraderet mode"""
        return self.degraded_services.copy()

# Global application status
app_status = VinkeljernetAppStatus()

def get_app_status() -> VinkeljernetAppStatus:
    """Returnerer global app status instance"""
    return app_status

class FaultTolerantAngleGenerationOrchestrator:
    """
    Orkestrerer hele vinkelgenererings processen med fejltolerance og 
    graceful degradation på alle niveauer.
    
    Denne klasse håndterer:
    1. Paralleliseret vinkelgenerering med fejlhåndtering
    2. Berigelse af vinkler med kildeforslag 
    3. Prioritering og filtrering af vinkler
    4. Fallback strategier for alle trin
    """
    
    def __init__(
        self,
        topic_info_service: FaultTolerantService = None,
        angle_generator: FaultTolerantAngleGenerator = None,
        source_suggestion_service: FaultTolerantService = None,
        cache_dir: str = "./.cache/vinklerjernet",
        fallback_strategy: str = "graceful"  # graceful, strict, or minimal
    ):
        """
        Initialiserer orkestrerings modulet
        
        Args:
            topic_info_service: FaultTolerantService til topic information
            angle_generator: FaultTolerantAngleGenerator til vinkelgenerering
            source_suggestion_service: FaultTolerantService til kildeforslag
            cache_dir: Base directory for cache
            fallback_strategy: Strategi for fejlhåndtering og fallback
        """
        # Opret eller brug service for topic information
        self.topic_info_service = topic_info_service or FaultTolerantService(
            name="topic_info", 
            cache_ttl=86400,
            cache_dir=f"{cache_dir}/topic_info",
            failure_threshold=3
        )
        
        # Opret eller brug vinkelgenerator
        self.angle_generator = angle_generator or FaultTolerantAngleGenerator(
            service_name="angle_generator",
            cache_ttl=86400,
            cache_dir=f"{cache_dir}/angles",
            failure_threshold=3
        )
        
        # Opret eller brug service for kildeforslag
        self.source_suggestion_service = source_suggestion_service or FaultTolerantService(
            name="source_suggestions",
            cache_ttl=86400 * 7,  # 7 dage
            cache_dir=f"{cache_dir}/sources",
            failure_threshold=3
        )
        
        self.fallback_strategy = fallback_strategy
        
        # Register med app status
        app_status.register_component("angle_orchestrator", self)
        
        logger.info(f"Initialized FaultTolerantAngleGenerationOrchestrator with {fallback_strategy} fallback strategy")
    
    async def generate_angles_with_sources(
        self,
        topic: str,
        profile: Any,
        topic_info_func: Callable,
        generate_angles_func: Callable,
        source_suggestion_func: Callable = None,
        bypass_cache: bool = False,
        max_angles: int = 10,
        max_sources: int = 3,
        progress_callback: Callable[[int, str], Awaitable[None]] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Genererer vinkler og ekspertkilder med fuld fejlhåndtering og fallbacks.
        
        Args:
            topic: Emnet at generere vinkler for
            profile: Redaktionel DNA profil
            topic_info_func: Funktion til at hente baggrundsinformation
            generate_angles_func: Funktion til at generere vinkler
            source_suggestion_func: Funktion til at generere kildeforslag
            bypass_cache: Om cache skal ignoreres
            max_angles: Maksimalt antal vinkler at returnere
            max_sources: Maksimalt antal vinkler at berige med kilder
            progress_callback: Callback funktion til fremskridtsrapportering
            
        Returns:
            Tuple af (angles, metadata)
        """
        metadata = {
            "topic": topic,
            "started_at": datetime.now().isoformat(),
            "degraded": False,
            "error": None,
            "steps_completed": [],
            "steps_failed": [],
            "fallbacks_used": []
        }
        
        # Asynkron progress update
        async def update_progress(percent: int, message: str) -> None:
            if progress_callback:
                await progress_callback(percent, message)
            else:
                logger.debug(f"Progress {percent}%: {message}")
        
        await update_progress(5, "Starter vinkelgenerering...")
        
        try:
            # 1. Hent baggrundsinformation om emnet
            await update_progress(10, "Henter baggrundsinformation...")
            
            topic_info_result, topic_info_metadata = await self.topic_info_service.call(
                func=topic_info_func,
                cache_key=f"topic_info_{topic.lower().replace(' ', '_')}",
                use_cache=not bypass_cache,
                fallback=f"Emnet handler om {topic}.",
                args=[topic],
                kwargs={"bypass_cache": bypass_cache}
            )
            
            if topic_info_metadata.get("error"):
                metadata["steps_failed"].append("topic_info")
                if topic_info_metadata.get("degraded"):
                    metadata["degraded"] = True
                    metadata["fallbacks_used"].append("simple_topic_info")
            else:
                metadata["steps_completed"].append("topic_info")
            
            # 2. Generer vinkler med fejltolerante service
            await update_progress(20, "Genererer vinkler...")
            
            angles, angle_metadata = await self.angle_generator.generate_angles(
                topic=topic,
                profile=profile,
                generate_func=generate_angles_func,
                bypass_cache=bypass_cache,
                topic_info=topic_info_result
            )
            
            metadata.update({
                "angle_count": len(angles),
                "angle_source": angle_metadata.get("source"),
                "angle_cached": angle_metadata.get("cached", False)
            })
            
            if angle_metadata.get("fallback_used"):
                metadata["fallbacks_used"].append("fallback_angles")
                metadata["degraded"] = True
                metadata["steps_failed"].append("angle_generation")
            else:
                metadata["steps_completed"].append("angle_generation")
            
            # Hvis vi ikke har vinkler, returner tidligt med fejl
            if not angles:
                metadata["error"] = "Ingen vinkler kunne genereres"
                metadata["degraded"] = True
                return [], metadata
            
            # Begræns antallet af vinkler
            angles = angles[:max_angles]
            
            # 3. Berig vinkler med kildeforslag hvis funktionen er tilgængelig
            if source_suggestion_func and (self.fallback_strategy != "minimal" or not metadata["degraded"]):
                await update_progress(50, "Henter kildeforslag...")
                
                # Berig kun de første max_sources vinkler
                angles_to_enrich = angles[:max_sources]
                enriched_count = 0
                
                for i, angle in enumerate(angles_to_enrich):
                    try:
                        headline = angle.get("overskrift", f"Vinkel om {topic}")
                        description = angle.get("beskrivelse", "")
                        
                        # Generer cache nøgle
                        cache_key = f"source_{topic.lower().replace(' ', '_')}_angle_{i}"
                        
                        # Hent kildeforslag
                        sources, source_metadata = await self.source_suggestion_service.call(
                            func=source_suggestion_func,
                            cache_key=cache_key,
                            use_cache=not bypass_cache,
                            fallback={
                                "experts": [],
                                "institutions": [],
                                "data_sources": []
                            },
                            args=[],
                            kwargs={
                                "topic": topic,
                                "angle_headline": headline,
                                "angle_description": description,
                                "bypass_cache": bypass_cache
                            }
                        )
                        
                        # Tilføj kilder til vinklen
                        angles[i]["ekspertKilder"] = sources
                        angles[i]["harEkspertKilder"] = bool(
                            sources.get("experts", []) or 
                            sources.get("institutions", []) or 
                            sources.get("data_sources", [])
                        )
                        enriched_count += 1
                        
                        # Opdater progress
                        source_progress = 50 + (enriched_count * 30) // max_sources
                        await update_progress(source_progress, f"Henter kildeforslag ({enriched_count}/{max_sources})...")
                        
                        if source_metadata.get("degraded"):
                            metadata["degraded"] = True
                            
                    except Exception as e:
                        logger.error(f"Error enriching angle {i} with sources: {str(e)}")
                        angles[i]["ekspertKilder"] = {
                            "experts": [],
                            "institutions": [],
                            "data_sources": [],
                            "error": str(e)
                        }
                        angles[i]["harEkspertKilder"] = False
                
                # Check om vi har beriget nogen vinkler
                if enriched_count > 0:
                    metadata["steps_completed"].append("source_suggestions")
                    metadata["enriched_angles"] = enriched_count
                else:
                    metadata["steps_failed"].append("source_suggestions")
                    metadata["fallbacks_used"].append("no_sources")
            
            # 4. Sorter og prioriter vinkler
            await update_progress(90, "Prioriterer vinkler...")
            
            try:
                # Check om angle_processor modulet er tilgængeligt
                from angle_processor import filter_and_rank_angles
                ranked_angles = filter_and_rank_angles(angles, profile, max_angles)
                metadata["steps_completed"].append("angle_ranking")
            except (ImportError, ModuleNotFoundError):
                # Fallback til simpel sortering
                ranked_angles = sorted(
                    angles, 
                    key=lambda a: len(a.get("begrundelse", "")) + len(a.get("beskrivelse", "")),
                    reverse=True
                )
                metadata["fallbacks_used"].append("simple_ranking")
            
            await update_progress(100, "Færdig!")
            metadata["completed_at"] = datetime.now().isoformat()
            
            return ranked_angles, metadata
            
        except Exception as e:
            logger.error(f"Error in generate_angles_with_sources: {str(e)}")
            metadata["error"] = str(e)
            metadata["degraded"] = True
            
            # Forsøg at returnere fallback vinkler
            fallback_angles = self.angle_generator._create_simple_fallback_angles(topic)
            return fallback_angles, metadata
    
    def get_stats(self) -> Dict[str, Any]:
        """Få statistik fra orkestrering"""
        return {
            "angle_generator_stats": self.angle_generator.get_stats(),
            "topic_info_service": self.topic_info_service.get_status(),
            "source_service": self.source_suggestion_service.get_status(),
            "fallback_strategy": self.fallback_strategy
        }
        
def create_api_health_check_func(api_name: str, api_client: Any, test_endpoint: str = None) -> Callable:
    """
    Opretter en health check funktion for et API
    
    Args:
        api_name: API navn
        api_client: API klientobjekt
        test_endpoint: Endpoint at test (hvis None bruges en no-op test)
        
    Returns:
        Health check funktion
    """
    async def health_check() -> Dict[str, Any]:
        """
        Udfører health check på API'et
        
        Returns:
            Dict med health check resultat
        """
        result = {
            "component": api_name,
            "healthy": True,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Hvis vi har en circuit breaker for dette API, tjek dens status
            circuit_breaker = CircuitBreaker.get(api_name)
            if circuit_breaker and circuit_breaker.is_open:
                result["healthy"] = False
                result["severity"] = "error"
                result["message"] = f"Circuit breaker for {api_name} er åben"
                return result
            
            # Hvis vi har et test endpoint, test API forbindelse
            if test_endpoint and hasattr(api_client, 'request'):
                # Implementér en enkel test request
                pass
            
        except Exception as e:
            result["healthy"] = False
            result["severity"] = "error"
            result["message"] = f"Health check fejlet: {str(e)}"
        
        return result
    
    return health_check