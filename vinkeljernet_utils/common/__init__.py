"""
Vinkeljernet Utils - Common Module

Dette modul indeholder fælleskomponenter til konfiguration, logging og statistik.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum

# Konfigurer pakke-loggerens format
logger = logging.getLogger("vinkeljernet_utils")


class LogLevel(Enum):
    """Log niveauer til konfiguration."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


@dataclass
class PerformanceStats:
    """Statistik for pakkeydelse."""
    json_parsing_successes: int = 0
    json_parsing_failures: int = 0
    json_parsing_time: float = 0.0
    
    expert_sources_successes: int = 0
    expert_sources_failures: int = 0
    expert_sources_time: float = 0.0
    
    cache_hits: int = 0
    cache_misses: int = 0
    
    function_calls: Dict[str, int] = field(default_factory=dict)
    
    start_time: float = field(default_factory=time.time)
    
    @property
    def json_parsing_success_rate(self) -> float:
        """Beregn succesraten for JSON-parsing."""
        total = self.json_parsing_successes + self.json_parsing_failures
        if total == 0:
            return 100.0
        return (self.json_parsing_successes / total) * 100
    
    @property
    def expert_sources_success_rate(self) -> float:
        """Beregn succesraten for indhentning af ekspertkilder."""
        total = self.expert_sources_successes + self.expert_sources_failures
        if total == 0:
            return 100.0
        return (self.expert_sources_successes / total) * 100
    
    @property
    def cache_hit_rate(self) -> float:
        """Beregn cache-hit-raten."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return (self.cache_hits / total) * 100
    
    @property
    def uptime(self) -> float:
        """Beregn oppetiden i sekunder."""
        return time.time() - self.start_time
    
    def record_json_parsing(self, success: bool, elapsed_time: float) -> None:
        """
        Registrer en JSON-parsing-hændelse.
        
        Args:
            success: Om parsing lykkedes
            elapsed_time: Tid brugt i sekunder
        """
        if success:
            self.json_parsing_successes += 1
        else:
            self.json_parsing_failures += 1
        self.json_parsing_time += elapsed_time
    
    def record_expert_sources(self, success: bool, elapsed_time: float) -> None:
        """
        Registrer en ekspertkilde-indhentningshændelse.
        
        Args:
            success: Om indhentningen lykkedes
            elapsed_time: Tid brugt i sekunder
        """
        if success:
            self.expert_sources_successes += 1
        else:
            self.expert_sources_failures += 1
        self.expert_sources_time += elapsed_time
    
    def record_cache_access(self, hit: bool) -> None:
        """
        Registrer et cache-opslag.
        
        Args:
            hit: Om det var et cache-hit eller miss
        """
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
    
    def record_function_call(self, function_name: str) -> None:
        """
        Registrer et funktionskald.
        
        Args:
            function_name: Navn på den kaldte funktion
        """
        self.function_calls[function_name] = self.function_calls.get(function_name, 0) + 1


@dataclass
class PackageConfig:
    """Konfiguration for vinkeljernet_utils-pakken."""
    debug: bool = False
    cache_enabled: bool = True
    timeout: int = 30
    log_level: LogLevel = LogLevel.INFO
    
    # JSON parsing options
    json_max_recursion: int = 5
    json_recovery_strategies: List[str] = field(default_factory=lambda: [
        "direct", "extract", "fix_format", "partial"
    ])
    
    # Expert source options
    expert_sources_max_count: int = 5
    expert_sources_cache_ttl: int = 86400  # 24 hours
    
    # Statistics object
    stats: PerformanceStats = field(default_factory=PerformanceStats)


# Global pakkekonfiguration og statistik
_config = None
_stats = PerformanceStats()


def setup_logging(log_level: str = "INFO", debug: bool = False) -> None:
    """
    Konfigurer logging for vinkeljernet_utils.
    
    Args:
        log_level: Logniveau som string ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        debug: Aktivér debug-tilstand, som tilsidesætter log_level til DEBUG
    """
    level = getattr(logging, log_level.upper()) if not debug else logging.DEBUG
    
    # Konfigurer root-logger for pakken
    logger.setLevel(level)
    
    # Tilføj handler, hvis ingen eksisterer
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    logger.info(f"Vinkeljernet Utils logging initialized with level: {logging.getLevelName(level)}")


def configure_package(
    debug: bool = False,
    cache_enabled: bool = True,
    timeout: int = 30,
    log_level: str = "INFO"
) -> PackageConfig:
    """
    Konfigurer vinkeljernet_utils-pakken.
    
    Args:
        debug: Aktivér debug-tilstand
        cache_enabled: Aktivér caching
        timeout: Standard timeout for API-kald i sekunder
        log_level: Logniveau
        
    Returns:
        PackageConfig: Den aktuelle konfiguration
    """
    global _config
    
    # Konverter logniveau-string til enum
    try:
        log_level_enum = LogLevel[log_level.upper()]
    except KeyError:
        logger.warning(f"Invalid log level: {log_level}. Using INFO")
        log_level_enum = LogLevel.INFO
    
    # Opret konfiguration
    _config = PackageConfig(
        debug=debug,
        cache_enabled=cache_enabled,
        timeout=timeout,
        log_level=log_level_enum,
        stats=_stats  # Brug den globale statistik-instans
    )
    
    logger.info("Vinkeljernet Utils configured")
    if debug:
        logger.debug(f"Configuration: {_config}")
    
    return _config


def get_package_config() -> Optional[PackageConfig]:
    """
    Hent den aktuelle pakkekonfiguration.
    
    Returns:
        PackageConfig eller None hvis ikke konfigureret
    """
    return _config


def get_performance_stats() -> PerformanceStats:
    """
    Hent ydelses- og brugsstatistik for pakken.
    
    Returns:
        PerformanceStats: De aktuelle statistikker
    """
    return _stats