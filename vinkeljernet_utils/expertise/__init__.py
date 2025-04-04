"""
Vinkeljernet Utils - Expertise Module

Dette modul håndterer indhentning og validering af ekspertkilder til nyhedsvinkler.
Modulet inkluderer:
- Modeller til ekspertkildedata
- Funktioner til at indhente og validere ekspertkilder
- Caching af ekspertkilder for forbedret ydelse
- Fejlhåndtering og fallback-mekanismer
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple, Callable

# Import JSON parsing utilities
from vinkeljernet_utils.json_parsing import safe_parse_json
from vinkeljernet_utils.common import get_performance_stats

# Module logger
logger = logging.getLogger("vinkeljernet_utils.expertise")


@dataclass
class ExpertContact:
    """Kontaktinformation for en ekspert."""
    type: str = "email"  # email, telefon, web
    value: str = ""
    notes: str = ""


@dataclass
class ExpertSource:
    """Model for en ekspertkilde."""
    name: str = ""
    title: str = ""
    organization: str = ""
    expertise: str = ""
    contact: Optional[ExpertContact] = None
    relevance: str = ""
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExpertSource':
        """
        Opret en ExpertSource fra en dict.
        
        Args:
            data: Dictionary med ekspertkildedata
            
        Returns:
            ExpertSource: Oprettet ekspertkilde
        """
        contact = None
        if "contact" in data and data["contact"]:
            if isinstance(data["contact"], dict):
                contact = ExpertContact(**data["contact"])
            else:
                # Hvis contact bare er en streng
                contact = ExpertContact(type="unknown", value=str(data["contact"]))
        
        return cls(
            name=data.get("name", data.get("navn", "")),
            title=data.get("title", data.get("titel", "")),
            organization=data.get("organization", data.get("organisation", "")),
            expertise=data.get("expertise", data.get("ekspertise", "")),
            contact=contact,
            relevance=data.get("relevance", data.get("relevans", ""))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Konverter til dictionary.
        
        Returns:
            Dict: Dictionary-repræsentation
        """
        result = {
            "name": self.name,
            "title": self.title,
            "organization": self.organization,
            "expertise": self.expertise,
            "relevance": self.relevance
        }
        
        if self.contact:
            result["contact"] = {
                "type": self.contact.type,
                "value": self.contact.value,
                "notes": self.contact.notes
            }
        
        return result


@dataclass
class Institution:
    """Model for en institution."""
    name: str = ""
    type: str = ""  # universitet, myndighed, NGO, etc.
    relevance: str = ""
    contact_person: str = ""
    contact_info: str = ""
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Institution':
        """
        Opret en Institution fra en dict.
        
        Args:
            data: Dictionary med institutionsdata
            
        Returns:
            Institution: Oprettet institution
        """
        return cls(
            name=data.get("name", data.get("navn", "")),
            type=data.get("type", ""),
            relevance=data.get("relevance", data.get("relevans", "")),
            contact_person=data.get("contact_person", data.get("kontaktperson", "")),
            contact_info=data.get("contact", data.get("kontakt", ""))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Konverter til dictionary.
        
        Returns:
            Dict: Dictionary-repræsentation
        """
        return {
            "name": self.name,
            "type": self.type,
            "relevance": self.relevance,
            "contact_person": self.contact_person,
            "contact_info": self.contact_info
        }


@dataclass
class DataSource:
    """Model for en datakilde."""
    title: str = ""
    publisher: str = ""
    description: str = ""
    link: str = ""
    last_updated: str = ""
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataSource':
        """
        Opret en DataSource fra en dict.
        
        Args:
            data: Dictionary med datakildedata
            
        Returns:
            DataSource: Oprettet datakilde
        """
        return cls(
            title=data.get("title", data.get("titel", "")),
            publisher=data.get("publisher", data.get("udgiver", "")),
            description=data.get("description", data.get("beskrivelse", "")),
            link=data.get("link", data.get("url", "")),
            last_updated=data.get("last_updated", data.get("senest_opdateret", ""))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Konverter til dictionary.
        
        Returns:
            Dict: Dictionary-repræsentation
        """
        return {
            "title": self.title,
            "publisher": self.publisher,
            "description": self.description,
            "link": self.link,
            "last_updated": self.last_updated
        }


@dataclass
class ExpertSourceResult:
    """Samlet resultat fra ekspertkilde-indhentning."""
    experts: List[ExpertSource] = field(default_factory=list)
    institutions: List[Institution] = field(default_factory=list)
    data_sources: List[DataSource] = field(default_factory=list)
    error: str = ""
    
    @property
    def has_error(self) -> bool:
        """Om resultatet indeholder en fejl."""
        return bool(self.error)
    
    @property
    def is_empty(self) -> bool:
        """Om resultatet er tomt (ingen data)."""
        return not (self.experts or self.institutions or self.data_sources)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExpertSourceResult':
        """
        Opret et ExpertSourceResult fra en dict.
        
        Args:
            data: Dictionary med ekspertkildedata
            
        Returns:
            ExpertSourceResult: Oprettet resultat
        """
        experts = []
        institutions = []
        data_sources = []
        error = data.get("error", "")
        
        # Parse experts
        raw_experts = data.get("experts", data.get("eksperter", []))
        if isinstance(raw_experts, list):
            for item in raw_experts:
                try:
                    experts.append(ExpertSource.from_dict(item))
                except Exception as e:
                    logger.warning(f"Error parsing expert: {e}")
        
        # Parse institutions
        raw_institutions = data.get("institutions", data.get("institutioner", []))
        if isinstance(raw_institutions, list):
            for item in raw_institutions:
                try:
                    institutions.append(Institution.from_dict(item))
                except Exception as e:
                    logger.warning(f"Error parsing institution: {e}")
        
        # Parse data sources
        raw_data_sources = data.get("data_sources", data.get("datakilder", []))
        if isinstance(raw_data_sources, list):
            for item in raw_data_sources:
                try:
                    data_sources.append(DataSource.from_dict(item))
                except Exception as e:
                    logger.warning(f"Error parsing data source: {e}")
        
        return cls(
            experts=experts,
            institutions=institutions,
            data_sources=data_sources,
            error=error
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Konverter til dictionary.
        
        Returns:
            Dict: Dictionary-repræsentation
        """
        return {
            "experts": [expert.to_dict() for expert in self.experts],
            "institutions": [institution.to_dict() for institution in self.institutions],
            "data_sources": [source.to_dict() for source in self.data_sources],
            "error": self.error
        }


# Cache for ekspertkilderesultater
_expert_source_cache: Dict[str, Tuple[ExpertSourceResult, float]] = {}


class ExpertSourceManager:
    """Manager til at indhente og validere ekspertkilder."""
    
    def __init__(self, api_client=None, cache_ttl: int = 86400):
        """
        Initialize expert source manager.
        
        Args:
            api_client: Client til at kalde API'er (optional)
            cache_ttl: Cache time-to-live i sekunder (default: 24 timer)
        """
        self.api_client = api_client
        self.cache_ttl = cache_ttl
    
    async def get_expert_sources(
        self,
        topic: str,
        angle_headline: str,
        angle_description: str,
        bypass_cache: bool = False,
        progress_callback: Optional[Callable] = None
    ) -> ExpertSourceResult:
        """
        Indhent ekspertkilder til en nyhedsvinkel.
        
        Args:
            topic: Nyhedsemnet
            angle_headline: Overskrift på vinklen
            angle_description: Beskrivelse af vinklen
            bypass_cache: Om cache skal ignoreres
            progress_callback: Callback-funktion til fremskridtsrapportering
            
        Returns:
            ExpertSourceResult: Resultat med ekspertkilder
        """
        # Create cache key
        cache_key = f"expert_sources:{topic}:{angle_headline}"
        stats = get_performance_stats()
        
        # Check cache first
        if not bypass_cache and cache_key in _expert_source_cache:
            result, timestamp = _expert_source_cache[cache_key]
            
            # Check if cache is still valid
            if time.time() - timestamp < self.cache_ttl:
                logger.info(f"Using cached expert sources for '{angle_headline}'")
                stats.record_cache_access(True)
                return result
        
        stats.record_cache_access(False)
        
        # If we have a direct API client reference, use it
        if self.api_client and hasattr(self.api_client, "generate_expert_source_suggestions"):
            start_time = time.time()
            
            try:
                # Call the API client
                api_result = await self.api_client.generate_expert_source_suggestions(
                    topic=topic,
                    angle_headline=angle_headline,
                    angle_description=angle_description,
                    bypass_cache=bypass_cache,
                    progress_callback=progress_callback
                )
                
                elapsed_time = time.time() - start_time
                
                # Parse and validate result
                if isinstance(api_result, dict):
                    result = ExpertSourceResult.from_dict(api_result)
                    
                    # Cache the result
                    _expert_source_cache[cache_key] = (result, time.time())
                    
                    stats.record_expert_sources(True, elapsed_time)
                    return result
                else:
                    logger.warning(f"Invalid API result format: {type(api_result)}")
                    error_result = ExpertSourceResult(error="Invalid API result format")
                    stats.record_expert_sources(False, elapsed_time)
                    return error_result
                    
            except Exception as e:
                elapsed_time = time.time() - start_time
                logger.error(f"Error fetching expert sources: {e}")
                error_result = ExpertSourceResult(error=f"API error: {str(e)}")
                stats.record_expert_sources(False, elapsed_time)
                return error_result
        
        # If no direct API client, return empty result with error
        logger.warning("No API client available for expert sources")
        return ExpertSourceResult(error="No API client available")
    
    def validate_expert_sources(self, result: ExpertSourceResult) -> Tuple[bool, List[str]]:
        """
        Valider ekspertkilder for kvalitet og fuldstændighed.
        
        Args:
            result: Ekspertkilderesultat at validere
            
        Returns:
            Tuple med:
            - Bool der angiver om validering bestod
            - Liste af fejl/advarsler
        """
        issues = []
        
        # Check for error
        if result.has_error:
            issues.append(f"Error in result: {result.error}")
            return False, issues
        
        # Check for empty result
        if result.is_empty:
            issues.append("Result contains no expert sources, institutions, or data sources")
            return False, issues
        
        # Validate experts
        for i, expert in enumerate(result.experts):
            if not expert.name:
                issues.append(f"Expert #{i+1} has no name")
            if not expert.organization:
                issues.append(f"Expert #{i+1} ({expert.name}) has no organization")
        
        # Validate institutions
        for i, institution in enumerate(result.institutions):
            if not institution.name:
                issues.append(f"Institution #{i+1} has no name")
        
        # Validate data sources
        for i, source in enumerate(result.data_sources):
            if not source.title:
                issues.append(f"Data source #{i+1} has no title")
        
        # Consider validation passed if there are no critical issues
        passed = len(issues) == 0
        
        return passed, issues


# Singleton instance for convenience
_manager = ExpertSourceManager()


async def get_expert_sources(
    topic: str,
    angle_headline: str,
    angle_description: str,
    api_client=None,
    bypass_cache: bool = False,
    progress_callback: Optional[Callable] = None
) -> ExpertSourceResult:
    """
    Convenience-funktion til at indhente ekspertkilder uden at oprette en manager-instans.
    
    Args:
        topic: Nyhedsemnet
        angle_headline: Overskrift på vinklen
        angle_description: Beskrivelse af vinklen
        api_client: Client til at kalde API'er (optional)
        bypass_cache: Om cache skal ignoreres
        progress_callback: Callback-funktion til fremskridtsrapportering
        
    Returns:
        ExpertSourceResult: Resultat med ekspertkilder
    """
    global _manager
    
    # If API client provided, create a new manager instance
    if api_client and api_client is not _manager.api_client:
        manager = ExpertSourceManager(api_client)
    else:
        # Use the singleton instance
        manager = _manager
        
        # Update the API client if provided
        if api_client:
            manager.api_client = api_client
    
    return await manager.get_expert_sources(
        topic=topic,
        angle_headline=angle_headline,
        angle_description=angle_description,
        bypass_cache=bypass_cache,
        progress_callback=progress_callback
    )


def validate_expert_sources(result: ExpertSourceResult) -> Tuple[bool, List[str]]:
    """
    Convenience-funktion til at validere ekspertkilder uden at oprette en manager-instans.
    
    Args:
        result: Ekspertkilderesultat at validere
        
    Returns:
        Tuple med:
        - Bool der angiver om validering bestod
        - Liste af fejl/advarsler
    """
    return _manager.validate_expert_sources(result)


def parse_expert_sources_json(json_str: str, context: str = "expert sources") -> ExpertSourceResult:
    """
    Parse og valider JSON-strengen til et ExpertSourceResult-objekt.
    
    Args:
        json_str: JSON-streng med ekspertkildedata
        context: Kontekst til fejlbeskeder
        
    Returns:
        ExpertSourceResult: Parseret resultat
    """
    # Use the safe JSON parser
    parsed_data = safe_parse_json(
        json_str, 
        context=context,
        fallback={
            "experts": [],
            "institutions": [],
            "data_sources": [],
            "error": "Failed to parse expert sources JSON"
        }
    )
    
    # If there was a parsing error, include it in the result
    if "error" in parsed_data:
        error = parsed_data.pop("error", "")
        parsed_data["error"] = error
    
    # Convert to ExpertSourceResult
    return ExpertSourceResult.from_dict(parsed_data)


def clear_expertise_cache() -> int:
    """
    Ryd ekspertkilde-cachen.
    
    Returns:
        int: Antal fjernede cache-elementer
    """
    global _expert_source_cache
    count = len(_expert_source_cache)
    _expert_source_cache.clear()
    logger.info(f"Cleared expert source cache ({count} items)")
    return count