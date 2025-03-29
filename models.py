"""
Models module for Vinkeljernet project.

This module defines Pydantic models that represent data structures used
throughout the application, particularly for validating YAML configurations.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Dict

class RedaktionelDNA(BaseModel):
    """
    Represents the editorial DNA/profile of a news outlet.
    
    This model captures the core principles, news prioritization criteria,
    tone and style, focus areas, and no-go areas that define the editorial
    identity and approach of a news publication.
    
    Example from YAML:
        kerneprincipper:
          - Aktualitet
          - Folkelig relevans
        nyhedsprioritering:
          sensation: 5
          identifikation: 4
          konflikt: 4
        tone_og_stil: "Direkte, letforståelig"
        fokusområder:
          - Politisk magt
        nogo_områder: []
    """
    kerneprincipper: List[Dict[str, str]] = Field(
        description="Core principles guiding editorial decisions"
    )
    nyhedsprioritering: Dict[str, int] = Field(
        description="News prioritization criteria with scores from 1-5"
    )
    tone_og_stil: str = Field(
        min_length=1,
        description="Tone and style of the editorial content"
    )
    fokusområder: List[str] = Field(
        default_factory=list,
        description="Areas of special focus for the publication"
    )
    nogo_områder: List[str] = Field(
        default_factory=list,
        description="Topics or areas the publication avoids covering"
    )

    @field_validator('kerneprincipper')
    def validate_kerneprincipper(cls, v):
        if not v:
            raise ValueError("At least one core principle must be provided")
        return v

    @field_validator('nyhedsprioritering')
    def validate_nyhedsprioritering_values(cls, v):
        for key, score in v.items():
            if not 1 <= score <= 5:
                raise ValueError(f"Score for '{key}' must be between 1 and 5, got {score}")
        return v
        
    @field_validator('tone_og_stil')
    def validate_tone_og_stil(cls, v):
        if not v.strip():
            raise ValueError("Tone and style must not be empty")
        return v