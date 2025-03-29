"""
Unit tests for angle processing functionality.
"""

import pytest
from angle_processor import filter_and_rank_angles, hits_nogo_areas, calculate_angle_score


def test_hits_nogo_areas():
    """Test detection of no-go areas in angles."""
    # Angle that hits a no-go area
    angle_hits = {
        "overskrift": "Kendis skandale afsløret",
        "beskrivelse": "Privat information om kendte personer lækket."
    }
    
    # Angle that doesn't hit a no-go area
    angle_safe = {
        "overskrift": "Klimaforandringer påvirker landbrug",
        "beskrivelse": "Nye data viser effekter af klimaforandringer på landbruget."
    }
    
    nogo_areas = [
        "Privat information om kendte personer",
        "Usaglige personangreb"
    ]
    
    # Test that the first angle hits a no-go area
    assert hits_nogo_areas(angle_hits, nogo_areas) is True
    
    # Test that the second angle doesn't hit a no-go area
    assert hits_nogo_areas(angle_safe, nogo_areas) is False


def test_calculate_angle_score():
    """Test angle score calculation."""
    # Define test angle with news criteria
    angle = {
        "nyhedskriterier": ["Aktualitet", "Konflikt", "Sensation"]
    }
    
    # Define news criteria priorities
    priorities = {
        "Aktualitet": 5,
        "Væsentlighed": 4,
        "Konflikt": 3,
        "Identifikation": 2,
        "Sensation": 1
    }
    
    # Calculate expected score: 5 (Aktualitet) + 3 (Konflikt) + 1 (Sensation) = 9
    expected_score = 9
    
    # Test the score calculation
    score = calculate_angle_score(angle, priorities)
    assert score == expected_score
    
    # Test with an angle that already has a kriterieScore
    angle_with_score = {
        "kriterieScore": 10,
        "nyhedskriterier": ["Aktualitet", "Konflikt"]
    }
    
    # Expected: 10 (base) + 5 (Aktualitet) + 3 (Konflikt) = 18
    score_with_base = calculate_angle_score(angle_with_score, priorities)
    assert score_with_base == 18
    
    # Test with no matching criteria
    angle_no_match = {
        "nyhedskriterier": ["NotInPriorities"]
    }
    score_no_match = calculate_angle_score(angle_no_match, priorities)
    assert score_no_match == 0


def test_filter_and_rank_angles(sample_profile, sample_angles):
    """Test filtering and ranking angles."""
    # Add an angle that hits a no-go area
    nogo_angle = {
        "overskrift": "Privatlivets fred krænket",
        "beskrivelse": "Indhold der krænker privatlivets fred uden samfundsmæssig relevans.",
        "nyhedskriterier": ["Sensation"],
        "begrundelse": "Test",
        "startSpørgsmål": ["Question?"]
    }
    
    test_angles = sample_angles + [nogo_angle]
    
    # Test filtering and ranking
    result = filter_and_rank_angles(test_angles, sample_profile, num_angles=2)
    
    # Should return at least 1 angle and not include the nogo_angle
    assert len(result) >= 1
    for angle in result:
        assert "privatlivets fred" not in angle["overskrift"].lower()
        assert "privatlivets fred" not in angle["beskrivelse"].lower()


def test_filter_all_angles_hit_nogo(sample_profile):
    """Test case when all angles hit no-go areas."""
    all_nogo_angles = [
        {
            "overskrift": "Privatlivets fred krænket",
            "beskrivelse": "Indhold der krænker privatlivets fred uden samfundsmæssig relevans.",
            "nyhedskriterier": ["Sensation"],
            "begrundelse": "Test",
            "startSpørgsmål": ["Question?"]
        },
        {
            "overskrift": "Anonyme kilder hævder",
            "beskrivelse": "Historier baseret på anonyme kilder uden faktuel verifikation",
            "nyhedskriterier": ["Sensation"],
            "begrundelse": "Test",
            "startSpørgsmål": ["Question?"]
        }
    ]
    
    # Test filtering and ranking when all angles hit no-go areas
    result = filter_and_rank_angles(all_nogo_angles, sample_profile, num_angles=2)
    
    # Should return empty list as all angles hit no-go areas
    assert result == []


def test_ensure_diverse_angles(sample_angles, sample_profile):  # Add sample_profile parameter
    """Test that angle diversity is maintained."""
    # Create two very similar angles
    similar_angles = [
        {
            "overskrift": "Klimaforandringer truer Danish farming",
            "beskrivelse": "Danish farmers are affected by climate change",
            "nyhedskriterier": ["Væsentlighed", "Konflikt"],
            "begrundelse": "Test",
            "startSpørgsmål": ["Question?"]
        },
        {
            "overskrift": "Danish climate changes affect farmers",
            "beskrivelse": "Climate changes are affecting Danish farmers",
            "nyhedskriterier": ["Væsentlighed", "Konflikt"],
            "begrundelse": "Test",
            "startSpørgsmål": ["Question?"]
        }
    ]
    
    # Add a very different angle
    diverse_angle = {
        "overskrift": "New COVID variant discovered",
        "beskrivelse": "Scientists have discovered a new COVID variant",
        "nyhedskriterier": ["Aktualitet", "Sensation"],
        "begrundelse": "Test",
        "startSpørgsmål": ["Question?"]
    }
    
    # Create test angles with similar angles plus one diverse angle
    test_angles = similar_angles + [diverse_angle]
    
    # Modify the angles to have same score for testing diversity only
    for angle in test_angles:
        angle["calculatedScore"] = 10

    # Filter and rank, selecting 2 angles
    result = filter_and_rank_angles(test_angles, sample_profile, num_angles=2)
    
    # The result should include the diverse angle and one of the similar angles
    assert len(result) == 2
    
    # Check that we have one angle about COVID and one about climate/farmers
    covid_angles = [a for a in result if "COVID" in a["overskrift"]]
    climate_angles = [a for a in result if "climate" in a["overskrift"].lower() or "klima" in a["overskrift"].lower()]
    
    assert len(covid_angles) == 1
    assert len(climate_angles) == 1