"""
Unit tests for the configuration loading and validation.
"""

import pytest
from pathlib import Path
import yaml

from config_loader import load_and_validate_profile


def test_load_valid_yaml_profile(sample_yaml_path):
    """Test loading a valid YAML profile."""
    profile = load_and_validate_profile(sample_yaml_path)
    
    # Check that the profile was loaded correctly
    assert len(profile.kerneprincipper) == 2
    assert profile.tone_og_stil == "Direkte, skarp, kontant og letforståelig tone."
    assert profile.nyhedsprioritering["Aktualitet"] == 5
    assert len(profile.fokusområder) == 3
    assert len(profile.nogo_områder) == 2


def test_load_nonexistent_yaml_file():
    """Test loading a nonexistent YAML file."""
    with pytest.raises(FileNotFoundError):
        load_and_validate_profile(Path("nonexistent_file.yaml"))


def test_load_invalid_yaml_file(tmp_path):
    """Test loading an invalid YAML file."""
    invalid_yaml = tmp_path / "invalid.yaml"
    invalid_yaml.write_text("this: is: not: valid: yaml:")
    
    with pytest.raises(yaml.YAMLError):
        load_and_validate_profile(invalid_yaml)


def test_missing_required_field(tmp_path):
    """Test validation of a YAML file with missing required fields."""
    incomplete_yaml = tmp_path / "incomplete.yaml"
    incomplete_yaml.write_text("""
    # Missing tone_og_stil field
    kerneprincipper:
      - Afslørende: Description
    nyhedsprioritering:
      Aktualitet: 5
    fokusområder:
      - Politik
    nogo_områder:
      - None
    """)
    
    with pytest.raises(ValueError):
        load_and_validate_profile(incomplete_yaml)


def test_invalid_kerneprincipper_format(tmp_path):
    """Test validation of invalid kerneprincipper format."""
    invalid_yaml = tmp_path / "invalid_principles.yaml"
    invalid_yaml.write_text("""
    kerneprincipper:
      - Just a string, not a key-value pair
    tone_og_stil: Test
    nyhedsprioritering:
      Aktualitet: 5
    fokusområder:
      - Politik
    nogo_områder:
      - None
    """)
    
    with pytest.raises(ValueError):
        load_and_validate_profile(invalid_yaml)


def test_invalid_nyhedsprioritering_values(tmp_path):
    """Test validation of invalid nyhedsprioritering values."""
    invalid_yaml = tmp_path / "invalid_priorities.yaml"
    invalid_yaml.write_text("""
    kerneprincipper:
      - Afslørende: Description
    tone_og_stil: Test
    nyhedsprioritering:
      Aktualitet: "not a number"
    fokusområder:
      - Politik
    nogo_områder:
      - None
    """)
    
    with pytest.raises(ValueError):
        load_and_validate_profile(invalid_yaml)