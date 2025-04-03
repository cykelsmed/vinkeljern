"""
Tests for the core module of the Vinkeljernet application.

This module contains unit tests for the core functionality of the application.
"""

import asyncio
import unittest
from unittest import mock
from pathlib import Path
from typing import Dict, List, Any

from models import RedaktionelDNA
from vinkeljernet.core import safe_process_angles


class TestCore(unittest.TestCase):
    """Test cases for the core module."""

    def test_safe_process_angles_normal(self):
        """Test safe_process_angles with normal input."""
        # Mock profile
        profile = mock.MagicMock(spec=RedaktionelDNA)
        
        # Mock angles
        angles = [
            {
                "overskrift": "Test Headline 1",
                "beskrivelse": "Test description 1",
                "nyhedskriterier": ["aktualitet", "væsentlighed"],
                "begrundelse": "Test rationale 1"
            },
            {
                "overskrift": "Test Headline 2",
                "beskrivelse": "Test description 2",
                "nyhedskriterier": ["aktualitet"],
                "begrundelse": "Test rationale 2"
            },
            {
                "overskrift": "Test Headline 3",
                "beskrivelse": "Test description 3",
                "nyhedskriterier": ["væsentlighed", "konflikt", "identifikation"],
                "begrundelse": "Test rationale 3"
            }
        ]
        
        # Mock filter_and_rank_angles
        with mock.patch('vinkeljernet.core.filter_and_rank_angles') as mock_filter:
            # Set up the mock to return a subset of angles
            mock_filter.return_value = angles[:2]
            
            # Call the function
            result = safe_process_angles(angles, profile, 2)
            
            # Verify the result
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["overskrift"], "Test Headline 1")
            self.assertEqual(result[1]["overskrift"], "Test Headline 2")
            
            # Verify filter_and_rank_angles was called with the right arguments
            mock_filter.assert_called_once_with(angles, profile, 2)

    def test_safe_process_angles_error(self):
        """Test safe_process_angles with error in filter_and_rank_angles."""
        # Mock profile
        profile = mock.MagicMock(spec=RedaktionelDNA)
        
        # Mock angles
        angles = [
            {
                "overskrift": "Test Headline 1",
                "beskrivelse": "Test description 1",
                "nyhedskriterier": ["aktualitet", "væsentlighed"],
                "begrundelse": "Test rationale 1"
            },
            {
                "overskrift": "Test Headline 2",
                "beskrivelse": "Test description 2",
                "nyhedskriterier": ["aktualitet"],
                "begrundelse": "Test rationale 2"
            },
            {
                "overskrift": "Test Headline 3",
                "beskrivelse": "Test description 3",
                "nyhedskriterier": ["væsentlighed", "konflikt", "identifikation"],
                "begrundelse": "Test rationale 3"
            }
        ]
        
        # Mock filter_and_rank_angles to raise an AttributeError
        with mock.patch('vinkeljernet.core.filter_and_rank_angles') as mock_filter:
            mock_filter.side_effect = AttributeError("Test error")
            
            # Call the function
            result = safe_process_angles(angles, profile, 2)
            
            # Verify the result - should fall back to sorting by number of criteria
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["overskrift"], "Test Headline 3")  # Has 3 criteria
            self.assertEqual(result[1]["overskrift"], "Test Headline 1")  # Has 2 criteria


if __name__ == "__main__":
    unittest.main()