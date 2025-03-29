"""
Integration tests for the full Vinkeljernet workflow.
"""

import pytest
import asyncio
import json
import os
from unittest.mock import patch, AsyncMock
from pathlib import Path
from pydantic import field_validator

from api_clients import fetch_topic_information, generate_angles
from angle_processor import filter_and_rank_angles
from config_loader import load_and_validate_profile


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_workflow(sample_yaml_path, mock_perplexity_response, mock_openai_response):
    """Test the full workflow from profile loading to angle filtering."""
    # Patch API functions to use mocks
    with patch('aiohttp.ClientSession') as mock_session, \
         patch('api_clients.create_secure_api_session'), \
         patch('api_clients.OpenAI', return_value=mock_openai_response), \
         patch('api_clients.parse_angles_from_response', return_value=[
             {
                 "overskrift": "Test Headline 1",
                 "beskrivelse": "Test description 1",
                 "begrundelse": "Test rationale 1",
                 "nyhedskriterier": ["Aktualitet", "Konflikt"],
                 "startSpørgsmål": ["Question 1?", "Question 2?"]
             },
             {
                 "overskrift": "Test Headline 2",
                 "beskrivelse": "Test description 2",
                 "begrundelse": "Test rationale 2",
                 "nyhedskriterier": ["Væsentlighed", "Identifikation"],
                 "startSpørgsmål": ["Question 3?", "Question 4?"]
             }
         ]):
         
        # Setup session mock
        session_instance = mock_session.return_value
        session_instance.__aenter__.return_value = session_instance
        session_instance.post.return_value.__aenter__.return_value = mock_perplexity_response
        
        # 1. Load profile
        profile = load_and_validate_profile(sample_yaml_path)
        
        # 2. Fetch topic information
        topic_info = await fetch_topic_information("climate change", dev_mode=True)
        
        # 3. Generate angles
        angles = generate_angles("climate change", topic_info, profile)
        
        # 4. Filter and rank angles
        ranked_angles = filter_and_rank_angles(angles, profile, num_angles=2)
        
        # Verify the full workflow results
        assert len(ranked_angles) == 2
        assert "Test Headline" in ranked_angles[0]["overskrift"]
        assert isinstance(ranked_angles[0]["calculatedScore"], (int, float))


@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow_with_perplexity_failure(sample_yaml_path, mock_openai_response):
    """Test the workflow when Perplexity API fails but OpenAI still works."""
    # Setup mocks where Perplexity fails but OpenAI works
    with patch('aiohttp.ClientSession') as mock_session, \
         patch('api_clients.create_secure_api_session'), \
         patch('api_clients.OpenAI', return_value=mock_openai_response), \
         patch('api_clients.parse_angles_from_response', return_value=[
             {
                 "overskrift": "Test Headline",
                 "beskrivelse": "Test description",
                 "begrundelse": "Test rationale",
                 "nyhedskriterier": ["Aktualitet", "Konflikt"],
                 "startSpørgsmål": ["Question 1?", "Question 2?"]
             }
         ]):
         
        # Setup session mock to fail
        session_instance = mock_session.return_value
        session_instance.__aenter__.return_value = session_instance
        session_instance.post.side_effect = Exception("API Error")
        
        # 1. Load profile
        profile = load_and_validate_profile(sample_yaml_path)
        
        # 2. Fetch topic information (will fail)
        topic_info = await fetch_topic_information("climate change", dev_mode=True)
        assert topic_info is None  # Should be None due to failure
        
        # 3. Generate angles (should still work with None topic_info)
        angles = generate_angles("climate change", topic_info, profile)
        
        # 4. Filter and rank angles
        ranked_angles = filter_and_rank_angles(angles, profile, num_angles=1)
        
        # Verify we still got results despite Perplexity failure
        assert len(ranked_angles) == 1
        assert "Test Headline" in ranked_angles[0]["overskrift"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow_with_openai_failure(sample_yaml_path, mock_perplexity_response):
    """Test the workflow when OpenAI API fails."""
    # Setup mocks where Perplexity works but OpenAI fails
    with patch('aiohttp.ClientSession') as mock_session, \
         patch('api_clients.create_secure_api_session', return_value=AsyncMock()), \
         patch('api_clients.fetch_topic_information', return_value="Mocked topic information"), \
         patch('api_clients.OpenAI') as mock_openai:
         
        # Setup OpenAI to fail
        mock_openai.return_value.chat.completions.create.side_effect = Exception("OpenAI API Error")
        
        # 1. Load profile
        profile = load_and_validate_profile(sample_yaml_path)
        
        # 2. Get topic info directly from our mock
        topic_info = "Mocked topic information"  # Skip the actual API call
        assert topic_info is not None  # Should have info
        
        # 3. Generate angles (will fail)
        angles = generate_angles("climate change", topic_info, profile)
        assert angles == []  # Should be empty due to failure
        
        # 4. Can't filter empty angles
        ranked_angles = filter_and_rank_angles(angles, profile, num_angles=1)
        assert ranked_angles == []  # Should also be empty