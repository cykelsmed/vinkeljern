"""
Unit tests for the API clients module.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import aiohttp
import json

from api_clients import fetch_topic_information, generate_angles


@pytest.mark.asyncio
async def test_fetch_topic_information_success(monkeypatch, mock_perplexity_response):
    """Test successful topic information retrieval."""
    # Setup mock for ClientSession
    session_mock = AsyncMock()
    session_mock.__aenter__.return_value = session_mock
    session_mock.post.return_value.__aenter__.return_value = mock_perplexity_response
    
    # Patch ClientSession and create_secure_api_session
    with patch('aiohttp.ClientSession', return_value=session_mock), \
         patch('api_clients.create_secure_api_session', return_value=session_mock):
        
        # Call the function
        result = await fetch_topic_information("climate change", dev_mode=True)
        
        # Verify the result
        assert result == "Sample research content about climate change"
        
        # Verify that the API was called with the right parameters
        session_mock.post.assert_called_once()
        args, kwargs = session_mock.post.call_args
        assert "perplexity.ai" in args[0]
        assert "climate change" in str(kwargs)


@pytest.mark.asyncio
async def test_fetch_topic_information_error(monkeypatch, mock_perplexity_error_response):
    """Test error handling in topic information retrieval."""
    # Setup mock for ClientSession
    session_mock = AsyncMock()
    session_mock.__aenter__.return_value = session_mock
    session_mock.post.return_value.__aenter__.return_value = mock_perplexity_error_response
    
    # Patch ClientSession and create_secure_api_session
    with patch('aiohttp.ClientSession', return_value=session_mock), \
         patch('api_clients.create_secure_api_session', return_value=session_mock), \
         patch('api_clients.log_error') as mock_log_error, \
         patch('api_clients.display_api_response_error') as mock_display_error:
        
        # Call the function
        result = await fetch_topic_information("climate change", dev_mode=True)
        
        # Verify the result is None due to the error
        assert result is None
        
        # Verify error logging was called
        mock_display_error.assert_called_once()


def test_generate_angles_success(monkeypatch, sample_profile, sample_topic_info, mock_openai_response):
    """Test successful angle generation."""
    # Patch OpenAI client
    with patch('api_clients.OpenAI', return_value=mock_openai_response), \
         patch('api_clients.parse_angles_from_response', return_value=[
             {
                 "overskrift": "Test Headline",
                 "beskrivelse": "Test description",
                 "begrundelse": "Test rationale",
                 "nyhedskriterier": ["Aktualitet", "Konflikt"],
                 "startSpørgsmål": ["Question 1?", "Question 2?"]
             }
         ]):
        
        # Call the function
        result = generate_angles("climate change", sample_topic_info, sample_profile)
        
        # Verify results
        assert len(result) == 1
        assert result[0]["overskrift"] == "Test Headline"
        assert "Aktualitet" in result[0]["nyhedskriterier"]


def test_generate_angles_with_none_topic_info(monkeypatch, sample_profile, mock_openai_response):
    """Test angle generation with None topic_info."""
    # Patch OpenAI client
    with patch('api_clients.OpenAI', return_value=mock_openai_response), \
         patch('api_clients.parse_angles_from_response', return_value=[
             {
                 "overskrift": "Test Headline",
                 "beskrivelse": "Test description",
                 "begrundelse": "Test rationale",
                 "nyhedskriterier": ["Aktualitet", "Konflikt"],
                 "startSpørgsmål": ["Question 1?", "Question 2?"]
             }
         ]):
        
        # Call the function with None topic_info
        result = generate_angles("climate change", None, sample_profile)
        
        # Verify results - the function should handle None topic_info gracefully
        assert len(result) == 1
        assert result[0]["overskrift"] == "Test Headline"


def test_generate_angles_error(monkeypatch, sample_profile, sample_topic_info, mock_openai_error):
    """Test error handling in angle generation."""
    # Patch OpenAI client with error
    with patch('api_clients.OpenAI', return_value=mock_openai_error), \
         patch('rich.print') as mock_print:
        
        # Call the function
        result = generate_angles("climate change", sample_topic_info, sample_profile)
        
        # Verify empty result due to error
        assert result == []
        
        # Verify error was printed
        mock_print.assert_called()