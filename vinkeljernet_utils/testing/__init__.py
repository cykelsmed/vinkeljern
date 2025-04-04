"""
Vinkeljernet Utils - Testing Module

Dette modul indeholder testværktøjer til validering af JSON-parsing og ekspertkilde-funktionalitet.
"""

import asyncio
import json
import logging
import os
import unittest
from typing import Dict, List, Any, Optional, Tuple, Callable

from vinkeljernet_utils.json_parsing import (
    robust_json_parse,
    safe_parse_json,
    JSONParsingConfig
)

from vinkeljernet_utils.expertise import (
    ExpertSourceResult,
    ExpertSource,
    Institution,
    DataSource
)

logger = logging.getLogger("vinkeljernet_utils.testing")


class MockAPIClient:
    """Mock API-klient til test af ekspertkilde-indhentning."""
    
    def __init__(self, response_data: Dict[str, Any] = None, simulate_error: bool = False):
        """
        Initialiser mock-klienten.
        
        Args:
            response_data: Foruddefineret svarsdata
            simulate_error: Om der skal simuleres en fejl
        """
        self.response_data = response_data or {
            "experts": [
                {
                    "name": "Test Expert",
                    "title": "Professor",
                    "organization": "Test University",
                    "expertise": "Testing",
                    "contact": {"type": "email", "value": "test@example.com"},
                    "relevance": "High"
                }
            ],
            "institutions": [
                {
                    "name": "Test Institution",
                    "type": "University",
                    "relevance": "High",
                    "contact_person": "Test Contact",
                    "contact": "info@example.com"
                }
            ],
            "data_sources": [
                {
                    "title": "Test Data",
                    "publisher": "Test Publisher",
                    "description": "Test description",
                    "link": "https://example.com",
                    "last_updated": "2023-01-01"
                }
            ]
        }
        self.simulate_error = simulate_error
        self.called_with = []
    
    async def generate_expert_source_suggestions(
        self,
        topic: str,
        angle_headline: str,
        angle_description: str,
        bypass_cache: bool = False,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Mock-implementering af generate_expert_source_suggestions.
        
        Args:
            topic: Nyhedsemnet
            angle_headline: Overskrift på vinklen
            angle_description: Beskrivelse af vinklen
            bypass_cache: Om cache skal ignoreres
            progress_callback: Callback-funktion til fremskridtsrapportering
            
        Returns:
            Dict: Mockdata eller fejl
        """
        # Record the call
        self.called_with.append({
            "topic": topic,
            "angle_headline": angle_headline,
            "angle_description": angle_description,
            "bypass_cache": bypass_cache
        })
        
        # Update progress if callback provided
        if progress_callback:
            await progress_callback(50)
        
        # Simulate delay
        await asyncio.sleep(0.1)
        
        # Simulate error if requested
        if self.simulate_error:
            raise RuntimeError("Simulated API error")
        
        # Update progress if callback provided
        if progress_callback:
            await progress_callback(100)
        
        # Return predefined response data
        return self.response_data


def load_test_json_samples() -> Dict[str, str]:
    """
    Indlæs JSON-testeksempler fra testmappe.
    
    Returns:
        Dict: Dictionary med testeksempler
    """
    samples = {}
    
    # Define sample JSON strings
    samples["valid"] = '{"key1": "value1", "key2": 123}'
    samples["array"] = '[{"name": "item1"}, {"name": "item2"}]'
    samples["with_prefix"] = 'Here is the result: {"key1": "value1"}'
    samples["with_markdown"] = '```json\n{"key1": "value1"}\n```'
    samples["invalid"] = '{key1: value1, key2: 123}'  # Missing quotes
    samples["empty"] = ''
    samples["single_quote"] = "{'key1': 'value1', 'key2': 123}"  # Python-style quotes
    samples["mixed"] = '{"key1": null, "key2": None}'  # Mix of null and None
    samples["trailing_comma"] = '{"key1": "value1", "key2": 123,}'  # Invalid trailing comma
    samples["python_booleans"] = '{"key1": True, "key2": False}'  # Python-style booleans
    
    # Expert sources examples
    samples["expert_valid"] = json.dumps({
        "experts": [
            {
                "name": "Dr. Jane Smith",
                "title": "Professor of Economics",
                "organization": "Copenhagen University",
                "expertise": "Macroeconomics",
                "contact": {"type": "email", "value": "j.smith@example.com"},
                "relevance": "Expert on monetary policy"
            }
        ],
        "institutions": [
            {
                "name": "Danish Economic Council",
                "type": "Government",
                "relevance": "Provides official economic forecasts",
                "contact_person": "Press Office",
                "contact": "press@example.com"
            }
        ],
        "data_sources": [
            {
                "title": "Quarterly Economic Report",
                "publisher": "Statistics Denmark",
                "description": "Official economic statistics",
                "link": "https://example.com/stats",
                "last_updated": "2023-03-15"
            }
        ]
    })
    
    samples["expert_malformed"] = '{"experts": [{"name": "Test Expert", "title": "Professor",' \
                                 '"organization": "Test University",}], "institutions": []}'
    
    samples["expert_empty"] = '{"experts": [], "institutions": [], "data_sources": []}'
    
    return samples


class JSONParsingTests(unittest.TestCase):
    """Testsuite til JSON-parsing-funktioner."""
    
    def setUp(self):
        """Forbered testmiljø."""
        self.samples = load_test_json_samples()
    
    def test_robust_json_parse_valid(self):
        """Test parsing af gyldig JSON."""
        results, error = robust_json_parse(self.samples["valid"], "test")
        self.assertIsNone(error)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["key1"], "value1")
        self.assertEqual(results[0]["key2"], 123)
    
    def test_robust_json_parse_array(self):
        """Test parsing af JSON-array."""
        results, error = robust_json_parse(self.samples["array"], "test")
        self.assertIsNone(error)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["name"], "item1")
        self.assertEqual(results[1]["name"], "item2")
    
    def test_robust_json_parse_with_prefix(self):
        """Test parsing af JSON med præfiks."""
        results, error = robust_json_parse(self.samples["with_prefix"], "test")
        self.assertIsNone(error)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["key1"], "value1")
    
    def test_robust_json_parse_with_markdown(self):
        """Test parsing af JSON i markdown."""
        results, error = robust_json_parse(self.samples["with_markdown"], "test")
        self.assertIsNone(error)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["key1"], "value1")
    
    def test_robust_json_parse_invalid(self):
        """Test parsing af ugyldig JSON."""
        # Either we get results with fix-ups or an error message
        results, error = robust_json_parse(self.samples["invalid"], "test")
        if error is None:
            # If parsing succeeded, validate content
            self.assertTrue(len(results) > 0)
            self.assertTrue(isinstance(results[0], dict))
        else:
            # If parsing failed, ensure error message is informative
            self.assertIn("parsing", error.lower())
    
    def test_robust_json_parse_empty(self):
        """Test parsing af tom string."""
        results, error = robust_json_parse(self.samples["empty"], "test")
        self.assertIsNotNone(error)
        self.assertEqual(len(results), 0)
        self.assertIn("empty", error.lower())
    
    def test_safe_parse_json(self):
        """Test sikker parsing med fallback."""
        # Valid JSON should be parsed correctly
        result = safe_parse_json(self.samples["valid"], "test")
        self.assertEqual(result["key1"], "value1")
        self.assertEqual(result["key2"], 123)
        
        # Invalid JSON should use fallback
        fallback = {"status": "error", "data": []}
        result = safe_parse_json(self.samples["empty"], "test", fallback)
        self.assertEqual(result, fallback)
        self.assertTrue("error" in result)


class ExpertSourceTests(unittest.TestCase):
    """Testsuite til ekspertkildefunktioner."""
    
    def setUp(self):
        """Forbered testmiljø."""
        self.samples = load_test_json_samples()
        
        # Create sample expert sources
        self.expert = ExpertSource(
            name="Test Expert",
            title="Professor",
            organization="Test University",
            expertise="Testing",
            relevance="High"
        )
        
        self.institution = Institution(
            name="Test Institution",
            type="University",
            relevance="High",
            contact_person="Test Contact",
            contact_info="info@example.com"
        )
        
        self.data_source = DataSource(
            title="Test Data",
            publisher="Test Publisher",
            description="Test description",
            link="https://example.com",
            last_updated="2023-01-01"
        )
        
        self.result = ExpertSourceResult(
            experts=[self.expert],
            institutions=[self.institution],
            data_sources=[self.data_source]
        )
    
    def test_expert_source_from_dict(self):
        """Test konvertering af dict til ExpertSource."""
        data = {
            "name": "Test Expert",
            "title": "Professor",
            "organization": "Test University",
            "expertise": "Testing",
            "relevance": "High"
        }
        expert = ExpertSource.from_dict(data)
        self.assertEqual(expert.name, "Test Expert")
        self.assertEqual(expert.title, "Professor")
        self.assertEqual(expert.organization, "Test University")
    
    def test_expert_source_to_dict(self):
        """Test konvertering af ExpertSource til dict."""
        data = self.expert.to_dict()
        self.assertEqual(data["name"], "Test Expert")
        self.assertEqual(data["title"], "Professor")
        self.assertEqual(data["organization"], "Test University")
    
    def test_expert_source_result_from_dict(self):
        """Test konvertering af dict til ExpertSourceResult."""
        data = json.loads(self.samples["expert_valid"])
        result = ExpertSourceResult.from_dict(data)
        self.assertEqual(len(result.experts), 1)
        self.assertEqual(len(result.institutions), 1)
        self.assertEqual(len(result.data_sources), 1)
        self.assertEqual(result.experts[0].name, "Dr. Jane Smith")
    
    def test_expert_source_result_to_dict(self):
        """Test konvertering af ExpertSourceResult til dict."""
        data = self.result.to_dict()
        self.assertIn("experts", data)
        self.assertIn("institutions", data)
        self.assertIn("data_sources", data)
        self.assertEqual(len(data["experts"]), 1)
        self.assertEqual(data["experts"][0]["name"], "Test Expert")
    
    def test_parse_expert_sources_json(self):
        """Test parsing af JSON til ExpertSourceResult."""
        from vinkeljernet_utils.expertise import parse_expert_sources_json
        
        # Valid JSON should be parsed correctly
        result = parse_expert_sources_json(self.samples["expert_valid"])
        self.assertEqual(len(result.experts), 1)
        self.assertEqual(result.experts[0].name, "Dr. Jane Smith")
        
        # Malformed JSON should be handled gracefully
        result = parse_expert_sources_json(self.samples["expert_malformed"])
        self.assertTrue(hasattr(result, "experts"))
        
        # Empty result should be valid but empty
        result = parse_expert_sources_json(self.samples["expert_empty"])
        self.assertEqual(len(result.experts), 0)
        self.assertEqual(len(result.institutions), 0)
        self.assertEqual(len(result.data_sources), 0)


async def run_async_tests():
    """Kør asynkrone tests."""
    from vinkeljernet_utils.expertise import get_expert_sources
    
    # Test get_expert_sources with mock client
    mock_client = MockAPIClient()
    result = await get_expert_sources(
        topic="Test Topic",
        angle_headline="Test Headline",
        angle_description="Test Description",
        api_client=mock_client
    )
    
    # Verify the result
    assert len(result.experts) == 1
    assert result.experts[0].name == "Test Expert"
    assert len(mock_client.called_with) == 1
    assert mock_client.called_with[0]["topic"] == "Test Topic"
    
    # Test error handling
    error_client = MockAPIClient(simulate_error=True)
    error_result = await get_expert_sources(
        topic="Test Topic",
        angle_headline="Test Headline",
        angle_description="Test Description",
        api_client=error_client
    )
    
    # Verify error handling
    assert error_result.has_error
    assert "error" in error_result.error.lower()
    
    return "Async tests completed successfully"


def run_tests(verbose: bool = False) -> Tuple[bool, str]:
    """
    Kør alle tests for pakken.
    
    Args:
        verbose: Om detaljeret output skal vises
        
    Returns:
        Tuple med:
        - Bool der angiver om alle tests bestod
        - Resultatbesked
    """
    # Configure unittest output
    verbosity = 2 if verbose else 1
    
    # Create test suite
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(JSONParsingTests))
    suite.addTest(unittest.makeSuite(ExpertSourceTests))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    # Run async tests
    try:
        asyncio.run(run_async_tests())
        async_success = True
    except Exception as e:
        logger.error(f"Async tests failed: {e}")
        async_success = False
    
    # Determine success
    all_success = result.wasSuccessful() and async_success
    
    # Create result message
    if all_success:
        message = f"All tests passed ({result.testsRun} synchronous, 2 asynchronous)"
    else:
        message = f"Tests failed: {len(result.failures)} failures, {len(result.errors)} errors"
        if not async_success:
            message += ", async tests failed"
    
    return all_success, message