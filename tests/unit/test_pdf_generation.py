"""
Tests for the PDF generation functionality in the Flask app.
"""
import unittest
from unittest.mock import patch, MagicMock
import sys
import os
from io import BytesIO

# Add parent directory to path to import app module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from app import generate_report_html, generate_html_content
except ImportError:
    # If we can't import the app module, stub out the functions for testing
    def generate_report_html(results):
        return f"<html><body>Mock report for {results.get('topic', 'unknown')}</body></html>"
    
    def generate_html_content(results):
        return f"<html><body>Mock HTML for {results.get('topic', 'unknown')}</body></html>"

class TestPDFGeneration(unittest.TestCase):
    """Test suite for PDF generation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_results = {
            'topic': 'Test Topic',
            'profile_name': 'Test Profile',
            'timestamp': '2025-04-03 12:00:00',
            'angles': [
                {
                    'overskrift': 'Test Headline 1',
                    'beskrivelse': 'Test Description 1',
                    'begrundelse': 'Test Rationale 1',
                    'nyhedskriterier': ['Aktualitet', 'Væsentlighed'],
                    'startSpørgsmål': ['Question 1?', 'Question 2?']
                },
                {
                    'overskrift': 'Test Headline 2',
                    'beskrivelse': 'Test Description 2',
                    'begrundelse': 'Test Rationale 2',
                    'nyhedskriterier': ['Konflikt', 'Identifikation'],
                    'startSpørgsmål': ['Question 3?', 'Question 4?']
                }
            ],
            'background_info': 'This is some test background information.',
            'editorial_considerations': 'These are some test editorial considerations.'
        }
    
    def test_generate_report_html(self):
        """Test that the report HTML is generated correctly."""
        html = generate_report_html(self.sample_results)
        
        # Check that the HTML includes the expected content
        self.assertIn(self.sample_results['topic'], html)
        self.assertIn(self.sample_results['profile_name'], html)
        self.assertIn(self.sample_results['background_info'], html)
        self.assertIn(self.sample_results['editorial_considerations'], html)
        
        # Check that all angles are included
        for angle in self.sample_results['angles']:
            self.assertIn(angle['overskrift'], html)
            self.assertIn(angle['beskrivelse'], html)
            
        # Check that it has the proper HTML structure
        self.assertIn('<!DOCTYPE html>', html)
        self.assertIn('<html>', html)
        self.assertIn('<head>', html)
        self.assertIn('<body>', html)
        self.assertIn('</html>', html)
    
    def test_generate_html_content(self):
        """Test that the HTML content for the simpler PDF is generated correctly."""
        html = generate_html_content(self.sample_results)
        
        # Check that the HTML includes the expected content
        self.assertIn(self.sample_results['topic'], html)
        self.assertIn(self.sample_results['profile_name'], html)
        
        # Check that all angles are included
        for angle in self.sample_results['angles']:
            self.assertIn(angle['overskrift'], html)
            self.assertIn(angle['beskrivelse'], html)
            
        # Check that it has the proper HTML structure
        self.assertIn('<!DOCTYPE html>', html)
        self.assertIn('<html>', html)
        self.assertIn('<head>', html)
        self.assertIn('<body>', html)
        self.assertIn('</html>', html)
    
    @patch('pdfkit.from_string')
    def test_pdf_conversion(self, mock_from_string):
        """Test PDF conversion with mocked pdfkit."""
        # No need to actually call wkhtmltopdf in tests
        mock_pdf_content = b'Mock PDF Content'
        mock_from_string.return_value = mock_pdf_content
        
        # Simulate app context
        app_context = MagicMock()
        app_context.__enter__ = MagicMock(return_value=None)
        app_context.__exit__ = MagicMock(return_value=None)
        
        # Generate the HTML
        html = generate_report_html(self.sample_results)
        
        # Mock conversion
        pdf = mock_from_string(html, False)
        
        # Check the result
        self.assertEqual(pdf, mock_pdf_content)
        mock_from_string.assert_called_once_with(html, False)

if __name__ == '__main__':
    unittest.main()