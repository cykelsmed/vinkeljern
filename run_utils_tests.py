#!/usr/bin/env python
"""
Testrunner for vinkeljernet_utils-pakken.
"""

import sys
import logging
from vinkeljernet_utils import setup
from vinkeljernet_utils.testing import run_tests

# Konfigurer logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Kør alle tests for vinkeljernet_utils-pakken."""
    print("Running tests for vinkeljernet_utils...")
    
    # Konfigurer pakken i testmiljø
    setup(debug=True, log_level="DEBUG")
    
    # Kør tests
    success, message = run_tests(verbose=True)
    
    print(f"\n{message}")
    
    # Returner exit kode baseret på testresultat
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())