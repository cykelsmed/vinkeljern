#!/usr/bin/env python3
"""
Script to run tests for Vinkeljernet.
"""

import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run tests for Vinkeljernet")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration", action="store_true", help="Run only integration tests")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage report")
    args = parser.parse_args()
    
    cmd = ["pytest", "-v"]
    
    if args.unit:
        cmd.append("-m 'unit'")
    elif args.integration:
        cmd.append("-m 'integration'")
    
    if args.coverage:
        cmd = ["coverage", "run", "-m"] + cmd
        run_cmd(" ".join(cmd))
        subprocess.run("coverage report", shell=True)
        subprocess.run("coverage html", shell=True)
        print("\nCoverage HTML report generated in htmlcov/index.html")
    else:
        run_cmd(" ".join(cmd))

def run_cmd(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True)

if __name__ == "__main__":
    main()