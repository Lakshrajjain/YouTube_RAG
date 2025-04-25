#!/usr/bin/env python
"""
Script to generate a test coverage report for the YouTube RAG Pipeline.
"""
import os
import sys
import subprocess
import time

def run_coverage():
    """Run test coverage."""
    print("Running test coverage...")
    
    # Run pytest with coverage
    result = subprocess.run(
        ["pytest", "--cov=src", "--cov-report=term", "--cov-report=html", "tests/"],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    
    if result.returncode != 0:
        print("Coverage run failed!")
        print(result.stderr)
        return False
    
    print("Coverage report generated in htmlcov/ directory")
    return True

def main():
    """Main function."""
    start_time = time.time()
    
    success = run_coverage()
    
    end_time = time.time()
    
    print(f"Coverage completed in {end_time - start_time:.2f} seconds")
    
    if success:
        print("Coverage report generated successfully!")
        return 0
    else:
        print("Coverage report generation failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
