#!/usr/bin/env python
"""
Script to run all tests for the YouTube RAG Pipeline.
"""
import os
import sys
import argparse
import subprocess
import time

def run_unit_tests():
    """Run unit tests."""
    print("Running unit tests...")
    result = subprocess.run(["pytest", "tests/unit", "-v"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("Unit tests failed!")
        print(result.stderr)
        return False
    return True

def run_integration_tests():
    """Run integration tests."""
    print("Running integration tests...")
    result = subprocess.run(["pytest", "tests/integration", "-v"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("Integration tests failed!")
        print(result.stderr)
        return False
    return True

def run_load_tests():
    """Run load tests."""
    print("Running load tests...")
    result = subprocess.run(["pytest", "tests/load", "-v"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("Load tests failed!")
        print(result.stderr)
        return False
    return True

def run_all_tests():
    """Run all tests."""
    unit_success = run_unit_tests()
    integration_success = run_integration_tests()
    load_success = run_load_tests()
    
    return unit_success and integration_success and load_success

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run tests for the YouTube RAG Pipeline.")
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--load", action="store_true", help="Run load tests")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    # If no arguments are provided, run all tests
    if not (args.unit or args.integration or args.load or args.all):
        args.all = True
    
    start_time = time.time()
    
    if args.all:
        success = run_all_tests()
    else:
        success = True
        if args.unit:
            success = success and run_unit_tests()
        if args.integration:
            success = success and run_integration_tests()
        if args.load:
            success = success and run_load_tests()
    
    end_time = time.time()
    
    print(f"Tests completed in {end_time - start_time:.2f} seconds")
    
    if success:
        print("All tests passed!")
        return 0
    else:
        print("Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
