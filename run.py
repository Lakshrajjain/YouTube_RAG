#!/usr/bin/env python3
"""
Main script to run the YouTube RAG Pipeline.
"""
import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_api():
    """Run the FastAPI backend."""
    logger.info("Starting API server...")
    subprocess.Popen(
        ["uvicorn", "src.api.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"],
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    logger.info("API server started at http://localhost:8000")

def run_ui():
    """Run the Streamlit UI."""
    logger.info("Starting Streamlit UI...")
    subprocess.Popen(
        ["streamlit", "run", "src/ui/app.py"],
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    logger.info("Streamlit UI started at http://localhost:8501")

def build_extension():
    """Build the Chrome extension."""
    logger.info("Building Chrome extension...")
    extension_dir = Path("src/chrome_extension")
    build_script = extension_dir / "build.sh"
    
    if not build_script.exists():
        logger.error(f"Build script not found at {build_script}")
        return False
    
    result = subprocess.run(
        [str(build_script)],
        cwd=str(extension_dir),
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    
    if result.returncode == 0:
        logger.info(f"Chrome extension built successfully in {extension_dir}/dist")
        return True
    else:
        logger.error("Failed to build Chrome extension")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run the YouTube RAG Pipeline")
    parser.add_argument("--api-only", action="store_true", help="Run only the API server")
    parser.add_argument("--ui-only", action="store_true", help="Run only the Streamlit UI")
    parser.add_argument("--build-extension", action="store_true", help="Build the Chrome extension")
    
    args = parser.parse_args()
    
    if args.build_extension:
        build_extension()
        return
    
    if args.api_only:
        run_api()
    elif args.ui_only:
        run_ui()
    else:
        # Run both API and UI
        run_api()
        time.sleep(2)  # Wait for API to start
        run_ui()
    
    # Keep the script running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        sys.exit(0)

if __name__ == "__main__":
    main()
