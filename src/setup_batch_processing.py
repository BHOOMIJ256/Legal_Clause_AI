#!/usr/bin/env python3
"""
Setup script for batch processing functionality.
This script ensures all necessary dependencies are installed and directories are created.
"""

import subprocess
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        return False
    logger.info(f"Python version: {sys.version}")
    return True

def install_requirements():
    """Install required packages."""
    requirements = [
        "torch",
        "transformers",
        "spacy",
        "python-docx",
        "PyPDF2",
        "tqdm",
        "numpy",
        "scikit-learn",
        "sentence-transformers"
    ]
    
    logger.info("Installing required packages...")
    
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            logger.info(f"Installed {package}")
        except subprocess.CalledProcessError:
            logger.error(f"Failed to install {package}")
            return False
    
    return True

def download_spacy_model():
    """Download spaCy model for NLP processing."""
    try:
        logger.info("Downloading spaCy model...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        logger.info("spaCy model downloaded successfully")
        return True
    except subprocess.CalledProcessError:
        logger.error("Failed to download spaCy model")
        return False

def create_directories():
    """Create necessary directories."""
    directories = [
        "data/agreements",
        "data/processed",
        "data/raw",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def create_initial_standard_clauses():
    """Create initial standard clauses file if it doesn't exist."""
    standard_clauses_file = Path("data/raw/standard_clauses.json")
    
    if not standard_clauses_file.exists():
        import json
        initial_clauses = [
            {
                "title": "Force Majeure",
                "text": "Neither party shall be liable for any failure or delay in performance under this Agreement due to circumstances beyond its reasonable control, including but not limited to acts of God, war, terrorism, riots, fire, natural disaster, or government action.",
                "category": "liability",
                "variants": []
            },
            {
                "title": "Confidentiality",
                "text": "Each party agrees to maintain the confidentiality of any proprietary or confidential information disclosed by the other party during the term of this Agreement and for a period of five years thereafter.",
                "category": "confidentiality",
                "variants": []
            },
            {
                "title": "Termination",
                "text": "Either party may terminate this Agreement upon thirty days written notice to the other party. Upon termination, all rights and obligations under this Agreement shall cease, except for those provisions that by their nature survive termination.",
                "category": "termination",
                "variants": []
            }
        ]
        
        with open(standard_clauses_file, 'w') as f:
            json.dump(initial_clauses, f, indent=2)
        
        logger.info("Created initial standard clauses file")

def main():
    """Run complete setup."""
    logger.info("Setting up batch processing environment...")
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Download spaCy model
    if not download_spacy_model():
        return False
    
    # Create initial standard clauses
    create_initial_standard_clauses()
    
    logger.info("Setup completed successfully!")
    logger.info("\nNext steps:")
    logger.info("1. Place your contracts in a directory or zip file")
    logger.info("2. Run: python src/run_batch_processing.py --source /path/to/your/contracts")
    logger.info("3. Or run: python src/example_batch_usage.py for examples")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)  #Exit with 0 if successful, 1 if not