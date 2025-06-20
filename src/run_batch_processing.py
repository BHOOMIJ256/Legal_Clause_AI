#!/usr/bin/env python3
"""
Main execution script for batch processing of legal contracts.
This script demonstrates how to use the DocumentUploader and ParallelClauseProcessor
to efficiently process large numbers of contracts.
"""

import os
import sys
from pathlib import Path
import argparse
import logging
from typing import Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from utils.batch_uploader import DocumentUploader
from processing.parallel_pipeline import ParallelClauseProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        "data/agreements",
        "data/processed", 
        "data/raw"  # Ensure the parent of standard_clauses.json exists
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")

def upload_contracts(source_path: str, max_files: Optional[int] = None):
    """
    Upload contracts from source directory or zip file.
    
    Args:
        source_path: Path to directory or zip file containing contracts
        max_files: Maximum number of files to upload (None for all)
    """
    logger.info(f"Starting document upload from: {source_path}")
    
    uploader = DocumentUploader(
        source_dir=source_path,
        target_dir="data/agreements",
        supported_extensions=[".docx", ".pdf", ".txt"]
    )
    
    # Check if source is a zip file
    if source_path.endswith('.zip'):
        uploaded = uploader.upload_from_zip(source_path, max_files)
        logger.info(f"Uploaded {uploaded} documents from zip file")
    else:
        uploaded = uploader.upload_documents(recursive=True, max_files=max_files)
        logger.info(f"Uploaded {uploaded} documents from directory")
    
    return uploaded

def process_contracts_parallel(max_workers: Optional[int] = None):
    """
    Process all uploaded contracts in parallel.
    
    Args:
        max_workers: Number of parallel workers (None for auto-detect)
    """
    logger.info("Starting parallel contract processing")
    
    processor = ParallelClauseProcessor(
        contracts_dir="data/agreements",
        output_dir="data/processed",
        standard_clauses_file="data/raw/standard_clauses.json",
        max_workers=max_workers
    )
    
    # Run the complete pipeline
    processor.run_pipeline()
    
    logger.info("Parallel processing completed")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Batch process legal contracts")
    parser.add_argument(
        "--source", 
        type=str, 
        required=True,
        help="Path to directory or zip file containing contracts"
    )
    parser.add_argument(
        "--max-files", 
        type=int, 
        default=None,
        help="Maximum number of files to process (default: all)"
    )
    parser.add_argument(
        "--workers", 
        type=int, 
        default=None,
        help="Number of parallel workers (default: auto-detect)"
    )
    parser.add_argument(
        "--upload-only", 
        action="store_true",
        help="Only upload documents, don't process them"
    )
    parser.add_argument(
        "--process-only", 
        action="store_true",
        help="Only process already uploaded documents"
    )
    
    args = parser.parse_args()
    
    # Setup directories
    setup_directories()
    
    if not args.process_only:
        # Upload contracts
        uploaded = upload_contracts(args.source, args.max_files)
        if uploaded == 0:
            logger.error("No documents were uploaded. Please check the source path.")
            return
    
    if not args.upload_only:
        # Process contracts
        process_contracts_parallel(args.workers)
    
    logger.info("Batch processing completed successfully!")

if __name__ == "__main__":
    main() 