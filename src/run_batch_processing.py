#!/usr/bin/env python3
"""
Main execution script for batch processing of legal contracts (sequential version).
Uploads contracts and processes them one by one using ClauseProcessingPipeline.
"""

import os
import sys
from pathlib import Path
import argparse
import logging
from typing import Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from src.utils.batch_uploader import DocumentUploader
from src.processing.pipeline import ClauseProcessingPipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        "data/agreements",
        "data/processed", 
        "data/raw"         # Ensure the parent of standard_clauses.json exists
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

def process_contracts_sequential(max_files: Optional[int] = None):
    """
    Process all uploaded contracts sequentially (optionally limit number).
    """
    logger.info("Starting sequential contract processing")
    
    pipeline = ClauseProcessingPipeline(
        contracts_dir="data/agreements",
        output_dir="data/processed",
        standard_clauses_file="data/raw/standard_clauses.json"
    )
    # If max_files is set, process only that many files
    contract_files = [f for f in Path("data/agreements").glob("**/*.docx") if not f.name.startswith("._")]
    if max_files is not None:
        contract_files = contract_files[:max_files]
    results = []
    for contract_path in contract_files:
        result = pipeline.process_contract(contract_path)
        results.append(result)
        pipeline._save_standard_clauses()  # Save after each contract
        with open(Path("data/processed") / "processing_results.json", 'w') as f:
            import json
            json.dump(results, f, indent=2)
    logger.info("Sequential processing completed")

    # --- ADD THIS SUMMARY BLOCK ---
    logger.info(f"Total contracts processed: {len(results)}")
    matched = sum(r.get('matched_clauses', 0) for r in results if r.get('status') == 'success')
    new = sum(r.get('new_standard_clauses', 0) for r in results if r.get('status') == 'success')
    logger.info(f"Total clauses matched: {matched}")
    logger.info(f"New standard clauses added: {new}")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Batch process legal contracts (sequential)")
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
        # Process contracts sequentially
        process_contracts_sequential(args.max_files)
    
    logger.info("Batch processing completed successfully!")

if __name__ == "__main__":
    main() 