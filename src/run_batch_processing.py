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

def get_already_processed_files() -> set:
    """Get list of files that have already been processed successfully."""
    processed_files = set()
    
    # Check if processing results exist
    results_file = Path("data/processed/processing_results.json")
    if results_file.exists():
        try:
            import json
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            for result in results:
                if result.get("status") == "success":
                    processed_files.add(result.get("contract_name", ""))
            
            logger.info(f"Found {len(processed_files)} already processed files")
        except Exception as e:
            logger.error(f"Error reading processing results: {e}")
    
    return processed_files

def process_contracts_sequential(max_files: Optional[int] = None, skip_processed: bool = False):
    """
    Process all uploaded contracts sequentially (optionally limit number).
    
    Args:
        max_files: Maximum number of files to process
        skip_processed: If True, skip files that were already successfully processed
    """
    logger.info("Starting sequential contract processing")
    
    pipeline = ClauseProcessingPipeline(
        contracts_dir="data/agreements",
        output_dir="data/processed",
        standard_clauses_file="data/raw/standard_clauses.json"
    )
    
    # Find all supported files (.docx, .pdf, .txt)
    contract_files = []
    for ext in ("*.docx", "*.pdf", "*.txt"):
        contract_files.extend([f for f in Path("data/agreements").glob(f"**/{ext}") if not f.name.startswith("._")])
    contract_files = sorted(contract_files, key=lambda x: str(x))  # Sort for consistency
    
    # Filter out already processed files if requested
    if skip_processed:
        processed_files = get_already_processed_files()
        original_count = len(contract_files)
        contract_files = [f for f in contract_files if f.name not in processed_files]
        skipped_count = original_count - len(contract_files)
        logger.info(f"Skipped {skipped_count} already processed files")
    
    if max_files is not None:
        contract_files = contract_files[:max_files]
    
    logger.info(f"Processing {len(contract_files)} files")
    
    # Load existing results
    results_file = Path("data/processed/processing_results.json")
    existing_results = []
    if results_file.exists():
        try:
            import json
            with open(results_file, 'r') as f:
                existing_results = json.load(f)
        except Exception as e:
            logger.error(f"Error loading existing results: {e}")
    
    # Process files
    new_results = []
    for contract_path in contract_files:
        try:
            logger.info(f"Processing file: {contract_path}")
            result = pipeline.process_contract(contract_path)
            new_results.append(result)
            
            # Save progress after each file
            pipeline._save_standard_clauses()
            
            # Save updated results
            all_results = existing_results + new_results
            with open(results_file, 'w') as f:
                import json
                json.dump(all_results, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error processing file {contract_path}: {e}")
            new_results.append({
                "status": "error",
                "contract_name": contract_path.name,
                "message": str(e)
            })
            continue

    logger.info("Sequential processing completed")

    # Summary
    successful = sum(1 for r in new_results if r.get("status") == "success")
    errors = sum(1 for r in new_results if r.get("status") == "error")
    
    logger.info(f"Total contracts processed in this run: {len(new_results)}")
    logger.info(f"Successfully processed: {successful} files")
    logger.info(f"Errors: {errors} files")
    logger.info(f"Total files now processed: {len(existing_results) + len(new_results)}")
    
    matched = sum(r.get('matched_clauses', 0) for r in new_results if r.get('status') == 'success')
    new_clauses = sum(r.get('new_standard_clauses', 0) for r in new_results if r.get('status') == 'success')
    logger.info(f"Total clauses matched: {matched}")
    logger.info(f"New standard clauses added: {new_clauses}")

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
    parser.add_argument(
        "--skip-processed", 
        action="store_true",
        help="Skip files that were already successfully processed"
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
        process_contracts_sequential(args.max_files, args.skip_processed)
    
    logger.info("Batch processing completed successfully!")

if __name__ == "__main__":
    main() 