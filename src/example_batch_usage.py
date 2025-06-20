#!/usr/bin/env python3
"""
Example usage script for batch processing legal contracts.
This script shows how to use the DocumentUploader and ParallelClauseProcessor
for efficient processing of large numbers of contracts.
"""

import os
import sys
from pathlib import Path
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from utils.batch_uploader import DocumentUploader
from processing.parallel_pipeline import ParallelClauseProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_1_upload_from_directory():
    """Example 1: Upload contracts from a directory."""
    print("\n=== Example 1: Upload from Directory ===")
    
    # Initialize uploader
    uploader = DocumentUploader(
        source_dir="path/to/your/contracts",  # Replace with your actual path
        target_dir="data/agreements",
        supported_extensions=[".docx", ".pdf", ".txt"]
    )
    
    # Upload all documents recursively
    uploaded = uploader.upload_documents(recursive=True)
    print(f"Uploaded {uploaded} documents from directory")
    
    # Or upload with a limit
    # uploaded = uploader.upload_documents(recursive=True, max_files=100)
    # print(f"Uploaded {uploaded} documents (limited to 100)")

def example_2_upload_from_zip():
    """Example 2: Upload contracts from a zip file."""
    print("\n=== Example 2: Upload from Zip File ===")
    
    # Initialize uploader
    uploader = DocumentUploader(
        source_dir="path/to/your/contracts.zip",  # Replace with your actual zip path
        target_dir="data/agreements"
    )
    
    # Upload from zip file
    uploaded = uploader.upload_from_zip("path/to/your/contracts.zip")
    print(f"Uploaded {uploaded} documents from zip file")

def example_3_parallel_processing():
    """Example 3: Process contracts in parallel."""
    print("\n=== Example 3: Parallel Processing ===")
    
    # Initialize parallel processor
    processor = ParallelClauseProcessor(
        contracts_dir="data/agreements",
        output_dir="data/processed",
        standard_clauses_file="data/raw/standard_clauses.json",
        max_workers=4  # Use 4 parallel workers
    )
    
    # Run the complete pipeline
    processor.run_pipeline()
    print("Parallel processing completed!")

def example_4_complete_workflow():
    """Example 4: Complete workflow for 500 contracts."""
    print("\n=== Example 4: Complete Workflow for 500 Contracts ===")
    
    # Step 1: Upload contracts
    print("Step 1: Uploading contracts...")
    uploader = DocumentUploader(
        source_dir="path/to/your/500_contracts",  # Replace with your actual path
        target_dir="data/agreements"
    )
    
    uploaded = uploader.upload_documents(recursive=True)
    print(f"Uploaded {uploaded} contracts")
    
    # Step 2: Process in parallel
    print("Step 2: Processing contracts in parallel...")
    processor = ParallelClauseProcessor(
        contracts_dir="data/agreements",
        output_dir="data/processed",
        standard_clauses_file="data/raw/standard_clauses.json",
        max_workers=None  # Auto-detect number of CPU cores
    )
    
    processor.run_pipeline()
    print("Processing completed!")
    
    # Step 3: Check results
    print("Step 3: Results saved to data/processed/")
    print("- processing_results.json: Detailed results for each contract")
    print("- training_dataset.json: Generated training dataset")
    print("- Updated standard_clauses.json: Enhanced clause library")

def example_5_memory_efficient_processing():
    """Example 5: Memory-efficient processing for large datasets."""
    print("\n=== Example 5: Memory-Efficient Processing ===")
    
    # Process in batches to manage memory
    batch_size = 50
    
    # Upload first batch
    uploader = DocumentUploader(
        source_dir="path/to/your/large_contract_collection",
        target_dir="data/agreements"
    )
    
    # Upload and process in batches
    for batch_num in range(0, 500, batch_size):
        print(f"Processing batch {batch_num//batch_size + 1}...")
        
        # Upload batch
        uploaded = uploader.upload_documents(
            recursive=True, 
            max_files=batch_size
        )
        
        if uploaded == 0:
            break
        
        # Process batch
        processor = ParallelClauseProcessor(
            contracts_dir="data/agreements",
            output_dir=f"data/processed/batch_{batch_num//batch_size + 1}",
            standard_clauses_file="data/raw/standard_clauses.json"
        )
        
        processor.run_pipeline()
        
        # Clean up processed files to save space
        import shutil
        for file in Path("data/agreements").glob("*.docx"):
            file.unlink()
        
        print(f"Batch {batch_num//batch_size + 1} completed")

def main():
    """Run examples based on user choice."""
    print("Legal Clause AI - Batch Processing Examples")
    print("=" * 50)
    print("Choose an example to run:")
    print("1. Upload from directory")
    print("2. Upload from zip file")
    print("3. Parallel processing")
    print("4. Complete workflow for 500 contracts")
    print("5. Memory-efficient processing")
    print("6. Run all examples")
    
    choice = input("\nEnter your choice (1-6): ").strip()
    
    if choice == "1":
        example_1_upload_from_directory()
    elif choice == "2":
        example_2_upload_from_zip()
    elif choice == "3":
        example_3_parallel_processing()
    elif choice == "4":
        example_4_complete_workflow()
    elif choice == "5":
        example_5_memory_efficient_processing()
    elif choice == "6":
        example_1_upload_from_directory()
        example_2_upload_from_zip()
        example_3_parallel_processing()
        example_4_complete_workflow()
        example_5_memory_efficient_processing()
    else:
        print("Invalid choice. Please run the script again.")

if __name__ == "__main__":
    main() 