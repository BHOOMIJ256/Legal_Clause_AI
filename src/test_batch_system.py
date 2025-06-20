#!/usr/bin/env python3
"""
Test script for batch processing system.
This script creates a small test dataset and verifies the system works correctly.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import json
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from utils.batch_uploader import DocumentUploader
from processing.parallel_pipeline import ParallelClauseProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_contracts():
    """Create test contract files for testing."""
    test_dir = Path("test_contracts")
    test_dir.mkdir(exist_ok=True)
    
    # Create test contracts 
    test_contracts = [
        {
            "name": "test_contract_1.txt",
            "content": """
            AGREEMENT FOR TRANSPORTATION SERVICES
            
            SECTION 1: SERVICES
            The Carrier shall provide transportation services for the Shipper's goods from origin to destination.
            
            SECTION 2: PAYMENT
            Payment shall be made within 30 days of invoice receipt.
            
            SECTION 3: LIABILITY
            The Carrier's liability shall be limited to the actual value of the goods transported.
            
            SECTION 4: FORCE MAJEURE
            Neither party shall be liable for delays due to circumstances beyond their control.
            """
        },
        {
            "name": "test_contract_2.txt",
            "content": """
            SUPPLY AGREEMENT
            
            ARTICLE 1: SUPPLY TERMS
            The Supplier shall deliver goods according to the specifications provided.
            
            ARTICLE 2: DELIVERY
            Delivery shall be made within 5 business days of order confirmation.
            
            ARTICLE 3: QUALITY
            All goods must meet industry standards and specifications.
            
            ARTICLE 4: TERMINATION
            Either party may terminate this agreement with 30 days written notice.
            """
        },
        {
            "name": "test_contract_3.txt",
            "content": """
            CONFIDENTIALITY AGREEMENT
            
            CLAUSE 1: CONFIDENTIAL INFORMATION
            Each party agrees to maintain confidentiality of proprietary information.
            
            CLAUSE 2: NON-DISCLOSURE
            Confidential information shall not be disclosed to third parties.
            
            CLAUSE 3: DURATION
            This agreement shall remain in effect for 5 years after termination.
            
            CLAUSE 4: REMEDIES
            Breach of this agreement may result in legal action and damages.
            """
        }
    ]
    
    # Write test contracts
    for contract in test_contracts:
        with open(test_dir / contract["name"], 'w') as f:
            f.write(contract["content"])
    
    logger.info(f"Created {len(test_contracts)} test contracts in {test_dir}")
    return test_dir

def test_uploader():
    """Test the document uploader."""
    logger.info("Testing DocumentUploader...")
    
    test_dir = create_test_contracts()
    
    uploader = DocumentUploader(
        source_dir=str(test_dir),
        target_dir="data/agreements",
        supported_extensions=[".txt"]
    )
    
    uploaded = uploader.upload_documents(recursive=False)
    logger.info(f"Uploaded {uploaded} test documents")
    
    # Clean up test directory
    shutil.rmtree(test_dir)
    
    return uploaded > 0

def test_processor():
    """Test the parallel processor."""
    logger.info("Testing ParallelClauseProcessor...")
    
    # Check if we have uploaded documents
    agreements_dir = Path("data/agreements")
    if not agreements_dir.exists() or not list(agreements_dir.glob("*.txt")):
        logger.warning("No test documents found. Please run uploader test first.")
        return False
    
    processor = ParallelClauseProcessor(
        contracts_dir="data/agreements",
        output_dir="data/processed",
        standard_clauses_file="data/raw/standard_clauses.json",
        max_workers=1  # Use single worker for testing
    )
    
    try:
        # Process contracts
        results = processor.process_all_contracts()
        
        # Check results
        successful = sum(1 for r in results if r["status"] == "success")
        logger.info(f"Successfully processed {successful}/{len(results)} contracts")
        
        return successful > 0
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        return False

def test_complete_workflow():
    """Test the complete workflow."""
    logger.info("Testing complete workflow...")
    
    # Step 1: Test uploader
    if not test_uploader():
        logger.error("Uploader test failed")
        return False
    
    # Step 2: Test processor
    if not test_processor():
        logger.error("Processor test failed")
        return False
    
    # Step 3: Check outputs
    output_files = [
        "data/processed/processing_results.json",
        "data/raw/standard_clauses.json"
    ]
    
    for file_path in output_files:
        if not Path(file_path).exists():
            logger.error(f"Expected output file not found: {file_path}")
            return False
    
    logger.info("Complete workflow test passed!")
    return True

def cleanup_test_data():
    """Clean up test data."""
    logger.info("Cleaning up test data...")
    
    # Remove test agreements
    agreements_dir = Path("data/agreements")
    if agreements_dir.exists():
        for file in agreements_dir.glob("*.txt"):
            file.unlink()
    
    # Remove test results
    processed_dir = Path("data/processed")
    if processed_dir.exists():
        for file in processed_dir.glob("*.json"):
            file.unlink()

def main():
    """Run all tests."""
    logger.info("Starting batch processing system tests...")
    
    # Create necessary directories
    Path("data/agreements").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    
    # Run tests
    tests = [
        ("Uploader Test", test_uploader),
        ("Processor Test", test_processor),
        ("Complete Workflow Test", test_complete_workflow)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = test_func()
            results.append((test_name, result))
            logger.info(f"{test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            logger.error(f"{test_name}: ERROR - {str(e)}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! The batch processing system is working correctly.")
        logger.info("\nYou can now use the system with your 500 contracts:")
        logger.info("python src/run_batch_processing.py --source /path/to/your/contracts")
    else:
        logger.error("❌ Some tests failed. Please check the logs above.")
    
    # Cleanup
    cleanup_test_data()
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 