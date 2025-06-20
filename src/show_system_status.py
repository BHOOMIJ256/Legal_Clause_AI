#!/usr/bin/env python3
"""
System Status Checker
Shows the current status of the batch processing system and available tools.
"""

import os
import sys
from pathlib import Path
import json

def check_system_status():
    """Check the current status of the batch processing system."""
    print("🔍 Legal Clause AI - Batch Processing System Status")
    print("=" * 60)
    
    # Check main directories for batch processing
    directories = [
        "data/agreements",
        "data/processed", 
        "data/raw"
    ]
    
    print("\n📁 Directory Status:")
    for directory in directories:
        path = Path(directory)
        if path.exists():
            files = len(list(path.glob("*")))
            print(f"  ✅ {directory} - {files} files")
        else:
            print(f"  ❌ {directory} - Not created")
    
    # Check key files
    key_files = [
        "src/utils/batch_uploader.py",
        "src/processing/parallel_pipeline.py",
        "src/processing/pipeline.py",
        "src/run_batch_processing.py",
        "src/example_batch_usage.py",
        "src/setup_batch_processing.py",
        "src/test_batch_system.py"
    ]
    
    print("\n📄 Key Files Status:")
    for file_path in key_files:
        path = Path(file_path)
        if path.exists():
            size = path.stat().st_size
            print(f"  ✅ {file_path} - {size} bytes")
        else:
            print(f"  ❌ {file_path} - Missing")
    
    # Check standard clauses
    standard_clauses_file = Path("data/raw/standard_clauses.json")
    if standard_clauses_file.exists():
        try:
            with open(standard_clauses_file, 'r') as f:
                clauses = json.load(f)
            print(f"\n📋 Standard Clauses: {len(clauses['clauses']) if isinstance(clauses, dict) and 'clauses' in clauses else len(clauses)} clauses loaded")
        except:
            print(f"\n📋 Standard Clauses: File exists but could not be read")
    else:
        print(f"\n📋 Standard Clauses: No file found")
    
    # Check processing results
    results_file = Path("data/processed/processing_results.json")
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            successful = sum(1 for r in results if r.get("status") == "success")
            print(f"📊 Processing Results: {successful}/{len(results)} successful")
        except:
            print(f"📊 Processing Results: File exists but could not be read")
    else:
        print(f"📊 Processing Results: No results found")

def show_available_commands():
    """Show available commands for using the system."""
    print("\n🚀 Available Commands:")
    print("=" * 60)
    
    commands = [
        ("Setup System", "python src/setup_batch_processing.py", "Install dependencies and create directories"),
        ("Test System", "python src/test_batch_system.py", "Test with sample contracts"),
        ("Process Contracts", "python src/run_batch_processing.py --source /path/to/contracts", "Process all contracts"),
        ("Test with 10 Files", "python src/run_batch_processing.py --source /path/to/contracts --max-files 10", "Test with limited files"),
        ("Upload Only", "python src/run_batch_processing.py --source /path/to/contracts --upload-only", "Only upload, don't process"),
        ("Process Only", "python src/run_batch_processing.py --process-only", "Process already uploaded files"),
        ("Examples", "python src/example_batch_usage.py", "Interactive examples"),
        ("Check Status", "python src/show_system_status.py", "Show this status")
    ]
    
    for name, command, description in commands:
        print(f"\n📝 {name}:")
        print(f"   Command: {command}")
        print(f"   Description: {description}")

def show_usage_examples():
    """Show specific usage examples."""
    print("\n💡 Usage Examples:")
    print("=" * 60)
    
    examples = [
        {
            "scenario": "Process 500 contracts from a directory",
            "command": "python src/run_batch_processing.py --source /Users/bhoomijain/Desktop/MyContracts",
            "description": "Process all contracts in the MyContracts directory"
        },
        {
            "scenario": "Process contracts from a zip file",
            "command": "python src/run_batch_processing.py --source /Users/bhoomijain/Desktop/contracts.zip",
            "description": "Extract and process contracts from a zip file"
        },
        {
            "scenario": "Test with first 10 contracts",
            "command": "python src/run_batch_processing.py --source /Users/bhoomijain/Desktop/MyContracts --max-files 10",
            "description": "Test the system with only 10 contracts first"
        },
        {
            "scenario": "Use 6 parallel workers",
            "command": "python src/run_batch_processing.py --source /Users/bhoomijain/Desktop/MyContracts --workers 6",
            "description": "Use 6 CPU cores for parallel processing"
        },
        {
            "scenario": "Memory-efficient processing",
            "command": "python src/run_batch_processing.py --source /Users/bhoomijain/Desktop/MyContracts --workers 2",
            "description": "Use fewer workers to reduce memory usage"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['scenario']}:")
        print(f"   {example['command']}")
        print(f"   {example['description']}")

def show_performance_info():
    """Show performance expectations."""
    print("\n⚡ Performance Expectations:")
    print("=" * 60)
    
    print("📊 For 500 Contracts:")
    print("  • Processing Speed: ~30-60 seconds per contract")
    print("  • Total Time: ~4-8 hours")
    print("  • Memory Usage: ~2-4 GB RAM")
    print("  • Disk Space: ~2-3x contract size")
    
    print("\n🔧 Optimization Tips:")
    print("  • Use SSD storage for faster I/O")
    print("  • Set workers = CPU cores - 1")
    print("  • Monitor system resources during processing")
    print("  • Test with 10 contracts first")

def main():
    """Main function to show system status."""
    check_system_status()
    show_available_commands()
    show_usage_examples()
    show_performance_info()
    
    print("\n" + "=" * 60)
    print("🎯 Next Steps:")
    print("1. Run: python src/setup_batch_processing.py")
    print("2. Test: python src/test_batch_system.py")
    print("3. Process: python src/run_batch_processing.py --source /path/to/your/contracts")
    print("=" * 60)

if __name__ == "__main__":
    main() 