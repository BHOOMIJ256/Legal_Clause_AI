# Legal Clause AI - Batch Processing Guide

This guide explains how to use the batch processing system to efficiently process large numbers of legal contracts (like your 500 contracts) using parallel processing and memory-efficient techniques.

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Run the setup script to install dependencies and create directories
python src/setup_batch_processing.py
```

### 2. Process Your Contracts
```bash
# Process all contracts from a directory
python src/run_batch_processing.py --source /path/to/your/contracts

# Process contracts from a zip file
python src/run_batch_processing.py --source /path/to/your/contracts.zip

# Limit to first 100 files for testing
python src/run_batch_processing.py --source /path/to/your/contracts --max-files 100

# Use specific number of parallel workers
python src/run_batch_processing.py --source /path/to/your/contracts --workers 8
```

## 📁 Supported File Formats

The system supports:
- **DOCX** files (Microsoft Word documents)
- **PDF** files (Portable Document Format)
- **TXT** files (Plain text files)
- **ZIP** files (Compressed archives containing the above formats)

## ⚡ Parallel Processing Benefits

### Efficiency Features:
- **Multi-core Processing**: Uses all available CPU cores
- **Memory Management**: Processes one contract at a time to avoid memory issues
- **Progress Tracking**: Real-time progress bars and status updates
- **Error Handling**: Continues processing even if individual contracts fail
- **Incremental Saving**: Saves results after each contract is processed

### Performance Comparison:
- **Sequential Processing**: ~2-3 minutes per contract
- **Parallel Processing**: ~30-60 seconds per contract (depending on CPU cores)
- **500 Contracts**: ~4-8 hours vs ~25-40 hours

## 🔧 Advanced Usage

### 1. Upload Only (Don't Process)
```bash
python src/run_batch_processing.py --source /path/to/your/contracts --upload-only
```

### 2. Process Already Uploaded Documents
```bash
python src/run_batch_processing.py --process-only
```

### 3. Memory-Efficient Processing for Large Datasets
```python
# Use the example script for batch processing
python src/example_batch_usage.py
# Choose option 5 for memory-efficient processing
```

### 4. Custom Configuration
```python
from src.utils.batch_uploader import DocumentUploader
from src.processing.parallel_pipeline import ParallelClauseProcessor

# Custom uploader
uploader = DocumentUploader(
    source_dir="your/contracts/path",
    target_dir="custom/upload/dir",
    supported_extensions=[".docx", ".pdf", ".txt"]
)

# Custom processor
processor = ParallelClauseProcessor(
    contracts_dir="custom/upload/dir",
    output_dir="custom/output/dir",
    standard_clauses_file="custom/standard_clauses.json",
    max_workers=6  # Use 6 parallel workers
)
```

## 📊 Output Structure

After processing, you'll find:

```
data/
├── agreements/           # Uploaded contract files
├── processed/           # Processing results
│   ├── processing_results.json    # Detailed results for each contract
│   └── training_dataset.json      # Generated training dataset
└── standard_clauses.json          # Enhanced clause library
```

### Processing Results Format:
```json
{
  "status": "success",
  "contract_name": "contract_001.docx",
  "total_clauses": 15,
  "matched_clauses": 12,
  "new_standard_clauses": 3,
  "processing_time": 45.2
}
```

## 🛠️ Troubleshooting

### Common Issues:

1. **Memory Errors**
   ```bash
   # Reduce number of workers
   python src/run_batch_processing.py --source /path/to/contracts --workers 2
   ```

2. **File Format Not Supported**
   ```bash
   # Check supported extensions in batch_uploader.py
   # Add your format to supported_extensions list
   ```

3. **Processing Hangs**
   ```bash
   # Check logs in data/processed/processing_results.json
   # Restart with fewer workers
   ```

4. **Dependencies Missing**
   ```bash
   # Re-run setup
   python src/setup_batch_processing.py
   ```

## 📈 Performance Optimization

### For 500 Contracts:

1. **Optimal Worker Count**: Use `CPU cores - 1`
   ```bash
   # Auto-detect (recommended)
   python src/run_batch_processing.py --source /path/to/contracts
   
   # Manual (if you know your CPU cores)
   python src/run_batch_processing.py --source /path/to/contracts --workers 7
   ```

2. **Batch Processing for Very Large Collections**:
   ```python
   # Process in batches of 50
   batch_size = 50
   for batch in range(0, 500, batch_size):
       # Process batch
       # Clean up memory
       # Continue to next batch
   ```

3. **Storage Optimization**:
   - Use SSD for faster I/O
   - Ensure sufficient disk space (2-3x contract size)
   - Clean up processed files after analysis

## 🔍 Monitoring Progress

### Real-time Monitoring:
- Progress bars show current status
- Logs show detailed information
- Results saved incrementally

### Check Progress:
```bash
# View processing results
cat data/processed/processing_results.json

# Count processed files
ls data/agreements/ | wc -l

# Check for errors
grep "error" data/processed/processing_results.json
```

## 📋 Example Workflows

### Workflow 1: Complete Processing
```bash
# 1. Setup
python src/setup_batch_processing.py

# 2. Process all contracts
python src/run_batch_processing.py --source /path/to/500_contracts

# 3. Check results
ls data/processed/
```

### Workflow 2: Test with Sample
```bash
# 1. Test with 10 contracts first
python src/run_batch_processing.py --source /path/to/contracts --max-files 10

# 2. If successful, process all
python src/run_batch_processing.py --source /path/to/contracts
```

### Workflow 3: Memory-Constrained Environment
```bash
# 1. Use fewer workers
python src/run_batch_processing.py --source /path/to/contracts --workers 2

# 2. Or use batch processing
python src/example_batch_usage.py
# Choose option 5
```

## 🎯 Best Practices

1. **Always test with a small sample first**
2. **Monitor system resources during processing**
3. **Keep backups of original contracts**
4. **Use SSD storage for better performance**
5. **Clean up temporary files after processing**
6. **Review results before proceeding with analysis**

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review logs in `data/processed/`
3. Try with fewer workers or smaller batches
4. Ensure all dependencies are installed correctly

## 🚀 Next Steps

After successful processing:
1. Review the generated training dataset
2. Analyze the enhanced standard clauses library
3. Use the results for model training
4. Build APIs or web interfaces for clause analysis
5. Implement continuous learning from new contracts

---

**Happy Processing! 🎉** 