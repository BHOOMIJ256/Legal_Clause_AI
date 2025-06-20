# Quick Start Guide - Processing 500 Contracts

This guide shows you exactly how to process your 500 contracts efficiently using the batch processing system.

## 🚀 Step-by-Step Instructions

### Step 1: Setup (One-time)
```bash
# Navigate to your project directory
cd /Users/bhoomijain/Desktop/Legal_Clause_AI

# Run the setup script
python src/setup_batch_processing.py
```

### Step 2: Test the System (Recommended)
```bash
# Test with a small sample first
python src/test_batch_system.py
```

### Step 3: Process Your Contracts

#### Option A: Process All 500 Contracts
```bash
# Replace with your actual contracts path
python src/run_batch_processing.py --source /path/to/your/500_contracts
```

#### Option B: Test with First 10 Contracts
```bash
# Test with 10 contracts first
python src/run_batch_processing.py --source /path/to/your/500_contracts --max-files 10
```

#### Option C: Process from Zip File
```bash
# If your contracts are in a zip file
python src/run_batch_processing.py --source /path/to/your/contracts.zip
```

#### Option D: Use Specific Number of Workers
```bash
# Use 6 parallel workers (adjust based on your CPU)
python src/run_batch_processing.py --source /path/to/your/500_contracts --workers 6
```

## 📊 Monitor Progress

### Real-time Monitoring
- Progress bars will show current status
- Logs will display detailed information
- Results are saved after each contract

### Check Progress
```bash
# View processing results
cat data/processed/processing_results.json

# Count processed files
ls data/agreements/ | wc -l

# Check for errors
grep "error" data/processed/processing_results.json
```

## 📁 Expected Output

After processing, you'll find:
```
data/
├── agreements/           # Your uploaded contracts
├── processed/
│   ├── processing_results.json    # Results for each contract
│   └── training_dataset.json      # Generated training dataset
└── standard_clauses.json          # Enhanced clause library
```

## ⚡ Performance Expectations

- **Processing Speed**: ~30-60 seconds per contract (depending on CPU cores)
- **Total Time for 500 Contracts**: ~4-8 hours
- **Memory Usage**: ~2-4 GB RAM
- **Disk Space**: ~2-3x the size of your contracts

## 🛠️ Troubleshooting

### If Processing is Slow
```bash
# Use more workers (if you have a powerful CPU)
python src/run_batch_processing.py --source /path/to/contracts --workers 8
```

### If You Get Memory Errors
```bash
# Use fewer workers
python src/run_batch_processing.py --source /path/to/contracts --workers 2
```

### If Processing Hangs
```bash
# Check the logs
cat data/processed/processing_results.json

# Restart with fewer workers
python src/run_batch_processing.py --source /path/to/contracts --workers 1
```

## 🎯 Best Practices

1. **Always test with 10 contracts first**
2. **Monitor your system resources**
3. **Keep backups of original contracts**
4. **Use SSD storage for better performance**
5. **Don't interrupt the process once started**

## 📞 Need Help?

If you encounter issues:
1. Check the troubleshooting section above
2. Review the detailed logs in `data/processed/`
3. Try with fewer workers or smaller batches
4. Ensure all dependencies are installed correctly

## 🎉 Success!

Once processing is complete:
- Review the generated training dataset
- Analyze the enhanced standard clauses library
- Use the results for your AI model training
- Build APIs or web interfaces for clause analysis

---

**You're all set! Start with Step 1 and you'll have your 500 contracts processed efficiently. 🚀** 