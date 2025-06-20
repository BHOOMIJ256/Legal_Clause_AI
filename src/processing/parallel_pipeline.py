import os
from pathlib import Path
from typing import List, Dict, Tuple
import json
from tqdm import tqdm
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import traceback

from processing.pipeline import ClauseProcessingPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParallelClauseProcessor:
    def __init__(self, 
                 contracts_dir: str = "data/agreements",
                 output_dir: str = "data/processed",
                 standard_clauses_file: str = "data/raw/standard_clauses.json",
                 max_workers: int = None):
        self.contracts_dir = Path(contracts_dir)
        self.output_dir = Path(output_dir)
        self.standard_clauses_file = Path(standard_clauses_file)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set number of workers (default to CPU count - 1)
        self.max_workers = max_workers or (cpu_count() - 1)
        
        # Initialize pipeline for each worker
        self.pipeline = ClauseProcessingPipeline(
            contracts_dir=contracts_dir,
            output_dir=output_dir,
            standard_clauses_file=standard_clauses_file
        )
    
    def process_contract_worker(self, contract_path: Path) -> Dict:
        """Worker function to process a single contract."""
        try:
            return self.pipeline.process_contract(contract_path)
        except Exception as e:
            logger.error(f"Error processing {contract_path}: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "status": "error",
                "contract_name": contract_path.name,
                "message": str(e)
            }
    
    def process_all_contracts(self) -> List[Dict]:
        """Process all contracts in parallel."""
        contract_files = [f for f in self.contracts_dir.glob("**/*.docx") if not f.name.startswith("._")]
        results = []
        
        logger.info(f"Found {len(contract_files)} contracts to process")
        logger.info(f"Using {self.max_workers} workers")
        
        # Process contracts in parallel
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_contract = {
                executor.submit(self.process_contract_worker, contract_path): contract_path
                for contract_path in contract_files
            }
            
            # Process results as they complete
            for future in tqdm(as_completed(future_to_contract), 
                             total=len(contract_files),
                             desc="Processing contracts"):
                contract_path = future_to_contract[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Save progress after each contract
                    self._save_progress(results)
                    
                except Exception as e:
                    logger.error(f"Error processing {contract_path}: {str(e)}")
                    results.append({
                        "status": "error",
                        "contract_name": contract_path.name,
                        "message": str(e)
                    })
        
        return results
    
    def _save_progress(self, results: List[Dict]):
        """Save processing progress."""
        # Save results
        with open(self.output_dir / "processing_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Update standard clauses
        self.pipeline._save_standard_clauses()
    
    def run_pipeline(self):
        """Run the complete parallel pipeline."""
        # Process all contracts
        results = self.process_all_contracts()
        
        # Generate training dataset
        positive_pairs, negative_pairs = self.pipeline.generate_training_dataset()
        
        # Print summary
        total_contracts = len(results)
        successful = sum(1 for r in results if r["status"] == "success")
        total_clauses = sum(r.get("total_clauses", 0) for r in results if r["status"] == "success")
        matched_clauses = sum(r.get("matched_clauses", 0) for r in results if r["status"] == "success")
        new_standard_clauses = sum(r.get("new_standard_clauses", 0) for r in results if r["status"] == "success")
        
        logger.info("\nProcessing Summary:")
        logger.info(f"Total contracts processed: {total_contracts}")
        logger.info(f"Successfully processed: {successful}")
        logger.info(f"Total clauses extracted: {total_clauses}")
        logger.info(f"Clauses matched with standards: {matched_clauses}")
        logger.info(f"New standard clauses added: {new_standard_clauses}")
        logger.info(f"\nDataset Summary:")
        logger.info(f"Positive pairs: {len(positive_pairs)}")
        logger.info(f"Negative pairs: {len(negative_pairs)}")
        logger.info(f"\nResults saved to: {self.output_dir}")

def main():
    # Initialize parallel processor
    processor = ParallelClauseProcessor()
    
    # Run parallel pipeline
    processor.run_pipeline()

if __name__ == "__main__":
    main() 