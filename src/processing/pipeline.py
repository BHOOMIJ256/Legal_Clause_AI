import os
from pathlib import Path
from typing import List, Dict, Tuple
import json
from tqdm import tqdm
import logging

# Update imports to reflect new structure
from ..models.clause_analyzer import ClauseAnalyzer
from .document_handler import DocumentHandler
from .document_processor import DocumentProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClauseProcessingPipeline:
    def __init__(self, 
                 contracts_dir: str = "data/agreements",
                 output_dir: str = "data/processed",
                 standard_clauses_file: str = "data/standard_clauses.json"):
        self.contracts_dir = Path(contracts_dir)
        self.output_dir = Path(output_dir)
        self.standard_clauses_file = Path(standard_clauses_file)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.document_handler = DocumentHandler()
        self.document_processor = DocumentProcessor()
        self.clause_analyzer = ClauseAnalyzer()
        
        # Load existing standard clauses
        self.standard_clauses = self._load_standard_clauses()
        
    def _load_standard_clauses(self) -> List[Dict]:
        """Load existing standard clauses."""
        if self.standard_clauses_file.exists():
            with open(self.standard_clauses_file, 'r') as f:
                return json.load(f)
        return []
    
    def _save_standard_clauses(self):
        """Save updated standard clauses."""
        with open(self.standard_clauses_file, 'w') as f:
            json.dump(self.standard_clauses, f, indent=2)
    
    def process_contract(self, contract_path: Path) -> Dict:
        """Process a single contract and return analysis results."""
        try:
            # Extract clauses
            clauses = self.document_handler.extract_clauses(str(contract_path))
            if not clauses:
                return {"status": "error", "message": "No clauses extracted"}
            
            # Process clauses
            processed_clauses = self.document_processor.process_clauses(clauses)
            
            # Analyze clauses
            analysis_results = self.clause_analyzer.analyze_document(processed_clauses)
            
            # Update standard clauses
            self._update_standard_clauses(processed_clauses, analysis_results)
            
            return {
                "status": "success",
                "contract_name": contract_path.name,
                "total_clauses": len(clauses),
                "matched_clauses": sum(1 for r in analysis_results if r["analysis_status"] == "match_found"),
                "new_standard_clauses": len(self.standard_clauses) - len(self._load_standard_clauses())
            }
            
        except Exception as e:
            return {
                "status": "error",
                "contract_name": contract_path.name,
                "message": str(e)
            }
    
    def _update_standard_clauses(self, processed_clauses: List[Dict], analysis_results: List[Dict]):
        """Update standard clauses based on analysis results."""
        for clause, result in zip(processed_clauses, analysis_results):
            if result["analysis_status"] == "no_match":
                # Add as new standard clause if it's not a match
                new_clause = {
                    "title": clause.get("title", "Unknown"),
                    "text": clause.get("text", ""),
                    "category": self.clause_analyzer._categorize_clause(clause.get("text", "")),
                    "variants": []
                }
                self.standard_clauses.append(new_clause)
            elif result["analysis_status"] == "match_found":
                # Update existing standard clause with new variant if significantly different
                matching_clause = result["matching_standard_clause"]
                if result["similarity_score"] < 0.95:  # If not too similar
                    for std_clause in self.standard_clauses:
                        if std_clause["text"] == matching_clause["text"]:
                            std_clause["variants"].append(clause.get("text", ""))
                            break
    
    def process_all_contracts(self) -> List[Dict]:
        """Process all contracts in the directory."""
        contract_files = list(self.contracts_dir.glob("**/*.docx"))
        results = []
        
        logger.info(f"Found {len(contract_files)} contracts to process")
        
        for contract_path in tqdm(contract_files, desc="Processing contracts"):
            result = self.process_contract(contract_path)
            results.append(result)
            
            # Save progress after each contract
            self._save_standard_clauses()
            
            # Save detailed results
            with open(self.output_dir / "processing_results.json", 'w') as f:
                json.dump(results, f, indent=2)
        
        return results
    
    def generate_training_dataset(self) -> Tuple[List[Dict], List[Dict]]:
        """Generate training dataset from standard clauses."""
        # Create positive pairs (standard clause with its variants)
        positive_pairs = []
        for clause in self.standard_clauses:
            positive_pairs.append({
                "text": clause["text"],
                "label": 1,
                "category": clause["category"]
            })
            for variant in clause.get("variants", []):
                positive_pairs.append({
                    "text": variant,
                    "label": 1,
                    "category": clause["category"]
                })
        
        # Create negative pairs (different standard clauses)
        negative_pairs = []
        for i, clause1 in enumerate(self.standard_clauses):
            for clause2 in self.standard_clauses[i+1:]:
                negative_pairs.append({
                    "text": clause1["text"],
                    "label": 0,
                    "category": clause1["category"]
                })
                negative_pairs.append({
                    "text": clause2["text"],
                    "label": 0,
                    "category": clause2["category"]
                })
        
        # Save dataset
        dataset = {
            "positive_pairs": positive_pairs,
            "negative_pairs": negative_pairs
        }
        with open(self.output_dir / "training_dataset.json", 'w') as f:
            json.dump(dataset, f, indent=2)
        
        return positive_pairs, negative_pairs
    
    def run_pipeline(self):
        """Run the complete pipeline."""
        # Process all contracts
        results = self.process_all_contracts()
        
        # Generate training dataset
        positive_pairs, negative_pairs = self.generate_training_dataset()
        
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
    # Initialize pipeline
    pipeline = ClauseProcessingPipeline()
    
    # Run complete pipeline
    pipeline.run_pipeline()

if __name__ == "__main__":
    main() 