import os
from pathlib import Path
from typing import List, Dict, Tuple
import json
from tqdm import tqdm
import logging

# Update imports to reflect new structure
from models.clause_analyzer import ClauseAnalyzer
from processing.document_handler import DocumentHandler
from processing.document_processor import DocumentProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClauseProcessingPipeline:
    def __init__(self, 
                 contracts_dir: str = "data/agreements",
                 output_dir: str = "data/processed",
                 standard_clauses_file: str = "data/raw/standard_clauses.json"):
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
                data = json.load(f)
                # Handle both formats: {"clauses": [...]} and [...]
                if isinstance(data, dict) and 'clauses' in data:
                    return data['clauses']
                elif isinstance(data, list):
                    return data
                else:
                    return []
        return []
    
    def _save_standard_clauses(self):
        """Save updated standard clauses as a dict with a 'clauses' key."""
        # Assign clause_id and clause_number to new clauses if missing
        for idx, clause in enumerate(self.standard_clauses, 1):
            if 'clause_id' not in clause or not clause['clause_id']:
                clause['clause_id'] = f"CLS{idx:03d}"
            if 'clause_number' not in clause or not clause['clause_number']:
                clause['clause_number'] = idx
        with open(self.standard_clauses_file, 'w') as f:
            json.dump({"clauses": self.standard_clauses}, f, indent=2)
    
    def process_contract(self, contract_path: Path) -> Dict:
        """Process a single contract and return analysis results."""
        try:
            # Determine file type from extension
            file_type = contract_path.suffix.lower().replace('.', '')  # 'docx', 'pdf', 'txt'
            if file_type not in ['docx', 'pdf', 'txt']:
                logging.warning(f"Skipping file {contract_path.name}: unsupported file type '{file_type}'")
                return {"status": "skipped", "contract_name": contract_path.name, "message": f"Unsupported file type: {file_type}"}
            with open(contract_path, 'rb') as f:
                file_content = f.read()
            # Extract text from the document
            try:
                text = self.document_handler.process_document(file_content, file_type)
            except Exception as e:
                logging.warning(f"Skipping file {contract_path.name}: could not extract text ({e})")
                return {"status": "skipped", "contract_name": contract_path.name, "message": f"Text extraction failed: {e}"}
            # Segment/process the text into clauses
            clauses = self.document_processor.segment_document(text)
            logging.info(f"Extracted {len(clauses)} clauses from {contract_path.name}")
            if not clauses:
                logging.warning(f"No clauses extracted from {contract_path.name}")
                return {"status": "error", "contract_name": contract_path.name, "message": "No clauses extracted"}
            # Process clauses (add metadata, etc.)
            processed_clauses = self.document_processor.process_document(text)
            # Analyze clauses
            analysis_results = self.clause_analyzer.analyze_document(processed_clauses)
            # Log clause-to-standard matches
            for i, (clause, result) in enumerate(zip(processed_clauses, analysis_results), 1):
                if result["analysis_status"] == "match_found":
                    match_title = result["matching_standard_clause"].get("title", "")
                    match_score = result.get("similarity_score", 0)
                    logging.info(f"Contract: {contract_path.name} | Clause {i} ('{clause['title']}') matched standard clause '{match_title}' (score: {match_score:.2f})")
                else:
                    logging.info(f"Contract: {contract_path.name} | Clause {i} ('{clause['title']}') did not match any standard clause.")
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
            logging.error(f"Error processing contract {contract_path.name}: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return {
                "status": "error",
                "contract_name": contract_path.name,
                "message": str(e)
            }
    
    def _update_standard_clauses(self, processed_clauses: List[Dict], analysis_results: List[Dict]):
        """Update standard clauses based on analysis results."""
        for i, (clause, result) in enumerate(zip(processed_clauses, analysis_results), 1):
            if result["analysis_status"] == "no_match":
                new_clause = {
                    "title": clause.get("title", "Unknown"),
                    "text": clause.get("text", ""),
                    "category": self.clause_analyzer._categorize_clause(clause.get("text", "")),
                    "variants": []
                }
                self.standard_clauses.append(new_clause)
                logging.info(f"Added new standard clause from contract clause {i}: '{clause.get('title', '')}'")
            elif result["analysis_status"] == "match_found":
                matching_clause = result.get("matching_standard_clause", {})
                if isinstance(matching_clause, dict) and result["similarity_score"] < 0.95:
                    matching_text = matching_clause.get("text", "")
                    for std_clause in self.standard_clauses:
                        if isinstance(std_clause, dict) and std_clause.get("text") == matching_text:
                            if "variants" not in std_clause:
                                std_clause["variants"] = []
                            std_clause["variants"].append(clause.get("text", ""))
                            logging.info(f"Added variant to standard clause '{std_clause.get('title', '')}' from contract clause {i}: '{clause.get('title', '')}'")
                            break
    
    def process_all_contracts(self) -> List[Dict]:
        """Process all contracts in the directory."""
        contract_files = [f for f in self.contracts_dir.glob("**/*.docx") if not f.name.startswith("._")]
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
        positive_pairs = []
        for clause in self.standard_clauses:
            if not isinstance(clause, dict):
                continue
            clause_id = clause.get('clause_id') or clause.get('id')
            text = clause.get('text', '')
            category = clause.get('category', 'Uncategorized')
            positive_pairs.append({
                "id": clause_id,
                "text": text,
                "label": 1,
                "category": category
            })
            variations = clause.get('metadata', {}).get('variations', [])
            for variant in variations:
                if isinstance(variant, str):
                    positive_pairs.append({
                        "id": clause_id,
                        "text": variant,
                        "label": 1,
                        "category": category
                    })
                elif isinstance(variant, dict) and "text" in variant:
                    positive_pairs.append({
                        "id": clause_id,
                        "text": variant["text"],
                        "label": 1,
                        "category": category
                    })
        # Create negative pairs (different standard clauses)
        negative_pairs = []
        for i, clause1 in enumerate(self.standard_clauses):
            if not isinstance(clause1, dict):
                continue
            for clause2 in self.standard_clauses[i+1:]:
                if not isinstance(clause2, dict):
                    continue
                negative_pairs.append({
                    "id": clause1.get('clause_id') or clause1.get('id'),
                    "text": clause1.get("text", ""),
                    "label": 0,
                    "category": clause1.get('category', 'Uncategorized')
                })
                negative_pairs.append({
                    "id": clause2.get('clause_id') or clause2.get('id'),
                    "text": clause2.get("text", ""),
                    "label": 0,
                    "category": clause2.get('category', 'Uncategorized')
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