import os
from pathlib import Path
from data.clause_dataset_generator import ClauseDatasetGenerator

def main():
    # Initialize generator with thresholds
    generator = ClauseDatasetGenerator(
        similarity_threshold=0.85,  # Threshold for matching
        update_threshold=0.95      # Threshold for updating existing clauses
    )
    
    # Generate dataset and update standard clauses
    generator.generate_dataset(
        contract_dir="data/raw/contracts",
        standard_clauses_file="data/raw/standard_clauses.json",
        output_file="data/processed/clause_dataset.json"
    )

if __name__ == "__main__":
    main() 