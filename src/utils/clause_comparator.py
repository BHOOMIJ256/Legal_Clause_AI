from typing import Dict, List, Tuple
import difflib
from Levenshtein import ratio
import spacy
import json

class ClauseComparator:
    def __init__(self):
        # Load spaCy model for NLP processing
        self.nlp = spacy.load("en_core_web_sm")
        
    def load_standard_clauses(self, file_path: str) -> Dict:
        """Load standard clauses from JSON file."""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for comparison."""
        # Convert to lowercase and remove extra whitespace
        text = text.lower().strip()
        # Remove special characters
        text = ''.join(c for c in text if c.isalnum() or c.isspace())
        return text
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using Levenshtein ratio."""
        text1 = self.preprocess_text(text1)
        text2 = self.preprocess_text(text2)
        return ratio(text1, text2)
    
    def find_matching_clause(self, customer_clause: str, standard_clauses: Dict) -> Tuple[str, float]:
        """Find the most similar standard clause for a given customer clause."""
        best_match = None
        best_score = 0.0
        
        for clause in standard_clauses['clauses']:
            score = self.calculate_similarity(customer_clause, clause['text'])
            if score > best_score:
                best_score = score
                best_match = clause
        
        return best_match, best_score
    
    def find_differences(self, text1: str, text2: str) -> List[Dict]:
        """Find differences between two texts."""
        differences = []
        
        # Create a differ object
        d = difflib.Differ()
        
        # Compare the texts
        diff = list(d.compare(text1.splitlines(), text2.splitlines()))
        
        current_diff = {"type": None, "original": "", "revised": ""}
        
        for line in diff:
            if line.startswith('+ '):
                if current_diff["type"] == "addition":
                    current_diff["revised"] += line[2:] + "\n"
                else:
                    if current_diff["type"] is not None:
                        differences.append(current_diff)
                    current_diff = {
                        "type": "addition",
                        "original": "",
                        "revised": line[2:] + "\n"
                    }
            elif line.startswith('- '):
                if current_diff["type"] == "deletion":
                    current_diff["original"] += line[2:] + "\n"
                else:
                    if current_diff["type"] is not None:
                        differences.append(current_diff)
                    current_diff = {
                        "type": "deletion",
                        "original": line[2:] + "\n",
                        "revised": ""
                    }
            elif line.startswith('? '):
                continue
            else:
                if current_diff["type"] is not None:
                    differences.append(current_diff)
                current_diff = {"type": None, "original": "", "revised": ""}
        
        if current_diff["type"] is not None:
            differences.append(current_diff)
        
        return differences
    
    def analyze_clause(self, customer_clause: str, standard_clauses: Dict) -> Dict:
        """Analyze a customer clause against standard clauses."""
        # Find the best matching standard clause
        best_match, similarity_score = self.find_matching_clause(customer_clause, standard_clauses)
        
        if best_match is None:
            return {
                "status": "no_match",
                "message": "No matching standard clause found"
            }
        
        # Find differences between the clauses
        differences = self.find_differences(best_match['text'], customer_clause)
        
        return {
            "status": "match_found",
            "matching_clause": best_match,
            "similarity_score": similarity_score,
            "differences": differences
        } 