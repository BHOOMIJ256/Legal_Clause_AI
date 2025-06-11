from typing import Dict, List, Tuple, Optional
import spacy
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import re
import difflib
from Levenshtein import ratio

class ClauseAnalyzer:
    def __init__(self, use_bert: bool = True):
        """
        Initialize the ClauseAnalyzer.
        
        Args:
            use_bert (bool): Whether to use BERT for semantic analysis. If False, falls back to simpler methods.
        """
        # Initialize spaCy for basic NLP tasks
        self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize BERT if requested
        self.use_bert = use_bert
        if use_bert:
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.model = AutoModel.from_pretrained('bert-base-uncased')
        
        # Load standard clauses
        self.standard_clauses = self._load_standard_clauses()
        
        # Pre-compute embeddings if using BERT
        if use_bert:
            self.standard_embeddings = self._compute_standard_embeddings()
    
    def _load_standard_clauses(self) -> Dict:
        """Load standard clauses from JSON file."""
        with open('src/data/standard_clauses.json', 'r') as f:
            return json.load(f)
    
    def _compute_standard_embeddings(self) -> Dict[str, np.ndarray]:
        """Pre-compute BERT embeddings for all standard clauses."""
        embeddings = {}
        for clause in self.standard_clauses['clauses']:
            text = f"{clause['title']} {clause['text']}"
            embeddings[clause['clause_id']] = self._get_bert_embedding(text)
        return embeddings
    
    def _get_bert_embedding(self, text: str) -> np.ndarray:
        """Get BERT embedding for a given text."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[0][0].numpy()
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis."""
        # Convert to lowercase and remove extra whitespace
        text = text.lower().strip()
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^a-z0-9\s.,;:()]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text using spaCy."""
        doc = self.nlp(text)
        key_terms = []
        
        # Add named entities
        for ent in doc.ents:
            key_terms.append(ent.text.lower())
        
        # Add noun phrases
        for chunk in doc.noun_chunks:
            key_terms.append(chunk.text.lower())
        
        return list(set(key_terms))
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using either BERT or Levenshtein."""
        if self.use_bert:
            embedding1 = self._get_bert_embedding(text1)
            embedding2 = self._get_bert_embedding(text2)
            return float(cosine_similarity([embedding1], [embedding2])[0][0])
        else:
            text1 = self.preprocess_text(text1)
            text2 = self.preprocess_text(text2)
            return ratio(text1, text2)
    
    def find_matching_clause(self, customer_clause: str) -> Tuple[Optional[Dict], float]:
        """Find the most similar standard clause for a given customer clause."""
        best_match = None
        best_score = 0.0
        
        for clause in self.standard_clauses['clauses']:
            score = self.calculate_similarity(customer_clause, clause['text'])
            if score > best_score:
                best_score = score
                best_match = clause
        
        return best_match, best_score
    
    def _identify_clause_type(self, text: str) -> str:
        """Identify the type of clause based on content."""
        processed_text = self.preprocess_text(text)
        key_terms = self._extract_key_terms(processed_text)
        
        best_match = None
        best_score = 0.0
        
        for clause in self.standard_clauses['clauses']:
            category_terms = self._extract_key_terms(clause['title'] + " " + clause['text'])
            similarity = len(set(key_terms) & set(category_terms)) / len(set(key_terms) | set(category_terms))
            
            if similarity > best_score:
                best_score = similarity
                best_match = clause['category']
        
        return best_match if best_match else "Unknown"
    
    def find_differences(self, text1: str, text2: str) -> List[Dict]:
        """Find differences between two texts using difflib."""
        differences = []
        d = difflib.Differ()
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
    
    def _generate_revision(self, standard_clause: Dict, customer_clause: str, differences: List[Dict]) -> str:
        """Generate a suggested revision based on standard clause and differences."""
        revision = standard_clause['text']
        
        # Apply customer's language style where appropriate
        customer_doc = self.nlp(customer_clause)
        standard_doc = self.nlp(revision)
        
        # Keep customer's formatting and style where appropriate
        for diff in differences:
            if diff['type'] == 'additional_terms':
                # Consider incorporating some of the additional terms if they're relevant
                pass
        
        return revision
    
    def analyze_clause(self, customer_clause: str) -> Dict:
        """
        Analyze a customer clause and provide detailed analysis.
        
        Args:
            customer_clause (str): The customer's clause text to analyze
            
        Returns:
            Dict containing analysis results including:
            - status: "match_found" or "no_match"
            - clause_type: Identified type of clause
            - matching_clause: Best matching standard clause (if found)
            - similarity_score: Similarity score with best match
            - differences: List of differences found
            - suggested_revision: Suggested revision of the clause
        """
        # Preprocess the clause
        processed_clause = self.preprocess_text(customer_clause)
        
        # Identify clause type
        clause_type = self._identify_clause_type(processed_clause)
        
        # Find best matching standard clause
        best_match, similarity_score = self.find_matching_clause(processed_clause)
        
        if best_match is None:
            return {
                "status": "no_match",
                "message": "No matching standard clause found",
                "clause_type": clause_type
            }
        
        # Find differences
        differences = self.find_differences(best_match['text'], customer_clause)
        
        # Generate suggested revision
        suggested_revision = self._generate_revision(best_match, customer_clause, differences)
        
        return {
            "status": "match_found",
            "clause_type": clause_type,
            "matching_clause": best_match,
            "similarity_score": similarity_score,
            "differences": differences,
            "suggested_revision": suggested_revision
        } 