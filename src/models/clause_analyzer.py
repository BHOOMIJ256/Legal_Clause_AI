from typing import Dict, List, Tuple, Optional, Any
import spacy
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import re
import difflib
from Levenshtein import ratio
from sklearn.feature_extraction.text import TfidfVectorizer
from difflib import SequenceMatcher
import os

class ClauseAnalyzer:
    def __init__(self, standard_clauses_path: str = "src/data/transportation_standard_clauses.json"):
        """Initialize the ClauseAnalyzer with standard clauses."""
        self.standard_clauses = self._load_standard_clauses(standard_clauses_path)
        self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize LegalBERT
        self.tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
        self.model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
        
        # Pre-compute embeddings for standard clauses
        self.standard_embeddings = self._compute_standard_embeddings()
        
    def _load_standard_clauses(self, file_path: str) -> List[Dict]:
        """Load standard clauses from JSON file."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)['clauses']
        except Exception as e:
            print(f"Error loading standard clauses: {e}")
            return []

    def _compute_standard_embeddings(self) -> Dict[str, np.ndarray]:
        """Pre-compute LegalBERT embeddings for all standard clauses."""
        embeddings = {}
        for clause in self.standard_clauses:
            text = f"{clause['title']} {clause['text']}"
            embeddings[clause['clause_id']] = self._get_bert_embedding(text)
        return embeddings

    def _get_bert_embedding(self, text: str) -> np.ndarray:
        """Get LegalBERT embedding for a given text."""
        # Tokenize and prepare input
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Get model output
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use [CLS] token embedding as sentence representation
        return outputs.last_hidden_state[0][0].numpy()

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using LegalBERT embeddings."""
        try:
            # Get embeddings
            embedding1 = self._get_bert_embedding(text1)
            embedding2 = self._get_bert_embedding(text2)
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            
            # Apply threshold
            return similarity if similarity > 0.6 else 0.0
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0

    def _find_matching_clause(self, clause_text: str) -> Tuple[Optional[Dict], float]:
        """Find the best matching standard clause using LegalBERT embeddings."""
        best_match = None
        best_score = 0.0
        
        # Get embedding for input clause
        input_embedding = self._get_bert_embedding(clause_text)
        
        # Compare with all standard clauses
        for clause in self.standard_clauses:
            standard_embedding = self.standard_embeddings[clause['clause_id']]
            score = cosine_similarity([input_embedding], [standard_embedding])[0][0]
            
            if score > best_score:
                best_score = score
                best_match = clause
        
        return best_match, best_score

    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text using spaCy."""
        doc = self.nlp(text)
        return [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop]

    def analyze_clause(self, clause_text: str, clause_title: str = "") -> Dict[str, Any]:
        """Analyze a single clause using LegalBERT."""
        # Find matching standard clause
        matching_clause, similarity_score = self._find_matching_clause(clause_text)
        
        if matching_clause and similarity_score > 0.6:
            # Extract key terms
            key_terms = self._extract_key_terms(clause_text)
            
            # Find differences
            differences = self._find_differences(clause_text, matching_clause['text'])
            
            return {
                "clause_title": clause_title,
                "original_text": clause_text,
                "analysis_status": "match_found",
                "similarity_score": similarity_score,
                "matching_standard_clause": {
                    "title": matching_clause['title'],
                    "text": matching_clause['text']
                },
                "key_terms": key_terms,
                "differences": differences,
                "suggested_revision": self._generate_revision(clause_text, matching_clause)
            }
        else:
            return {
                "clause_title": clause_title,
                "original_text": clause_text,
                "analysis_status": "no_match",
                "similarity_score": similarity_score,
                "key_terms": self._extract_key_terms(clause_text)
            }

    def _find_differences(self, original: str, standard: str) -> List[Dict]:
        """Find differences between original and standard clause."""
        matcher = SequenceMatcher(None, original, standard)
        differences = []
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != 'equal':
                differences.append({
                    'type': tag,
                    'original': original[i1:i2],
                    'revised': standard[j1:j2]
                })
        
        return differences

    def _generate_revision(self, original: str, standard_clause: Dict) -> str:
        """Generate a suggested revision based on standard clause."""
        # Extract key terms from original
        original_terms = set(self._extract_key_terms(original))
        standard_terms = set(self._extract_key_terms(standard_clause['text']))
        
        # Keep original terms that are relevant
        relevant_terms = original_terms.intersection(standard_terms)
        
        # Generate revision
        revision = standard_clause['text']
        for term in relevant_terms:
            if term not in revision:
                revision += f"\nRelevant term from original: {term}"
        
        return revision

    def analyze_document(self, clauses: List[Dict]) -> List[Dict]:
        """Analyze multiple clauses in a document."""
        return [self.analyze_clause(clause['text'], clause.get('title', '')) 
                for clause in clauses]

    def preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis."""
        # Convert to lowercase and remove extra whitespace
        text = text.lower().strip()
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^a-z0-9\s.,;:()]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def _identify_clause_type(self, text: str) -> str:
        """Identify the type of clause based on content."""
        processed_text = self.preprocess_text(text)
        key_terms = self._extract_key_terms(processed_text)
        
        best_match = None
        best_score = 0.0
        
        for clause in self.standard_clauses:
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