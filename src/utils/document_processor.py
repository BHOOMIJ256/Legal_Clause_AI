import re
from typing import List, Dict

class DocumentProcessor:
    def __init__(self):
        self.clause_pattern = r'\d+\.\s*(.*?)(?=\d+\.|$)'
    
    def segment_document(self, text: str) -> List[Dict]:
        """Segment a document into individual clauses."""
        # Clean the text
        text = self._clean_text(text)
        
        # Find all clauses
        clauses = []
        matches = re.finditer(self.clause_pattern, text, re.DOTALL)
        
        for match in matches:
            clause_text = match.group(1).strip()
            if clause_text:
                clauses.append({
                    "text": clause_text,
                    "title": self._extract_clause_title(clause_text)
                })
        
        return clauses
    
    def _clean_text(self, text: str) -> str:
        """Clean the text by removing unnecessary whitespace and formatting."""
        # Remove page numbers and headers
        text = re.sub(r'Source:.*?\n', '', text)
        text = re.sub(r'\d+\nSource:.*?\n', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _extract_clause_title(self, clause_text: str) -> str:
        """Extract a title for the clause based on its content."""
        # Take first 50 characters or first sentence
        first_sentence = clause_text.split('.')[0]
        if len(first_sentence) > 50:
            return first_sentence[:50] + "..."
        return first_sentence 