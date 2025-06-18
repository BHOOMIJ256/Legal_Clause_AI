import re
from typing import List, Dict, Optional
import spacy
import datetime

class DocumentProcessor:
    def __init__(self):
        """Initialize the DocumentProcessor with improved clause patterns and NLP capabilities."""
        # Initialize spaCy for better text processing
        self.nlp = spacy.load("en_core_web_sm")
        
        # Enhanced clause patterns to catch more variations
        self.clause_patterns = [
            r'\d+\.\s*(.*?)(?=\d+\.|$)',  # Standard numbered clauses
            r'[A-Z]\.\s*(.*?)(?=[A-Z]\.|$)',  # Lettered clauses
            r'Article\s+\d+[.:]\s*(.*?)(?=Article\s+\d+[.:]|$)',  # Article-style clauses
            r'Section\s+\d+[.:]\s*(.*?)(?=Section\s+\d+[.:]|$)'   # Section-style clauses
        ]
        
        # Common legal terms to help identify clause boundaries
        self.legal_terms = {
            'definitions': ['defined', 'definition', 'means', 'refers to'],
            'obligations': ['shall', 'must', 'will', 'agrees to', 'undertakes to'],
            'rights': ['may', 'entitled to', 'right to', 'reserves the right'],
            'termination': ['terminate', 'termination', 'end', 'expire'],
            'liability': ['liable', 'liability', 'indemnify', 'indemnification']
        }
    
    def process_document(self, text: str) -> List[Dict]:
        """
        Process a document and return structured clause data.
        
        Args:
            text (str): The document text to process
            
        Returns:
            List[Dict]: List of clauses with their metadata
        """
        if not isinstance(text, str):
            raise ValueError(f"Text must be a string, got {type(text)}")
            
        # Split text into clauses
        clauses = self.segment_document(text)
        print(f"Segmented document into {len(clauses)} clauses")
        
        # Process each clause
        processed_clauses = []
        for i, clause in enumerate(clauses, 1):
            try:
                # Ensure text is a string
                clause_text = str(clause.get('text', '')).strip()
                if not clause_text:
                    print(f"Skipping empty clause {i}")
                    continue
                    
                # Create properly formatted clause dictionary
                processed_clause = {
                    'title': str(clause.get('title', '')),
                    'text': clause_text,  # This is the key field that ClauseAnalyzer expects
                    'type': self.identify_clause_type(clause_text),
                    'key_terms': self.extract_key_terms(clause_text)
                }
                
                # Validate the clause structure
                if not isinstance(processed_clause['text'], str):
                    print(f"Invalid text type in clause {i}, skipping")
                    continue
                    
                processed_clauses.append(processed_clause)
                print(f"Processed clause {i}: {processed_clause['title'][:50]}...")
                print(f"Clause structure: {processed_clause.keys()}")
            except Exception as e:
                print(f"Error processing clause {i}: {str(e)}")
                continue
        
        print(f"Successfully processed {len(processed_clauses)} clauses")
        return processed_clauses
    
    def segment_document(self, text: str) -> List[Dict]:
        """
        Segment document into clauses.
        
        Args:
            text (str): Document text
            
        Returns:
            List[Dict]: List of clauses with titles and text
        """
        # Split text into paragraphs
        paragraphs = text.split('\n\n')
        
        clauses = []
        current_title = ""
        current_text = []
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # Check if paragraph is a title
            if self.is_title(para):
                # Save previous clause if exists
                if current_text:
                    clauses.append({
                        'title': current_title,
                        'text': '\n'.join(current_text)
                    })
                current_title = para
                current_text = []
            else:
                current_text.append(para)
        
        # Add last clause
        if current_text:
            clauses.append({
                'title': current_title,
                'text': '\n'.join(current_text)
            })
        
        return clauses
    
    def is_title(self, text: str) -> bool:
        """Check if text is likely a clause title."""
        # Check for common title patterns
        title_patterns = [
            r'^\d+\.',  # Numbered sections
            r'^[A-Z][A-Z\s]+:',  # All caps followed by colon
            r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*:',  # Title case followed by colon
        ]
        
        return any(re.match(pattern, text) for pattern in title_patterns)
    
    def identify_clause_type(self, text: str) -> str:
        """Identify the type of clause based on content."""
        # Common clause types and their keywords
        clause_types = {
            'definition': ['means', 'defined as', 'refers to'],
            'obligation': ['shall', 'must', 'will', 'agree to'],
            'prohibition': ['shall not', 'must not', 'may not'],
            'termination': ['terminate', 'termination', 'end'],
            'payment': ['payment', 'fee', 'cost', 'price'],
            'liability': ['liability', 'damages', 'indemnify'],
            'confidentiality': ['confidential', 'secret', 'proprietary'],
            'default': ['default', 'breach', 'violation']
        }
        
        text_lower = text.lower()
        for clause_type, keywords in clause_types.items():
            if any(keyword in text_lower for keyword in keywords):
                return clause_type
        
        return 'other'
    
    def extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text using spaCy."""
        doc = self.nlp(text)
        key_terms = []
        
        # Add named entities
        for ent in doc.ents:
            key_terms.append(ent.text)
        
        # Add noun phrases
        for chunk in doc.noun_chunks:
            if not chunk.text.lower() in ['the', 'a', 'an']:
                key_terms.append(chunk.text)
        
        return list(set(key_terms))
    
    def get_timestamp(self) -> str:
        """Get current timestamp."""
        return datetime.datetime.now().isoformat()
    
    def _clean_text(self, text: str) -> str:
        """
        Clean the text by removing unnecessary elements and standardizing format.
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        # Remove page numbers and headers
        text = re.sub(r'Page\s+\d+', '', text)
        text = re.sub(r'\d+\nSource:.*?\n', '', text)
        
        # Remove extra whitespace and normalize line endings
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        
        # Remove common document artifacts
        text = re.sub(r'©.*?All rights reserved\.', '', text)
        text = re.sub(r'Confidential.*?Document', '', text)
        
        return text.strip()
    
    def _is_valid_clause(self, text: str) -> bool:
        """
        Check if a text segment is a valid clause.
        
        Args:
            text (str): Text to validate
            
        Returns:
            bool: True if valid clause, False otherwise
        """
        # Minimum length check
        if len(text) < 10:
            return False
        
        # Check for common legal terms
        text_lower = text.lower()
        has_legal_term = any(term in text_lower for terms in self.legal_terms.values() for term in terms)
        
        # Check for sentence structure
        doc = self.nlp(text)
        has_verb = any(token.pos_ == "VERB" for token in doc)
        
        return has_legal_term or has_verb
    
    def _process_clause(self, clause_text: str) -> Optional[Dict]:
        """
        Process a clause and extract relevant information.
        
        Args:
            clause_text (str): The clause text to process
            
        Returns:
            Optional[Dict]: Dictionary containing processed clause information
        """
        # Extract title and clean text
        title = self._extract_clause_title(clause_text)
        cleaned_text = self._clean_clause_text(clause_text)
        
        if not cleaned_text:
            return None
        
        # Identify clause type
        clause_type = self._identify_clause_type(cleaned_text)
        
        return {
            "text": cleaned_text,
            "title": title,
            "type": clause_type,
            "key_terms": self._extract_key_terms(cleaned_text)
        }
    
    def _extract_clause_title(self, clause_text: str) -> str:
        """
        Extract a meaningful title for the clause.
        
        Args:
            clause_text (str): The clause text
            
        Returns:
            str: Extracted title
        """
        # Try to find a title in the first sentence
        first_sentence = clause_text.split('.')[0].strip()
        
        # If first sentence is too long, take first 50 chars
        if len(first_sentence) > 50:
            return first_sentence[:50] + "..."
        
        return first_sentence
    
    def _clean_clause_text(self, text: str) -> str:
        """
        Clean and normalize clause text.
        
        Args:
            text (str): Raw clause text
            
        Returns:
            str: Cleaned clause text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common artifacts
        text = re.sub(r'^\d+\.\s*', '', text)
        text = re.sub(r'^[A-Z]\.\s*', '', text)
        
        return text.strip()
    
    def _identify_clause_type(self, text: str) -> str:
        """
        Identify the type of clause based on its content.
        
        Args:
            text (str): The clause text
            
        Returns:
            str: Identified clause type
        """
        text_lower = text.lower()
        
        # Check for clause type based on keywords
        for clause_type, terms in self.legal_terms.items():
            if any(term in text_lower for term in terms):
                return clause_type
        
        return "general"
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """
        Extract key terms from the clause text.
        
        Args:
            text (str): The clause text
            
        Returns:
            List[str]: List of key terms
        """
        doc = self.nlp(text)
        key_terms = []
        
        # Extract named entities
        for ent in doc.ents:
            key_terms.append(ent.text)
        
        # Extract noun phrases
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 4:  # Limit to reasonable phrase length
                key_terms.append(chunk.text)
        
        return list(set(key_terms))
    
    def _deduplicate_clauses(self, clauses: List[Dict]) -> List[Dict]:
        """
        Remove duplicate clauses and sort by position.
        
        Args:
            clauses (List[Dict]): List of clause dictionaries
            
        Returns:
            List[Dict]: Deduplicated and sorted clauses
        """
        # Use text as key for deduplication
        unique_clauses = {}
        for clause in clauses:
            if clause['text'] not in unique_clauses:
                unique_clauses[clause['text']] = clause
        
        return list(unique_clauses.values()) 
    

 