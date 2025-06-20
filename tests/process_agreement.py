import os
from typing import Dict, List, Optional
from src.processing.document_handler import DocumentHandler
from src.processing.document_processor import DocumentProcessor
from src.models.clause_analyzer import ClauseAnalyzer

def process_agreement(file_content: bytes = None, file_type: str = None, agreement_text: str = None) -> Dict:
    """
    Process an agreement and return analysis results.
    
    Args:
        file_content (bytes, optional): Raw file content if processing from file
        file_type (str, optional): Type of file ('pdf', 'docx', 'txt')
        agreement_text (str, optional): Direct text input if not processing from file
    
    Returns:
        Dict: Analysis results
    """
    try:
        print("\nInitializing components...")
        
        # Check if standard clauses file exists
        standard_clauses_path = os.path.join(os.path.dirname(__file__), "data", "transportation_standard_clauses.json")
        if not os.path.exists(standard_clauses_path):
            raise FileNotFoundError(f"Standard clauses file not found at: {standard_clauses_path}")
            
        # Initialize components
        document_handler = DocumentHandler()
        document_processor = DocumentProcessor()
        clause_analyzer = ClauseAnalyzer(standard_clauses_path=standard_clauses_path)
        
        # Get text content
        print("Getting text content...")
        if agreement_text:
            if not isinstance(agreement_text, str):
                raise ValueError(f"Agreement text must be a string, got {type(agreement_text)}")
            text_content = agreement_text
            print(f"Using provided agreement text ({len(text_content)} characters)")
        elif file_content and file_type:
            if not isinstance(file_content, bytes):
                raise ValueError(f"File content must be bytes, got {type(file_content)}")
            print(f"Processing file of type: {file_type}")
            text_content = document_handler.process_document(file_content, file_type)
            print(f"Extracted text content ({len(text_content)} characters)")
        else:
            raise ValueError("Either file_content and file_type, or agreement_text must be provided")
        
        if not isinstance(text_content, str):
            raise ValueError(f"Text content must be a string, got {type(text_content)}")
            
        if not text_content.strip():
            raise ValueError("No text content to process")
            
        # Process document
        print("\nProcessing document...")
        clauses = document_processor.process_document(text_content)
        print(f"Extracted {len(clauses)} clauses")
        
        # Analyze clauses
        print("\nAnalyzing clauses...")
        results = clause_analyzer.analyze_document(clauses)
        print(f"Analyzed {len(results)} clauses")
        
        return {
            'clauses': results,
            'metadata': {
                'total_clauses': len(results),
                'processed_at': document_processor.get_timestamp()
            }
        }
        
    except Exception as e:
        print(f"Error in process_agreement: {str(e)}")
        import traceback
        print("Full error traceback:")
        print(traceback.format_exc())
        return None

if __name__ == "__main__":
    # Example usage with direct text input
    agreement_text = """
    1. Party B shall provide parcel transportation services on highway line-haul routes based on the needs of Party A.
    2. Period of transportation services: this Agreement is valid for an indefinite term.
    3. Freight and payment method: Party A pays freight based on carload rate.
    """
    
    # Process and analyze the agreement
    results = process_agreement(agreement_text=agreement_text) 