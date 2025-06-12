import json
from utils.document_processor import DocumentProcessor
from utils.document_handler import DocumentHandler
from models.clause_analyzer import ClauseAnalyzer

def process_agreement(agreement_text: str = None, file_content: bytes = None, file_type: str = None):
    """
    Process and analyze an agreement.
    
    Args:
        agreement_text (str, optional): Direct text input of the agreement
        file_content (bytes, optional): File content as bytes if processing from a file
        file_type (str, optional): Type of file ('pdf', 'docx', or 'txt') if processing from a file
        
    Returns:
        list: Analysis results for each clause
    """
    # Initialize document processor and clause analyzer
    doc_processor = DocumentProcessor()
    analyzer = ClauseAnalyzer()
    
    # Load standard clauses
    with open('src/data/standard_clauses.json', 'r') as f:
        standard_clauses = json.load(f)
    
    # Get agreement text either from direct input or file
    if agreement_text is None and file_content is not None:
        doc_handler = DocumentHandler()
        agreement_text = doc_handler.process_document(file_content, file_type)
    elif agreement_text is None:
        raise ValueError("Either agreement_text or file_content must be provided")
    
    # Segment the document into clauses
    clauses = doc_processor.segment_document(agreement_text)
    
    # Analyze each clause
    analysis_results = []
    for clause in clauses:
        result = analyzer.analyze_clause(clause['text'])
        analysis_results.append({
            'original_clause': clause['text'],
            'clause_title': clause['title'],
            'analysis': result
        })
    
    # Print detailed analysis
    print("\nDetailed Clause Analysis:")
    print("=" * 80)
    
    for result in analysis_results:
        print(f"\nClause Title: {result['clause_title']}")
        print("-" * 40)
        print(f"Original Text: {result['original_clause'][:200]}...")
        print(f"Analysis Status: {result['analysis']['status']}")
        print(f"Similarity Score: {result['analysis']['similarity_score']:.2f}")
        
        if result['analysis']['matching_clause']:
            print("\nMatching Standard Clause:")
            print(f"Title: {result['analysis']['matching_clause']['title']}")
            print(f"Text: {result['analysis']['matching_clause']['text'][:200]}...")
        
        if result['analysis']['differences']:
            print("\nKey Differences:")
            for diff in result['analysis']['differences']:
                print(f"- {diff}")
        
        if result['analysis']['suggested_revision']:
            print("\nSuggested Revision:")
            print(result['analysis']['suggested_revision'])
        
        print("=" * 80)
    
    return analysis_results

if __name__ == "__main__":
    # Example usage with direct text input
    agreement_text = """
    1. Party B shall provide parcel transportation services on highway line-haul routes based on the needs of Party A.
    2. Period of transportation services: this Agreement is valid for an indefinite term.
    3. Freight and payment method: Party A pays freight based on carload rate.
    """
    
    # Process and analyze the agreement
    results = process_agreement(agreement_text=agreement_text) 