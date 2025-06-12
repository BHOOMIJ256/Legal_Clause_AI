from utils.document_processor import DocumentProcessor

def test_document_processor():
    # Initialize the processor
    processor = DocumentProcessor()
    
    # Test text with different clause formats
    test_text = """
    Road Transportation Agreement
    
    1. Party A shall provide parcel transportation services on highway line-haul routes based on the needs of Party B.
    
    2. Period of transportation services: this Agreement is valid for an indefinite term.
    
    A. Freight and payment method:
       (a) Verification of freight: Party A pays freight based on carload rate.
       (b) Party A shall not pay any other charges other than the freight.
    
    Article 1: Definitions
    "Agreement" means this Road Transportation Agreement.
    "Services" means the transportation services provided under this Agreement.
    
    Section 2: Obligations
    Party B shall maintain valid insurance coverage throughout the term of this Agreement.
    """
    
    # Process the document
    print("\nProcessing test document...")
    clauses = processor.segment_document(test_text)
    
    # Print results
    print("\nExtracted Clauses:")
    print("=" * 80)
    
    for i, clause in enumerate(clauses, 1):
        print(f"\nClause {i}:")
        print(f"Title: {clause['title']}")
        print(f"Type: {clause['type']}")
        print(f"Text: {clause['text']}")
        print(f"Key Terms: {', '.join(clause['key_terms'])}")
        print("-" * 40)

if __name__ == "__main__":
    test_document_processor() 