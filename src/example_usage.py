from process_agreement import process_agreement

def process_document_from_file(file_path: str, file_type: str):
    """
    Process a document file and get analysis without saving intermediate text.
    
    Args:
        file_path (str): Path to the document file
        file_type (str): Type of file ('pdf', 'docx', or 'txt')
    """
    try:
        # Read file directly into memory
        with open(file_path, 'rb') as f:
            file_content = f.read()
        
        # Process the document
        print(f"\nProcessing {file_type.upper()} document: {file_path}")
        results = process_agreement(file_content=file_content, file_type=file_type)
        
        # Print analysis results
        print("\nAnalysis Results:")
        print("=" * 80)
        
        for i, result in enumerate(results, 1):
            print(f"\nClause {i}:")
            print(f"Title: {result['clause_title']}")
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
            
            print("-" * 40)
        
        return results
    
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        return None

def process_direct_text(agreement_text: str):
    """
    Process agreement text directly without file handling.
    
    Args:
        agreement_text (str): The agreement text to analyze
    """
    try:
        # Process the text directly
        print("\nProcessing agreement text...")
        results = process_agreement(agreement_text=agreement_text)
        
        # Print analysis results
        print("\nAnalysis Results:")
        print("=" * 80)
        
        for i, result in enumerate(results, 1):
            print(f"\nClause {i}:")
            print(f"Title: {result['clause_title']}")
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
            
            print("-" * 40)
        
        return results
    
    except Exception as e:
        print(f"Error processing text: {str(e)}")
        return None

if __name__ == "__main__":
    # Example 1: Process a file
    file_path = "path/to/your/agreement.pdf"
    file_type = "pdf"
    results = process_document_from_file(file_path, file_type)
    
    # Example 2: Process direct text
    agreement_text = """
    1. Party A shall provide parcel transportation services on highway line-haul routes based on the needs of Party B.
    
    2. Period of transportation services: this Agreement is valid for an indefinite term.
    
    3. Freight and payment method: Party A pays freight based on carload rate.
    """
    results = process_direct_text(agreement_text)

    # Process a PDF file
    results = process_document_from_file("your_agreement.pdf", "pdf")

    # Process a DOCX file
    results = process_document_from_file("your_agreement.docx", "docx")

    # Your agreement text
    text = """
    1. Party A shall provide services.
    2. Party B agrees to pay within 30 days.
    """

    # Get analysis
    results = process_direct_text(text) 