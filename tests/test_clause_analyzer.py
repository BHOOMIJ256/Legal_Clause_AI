from models.clause_analyzer import ClauseAnalyzer
from typing import Dict

def test_clause_analysis(analyzer: ClauseAnalyzer, clause: Dict):
    """Test clause analysis with a given analyzer."""
    print(f"\nAnalyzing Clause: {clause['title']}")
    print("=" * 50)
    
    # Get analysis
    analysis = analyzer.analyze_clause(clause['text'])
    
    # Print results
    if analysis["status"] == "match_found":
        print(f"\nClause Type: {analysis['clause_type']}")
        print(f"Similarity Score: {analysis['similarity_score']:.2%}")
        
        print("\nMatching Standard Clause:")
        print(f"Title: {analysis['matching_clause']['title']}")
        print(f"Text: {analysis['matching_clause']['text']}")
        
        print("\nDifferences Found:")
        for diff in analysis["differences"]:
            print(f"\nType: {diff['type']}")
            if diff['original']:
                print(f"Original: {diff['original'].strip()}")
            if diff['revised']:
                print(f"Revised: {diff['revised'].strip()}")
        
        print("\nSuggested Revision:")
        print(analysis["suggested_revision"])
    else:
        print(f"\nStatus: {analysis['status']}")
        print(f"Message: {analysis['message']}")
        print(f"Identified Type: {analysis['clause_type']}")

def main():
    # Example customer clauses to test
    test_clauses = [
        {
            "title": "Payment Terms",
            "text": """
            All invoices must be paid within 45 days of receipt.
            Any disputes regarding the invoice must be notified within 10 days.
            Late payments will incur a 1.5% monthly interest charge.
            """
        },
        {
            "title": "Confidentiality",
            "text": """
            Both parties agree to maintain the confidentiality of all shared information.
            The confidentiality obligations shall remain in effect for 2 years after termination.
            """
        },
        {
            "title": "Service Level Agreement",
            "text": """
            Service levels will be as specified in the Statement of Work.
            The customer may request penalties for service level breaches.
            """
        }
    ]
    
    # Test with BERT-based analyzer
    print("\nTesting with BERT-based analyzer:")
    print("=" * 50)
    bert_analyzer = ClauseAnalyzer(use_bert=True)
    for clause in test_clauses:
        test_clause_analysis(bert_analyzer, clause)
    
    # Test with simple analyzer (no BERT)
    print("\nTesting with simple analyzer (no BERT):")
    print("=" * 50)
    simple_analyzer = ClauseAnalyzer(use_bert=False)
    for clause in test_clauses:
        test_clause_analysis(simple_analyzer, clause)

if __name__ == "__main__":
    main() #Run the main function