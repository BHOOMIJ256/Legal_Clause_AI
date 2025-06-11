import json
from utils.document_processor import DocumentProcessor
from models.clause_analyzer import ClauseAnalyzer

def process_agreement(agreement_text):
    # Initialize document processor and clause analyzer
    doc_processor = DocumentProcessor()
    analyzer = ClauseAnalyzer()
    
    # Load standard clauses
    with open('src/data/standard_clauses.json', 'r') as f:
        standard_clauses = json.load(f)
    
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
    
    # Save analysis results to file
    with open('agreement_analysis.json', 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    return analysis_results

if __name__ == "__main__":
    # Read the agreement text
    with open('road_transportation_agreement.txt', 'r') as f:
        agreement_text = f.read()
    
    # Process and analyze the agreement
    results = process_agreement(agreement_text) 