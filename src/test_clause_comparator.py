from utils.clause_comparator import ClauseComparator

def main():
    # Initialize the clause comparator
    comparator = ClauseComparator()
    
    # Load standard clauses
    standard_clauses = comparator.load_standard_clauses('data/standard_clauses.json')
    
    # Example customer clause
    customer_clause = """
    All payments must be made within 45 days of invoice receipt.
    Any disputes regarding the invoice must be notified within 10 days.
    Late payment will incur a 1.5% monthly interest charge.
    """
    
    # Analyze the customer clause
    analysis = comparator.analyze_clause(customer_clause, standard_clauses)
    
    # Print results
    print("\nClause Analysis Results:")
    print("-----------------------")
    
    if analysis["status"] == "match_found":
        print(f"\nMatching Clause: {analysis['matching_clause']['title']}")
        print(f"Similarity Score: {analysis['similarity_score']:.2%}")
        
        print("\nDifferences Found:")
        for diff in analysis["differences"]:
            print(f"\nType: {diff['type']}")
            if diff['original']:
                print(f"Original: {diff['original'].strip()}")
            if diff['revised']:
                print(f"Revised: {diff['revised'].strip()}")
    else:
        print(analysis["message"])

if __name__ == "__main__":
    main() 