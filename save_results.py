import json
import pandas as pd

# Load the JSON data
with open('data/processed/processing_results.json', 'r') as f:
    data = json.load(f)

rows = []
for contract in data:
    try:
        contract_name = contract.get('contract_name', 'Unknown')
        total_clauses = contract.get('total_clauses', 0)
        matched_clauses = contract.get('matched_clauses', 0)
        new_standard_clauses = contract.get('new_standard_clauses', 0)
        
        first_row = True
        for clause in contract.get('clauses', []):
        row = {
                'Contract Name': contract_name if first_row else '',
                'Total Clauses': total_clauses,
                'Matched Clauses': matched_clauses,
                'New Standard Clauses': new_standard_clauses,
                'Clause Title': clause.get('clause_title', ''),
                'Clause Type': clause.get('clause_type', ''),
                'Original Text': clause.get('original_text', ''),
                'Analysis Status': clause.get('analysis_status', ''),
                'Similarity Score': clause.get('similarity_score', 0.0),
                'Matching Standard Clause Title': clause.get('matching_standard_clause', {}).get('title', ''),
                'Matching Standard Clause Text': clause.get('matching_standard_clause', {}).get('text', ''),
                'Differences': str(clause.get('differences', [])),
                'Suggested Revision': clause.get('suggested_revision', ''),
                'Summary': clause.get('summary', ''),
                'Warning': clause.get('warning', '')
        }
        rows.append(row)
            first_row = False
    except Exception as e:
        print(f"Error processing contract {contract.get('contract_name', 'Unknown')}: {e}")
        continue

df = pd.DataFrame(rows)
df.to_excel('data/processed/processing_results_review.xlsx', index=False)
print("Excel file created: data/processed/processing_results_review.xlsx")
print(f"Total rows exported: {len(rows)}")