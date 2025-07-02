import json
import pandas as pd

# Load the JSON data
with open('data/processed/processing_results.json', 'r') as f:
    data = json.load(f)

rows = []
for contract in data:
    contract_name = contract['contract_name']
    for clause in contract['clauses']:
        row = {
            'Contract Name': contract_name,
            'Clause Title': clause['clause_title'],
            'Original Text': clause['original_text'],
            'Analysis Status': clause['analysis_status'],
            'Similarity Score': clause['similarity_score'],
            'Matching Standard Clause Title': clause['matching_standard_clause']['title'],
            'Matching Standard Clause Text': clause['matching_standard_clause']['text'],
            'Key Terms': ', '.join(clause['key_terms']),
            'Differences': str(clause['differences']),
            'Suggested Revision': clause['suggested_revision']
        }
        rows.append(row)

df = pd.DataFrame(rows)
df.to_excel('data/processed/processing_results_review.xlsx', index=False)
print("Excel file created: data/processed/processing_results_review.xlsx")