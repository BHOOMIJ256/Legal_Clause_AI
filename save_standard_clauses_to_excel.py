import json
import pandas as pd

# Load the summarized clauses
with open("data/raw/standard_clauses_summarized.json") as f:
    data = json.load(f)
    clauses = data["clauses"] if "clauses" in data else data

# Prepare data for DataFrame
rows = []
for clause in clauses:
    rows.append({
        "Title": clause.get("title", ""),
        "Text": clause.get("text", ""),
        "Summary": clause.get("summary", ""),
        "Category": clause.get("category", ""),
        "Clause ID": clause.get("clause_id", ""),
        "Clause Number": clause.get("clause_number", ""),
    })

df = pd.DataFrame(rows)

# Save to Excel
output_path = "data/raw/standard_clauses_summarized.xlsx"
df.to_excel(output_path, index=False)
print(f"Exported to {output_path}") 