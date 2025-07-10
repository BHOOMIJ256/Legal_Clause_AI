import json
import google.generativeai as genai

# Set your Gemini API key here
genai.configure(api_key="AIzaSyBb0Zzd1GLDeem5_7X5RNJ7M2C3Ze1hcPg")

def is_meaningful(text):
    return len(text.strip()) > 10 and any(c.isalpha() for c in text)

def summarize_clause(text):
    prompt = (
        "You are a legal expert. Given the following legal clause, write a concise summary (1-2 sentences) that captures the essential legal meaning, obligations, and intent of the clause.\n"
        "- Do NOT omit any critical legal terms, conditions, or exceptions.\n"
        "- Use clear, professional language suitable for a legal summary.\n"
        "- If the clause is extremely short and not meaningful (e.g., just a few words or not a complete thought), respond with: SKIP\n"
        "- If the clause is a standard legal phrase, summarize it in plain English.\n"
        "- Do NOT add any information that is not present in the original clause.\n\n"
        f"Clause:\n{text}\n\nSummary:"
    )
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

with open("data/raw/standard_clauses.json") as f:
    data = json.load(f)
    clauses = data["clauses"] if "clauses" in data else data

summarized_clauses = []
for clause in clauses:
    text = clause.get("text", "")
    if not is_meaningful(text):
        continue  # Skip non-meaningful clauses
    if len(text) > 300:
        try:
            summary = summarize_clause(text)
            clause["summary"] = summary
        except Exception as e:
            print(f"Error summarizing: {e}")
            clause["summary"] = ""
    else:
        clause["summary"] = text  # Use as-is for short/medium clauses
    summarized_clauses.append(clause)

with open("data/raw/standard_clauses_summarized.json", "w") as f:
    json.dump({"clauses": summarized_clauses}, f, indent=2)

print("Summarization complete! Output saved to data/raw/standard_clauses_summarized.json")
