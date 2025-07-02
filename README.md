

# Legal Clause AI

## Overview

**Legal Clause AI** is an AI-powered system designed to automate the review and analysis of legal agreements. It identifies, segments, and analyzes clauses in customer agreements, compares them to a set of standard clauses, highlights differences, and proposes revised clauses using the customer’s own language. The system supports multiple document formats (PDF, DOCX, TXT) and is built for extensibility and legal team collaboration.

---

## Features

- **Multi-format Support:** Process agreements in PDF, DOCX, and TXT formats.
- **Clause Segmentation:** Automatically detects and segments agreements into individual clauses.
- **Clause Matching:** Compares each clause to a customizable set of standard clauses using advanced NLP (TF-IDF/BERT).
- **Difference Highlighting:** Clearly identifies and presents differences between customer and standard clauses.
- **Revision Suggestions:** Proposes revised clauses, articulated in the customer’s language, to align with standards.
- **Key Term Extraction:** Extracts important terms and entities from each clause.
- **Batch Processing:** Analyze multiple agreements at once and save results for review.
- **Review-Ready Output:** Exports results in tabular formats (Excel, HTML, CSV) for easy legal team review.

---

## Workflow

1. **Place Agreements:** Add agreement files to `src/data/agreements/`.
2. **Run Batch Processing:** Execute `run_batch_processing.py` to analyze all agreements.
3. **Automated Analysis:** The system extracts, segments, matches, and analyzes each clause.
4. **Results Compilation:** Results are saved in `data/processed/processing_results.json`.
5. **Tabular Export:** Use the provided script to convert JSON results to Excel/HTML/CSV for legal review.
6. **Legal Review:** Legal team reviews, filters, and comments on the results in a familiar tabular format.

---

## Output Structure

Each processed agreement includes:
- **Contract Name**
- **Total Clauses Detected**
- **Matched Clauses**
- **Per-Clause Analysis:**
  - Clause Title
  - Original Text
  - Analysis Status
  - Similarity Score
  - Matching Standard Clause (title & text)
  - Key Terms
  - Differences (with type, original, and revised text)
  - Suggested Revision

---

## Evaluation

- **Manual Review:** Legal experts can review the tabular output to assess clause matching, difference highlighting, and revision quality.
- **Metrics:** Precision, recall, and acceptability of suggestions can be tracked using a gold standard dataset.
- **Continuous Improvement:** Feedback from legal reviewers is used to refine clause segmentation, matching, and revision logic.

---

## Getting Started

### Prerequisites

- Python 3.8+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### Usage

1. **Add Agreements:**  
   Place your agreement files in `src/data/agreements/`.

2. **Run Batch Processing:**  
   ```bash
   python run_batch_processing.py
   ```

3. **Export Results for Review:**  
   Use the provided script (e.g., `export_results_to_excel.py`) to convert JSON to Excel:
   ```bash
   python export_results_to_excel.py
   ```

4. **Review Results:**  
   Open the generated Excel/HTML/CSV file for legal review.

---

## Project Structure

```
src/
  data/
    agreements/                # Input agreements
    processed/                 # Processed results
    standard_clauses.json      # Standard clauses (customizable)
  models/
    clause_analyzer.py         # Core clause analysis logic
  utils/
    document_handler.py        # File extraction utilities
    document_processor.py      # Clause segmentation & key term extraction
    clause_comparator.py       # Clause comparison logic
  run_batch_processing.py      # Main batch processing script
  export_results_to_excel.py   # Script to export results for review
```

---

## Customization

- **Standard Clauses:**  
  Update `standard_clauses.json` to reflect your domain or organization’s preferred language and requirements.

- **Evaluation:**  
  Use the evaluation workflow to measure and improve model performance.

---

## Limitations & Future Work

- **Revision Suggestions:**  
  The quality of suggested revisions depends on the underlying NLP models and can be further improved.
- **Edge Cases:**  
  Highly non-standard or ambiguous clauses may require manual review.
- **UI:**  
  A web-based review interface can be added for enhanced collaboration.

---

## License

[MIT License] (or your preferred license)

---

## Contact

For questions, support, or contributions, please contact:  
**Your Name / Team**  
**your.email@domain.com**

---

Let me know if you want this as a file, or if you want to add/change any sections!
