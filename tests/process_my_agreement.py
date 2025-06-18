import os
import sys
from typing import Dict, List, Optional

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import from src directory
from src.process_agreement import process_agreement
from src.utils.document_handler import DocumentHandler
from src.models.clause_analyzer import ClauseAnalyzer

def get_supported_file_types():
    """Return list of supported file types."""
    return ['pdf', 'docx', 'txt']

def list_available_agreements() -> List[str]:
    """List all available agreements in the agreements directory."""
    # Use absolute path for agreements directory
    agreements_dir = os.path.join(project_root, "src", "data", "agreements")
    
    # Create the directory if it doesn't exist
    if not os.path.exists(agreements_dir):
        os.makedirs(agreements_dir, exist_ok=True)
        print(f"Created agreements directory at: {agreements_dir}")
        print("Please add your agreement files to this directory.")
        return []
    
    agreements = []
    for file in os.listdir(agreements_dir):
        if file.endswith(('.pdf', '.docx', '.txt')):
            agreements.append(file)
    
    return sorted(agreements)

def process_agreement_file(file_path: str) -> Optional[Dict]:
    """Process a single agreement file."""
    try:
        print(f"\nReading file: {file_path}")
        # Read the file
        with open(file_path, 'rb') as f:
            file_content = f.read()
        
        # Determine file type
        file_type = file_path.split('.')[-1].lower()
        print(f"File type: {file_type}")
        print(f"File content type: {type(file_content)}")
        print(f"File content length: {len(file_content)} bytes")
        
        # Process the agreement
        print("Processing agreement...")
        result = process_agreement(file_content=file_content, file_type=file_type)
        
        if result is None:
            print("No results returned from process_agreement")
            return None
            
        return result
        
    except Exception as e:
        print(f"Error processing agreement: {str(e)}")
        import traceback
        print("Full error traceback:")
        print(traceback.format_exc())
        return None

def display_results(results: Dict):
    """Display the analysis results in a formatted way."""
    if not results:
        print("No results to display.")
        return
    
    print("\nDetailed Clause Analysis:")
    print("=" * 80)
    
    for clause in results.get('clauses', []):
        print(f"\nClause Title: {clause.get('clause_title', 'N/A')}")
        print("-" * 40)
        print(f"Original Text: {clause.get('original_text', 'N/A')[:100]}...")
        print(f"Analysis Status: {clause.get('analysis_status', 'N/A')}")
        print(f"Similarity Score: {clause.get('similarity_score', 0):.2f}")
        
        if clause.get('analysis_status') == 'match_found':
            print("\nMatching Standard Clause:")
            print(f"Title: {clause['matching_standard_clause']['title']}")
            print(f"Text: {clause['matching_standard_clause']['text']}")
            
            print("\nKey Differences:")
            for diff in clause.get('differences', []):
                print(f"- {diff}")
            
            print("\nSuggested Revision:")
            print(clause.get('suggested_revision', 'N/A'))
        
        print("=" * 80)

def main():
    """Main function to process agreements."""
    # List available agreements
    agreements = list_available_agreements()
    if not agreements:
        print("No agreements found in the agreements directory.")
        print(f"Please add your agreement files to: {os.path.join(project_root, 'src', 'data', 'agreements')}")
        return
    
    print("\nAvailable agreements:")
    for i, agreement in enumerate(agreements, 1):
        print(f"{i}. {agreement}")
    
    while True:
        choice = input("\nEnter the number of the agreement to process (or 'q' to quit): ")
        
        if choice.lower() == 'q':
            break
        
        try:
            index = int(choice) - 1
            if 0 <= index < len(agreements):
                selected_agreement = agreements[index]
                print(f"\nProcessing agreement: {selected_agreement}")
                
                # Process the agreement using absolute path
                file_path = os.path.join(project_root, "src", "data", "agreements", selected_agreement)
                results = process_agreement_file(file_path)
                
                if results:
                    display_results(results)
                else:
                    print("Failed to process the agreement.")
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number or 'q' to quit.")

if __name__ == "__main__":
    main() 