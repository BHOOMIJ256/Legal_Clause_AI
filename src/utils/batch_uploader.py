import os
import shutil
from pathlib import Path
from typing import List, Optional
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentUploader:
    def __init__(self, 
                 source_dir: str,
                 target_dir: str = "data/agreements",
                 supported_extensions: List[str] = [".docx", ".pdf", ".txt"]):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.supported_extensions = supported_extensions
        
        # Create target directory if it doesn't exist
        self.target_dir.mkdir(parents=True, exist_ok=True)
    
    def upload_documents(self, 
                        recursive: bool = True,
                        max_files: Optional[int] = None) -> int:
        """
        Upload documents from source directory to target directory.
        
        Args:
            recursive: Whether to search subdirectories
            max_files: Maximum number of files to upload (None for all)
            
        Returns:
            Number of files uploaded
        """
        # Get all files
        if recursive:
            files = list(self.source_dir.rglob("*"))
        else:
            files = list(self.source_dir.glob("*"))
        
        # Filter for supported extensions
        files = [f for f in files if f.suffix.lower() in self.supported_extensions]
        
        if max_files:
            files = files[:max_files]
        
        # Upload files
        uploaded = 0
        for file in tqdm(files, desc="Uploading documents"):
            try:
                # Create a unique filename to avoid conflicts
                target_file = self.target_dir / f"{file.stem}_{uploaded}{file.suffix}"
                
                # Copy file
                shutil.copy2(file, target_file)
                uploaded += 1
                
            except Exception as e:
                logger.error(f"Error uploading {file}: {str(e)}")
                continue
        
        return uploaded
    
    def upload_from_zip(self, 
                       zip_path: str,
                       max_files: Optional[int] = None) -> int:
        """
        Upload documents from a zip file.
        
        Args:
            zip_path: Path to zip file
            max_files: Maximum number of files to upload
            
        Returns:
            Number of files uploaded
        """
        import zipfile
        
        uploaded = 0
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of files in zip
            files = [f for f in zip_ref.namelist() 
                    if Path(f).suffix.lower() in self.supported_extensions]
            
            if max_files:
                files = files[:max_files]
            
            # Extract files
            for file in tqdm(files, desc="Extracting documents"):
                try:
                    # Create a unique filename
                    target_file = self.target_dir / f"{Path(file).stem}_{uploaded}{Path(file).suffix}"
                    
                    # Extract file
                    with zip_ref.open(file) as source, open(target_file, 'wb') as target:
                        shutil.copyfileobj(source, target)
                    
                    uploaded += 1
                    
                except Exception as e:
                    logger.error(f"Error extracting {file}: {str(e)}")
                    continue
        
        return uploaded

def main():
    # Example usage
    uploader = DocumentUploader(
        source_dir="path/to/your/contracts",
        target_dir="data/agreements"
    )
    
    # Upload from directory
    uploaded = uploader.upload_documents(recursive=True)
    logger.info(f"Uploaded {uploaded} documents from directory")
    
    # Or upload from zip
    # uploaded = uploader.upload_from_zip("path/to/your/contracts.zip")
    # logger.info(f"Uploaded {uploaded} documents from zip")

if __name__ == "__main__":
    main() 