from typing import Dict, List, Optional
import PyPDF2
import docx
import io

class DocumentHandler:
    def __init__(self):
        """Initialize the DocumentHandler."""
        pass
    
    def extract_text_from_pdf(self, pdf_file: bytes) -> str:
        """
        Extract text from a PDF file in memory.
        
        Args:
            pdf_file (bytes): PDF file content as bytes
            
        Returns:
            str: Extracted text from the PDF
        """
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def extract_text_from_docx(self, docx_file: bytes) -> str:
        """
        Extract text from a DOCX file in memory.
        
        Args:
            docx_file (bytes): DOCX file content as bytes
            
        Returns:
            str: Extracted text from the DOCX
        """
        try:
            doc = docx.Document(io.BytesIO(docx_file))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error extracting text from DOCX: {str(e)}")
    
    def extract_text_from_txt(self, txt_file: bytes) -> str:
        """
        Extract text from a TXT file in memory.
        
        Args:
            txt_file (bytes): TXT file content as bytes
            
        Returns:
            str: Extracted text from the TXT
        """
        try:
            return txt_file.decode('utf-8')
        except Exception as e:
            raise Exception(f"Error extracting text from TXT: {str(e)}")
    
    def process_document(self, file_content: bytes, file_type: str) -> str:
        """
        Process a document and extract its text content.
        
        Args:
            file_content (bytes): Document content as bytes
            file_type (str): Type of document ('pdf', 'docx', or 'txt')
            
        Returns:
            str: Extracted text from the document
        """
        file_type = file_type.lower()
        
        if file_type == 'pdf':
            return self.extract_text_from_pdf(file_content)
        elif file_type == 'docx':
            return self.extract_text_from_docx(file_content)
        elif file_type == 'txt':
            return self.extract_text_from_txt(file_content)
        else:
            raise ValueError(f"Unsupported file type: {file_type}") 