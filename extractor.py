import httpx
import os
import io
import tempfile
import pdfplumber
import docx
import magic
from bs4 import BeautifulSoup
from typing import List, Dict, Generator
import logging
from config import Config
from typing import List, Any
logger = logging.getLogger(__name__)
import gc
class DocumentExtractor:
    """Memory-optimized document extractor"""
    
    def __init__(self):
        self.max_file_size = Config.MAX_DOCUMENT_SIZE_MB * 1024 * 1024  # Convert to bytes
    
    def download_file(self, url: str, output_dir: str) -> str:
        """Download file with memory optimization"""
        try:
            filename = os.path.basename(url.split("?")[0]) or "document"
            local_path = os.path.join(output_dir, filename)
            
            # Stream download to avoid loading entire file in memory
            with httpx.stream("GET", url, timeout=60.0) as response:
                response.raise_for_status()
                
                # Check file size
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > self.max_file_size:
                    raise ValueError(f"File too large: {int(content_length) / 1024 / 1024:.1f}MB")
                
                # Stream to file
                with open(local_path, "wb") as f:
                    downloaded = 0
                    for chunk in response.iter_bytes(chunk_size=8192):
                        downloaded += len(chunk)
                        if downloaded > self.max_file_size:
                            raise ValueError(f"File too large: >{Config.MAX_DOCUMENT_SIZE_MB}MB")
                        f.write(chunk)
            
            logger.info(f"Downloaded: {filename} ({downloaded / 1024 / 1024:.1f}MB)")
            return local_path
            
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            raise
    
    def extract_text_streaming(self, file_path: str) -> Generator[Dict[str, Any], None, None]:
        """Extract text in streaming fashion to manage memory"""
        try:
            mime_type = magic.from_file(file_path, mime=True)
            filename = os.path.basename(file_path)
            
            if mime_type == "application/pdf":
                yield from self._extract_pdf_streaming(file_path, filename)
            elif mime_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
                yield from self._extract_docx_streaming(file_path, filename)
            elif mime_type.startswith("text/"):
                yield from self._extract_text_streaming(file_path, filename)
            elif mime_type in ["text/html", "application/xhtml+xml"]:
                yield from self._extract_html_streaming(file_path, filename)
            else:
                logger.warning(f"Unsupported file type: {mime_type}")
                return
                
        except Exception as e:
            logger.error(f"Failed to extract text from {file_path}: {e}")
            raise
    
    def _extract_pdf_streaming(self, file_path: str, filename: str) -> Generator[Dict[str, Any], None, None]:
        """Stream PDF extraction page by page"""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    if not text.strip():
                        continue
                    
                    # Split into paragraphs
                    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
                    
                    for chunk_num, paragraph in enumerate(paragraphs):
                        if len(paragraph) < 50:  # Skip very short chunks
                            continue
                            
                        yield {
                            "doc_id": filename,
                            "page": page_num,
                            "chunk_id": f"{filename}_p{page_num}_c{chunk_num}",
                            "text": paragraph
                        }
                    
                    # Force garbage collection every 10 pages
                    if page_num % 10 == 0:
                        gc.collect()
                        
        except Exception as e:
            logger.error(f"PDF extraction failed for {file_path}: {e}")
            raise
    
    def _extract_docx_streaming(self, file_path: str, filename: str) -> Generator[Dict[str, Any], None, None]:
        """Stream DOCX extraction paragraph by paragraph"""
        try:
            doc = docx.Document(file_path)
            
            for chunk_num, paragraph in enumerate(doc.paragraphs):
                text = paragraph.text.strip()
                if len(text) < 50:  # Skip very short paragraphs
                    continue
                
                yield {
                    "doc_id": filename,
                    "page": None,
                    "chunk_id": f"{filename}_c{chunk_num}",
                    "text": text
                }
                
        except Exception as e:
            logger.error(f"DOCX extraction failed for {file_path}: {e}")
            raise
    
    def _extract_text_streaming(self, file_path: str, filename: str) -> Generator[Dict[str, Any], None, None]:
        """Stream text file extraction"""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            
            for chunk_num, paragraph in enumerate(paragraphs):
                if len(paragraph) < 50:
                    continue
                
                yield {
                    "doc_id": filename,
                    "page": None,
                    "chunk_id": f"{filename}_c{chunk_num}",
                    "text": paragraph
                }
                
        except Exception as e:
            logger.error(f"Text extraction failed for {file_path}: {e}")
            raise
    
    def _extract_html_streaming(self, file_path: str, filename: str) -> Generator[Dict[str, Any], None, None]:
        """Stream HTML extraction"""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                soup = BeautifulSoup(f, "html.parser")
            
            text = soup.get_text(separator="\n")
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            
            for chunk_num, paragraph in enumerate(paragraphs):
                if len(paragraph) < 50:
                    continue
                
                yield {
                    "doc_id": filename,
                    "page": None,
                    "chunk_id": f"{filename}_c{chunk_num}",
                    "text": paragraph
                }
                
        except Exception as e:
            logger.error(f"HTML extraction failed for {file_path}: {e}")
            raise
    
    def extract_from_url(self, url: str, temp_dir: str) -> List[Dict[str, Any]]:
        """Extract text from URL with full memory management"""
        file_path = None
        try:
            # Download file
            file_path = self.download_file(url, temp_dir)
            
            # Extract text in streaming fashion
            chunks = list(self.extract_text_streaming(file_path))
            
            logger.info(f"Extracted {len(chunks)} chunks from {url}")
            return chunks
            
        finally:
            # Cleanup downloaded file
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.info(f"Cleaned up: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup {file_path}: {e}")

# Global extractor instance
document_extractor = DocumentExtractor()

# =============================================================================

