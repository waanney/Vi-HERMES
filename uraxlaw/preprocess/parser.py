from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple

from uraxlaw.preprocess.models import DocumentMetadata


class DocumentParser:
    """Parser for legal documents (TXT, DOCX, etc.)"""

    def parse_txt(self, file_path: str | Path) -> str:
        """Parse TXT file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def extract_metadata(self, text: str, file_path: Optional[str | Path] = None) -> Tuple[str, Optional[DocumentMetadata]]:
        """
        Extract metadata from text and return clean text with metadata.

        Looks for "--- THÔNG TIN THÊM ---" section at the end of the file.
        If metadata section not found, tries to extract from file name and content.
        
        Args:
            text: Raw document text with potential metadata section
            file_path: Optional file path to extract metadata from filename

        Returns:
            Tuple of (clean_text, metadata)
            - clean_text: Text with metadata section removed
            - metadata: DocumentMetadata object if metadata found, None otherwise
        """
        # Find metadata section
        metadata_start = text.find("--- THÔNG TIN THÊM ---")
        has_metadata_section = metadata_start != -1
        
        if not has_metadata_section:
            # Try to extract from file name and content
            metadata = self._extract_from_filename_and_content(text, file_path)
            if metadata:
                return text, metadata
            return text, None

        # Split text into content and metadata
        content_text = text[:metadata_start].strip()
        metadata_text = text[metadata_start:]

        # Parse metadata fields
        metadata = {}
        
        # Extract field patterns - improved to handle multiline
        patterns = {
            "Ten": r"Ten:\s*(.+?)(?:\n(?!\s*[A-Z][a-z]+:)|$)",
            "So hieu": r"So hieu:\s*(.+?)(?:\n(?!\s*[A-Z][a-z]+:)|$)",
            "Loai van ban": r"Loai van ban:\s*(.+?)(?:\n(?!\s*[A-Z][a-z]+:)|$)",
            "Linh vuc nganh": r"Linh vuc nganh:\s*(.+?)(?:\n(?!\s*[A-Z][a-z]+:)|$)",
            "Noi ban hanh": r"Noi ban hanh:\s*(.+?)(?:\n(?!\s*[A-Z][a-z]+:)|$)",
            "Nguoi ky": r"Nguoi ky:\s*(.+?)(?:\n(?!\s*[A-Z][a-z]+:)|$)",
            "Ngay ban hanh": r"Ngay ban hanh:\s*(.+?)(?:\n(?!\s*[A-Z][a-z]+:)|$)",
            "Ngay hieu luc": r"Ngay hieu luc:\s*(.+?)(?:\n(?!\s*[A-Z][a-z]+:)|$)",
            "Ngay dang": r"Ngay dang:\s*(.+?)(?:\n(?!\s*[A-Z][a-z]+:)|$)",
            "So cong bao": r"So cong bao:\s*(.+?)(?:\n(?!\s*[A-Z][a-z]+:)|$)",
            "Tinh trang": r"Tinh trang:\s*(.+?)(?:\n(?!\s*[A-Z][a-z]+:)|$)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, metadata_text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            if match:
                value = match.group(1).strip()
                if value:
                    metadata[key] = value

        # Map to DocumentMetadata fields
        if not metadata:
            return content_text, None

        # Extract doc_id from So hieu or generate from other fields
        doc_id = metadata.get("So hieu", "").strip()
        if not doc_id:
            # Try to generate from Loai van ban and So hieu pattern
            loai = metadata.get("Loai van ban", "").strip()
            if "Thông tư" in loai and "So hieu" in metadata_text:
                # Extract number from So hieu
                so_hieu_match = re.search(r"So hieu:\s*([^\n]+)", metadata_text, re.IGNORECASE)
                if so_hieu_match:
                    doc_id = so_hieu_match.group(1).strip()

        # Map status
        tinh_trang = metadata.get("Tinh trang", "").strip().lower()
        status = None
        if "không còn phù hợp" in tinh_trang or "hết hiệu lực" in tinh_trang:
            status = "expired"
        elif "có hiệu lực" in tinh_trang or "đang hiệu lực" in tinh_trang:
            status = "effective"
        elif "đã sửa đổi" in tinh_trang:
            status = "amended"
        elif "dự thảo" in tinh_trang:
            status = "draft"

        # Extract effect_date (prefer Ngay hieu luc, fallback to Ngay ban hanh)
        effect_date = metadata.get("Ngay hieu luc") or metadata.get("Ngay ban hanh")
        
        # Normalize date format (DD/MM/YYYY -> YYYY-MM-DD)
        if effect_date:
            # Try DD/MM/YYYY format
            date_match = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", effect_date)
            if date_match:
                day, month, year = date_match.groups()
                effect_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            else:
                # Try DD-MM-YYYY format
                date_match = re.search(r"(\d{1,2})-(\d{1,2})-(\d{4})", effect_date)
                if date_match:
                    day, month, year = date_match.groups()
                    effect_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                else:
                    # Try "ngày DD tháng MM năm YYYY" format
                    date_match = re.search(r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})", effect_date, re.IGNORECASE)
                    if date_match:
                        day, month, year = date_match.groups()
                        effect_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                    else:
                        # Try YYYY-MM-DD format (already normalized)
                        date_match = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", effect_date)
                        if not date_match:
                            # Keep original if can't parse
                            effect_date = effect_date.strip()

        # Extract field from Linh vuc nganh
        field = metadata.get("Linh vuc nganh", "").strip()
        # Take first part if comma-separated
        if "," in field:
            field = field.split(",")[0].strip()

        # Create DocumentMetadata
        doc_metadata = DocumentMetadata(
            document_id=doc_id or "UNKNOWN",
            issuing_authority=metadata.get("Noi ban hanh", "").strip() or None,
            effect_date=effect_date or None,
            field=field or None,
            status=status,
            version=None,
            source_url=None,
        )

        return content_text, doc_metadata

    def _extract_from_filename_and_content(
        self, text: str, file_path: Optional[str | Path] = None
    ) -> Optional[DocumentMetadata]:
        """
        Extract metadata from file name and document content.
        
        Args:
            text: Document text content
            file_path: File path to extract metadata from filename
            
        Returns:
            DocumentMetadata object if extraction successful, None otherwise
        """
        doc_id = None
        doc_type = None
        issuing_authority = None
        effect_date = None
        field = None
        status = "effective"  # Default status

        # Extract from filename if provided
        if file_path:
            path = Path(file_path)
            filename = path.stem  # Without extension
            
            # Pattern: number_year_type_optional.txt
            # Examples: 649_QĐ-UBND.txt, 74_2024_QĐ-UBND_1.txt, 16_2024_QĐ-TTg.txt
            import re
            
            # Try to extract doc_id from filename
            # Pattern 1: number_year_type-agency_optional.txt -> number/year/type-agency
            # Examples: 74_2024_QĐ-UBND_1.txt -> 74/2024/QĐ-UBND
            match = re.match(r"(\d+)_(\d{4})_(.+?)(?:_\d+)?$", filename)
            if match:
                number, year, type_part = match.groups()
                # Normalize type part: keep format like QĐ-UBND, QĐ-TTg, etc.
                # Remove underscores and ensure proper format
                type_part = type_part.replace("_", "-")
                # Ensure format is type-agency_code (e.g., QĐ-UBND, NĐ-CP, TT-BTC)
                doc_id = f"{number}/{year}/{type_part}"
                doc_type = self._infer_doc_type(type_part)
            else:
                # Pattern 2: number_type-agency_optional.txt -> number/type-agency
                # Examples: 649_QĐ-UBND.txt -> 649/QĐ-UBND (missing year, extract from content)
                match = re.match(r"(\d+)_(.+?)(?:_\d+)?$", filename)
                if match:
                    number, type_part = match.groups()
                    # Normalize type part
                    type_part = type_part.replace("_", "-")
                    # Try to extract year from content first
                    year_match = re.search(r"(\d{4})", text[:500])  # Check first 500 chars
                    if year_match:
                        year = year_match.group(1)
                        doc_id = f"{number}/{year}/{type_part}"
                    else:
                        # Fallback: no year (will be normalized later)
                        doc_id = f"{number}/{type_part}"
                    doc_type = self._infer_doc_type(type_part)

        # Extract from content if not found in filename
        if not doc_id:
            # Try to find "Số: X/Y/Z" pattern in content
            # Format: "Số: 74/2024/QĐ-UBND" or "Số: 649/QĐ-UBND"
            so_match = re.search(r"Số:\s*([^\n]+)", text, re.IGNORECASE)
            if so_match:
                doc_id = so_match.group(1).strip()
                # Normalize doc_id to ensure proper format
                doc_id = self._normalize_doc_id(doc_id)
                # Infer doc_type from doc_id
                if not doc_type:
                    doc_type = self._infer_doc_type(doc_id)
        
        # Normalize doc_id if it doesn't have proper format
        if doc_id:
            doc_id = self._normalize_doc_id(doc_id)

        # Extract issuing authority from content
        # Look for "ỦY BAN NHÂN DÂN" or "THỦ TƯỚNG" patterns
        uy_ban_match = re.search(r"ỦY\s+BAN\s+NHÂN\s+DÂN\s+([^\n]+)", text, re.IGNORECASE | re.MULTILINE)
        if uy_ban_match:
            issuing_authority = f"Ủy ban nhân dân {uy_ban_match.group(1).strip()}"
        else:
            thu_tuong_match = re.search(r"THỦ\s+TƯỚNG\s+CHÍNH\s+PHỦ", text, re.IGNORECASE)
            if thu_tuong_match:
                issuing_authority = "Thủ tướng Chính phủ"
            else:
                # Look for other authorities
                bo_match = re.search(r"BỘ\s+([^\n]+)", text, re.IGNORECASE)
                if bo_match:
                    issuing_authority = f"Bộ {bo_match.group(1).strip()}"

        # Extract effect_date from content
        # Look for "ngày DD tháng MM năm YYYY" pattern
        date_match = re.search(r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})", text, re.IGNORECASE)
        if date_match:
            day, month, year = date_match.groups()
            effect_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"

        # If we have at least doc_id, create metadata
        if doc_id:
            return DocumentMetadata(
                document_id=doc_id,
                issuing_authority=issuing_authority,
                effect_date=effect_date,
                field=field,
                status=status,
                version=None,
                source_url=None,
            )
        
        return None

    def _normalize_doc_id(self, doc_id: str) -> str:
        """
        Normalize doc_id to Vietnamese standard format: number/year/type-agency_code
        
        Examples:
        - "74/2024/QĐ-UBND" -> "74/2024/QĐ-UBND" (already correct)
        - "649/QĐ-UBND" -> "649/QĐ-UBND" (no year, keep as is)
        - "14/2022/QĐ-UBND" -> "14/2022/QĐ-UBND" (already correct)
        - "Nghị định_24/2014" -> "24/2014/NĐ-CP" (normalize)
        - "Luật 38/2019/QH14" -> "38/2019/L-QH14" (normalize)
        """
        import re
        
        # Remove extra spaces
        doc_id = doc_id.strip()
        
        # Pattern 1: Already in correct format: number/year/type-agency
        if re.match(r"^\d+/\d{4}/[A-ZĐ]+-[A-Z]+", doc_id):
            return doc_id
        
        # Pattern 2: number/type-agency (missing year)
        if re.match(r"^\d+/[A-ZĐ]+-[A-Z]+", doc_id):
            return doc_id
        
        # Pattern 3: "Loại văn bản số number/year" -> "number/year/type-agency"
        match = re.search(r"(Luật|Nghị định|Thông tư|Quyết định|Nghị quyết)\s+(?:số\s+)?(\d+)/(\d{4})", doc_id, re.IGNORECASE)
        if match:
            doc_type_text, number, year = match.groups()
            # Map document type to agency code
            agency_map = {
                "Luật": "L-QH",
                "Nghị định": "NĐ-CP",
                "Thông tư": "TT-BTC",  # Default, will be updated if found in content
                "Quyết định": "QĐ-TTg",  # Default
                "Nghị quyết": "NQ-HĐND",  # Default
            }
            type_prefix = agency_map.get(doc_type_text.title(), "QĐ-TTg")
            return f"{number}/{year}/{type_prefix}"
        
        # Pattern 4: "Loại văn bản number/year" -> "number/year/type-agency"
        match = re.search(r"(Luật|Nghị định|Thông tư|Quyết định|Nghị quyết)\s+(\d+)/(\d{4})", doc_id, re.IGNORECASE)
        if match:
            doc_type_text, number, year = match.groups()
            agency_map = {
                "Luật": "L-QH",
                "Nghị định": "NĐ-CP",
                "Thông tư": "TT-BTC",
                "Quyết định": "QĐ-TTg",
                "Nghị quyết": "NQ-HĐND",
            }
            type_prefix = agency_map.get(doc_type_text.title(), "QĐ-TTg")
            return f"{number}/{year}/{type_prefix}"
        
        # Pattern 5: "number/year/type" -> ensure it has agency code
        match = re.match(r"^(\d+)/(\d{4})/(.+)$", doc_id)
        if match:
            number, year, type_part = match.groups()
            # If type_part doesn't have agency code, add default
            if "-" not in type_part:
                # Infer agency from type
                if "QĐ" in type_part.upper():
                    type_part = f"{type_part}-UBND"
                elif "NĐ" in type_part.upper():
                    type_part = f"{type_part}-CP"
                elif "TT" in type_part.upper():
                    type_part = f"{type_part}-BTC"
            return f"{number}/{year}/{type_part}"
        
        # Pattern 6: "Loại văn bản_number/year" -> "number/year/type-agency"
        match = re.search(r"(Luật|Nghị định|Thông tư|Quyết định|Nghị quyết)[_\s]+(\d+)/(\d{4})", doc_id, re.IGNORECASE)
        if match:
            doc_type_text, number, year = match.groups()
            agency_map = {
                "Luật": "L-QH",
                "Nghị định": "NĐ-CP",
                "Thông tư": "TT-BTC",
                "Quyết định": "QĐ-TTg",
                "Nghị quyết": "NQ-HĐND",
            }
            type_prefix = agency_map.get(doc_type_text.title(), "QĐ-TTg")
            return f"{number}/{year}/{type_prefix}"
        
        # Return as is if can't normalize
        return doc_id

    def _infer_doc_type(self, text: str) -> str:
        """Infer document type from text (doc_id or filename pattern)."""
        text_upper = text.upper()
        if "QĐ-UBND" in text_upper or "QĐ-UBND" in text_upper:
            return "Decision"
        elif "QĐ-TTG" in text_upper or "QĐ-TTg" in text_upper:
            return "Decision"
        elif "QĐ-B" in text_upper or "QĐ-" in text_upper:
            return "Decision"
        elif "TT-" in text_upper or "THÔNG TƯ" in text_upper:
            return "Circular"
        elif "NĐ-" in text_upper or "NGHỊ ĐỊNH" in text_upper:
            return "Decree"
        elif "LUẬT" in text_upper or "L-" in text_upper:
            return "Law"
        elif "NQ-" in text_upper or "NGHỊ QUYẾT" in text_upper:
            return "Resolution"
        elif "KH-" in text_upper or "KẾ HOẠCH" in text_upper:
            return "Plan"
        elif "CTr-" in text_upper or "CHƯƠNG TRÌNH" in text_upper:
            return "Program"
        elif "VBHN" in text_upper:
            return "VBHN"
        else:
            return "Decision"  # Default

    def parse_docx(self, file_path: str | Path) -> str:
        """Parse DOCX file using python-docx."""
        try:
            from docx import Document

            doc = Document(file_path)
            text_parts = []
            for para in doc.paragraphs:
                if para.text.strip():
                    text_parts.append(para.text)
            return "\n".join(text_parts)
        except ImportError:
            raise ImportError("python-docx is required for DOCX parsing. Install with: pip install python-docx")

    def parse(self, file_path: str | Path) -> str:
        """
        Auto-detect file type and parse.

        Args:
            file_path: Path to document file

        Returns:
            Extracted text content (metadata section removed)
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == ".txt":
            text = self.parse_txt(path)
        elif suffix in [".docx", ".doc"]:
            text = self.parse_docx(path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}. Supported: .txt, .docx, .doc")

        # Extract and remove metadata section
        clean_text, _ = self.extract_metadata(text, file_path)
        return clean_text

    def parse_with_metadata(self, file_path: str | Path) -> Tuple[str, Optional[DocumentMetadata]]:
        """
        Parse document and extract metadata.

        Args:
            file_path: Path to document file

        Returns:
            Tuple of (clean_text, metadata)
            - clean_text: Extracted text content (metadata section removed)
            - metadata: DocumentMetadata object if metadata found, None otherwise
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == ".txt":
            text = self.parse_txt(path)
        elif suffix in [".docx", ".doc"]:
            text = self.parse_docx(path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}. Supported: .txt, .docx, .doc")

        # Extract metadata (with file_path for filename extraction)
        clean_text, metadata = self.extract_metadata(text, file_path)
        return clean_text, metadata

