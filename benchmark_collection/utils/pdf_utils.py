from typing import List, Optional, Tuple, Dict
import PyPDF2
import re
import tiktoken
from dataclasses import dataclass
from collections import defaultdict
from docling.document_converter import DocumentConverter
import os

@dataclass
class PDFElement:
    type: str  # 'text', 'equation', 'table', 'figure', 'reference'
    content: str
    metadata: dict = None

class PDFParser:
    SECTION_PATTERNS = [
        r'^[0-9]+\.[0-9]* +[A-Z]',    # Numbered sections
        r'^[IVX]+\. +[A-Z]',          # Roman numerals
        r'^[A-Z][A-Za-z\s]{0,50}$',   # Uppercase words
        r'^[A-Z][A-Z\s&-]{2,50}$',    # All caps sections
        r'^[A-Z][a-z\s]{2,50}:',      # Title followed by colon
        r'^\d+(\.\d+)* [A-Z]'         # Hierarchical numbering
    ]
    
    SECTION_KEYWORDS = {
        "ABSTRACT", "INTRODUCTION", "BACKGROUND", "METHODOLOGY", 
        "METHOD", "RESULTS", "DISCUSSION", "CONCLUSION", 
        "REFERENCES", "RELATED WORK", "EXPERIMENTAL",
        "IMPLEMENTATION", "EVALUATION", "ANALYSIS",
        "ACKNOWLEDGMENTS", "APPENDIX"
    }

    INTRO_PATTERNS = [
        r'(?i)^1\.?\s*introduction',
        r'(?i)^I\.?\s*introduction',
        r'(?i)^introduction',
        r'(?i)^1\.?\s*INTRODUCTION',
        r'(?i)^I\.?\s*INTRODUCTION',
        r'(?i)^INTRODUCTION'
    ]

    def __init__(self):
        self.equations = []
        self.tables = []
        self.figures = []
        self.references = []
        self.docling_converter = DocumentConverter()  # Initialize Docling converter


    def _find_introduction(self, text: str) -> int:
        """Find the starting position of the introduction section."""
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if any(re.match(pattern, line.strip()) for pattern in self.INTRO_PATTERNS):
                return sum(len(l) + 1 for l in lines[:i])
        return 0

    def _remove_pre_introduction(self, text: str) -> str:
        """Remove content before the introduction section."""
        intro_pos = self._find_introduction(text)
        if intro_pos > 0:
            return text[intro_pos:].strip()
        return text

    def _simple_extraction(self, pdf_path: str) -> str:
        """Simple PDF text extraction method."""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ''
                for page in reader.pages:
                    text += page.extract_text() + '\n'
                return text
        except Exception as e:
            print(f"Error in simple extraction: {str(e)}")
            return ""

    def _extract_equations(self, text: str) -> str:
        """Extract mathematical equations from text."""
        equation_patterns = [
            r'\$.*?\$',              # Inline math
            r'\\\[.*?\\\]',          # Display math
            r'\{.*?\}',              # Bracketed expressions
        ]
        
        for pattern in equation_patterns:
            matches = re.finditer(pattern, text, re.DOTALL)
            for i, match in enumerate(matches):
                equation = match.group(0)
                self.equations.append(PDFElement('equation', equation))
                text = text.replace(equation, f'[EQUATION_{len(self.equations)}]')
        
        return text

    def _extract_tables(self, text: str) -> str:
        """Extract tables from text."""
        # Find table captions and content
        table_pattern = r'(Table \d+:|TABLE \d+:).*?(?=(?:Table \d+:|TABLE \d+:|$))'
        matches = re.finditer(table_pattern, text, re.DOTALL)
        
        for i, match in enumerate(matches):
            table = match.group(0)
            self.tables.append(PDFElement('table', table))
            text = text.replace(table, f'[TABLE_{len(self.tables)}]')
        
        return text

    def _extract_figures(self, text: str) -> str:
        """Extract figures and their captions."""
        figure_pattern = r'(Figure|Fig\.) \d+:?\.?\s*([^\.]+\.[^\.]+\.?)'
        matches = re.finditer(figure_pattern, text)
        
        for i, match in enumerate(matches):
            figure = match.group(0)
            self.figures.append(PDFElement('figure', figure))
            text = text.replace(figure, f'[FIGURE_{len(self.figures)}]')
        
        return text

    def _extract_references(self, text: str) -> str:
        """Extract references section."""
        reference_pattern = r'REFERENCES.*?(?=(?:[A-Z][A-Z\s]{3,}|$))'
        match = re.search(reference_pattern, text, re.DOTALL)
        
        if match:
            references = match.group(0)
            self.references.append(PDFElement('reference', self._clean_references(references)))
            text = text.replace(references, f'[REFERENCES_{len(self.references)}]')
        
        return text

    def _clean_references(self, text: str) -> str:
        """Clean reference section formatting."""
        # Remove citation numbers/brackets
        text = re.sub(r'\[\d+\]', '', text)
        
        # Remove DOIs and URLs
        text = re.sub(r'doi:[^\s]+', '', text)
        text = re.sub(r'https?://[^\s]+', '', text)
        
        # Fix author name formatting
        text = re.sub(r'([A-Z]\.), ([A-Z]\.)', r'\1,\2', text)
        
        # Clean up extra spaces and newlines
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text

    def read_pdf(self, pdf_path: str, use_docling: bool = True) -> str:
        """Main interface function to read and process PDF."""
        try:
            if use_docling:
                text = self._extract_text_with_docling(pdf_path)
                if text and len(text.strip()) > 100:
                    return text

            text = self._extract_text_with_fallback(pdf_path)
            text = self._process_text(text)
            text = self._remove_pre_introduction(text)
            text = re.sub(r'\[ (\d+)\]', r'[\1]', text)
            return text
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            return ""

    def _extract_text_with_docling(self, pdf_path: str) -> str:
        """Extract text using Docling backend."""
        try:
            # Check if corresponding .md file exists
            md_path = pdf_path.rsplit('.', 1)[0] + '.md'
            if os.path.exists(md_path):
                # Read existing markdown file
                with open(md_path, 'r', encoding='utf-8') as f:
                    return f.read()

            # Convert document using Docling
            result = self.docling_converter.convert(pdf_path)
            # Get markdown output
            text = result.document.export_to_markdown()
            
            # Save markdown output
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(text)
                
            return text
        except Exception as e:
            print(f"Docling extraction failed: {str(e)}")
            return ""
        
    def _extract_text_with_fallback(self, pdf_path: str) -> str:
        """Try multiple extraction methods with fallback."""
        try:
            # Try Docling first
            text = self._extract_text_with_docling(pdf_path)
            if text and len(text.strip()) > 100:
                return text

            # Fallback to existing methods
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = self._extract_with_layout(reader)
                
                if len(text.strip()) < 100 or self._has_merged_columns(text):
                    text = self._extract_with_columns(reader)
                
                return text
        except Exception as e:
            print(f"Extraction failed: {str(e)}")
            return self._simple_extraction(pdf_path)

    def _extract_with_layout(self, reader: PyPDF2.PdfReader) -> str:
        """Extract text while preserving layout."""
        text = ''
        for page in reader.pages:
            page_text = page.extract_text()
            text += page_text + '\n'
        return text

    def _extract_with_columns(self, reader: PyPDF2.PdfReader) -> str:
        """Extract text with column awareness."""
        text = ''
        for page in reader.pages:
            page_text = page.extract_text()
            columns = self._detect_and_split_columns(page_text)
            text += self._merge_columns(columns) + '\n'
        return text

    def _has_merged_columns(self, text: str) -> bool:
        """Check if text has incorrectly merged columns."""
        lines = text.split('\n')
        for line in lines:
            if len(line.strip()) > 100:
                words = line.split()
                for i in range(1, len(words) - 1):
                    if (words[i].endswith('.') and 
                        words[i+1][0].isupper() and 
                        i < len(words) * 0.7):
                        return True
        return False

    def _detect_and_split_columns(self, page_text: str) -> List[str]:
        """Detect and split text into columns."""
        lines = page_text.split('\n')
        if not lines:
            return ['']

        # Use statistical analysis to detect columns
        line_lengths = [len(line.strip()) for line in lines if line.strip()]
        if not line_lengths:
            return ['']

        avg_length = sum(line_lengths) / len(line_lengths)
        split_point = avg_length * 0.6

        columns = defaultdict(list)
        for line in lines:
            if len(line) > avg_length * 1.5:
                # Long line probably spans multiple columns
                mid = len(line) // 2
                columns[0].append(line[:mid].strip())
                columns[1].append(line[mid:].strip())
            else:
                # Assign to column based on starting position
                col = 1 if line.startswith(' ' * int(split_point)) else 0
                columns[col].append(line.strip())

        return ['\n'.join(col) for col in columns.values()]

    def _merge_columns(self, columns: List[str]) -> str:
        """Merge columns with proper formatting."""
        return '\n'.join(col.strip() for col in columns if col.strip())

    def _process_text(self, text: str) -> str:
        """Process and clean extracted text."""
        text = self._extract_special_elements(text)
        text = self._clean_text(text)
        text = self._restore_special_elements(text)
        return text

    def _extract_special_elements(self, text: str) -> str:
        """Extract equations, tables, figures, and references."""
        text = self._extract_equations(text)
        text = self._extract_tables(text)
        text = self._extract_figures(text)
        text = self._extract_references(text)
        return text

    def _clean_text(self, text: str) -> str:
        """Clean and format the text."""
        if not text:
            return ""

        # Basic cleanup
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        
        # Format spacing
        text = re.sub(r'\s+([.,;:!?)])', r'\1', text)
        text = re.sub(r'(\()\s+', r'\1', text)
        text = re.sub(r'\.(?=[A-Z])', '.\n', text)

        text = re.sub(r'\[ (\d+)\]', r'[\1]', text)
        
        # Handle paragraphs
        paragraphs = text.split('\n')
        processed_paragraphs = []
        for para in paragraphs:
            if self._is_section_title(para.strip()):
                processed_paragraphs.extend(['', para.strip().upper(), ''])
            else:
                processed_paragraphs.append(para.strip())
        
        return '\n'.join(p for p in processed_paragraphs if p)

    def _is_section_title(self, line: str) -> bool:
        """Enhanced section title detection."""
        line = line.strip()
        
        if not line or len(line) > 100 or len(line.split()) > 10:
            return False
            
        if any(re.match(pattern, line) for pattern in self.SECTION_PATTERNS):
            return True
            
        return any(keyword in line.upper() for keyword in self.SECTION_KEYWORDS)

    def _restore_special_elements(self, text: str) -> str:
        """Restore extracted special elements to the text."""
        # Restore in reverse order to maintain placement accuracy
        for element_list in [self.references, self.figures, self.tables, self.equations]:
            for i, element in enumerate(element_list):
                placeholder = f'[{element.type.upper()}_{i+1}]'
                text = text.replace(placeholder, element.content)
        return text

def read_pdf(pdf_path: str, use_docling: bool = True) -> str:
    """Interface function for PDF reading."""
    parser = PDFParser()
    return parser.read_pdf(pdf_path, use_docling)

def truncate_text(text: str, max_tokens: int = 128000, model: str = "gpt-4") -> str:
    """Truncate text to fit within token limit."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        truncated_tokens = tokens[:max_tokens]
        return encoding.decode(truncated_tokens)
    return text