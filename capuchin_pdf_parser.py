import pandas as pd
import numpy as np
from docling.document_converter import DocumentConverter
import re
from datetime import datetime
import os
from typing import Dict, List, Optional, Tuple

class CapuchinHealthDataParser:
    """Parser for extracting capuchin monkey health data from PDF documents using Docling."""
    
    def __init__(self):
        self.converter = DocumentConverter()
        
        # Common test name patterns to look for in documents
        self.test_patterns = {
            'RBC': r'(?:RBC|Red Blood Cells?|Erythrocytes?)\s*:?\s*([\d.]+)',
            'WBC': r'(?:WBC|White Blood Cells?|Leukocytes?)\s*:?\s*([\d.]+)',
            'Hemoglobin': r'(?:Hemoglobin|Hgb|HGB)\s*:?\s*([\d.]+)',
            'Hematocrit': r'(?:Hematocrit|HCT|Hct)\s*:?\s*([\d.]+)',
            'Glucose': r'(?:Glucose|Blood Sugar|GLU)\s*:?\s*([\d.]+)',
            'ALT': r'(?:ALT|SGPT|Alanine Aminotransferase)\s*:?\s*([\d.]+)',
            'AST': r'(?:AST|SGOT|Aspartate Aminotransferase)\s*:?\s*([\d.]+)',
            'Creatinine': r'(?:Creatinine|CREA)\s*:?\s*([\d.]+)',
            'BUN': r'(?:BUN|Blood Urea Nitrogen|Urea)\s*:?\s*([\d.]+)',
            'Weight': r'(?:Weight|Body Weight)\s*:?\s*([\d.]+)\s*(?:kg|kilograms?)',
            'BCS': r'(?:BCS|Body Condition Score)\s*:?\s*(\d+)',
        }
        
        # Unit patterns
        self.unit_patterns = {
            'mg/dL': r'mg/dL',
            'g/dL': r'g/dL',
            'M/µL': r'M/µL|10\^6/µL',
            'K/µL': r'K/µL|10\^3/µL',
            '%': r'%',
            'U/L': r'U/L|IU/L',
            'kg': r'kg|kilograms?',
            'mmol/L': r'mmol/L',
        }
        
    def parse_pdf(self, pdf_path: str) -> pd.DataFrame:
        """
        Parse a PDF document and extract capuchin health data.
        
        Args:
            pdf_path: Path to the PDF file or URL
            
        Returns:
            DataFrame with columns: date, test, result, units, age_years, sex, id
        """
        try:
            # Convert PDF to structured format
            result = self.converter.convert(pdf_path)
            
            # Get markdown content
            content = result.document.export_to_markdown()
            
            # Extract data from content
            extracted_data = self._extract_health_data(content)
            
            # Convert to DataFrame
            if extracted_data:
                df = pd.DataFrame(extracted_data)
                return self._standardize_dataframe(df)
            else:
                return pd.DataFrame(columns=['date', 'test', 'result', 'units', 'age_years', 'sex', 'id'])
                
        except Exception as e:
            print(f"Error parsing PDF {pdf_path}: {str(e)}")
            return pd.DataFrame(columns=['date', 'test', 'result', 'units', 'age_years', 'sex', 'id'])
    
    def _extract_health_data(self, content: str) -> List[Dict]:
        """Extract health data from parsed document content."""
        data_records = []
        
        # Split content into sections (could be by animal ID, date, or other markers)
        sections = self._split_into_sections(content)
        
        for section in sections:
            # Extract metadata
            animal_id = self._extract_animal_id(section)
            date = self._extract_date(section)
            age = self._extract_age(section)
            sex = self._extract_sex(section)
            
            # Extract test results
            test_results = self._extract_test_results(section)
            
            # Create records
            for test_name, value, units in test_results:
                record = {
                    'id': animal_id,
                    'date': date,
                    'test': test_name,
                    'result': value,
                    'units': units,
                    'age_years': age,
                    'sex': sex
                }
                data_records.append(record)
                
        return data_records
    
    def _split_into_sections(self, content: str) -> List[str]:
        """Split content into logical sections for processing."""
        # Look for common section markers
        section_markers = [
            r'Animal ID:?\s*\w+',
            r'Subject:?\s*\w+',
            r'Patient:?\s*\w+',
            r'Monkey:?\s*\w+',
            r'ID:?\s*\w+',
            r'\n={3,}\n',  # Horizontal rules
            r'\n-{3,}\n',
        ]
        
        # Try to split by animal ID or major sections
        pattern = '|'.join(section_markers)
        sections = re.split(pattern, content, flags=re.IGNORECASE)
        
        # If no clear sections found, treat entire content as one section
        if len(sections) <= 1:
            return [content]
            
        return [s.strip() for s in sections if s.strip()]
    
    def _extract_animal_id(self, section: str) -> str:
        """Extract animal ID from section."""
        patterns = [
            r'(?:Animal ID|Subject ID|Monkey ID|ID)\s*:?\s*(\w+)',
            r'(?:Name|Subject)\s*:?\s*(\w+)',
            r'#(\w+)',  # Sometimes IDs are marked with #
        ]
        
        for pattern in patterns:
            match = re.search(pattern, section, re.IGNORECASE)
            if match:
                return match.group(1)
                
        return "Unknown"
    
    def _extract_date(self, section: str) -> Optional[datetime]:
        """Extract date from section."""
        date_patterns = [
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})',
            r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})',
            r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})',
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, section, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                # Try multiple date parsing formats
                for fmt in ['%m/%d/%Y', '%Y-%m-%d', '%d/%m/%Y', '%B %d, %Y', '%d %B %Y']:
                    try:
                        return pd.to_datetime(date_str, format=fmt)
                    except:
                        continue
                # If specific formats fail, try pandas general parser
                try:
                    return pd.to_datetime(date_str)
                except:
                    pass
                    
        return None
    
    def _extract_age(self, section: str) -> Optional[float]:
        """Extract age in years from section."""
        patterns = [
            r'Age\s*:?\s*([\d.]+)\s*(?:years?|yrs?)',
            r'([\d.]+)\s*(?:years?|yrs?)\s*old',
            r'Age\s*:?\s*([\d.]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, section, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except:
                    pass
                    
        return None
    
    def _extract_sex(self, section: str) -> str:
        """Extract sex from section."""
        if re.search(r'\b(?:male|M)\b', section, re.IGNORECASE) and not re.search(r'\bfemale\b', section, re.IGNORECASE):
            return 'Male'
        elif re.search(r'\b(?:female|F)\b', section, re.IGNORECASE):
            return 'Female'
        return 'Unknown'
    
    def _extract_test_results(self, section: str) -> List[Tuple[str, str, str]]:
        """Extract test results from section."""
        results = []
        
        # First try to find results in table format
        table_results = self._extract_from_tables(section)
        results.extend(table_results)
        
        # Then look for inline test results
        for test_name, pattern in self.test_patterns.items():
            matches = re.finditer(pattern, section, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                value = match.group(1)
                
                # Try to find units near the value
                units = self._find_units_near_match(section, match.end())
                
                results.append((test_name, value, units))
                
        return results
    
    def _extract_from_tables(self, section: str) -> List[Tuple[str, str, str]]:
        """Extract test results from table-like structures in the text."""
        results = []
        
        # Look for patterns that might indicate tabular data
        # Pattern: test name followed by value and possibly units
        table_pattern = r'([A-Za-z\s]+?)\s+([\d.<>]+)\s*([A-Za-z/%µ]+)?'
        
        # Find lines that look like table rows
        lines = section.split('\n')
        for line in lines:
            # Skip empty lines and headers
            if not line.strip() or 'Test' in line or 'Result' in line:
                continue
                
            # Try to match table row pattern
            match = re.match(table_pattern, line.strip())
            if match:
                test_name = match.group(1).strip()
                value = match.group(2).strip()
                units = match.group(3).strip() if match.group(3) else ''
                
                # Check if this matches any known test
                for known_test in self.test_patterns.keys():
                    if known_test.lower() in test_name.lower():
                        results.append((known_test, value, units))
                        break
                        
        return results
    
    def _find_units_near_match(self, text: str, start_pos: int, search_distance: int = 20) -> str:
        """Find units near a matched value."""
        # Look ahead for units
        search_text = text[start_pos:start_pos + search_distance]
        
        for unit, pattern in self.unit_patterns.items():
            if re.search(pattern, search_text):
                return unit
                
        return ''
    
    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize the dataframe to match expected format."""
        # Ensure all required columns exist
        required_columns = ['date', 'test', 'result', 'units', 'age_years', 'sex', 'id']
        for col in required_columns:
            if col not in df.columns:
                df[col] = None
                
        # Reorder columns
        df = df[required_columns]
        
        # Convert date column to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
        # Fill missing dates with today's date as placeholder
        if df['date'].isna().all():
            df['date'] = pd.Timestamp.now()
            
        return df
    
    def parse_multiple_pdfs(self, pdf_paths: List[str]) -> pd.DataFrame:
        """
        Parse multiple PDF documents and combine results.
        
        Args:
            pdf_paths: List of paths to PDF files
            
        Returns:
            Combined DataFrame with all extracted data
        """
        all_data = []
        
        for pdf_path in pdf_paths:
            print(f"Parsing: {pdf_path}")
            df = self.parse_pdf(pdf_path)
            if not df.empty:
                all_data.append(df)
                
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            # Remove duplicates if any
            combined_df = combined_df.drop_duplicates()
            return combined_df
        else:
            return pd.DataFrame(columns=['date', 'test', 'result', 'units', 'age_years', 'sex', 'id'])


# Example usage function
def parse_capuchin_pdfs(pdf_files: List[str], output_csv: str = "capuchin_health_data.csv") -> pd.DataFrame:
    """
    Convenience function to parse PDFs and save to CSV.
    
    Args:
        pdf_files: List of PDF file paths or URLs
        output_csv: Output CSV filename
        
    Returns:
        DataFrame with parsed data
    """
    parser = CapuchinHealthDataParser()
    df = parser.parse_multiple_pdfs(pdf_files)
    
    if not df.empty:
        df.to_csv(output_csv, index=False)
        print(f"Saved {len(df)} records to {output_csv}")
    else:
        print("No data extracted from PDFs")
        
    return df