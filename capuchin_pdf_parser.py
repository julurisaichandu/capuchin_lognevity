import pandas as pd
import numpy as np
import warnings

# Suppress PyTorch MPS warnings on Apple Silicon
warnings.filterwarnings("ignore", message=".*pin_memory.*MPS.*", category=UserWarning)

from docling.document_converter import DocumentConverter
import re
from datetime import datetime
import os
from typing import Dict, List, Optional, Tuple

class CapuchinHealthDataParser:
    """Parser for extracting capuchin monkey health data from PDF documents using Docling."""
    
    def __init__(self):
        self.converter = DocumentConverter()
        
        # Comprehensive test name patterns based on your project data
        self.test_patterns = {
            # Hematology
            'RBC': r'(?:RBC|Red Blood Cells?|Erythrocytes?)\s*:?\s*([\d.<>]+)',
            'WBC': r'(?:WBC|White Blood Cells?|Leukocytes?)\s*:?\s*([\d.<>]+)',
            'Hemoglobin': r'(?:Hemoglobin|HEMOGLOBIN|Hgb|HGB)\s*:?\s*([\d.<>]+)',
            'Hematocrit': r'(?:Hematocrit|HEMATOCRIT|HCT|Hct)\s*:?\s*([\d.<>]+)',
            'MCV': r'(?:MCV)\s*:?\s*([\d.<>]+)',
            'MCH': r'(?:MCH)\s*:?\s*([\d.<>]+)',
            'MCHC': r'(?:MCHC)\s*:?\s*([\d.<>]+)',
            'RDW%': r'(?:RDW%?)\s*:?\s*([\d.<>]+)',
            'Platelets': r'(?:Platelets?|Platelet Count|PLT)\s*:?\s*([\d.<>]+)',
            'MPV': r'(?:MPV)\s*:?\s*([\d.<>]+)',
            
            # White Blood Cell Differential
            '% Neutrophils': r'(?:%\s*Neutrophils?|Neutrophils?\s*%)\s*:?\s*([\d.<>]+)',
            '% Lymphocytes': r'(?:%\s*Lymphocytes?|Lymphocytes?\s*%)\s*:?\s*([\d.<>]+)',
            '% Monocytes': r'(?:%\s*Monocytes?|Monocytes?\s*%)\s*:?\s*([\d.<>]+)',
            '% Eosinophils': r'(?:%\s*Eosinophils?|Eosinophils?\s*%)\s*:?\s*([\d.<>]+)',
            '% Basophils': r'(?:%\s*Basophils?|Basophils?\s*%)\s*:?\s*([\d.<>]+)',
            '% Bands': r'(?:%\s*Bands?|Bands?\s*%)\s*:?\s*([\d.<>]+)',
            
            # Absolute Counts
            'Neutrophils': r'(?:Neutrophils?|ABSOLUTE POLYS)\s*:?\s*([\d.<>]+)',
            'Lymphocytes': r'(?:Lymphocytes?|ABSOLUTE LYMPHS)\s*:?\s*([\d.<>]+)',
            'Monocytes': r'(?:Monocytes?|ABSOLUTE MONOS)\s*:?\s*([\d.<>]+)',
            'Eosinophils': r'(?:Eosinophils?|ABSOLUTE EOS)\s*:?\s*([\d.<>]+)',
            'Basophils': r'(?:Basophils?|ABSOLUTE BASOS)\s*:?\s*([\d.<>]+)',
            'Bands': r'(?:Bands?|ABSOLUTE BANDS)\s*:?\s*([\d.<>]+)',
            
            # Chemistry Panel
            'Glucose': r'(?:Glucose|GLU|Blood Sugar)\s*:?\s*([\d.<>]+)',
            'BUN': r'(?:BUN|Blood Urea Nitrogen|Urea Nitrogen)\s*:?\s*([\d.<>]+)',
            'Creatinine': r'(?:Creatinine|CREA)\s*:?\s*([\d.<>]+)',
            'BUN/Creatinine Ratio': r'(?:BUN[/:]\s*Creatinine(?:\s*Ratio)?)\s*:?\s*([\d.<>]+)',
            'Phosphorus': r'(?:Phosphorus|Phosphorous|PHOS)\s*:?\s*([\d.<>]+)',
            'Calcium': r'(?:Calcium|Ca)\s*:?\s*([\d.<>]+)',
            'Magnesium': r'(?:Magnesium|MAGNESIUM)\s*:?\s*([\d.<>]+)',
            
            # Electrolytes
            'Sodium': r'(?:Sodium|Na)\s*:?\s*([\d.<>]+)',
            'Potassium': r'(?:Potassium|K)\s*:?\s*([\d.<>]+)',
            'Chloride': r'(?:Chloride|Cl)\s*:?\s*([\d.<>]+)',
            'TCO2': r'(?:TCO2|Bicarbonate)\s*:?\s*([\d.<>]+)',
            'Anion Gap': r'(?:Anion Gap)\s*:?\s*([\d.<>]+)',
            'Na/K Ratio': r'(?:Na[/:]\s*K(?:\s*Ratio)?)\s*:?\s*([\d.<>]+)',
            
            # Protein Tests
            'Total Protein': r'(?:Total Protein|TP)\s*:?\s*([\d.<>]+)',
            'Albumin': r'(?:Albumin|ALB)\s*:?\s*([\d.<>]+)',
            'Globulin': r'(?:Globulin|GLOBULIN|GLOB)\s*:?\s*([\d.<>]+)',
            'A/G Ratio': r'(?:A/G(?:\s*Ratio)?|Alb[:/]Glob(?:\s*Ratio)?|Albumin[:/]Globulin(?:\s*Ratio)?)\s*:?\s*([\d.<>]+)',
            
            # Liver Function
            'ALT': r'(?:ALT|SGPT|Alanine Aminotransferase)\s*:?\s*([\d.<>]+)',
            'AST': r'(?:AST|SGOT|Aspartate Aminotransferase)\s*:?\s*([\d.<>]+)',
            'ALP': r'(?:ALP|ALKP|Alkaline Phosphatase|Alk Phosphatase)\s*:?\s*([\d.<>]+)',
            'GGT': r'(?:GGT|GGTP|Gamma-Glutamyl Transferase)\s*:?\s*([\d.<>]+)',
            'Total Bilirubin': r'(?:Total Bilirubin|Bilirubin Total|TBIL|Bilirubin - Total)\s*:?\s*([\d.<>]+)',
            'Bilirubin - Conjugated': r'(?:Bilirubin - Conjugated|Direct Bilirubin)\s*:?\s*([\d.<>]+)',
            'Bilirubin - Unconjugated': r'(?:Bilirubin - Unconjugated|Indirect Bilirubin)\s*:?\s*([\d.<>]+)',
            
            # Lipids
            'Cholesterol': r'(?:Cholesterol|CHOL)\s*:?\s*([\d.<>]+)',
            'Triglycerides': r'(?:Triglycerides|TRIGLYCERIDES)\s*:?\s*([\d.<>]+)',
            
            # Pancreatic
            'Amylase': r'(?:Amylase|AMYLASE|AMYL)\s*:?\s*([\d.<>]+)',
            'Lipase': r'(?:Lipase|LIPASE)\s*:?\s*([\d.<>]+)',
            
            # Muscle/Cardiac
            'CPK': r'(?:CPK|Creatine Kinase)\s*:?\s*([\d.<>]+)',
            
            # Kidney Function
            'IDEXX SDMA': r'(?:IDEXX SDMA|SDMA)\s*:?\s*([\d.<>]+)',
            'OSMOLALITY CALCULATED': r'(?:OSMOLALITY CALCULATED|Osmolality)\s*:?\s*([\d.<>]+)',
            
            # Thyroid
            'T3 Assay': r'(?:T3 Assay|T3)\s*:?\s*([\d.<>]+)',
            'T4 Total': r'(?:T4 Total|T4)\s*:?\s*([\d.<>]+)',
            'Thyroxine Free': r'(?:Thyroxine Free|Free T4)\s*:?\s*([\d.<>]+)',
            
            # Other
            'Fructosamine': r'(?:Fructosamine)\s*:?\s*([\d.<>]+)',
            'GIARDIA ELISA': r'(?:GIARDIA ELISA)\s*:?\s*(\w+)',
            'Hemolysis Index': r'(?:Hemolysis Index)\s*:?\s*(\w+)',
            'Lipemia Index': r'(?:Lipemia Index)\s*:?\s*(\w+)',
            'PLATELET EST': r'(?:PLATELET EST)\s*:?\s*(\w+)',
            
            # Physical Measurements
            'Weight': r'(?:Weight|Body Weight)\s*:?\s*([\d.<>]+)\s*(?:kg|kilograms?)',
            'BCS': r'(?:BCS|Body Condition Score)\s*:?\s*(\d+)',
        }
        
        # Comprehensive unit patterns based on your data
        self.unit_patterns = {
            'mg/dL': r'mg/dL',
            'g/dL': r'g/dL',
            'M/µL': r'M/µL|M/μL|MILL/CMM|10\^6/µL',
            'K/µL': r'K/µL|K/μL|THDS/CMM|10\^3/µL',
            '%': r'%',
            'U/L': r'U/L|IU/L',
            'kg': r'kg|kilograms?',
            'mmol/L': r'mmol/L',
            'mEq/L': r'mEq/L',
            'fL': r'fL',
            'pg': r'pg',
            '/µL': r'/µL|/μL',
            'µg/dL': r'µg/dL|μg/dL',
            'ug/dL': r'ug/dL',
            'ng/dL': r'ng/dL',
            'pmol/L': r'pmol/L',
            'micromoles/L': r'micromoles/L',
            'mOsm/L': r'mOsm/L',
            'RATIO': r'RATIO',
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
            r'(?:Animal ID|Subject ID|Monkey ID|Patient ID|ID)\s*:?\s*(\w+)',
            r'(?:LAB ID)\s*:?\s*(\d+)',  # IDEXX Lab ID
            r'(?:ORDER ID)\s*:?\s*(\d+)',  # IDEXX Order ID
            r'(?:Name|Subject)\s*:?\s*(\w+)',
            r'#(\w+)',  # Sometimes IDs are marked with #
            r'(?:PATIENT ID)\s*:?\s*(\w+)',  # From IDEXX format
        ]
        
        for pattern in patterns:
            match = re.search(pattern, section, re.IGNORECASE)
            if match:
                id_value = match.group(1).strip()
                if id_value:  # Only return if not empty
                    return id_value
                
        # If no ID found, try to extract from lab ID or order ID as fallback
        lab_id_match = re.search(r'LAB ID\s*:?\s*(\d+)', section, re.IGNORECASE)
        if lab_id_match:
            return f"LAB_{lab_id_match.group(1)}"
            
        return "Unknown"
    
    def _extract_date(self, section: str) -> Optional[datetime]:
        """Extract date from section."""
        # Look for specific date fields first
        date_field_patterns = [
            r'(?:Collection Date|DATE OF RESULT|Date of Result)\s*:?\s*([^\n]+)',
            r'(?:Date|DATE)\s*:?\s*([^\n]+)',
            r'(?:COLLECTION DATE)\s*:?\s*([^\n]+)',
            r'(?:DATE OF RECEIPT)\s*:?\s*([^\n]+)',
        ]
        
        for field_pattern in date_field_patterns:
            field_match = re.search(field_pattern, section, re.IGNORECASE)
            if field_match:
                date_str = field_match.group(1).strip()
                # Try to parse the date
                parsed_date = self._parse_date_string(date_str)
                if parsed_date:
                    return parsed_date
        
        # If no specific field found, look for date patterns anywhere
        date_patterns = [
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})',
            r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})',
            r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})',
            r'(\d{1,2}-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{4})',
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, section, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                parsed_date = self._parse_date_string(date_str)
                if parsed_date:
                    return parsed_date
                    
        return None
    
    def _parse_date_string(self, date_str: str) -> Optional[datetime]:
        """Parse a date string into a datetime object."""
        # Clean the date string
        date_str = date_str.strip()
        
        # Try multiple date parsing formats
        formats = [
            '%m/%d/%Y', '%Y-%m-%d', '%d/%m/%Y', '%m-%d-%Y', '%d-%m-%Y',
            '%B %d, %Y', '%d %B %Y', '%b %d, %Y', '%d %b %Y',
            '%d-%b-%Y', '%m/%d/%y', '%d/%m/%y',  # 2-digit year formats
        ]
        
        for fmt in formats:
            try:
                return pd.to_datetime(date_str, format=fmt)
            except:
                continue
        
        # If specific formats fail, try pandas general parser
        try:
            return pd.to_datetime(date_str)
        except:
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
        # Look for explicit sex/gender fields
        sex_patterns = [
            r'(?:Sex|Gender)\s*:?\s*(Male|Female|M|F)',
            r'(?:SEX|GENDER)\s*:?\s*(Male|Female|M|F)',
        ]
        
        for pattern in sex_patterns:
            match = re.search(pattern, section, re.IGNORECASE)
            if match:
                sex_value = match.group(1).upper()
                if sex_value in ['M', 'MALE']:
                    return 'Male'
                elif sex_value in ['F', 'FEMALE']:
                    return 'Female'
        
        # Fallback to looking for male/female in text
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
        
        # Define test name variations for better matching
        test_variations = {
            'ALB': ['ALB', 'Albumin'],
            'ALKP': ['ALKP', 'ALP', 'Alkaline Phosphatase', 'Alk Phosphatase'],
            'ALT': ['ALT', 'SGPT'],
            'AST': ['AST', 'SGOT'],
            'AMYL': ['AMYL', 'AMYLASE', 'Amylase'],
            'BUN': ['BUN', 'Blood Urea Nitrogen', 'Urea Nitrogen'],
            'Ca': ['Ca', 'Calcium'],
            'CHOL': ['CHOL', 'Cholesterol'],
            'CREA': ['CREA', 'Creatinine'],
            'Cl': ['Cl', 'Chloride'],
            'GGT': ['GGT', 'GGTP'],
            'GLOB': ['GLOB', 'GLOBULIN', 'Globulin'],
            'GLU': ['GLU', 'Glucose'],
            'HCT': ['HCT', 'HEMATOCRIT', 'Hematocrit'],
            'HGB': ['HGB', 'HEMOGLOBIN', 'Hemoglobin'],
            'K': ['K', 'Potassium'],
            'Na': ['Na', 'Sodium'],
            'PHOS': ['PHOS', 'Phosphorus', 'Phosphorous'],
            'PLT': ['PLT', 'Platelets', 'Platelet Count'],
            'TBIL': ['TBIL', 'Total Bilirubin', 'Bilirubin Total'],
            'TP': ['TP', 'Total Protein'],
            'WBC': ['WBC', 'White Blood Cell'],
            'RBC': ['RBC', 'Red Blood Cell'],
        }
        
        # Pattern for table rows with test results
        patterns = [
            # Pattern 1: Test name = value unit (e.g., "ALB = 4.87 g/dl")
            r'([A-Za-z][A-Za-z0-9\s/:%-]*?)\s*=\s*([\d.<>]+)\s*([A-Za-z/%µμ]+)?',
            # Pattern 2: Test name: value unit (e.g., "Glucose: 89 mg/dL")
            r'([A-Za-z][A-Za-z0-9\s/:%-]*?)\s*:\s*([\d.<>]+)\s*([A-Za-z/%µμ]+)?',
            # Pattern 3: Test name value unit (e.g., "RBC 5.46 M/µL")
            r'^([A-Za-z][A-Za-z0-9\s/:%-]*?)\s+([\d.<>]+)\s+([A-Za-z/%µμ]+)',
            # Pattern 4: Test name (tab/spaces) value (tab/spaces) unit
            r'([A-Za-z][A-Za-z0-9\s/:%-]*?)\s{2,}([\d.<>]+)\s*([A-Za-z/%µμ]+)?',
        ]
        
        # Split into lines and process each
        lines = section.split('\n')
        for line in lines:
            line = line.strip()
            
            # Skip empty lines, headers, and reference ranges
            if not line or any(header in line.lower() for header in ['test', 'result', 'reference', 'range', 'indicator']):
                continue
            
            # Try each pattern
            matched = False
            for pattern in patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    test_name_raw = match.group(1).strip()
                    value = match.group(2).strip()
                    units = match.group(3).strip() if match.group(3) else ''
                    
                    # Standardize test name
                    test_name = None
                    test_name_lower = test_name_raw.lower()
                    
                    # Check against known variations
                    for standard_name, variations in test_variations.items():
                        if any(var.lower() == test_name_lower for var in variations):
                            test_name = standard_name
                            break
                    
                    # If not found in variations, check against our test patterns
                    if not test_name:
                        for pattern_name in self.test_patterns.keys():
                            if pattern_name.lower() in test_name_lower or test_name_lower in pattern_name.lower():
                                test_name = pattern_name
                                break
                    
                    # If still not found, use the raw name
                    if not test_name:
                        test_name = test_name_raw
                    
                    results.append((test_name, value, units))
                    matched = True
                    break
            
            # If no pattern matched, try to extract from simpler formats
            if not matched:
                # Check if line contains a number that might be a result
                number_match = re.search(r'([\d.<>]+)', line)
                if number_match:
                    # Look for test name before the number
                    before_number = line[:number_match.start()].strip()
                    if before_number and len(before_number) > 1:
                        value = number_match.group(1)
                        # Look for units after the number
                        after_number = line[number_match.end():].strip()
                        units = ''
                        if after_number:
                            unit_match = re.match(r'([A-Za-z/%µμ]+)', after_number)
                            if unit_match:
                                units = unit_match.group(1)
                        
                        results.append((before_number, value, units))
        
        return results
    
    def _find_units_near_match(self, text: str, match_pos: int, search_distance: int = 50) -> str:
        """Find units near a matched value."""
        # Look ahead for units
        end_pos = min(match_pos + search_distance, len(text))
        search_text_after = text[match_pos:end_pos]
        
        # Also look behind for units (sometimes units come before values)
        start_pos = max(0, match_pos - search_distance)
        search_text_before = text[start_pos:match_pos]
        
        # Check after the value first (most common)
        for unit, pattern in self.unit_patterns.items():
            if re.search(pattern, search_text_after[:20]):  # Check immediate vicinity first
                return unit
        
        # If not found after, check before
        for unit, pattern in self.unit_patterns.items():
            if re.search(pattern, search_text_before[-20:]):  # Check end of before text
                return unit
        
        # Check larger area after if still not found
        for unit, pattern in self.unit_patterns.items():
            if re.search(pattern, search_text_after):
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