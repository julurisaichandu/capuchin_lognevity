#!/usr/bin/env python3
"""
Integration script to parse PDFs and prepare data for the Capuchin Health Dashboard.
This script uses the CapuchinHealthDataParser to extract data from PDFs and 
formats it for use with the main_2.py Streamlit dashboard.
"""

import sys
import os
from typing import List
from capuchin_pdf_parser import CapuchinHealthDataParser, parse_capuchin_pdfs

def main():
    """Main function to parse PDFs and prepare data for dashboard."""
    
    # Example PDF files to parse
    # Replace these with your actual PDF file paths or URLs
    pdf_files = [
        # Local files
        # "path/to/capuchin_health_report_2024.pdf",
        # "path/to/lab_results_january.pdf",
        
        # Or URLs
        # "https://example.com/capuchin_study.pdf",
    ]
    
    # Check if PDFs are provided as command line arguments
    if len(sys.argv) > 1:
        pdf_files = sys.argv[1:]
        print(f"Processing {len(pdf_files)} PDF files from command line arguments")
    else:
        print("No PDF files provided. Usage: python integrate_pdfs.py file1.pdf file2.pdf ...")
        print("\nExample with local files:")
        print("  python integrate_pdfs.py lab_report1.pdf lab_report2.pdf")
        print("\nExample with URLs:")
        print("  python integrate_pdfs.py https://example.com/report.pdf")
        return
    
    # Parse the PDFs
    print("\nStarting PDF parsing...")
    output_csv = "capuchin_health_data.csv"
    
    try:
        df = parse_capuchin_pdfs(pdf_files, output_csv)
        
        if df.empty:
            print("\nNo data was extracted. Please check:")
            print("1. The PDF files contain capuchin health data")
            print("2. The data format matches expected patterns")
            print("3. The files are accessible and not corrupted")
        else:
            print(f"\nSuccessfully extracted {len(df)} health records")
            print(f"\nData summary:")
            print(f"- Animals: {df['id'].nunique()}")
            print(f"- Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"- Tests performed: {', '.join(df['test'].unique()[:5])}" + 
                  (f" and {len(df['test'].unique())-5} more" if len(df['test'].unique()) > 5 else ""))
            
            print(f"\nData saved to: {output_csv}")
            print("\nTo view in the dashboard:")
            print("1. Run: streamlit run main_2.py")
            print(f"2. Upload the file: {output_csv}")
            
            # Optional: Show sample of extracted data
            print("\nSample of extracted data:")
            print(df.head(10))
            
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        print("\nPlease ensure:")
        print("1. Docling is properly installed: pip install docling")
        print("2. All required dependencies are installed")
        print("3. PDF files are accessible")


if __name__ == "__main__":
    main()