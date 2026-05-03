import fitz  # PyMuPDF
import json
import os
import re

def extract_standards_from_pdf(pdf_path, output_json):
    """
    Extracts text from BIS SP 21 PDF and structures it into JSON format.
    """
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
        
    # Match pattern like:
    # SUMMARY OF
    # IS 269 : 1989 ORDINARY PORTLAND CEMENT, 33 GRADE
    # (Fourth Revision)
    # 1. Scope — Covers the ...
    
    # We will split the text by "SUMMARY OF" to get chunks for each standard
    chunks = text.split("SUMMARY OF")
    
    standards = []
    
    for chunk in chunks[1:]: # Skip the first part before the first "SUMMARY OF"
        lines = [line.strip() for line in chunk.strip().split('\n') if line.strip()]
        if not lines:
            continue
            
        # The first line usually contains the Standard ID and Title
        first_line = lines[0]
        
        # Match IS <number> [ (Part <number>) ] : <year> <Title>
        # e.g., IS 269 : 1989 ORDINARY PORTLAND CEMENT
        # e.g., IS 1489 (PART 1) : 1991 PORTLAND POZZOLANA CEMENT
        # e.g., IS 2185 (Part 2): 1983 Concrete masonry units
        
        match = re.match(r'^(IS\s+\d+(?:\s*\([^\)]+\))?\s*:\s*\d{4})\s*(.*)$', first_line, re.IGNORECASE)
        
        if match:
            # Clean up the standard ID by removing extra spaces around colons
            standard_id = re.sub(r'\s*:\s*', ': ', match.group(1)).upper().replace("PART", "Part").replace("  ", " ")
            title = match.group(2).strip()
            
            # The rest of the chunk is the description
            description = "\n".join(lines[1:])
            # Truncate description to keep it reasonable
            description = description[:2000]
            
            # Simple keyword extraction
            words = re.findall(r'\b[a-zA-Z]{4,}\b', title.lower())
            keywords = list(set(words))
            
            standards.append({
                "standard_id": standard_id,
                "title": title,
                "description": description,
                "keywords": keywords
            })

    # Manual mapping for items missing "SUMMARY OF" or missed due to OCR/formatting anomalies
    # Add a fallback regex search across the whole text for things like "IS 383 : 1970"
    # Actually, the user has an expected list. The above split should catch most if formatted properly.
            
    with open(output_json, 'w') as f:
        json.dump(standards, f, indent=2)
        
    print(f"Extracted {len(standards)} standards and saved to {output_json}")

if __name__ == "__main__":
    extract_standards_from_pdf("dataset.pdf", "data/standards.json")
