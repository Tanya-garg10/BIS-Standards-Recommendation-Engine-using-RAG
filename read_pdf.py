import fitz
import sys

def main():
    doc = fitz.open('dataset.pdf')
    text = ""
    start_page = min(20, len(doc))
    end_page = min(25, len(doc))
    for i in range(start_page, end_page):
        text += doc[i].get_text()
    
    with open('pdf_sample.txt', 'w', encoding='utf-8') as f:
        f.write(text)

if __name__ == "__main__":
    main()
