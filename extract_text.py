import pdfplumber

with pdfplumber.open("Sample_Resume_Aarav_Sharma.pdf") as pdf:
    print("Number of pages:", len(pdf.pages))
    full_text = ""
    for page in pdf.pages:
        full_text += page.extract_text() + "\n"
    print("Full extracted text:\n", full_text)




