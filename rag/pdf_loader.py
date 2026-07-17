import fitz


def extract_pages(pdf_path):

    doc = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(doc, start=1):

        text = page.get_text("text")

        if text.strip():

            pages.append({

                "page": page_number,

                "text": text

            })

    return pages