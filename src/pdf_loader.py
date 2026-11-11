import pymupdf
import os


def load_pdf(file_path):

    """
    Extract text from a PDF document page by page.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")

    extracted_pages = []

    try:
        document = pymupdf.open(file_path)

        if document.page_count == 0:
            raise ValueError("The PDF contains no pages.")

        for page_number, page in enumerate(document, start=1):

            # Extract text from the current page
            text = page.get_text("text").strip()

            # Skip empty pages
            if not text:
                continue

            page_data = {
                "page_number": page_number,
                "text": text,
                "source": os.path.basename(file_path)
            }

            extracted_pages.append(page_data)

        document.close()

        if not extracted_pages:
            raise ValueError(
                "No extractable text found. "
                "The PDF may contain scanned images."
            )

        return extracted_pages

    except Exception:
        raise


if __name__ == "__main__":

    pdf_path = "data/attention-all-you-need-Paper.pdf"

    pages = load_pdf(pdf_path)

    print("PDF extraction completed!")
    print("Total pages extracted:", len(pages))

    for page in pages:

        print("\nPage:", page["page_number"])
        print("Text preview:")
        print(page["text"][:300])