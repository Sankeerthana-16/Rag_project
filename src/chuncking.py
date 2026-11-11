def split_text(text, chunk_size=700, overlap=150):
    chunks = []
    if not text:
        return chunks
    

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end].strip()
        if chunk:

            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks

if __name__ == "__main__":

    from src.pdf_loader import load_pdf

    pdf_path = "data/attention-all-you-need-Paper.pdf"

    pages = load_pdf(pdf_path)

    full_text = "\n".join(
        page["text"] for page in pages
    )

    chunks = split_text(full_text)

    print("Total chunks:", len(chunks))

    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i+1}:")
        print(chunk[:200])