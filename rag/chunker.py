from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,      # characters
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " "
        ]
    )

    texts = splitter.split_text(text)

    chunks = []

    for i, t in enumerate(texts):
        chunks.append({
            "chunk_id": i,
            "text": t
        })

    return chunks