from sentence_transformers import SentenceTransformer

# Load model once
model = SentenceTransformer("all-MiniLM-L6-v2")


def create_embedding(text):
    """
    Generate embedding for a text chunk
    """
    return model.encode(text)