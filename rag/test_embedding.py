from embedder import create_embedding

text  =input( "ask anything ")

embedding = create_embedding(text)

print("Embedding Dimension:", len(embedding))
print("\nFirst 10 values:")
print(embedding[:10])