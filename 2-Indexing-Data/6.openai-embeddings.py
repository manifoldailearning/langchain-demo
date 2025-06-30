from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

model = OpenAIEmbeddings(model="text-embedding-3-small")
print("\n--------------------------------")
print("Embedding list of texts")
embeddings = model.embed_documents([
    "Ramesh in our class has won cricket match",
    "who won the cricket match",
    "write a short story about a dog",
    "write a short story about a bird",
    "write a short story about a fish",
    "translate 'Hello, how are you?' to French",])

print("\nEmbeddings:")
for i, embedding in enumerate(embeddings):
    print(f"Embedding {i+1}:")
    print(embedding[:10])  # Print first 10 dimensions for brevity
    #print embedding dimensions
    print(f"Total dimensions: {len(embedding)}")