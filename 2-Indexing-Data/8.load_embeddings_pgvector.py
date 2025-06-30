from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
import uuid 
from langchain_postgres import PGVector
load_dotenv()

connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"

loader = TextLoader("./text_demo.txt")
docs = loader.load()

# split the documents into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=20,
    chunk_overlap=5,
    length_function=len
)

splitted_docs = splitter.split_documents(docs)
print(splitted_docs)

# create embeddings
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
embeddings = embeddings_model.embed_documents([chunk.page_content for chunk in splitted_docs])

print("\nEmbeddings:" )
for i, embedding in enumerate(embeddings):
    print(f"Embedding {i+1}:")
    print(embedding[:10])  # Print first 10 dimensions for brevity
    #print embedding dimensions
    print(f"Total dimensions: {len(embedding)}")

# create pgvector instance
collection_name = "my_docs"

vector_store = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)

# Similarity search
query = "Artificial Intelligence (AI)"

results = vector_store.similarity_search(query, k=3)
print("\nSimilarity Search Results:")
for i, result in enumerate(results):
    print(f"Result {i+1}:")
    print(f"Page Content: {result.page_content}")
    print(f"Metadata: {result.metadata}")
    print(f"Score: {result.score}\n")