from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import uuid 
from langchain_postgres import PGVector
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"

loader = TextLoader("text_demo.txt", encoding="utf8")
docs = loader.load()

#chunking

splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
docs = splitter.split_documents(docs)
print(f"Number of chunks: {len(docs)}")
# print chunks
for i, doc in enumerate(docs):
    print(f"Chunk {i}: {doc.page_content[:50]}...")  # Print first 50 characters of each chunk

# create embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# create vector store
vector_store = PGVector.from_documents(
    docs,embedding=embeddings,
    connection=connection)

#  similarity search
query = "What is Artificial Intelligence?"

results = vector_store.similarity_search(query, k=4)
print(f"Number of results: {len(results)}")
for i, result in enumerate(results):
    print(f"Result {i}: {result.page_content}...")  # Print first 50 characters of each result
print(f"Metadata: {result.metadata}")  # Print metadata associated with the result

# return similarity search with score
results_with_scores = vector_store.similarity_search_with_score(query, k=4)
print(f"Number of results with scores: {len(results_with_scores)}")
for i, (result, score) in enumerate(results_with_scores):
    print(f"Result {i}: {result.page_content[:50]}...")  # Print first 50 characters of each result
    print(f"Score: {score}")  # Print score associated with the result

# add a new document
ids = [str(uuid.uuid4()) for _ in range(2)]
vector_store.add_documents(
    [Document(page_content="I like cats", metadata={"location": "home", "topic": "pets"}),
     Document(page_content="I like dogs", metadata={"location": "home", "topic": "pets"})],
    ids=ids,
)

# return the entries in the vector store

# results = vector_store.similarity_search("", k=30)
# for i, result in enumerate(results):
#     print(f"Result {i}: {result.page_content[:50]}...")  # Print first 50 characters of each result
#     print(f"Metadata: {result.metadata}")  # Print metadata associated with the result

# return by ids
results_by_ids = vector_store.get_by_ids(ids)
for i, result in enumerate(results_by_ids):
    print(f"Result by ID {i}: {result.page_content[:50]}...")  # Print first 50 characters of each result
    print(f"Metadata: {result.metadata}")  # Print metadata associated with the result

# delete by ids
vector_store.delete([ids[0]]) # send as a list
# verify deletion
results_after_deletion = vector_store.get_by_ids(ids)
for i, result in enumerate(results_after_deletion):
    print(f"Result after deletion {i}: {result.page_content[:50]}...")  # Print first 50 characters of each result
    print(f"Metadata: {result.metadata}")  # Print metadata associated with the result