from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import uuid 
from langchain_postgres import PGVector
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.indexes import SQLRecordManager, index


load_dotenv()

connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
# create embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
collection_name = "my_collection"
namespace = "my_namespace"

# initialize vector store
vector_store = PGVector(
    collection_name=collection_name,
    embeddings=embeddings,
    connection=connection
)

# initialize record manager
record_manager = SQLRecordManager(
    db_url=connection,
    namespace=namespace)

# create schema
record_manager.create_schema()

# Create documents  
docs = [
    Document(page_content="there are cats in the house", metadata={"location": "home", "topic": "pets", "source": "file1.txt","id": 1}),
    Document(page_content="there are dogs also in the house", metadata={"location": "home", "topic": "pets","source": "file2.txt","id": 2})
]

# Index the documents

index_1 = index(
    docs,
    record_manager,
    vector_store,
    cleanup="incremental",
    source_id_key="source",
)

print(f"Index attempt 1: {index_1}")

# print results
results = vector_store.similarity_search("",k=10)
for result in results:
    print(f"Document: {result.page_content}, Metadata: {result.metadata}")

# Index the documents again with a same content
index_2 = index(
    docs,
    record_manager,
    vector_store,
    cleanup="incremental",
    source_id_key="source",
)

print(f"Index attempt 2: {index_2}")

# print results
results = vector_store.similarity_search("",k=10)
for result in results:
    print(f"Document: {result.page_content}, Metadata: {result.metadata}")

# Index the documents again with a different content
docs[0].page_content = "there are cats and dogs in the house"

index_3 = index(
    docs,
    record_manager,
    vector_store,
    cleanup="incremental",
    source_id_key="source",
)

print(f"Index attempt 3: {index_3}")

# print results
results = vector_store.similarity_search("",k=10)
for result in results:
    print(f"Document: {result.page_content}, Metadata: {result.metadata}")
