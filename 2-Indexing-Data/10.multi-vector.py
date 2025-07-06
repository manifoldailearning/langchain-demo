from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
import uuid 
from langchain_postgres import PGVector
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.document_loaders.parsers.pdf import PyPDFParser
from langchain_core.documents.base import Blob
# Load environment variables

load_dotenv()

connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
blob = Blob.from_path("./Arjun_Varma_Generative_AI_Resume.pdf")

parser = PyPDFParser()

documents = parser.lazy_parse(blob)
docs = []
for doc in documents:
    docs.append(doc)
# print(docs[0].page_content)
# print(docs[0].metadata)
print(docs)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Adjust chunk size as needed
    chunk_overlap=50  # Adjust overlap as needed
)

chunks = splitter.split_documents(docs)
print(f"Number of chunks created: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:")
    print(chunk.page_content)  # Print the content of each chunk
    print("--------------------------------")

# create vector store
vector_store = PGVector.from_documents(
    documents=chunks,
    embedding=embeddings,
    connection=connection,
    use_jsonb=True,  # Use JSONB for metadata storage
)
results = vector_store.similarity_search("", k=30)
for i, result in enumerate(results):
    print(f"Result {i}: {result.page_content[:50]}...")  # Print first 50 characters of each result
    print(f"Metadata: {result.metadata}")  # Print metadata associated with the result

prompt_text = "Summarize the following Document : {text}" # Document content will be passed here

prompt = ChatPromptTemplate.from_template(prompt_text)

llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.5)

summary_chain = {"text": lambda x: x.page_content} | prompt | llm | StrOutputParser()

summaries = summary_chain.batch(chunks, {"max_concurrency": 5})
print("------------------------------------")
print("Summaries:")
print(summaries)
print("------------------------------------")
# storage layer for parent documents
store = InMemoryStore()
id_key = "document_id" # linkage summaries to their parent chunks
doc_ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
store.mset(list(zip(doc_ids, chunks))) # Add full chunks to the Docstore

# each summary linked to its corresponding document
summary_docs = [
    Document(page_content=summary, metadata={"document_id": doc_ids[i]}) for i, summary in enumerate(summaries)]

# Create a MultiVectorRetriever to handle both the original chunks and the summaries
# index the summary in the vector store
retriever = MultiVectorRetriever(
    vectorstore=vector_store,
    docstore=store,
    id_key=id_key)

# Add the document summaries to the vector store
retriever.vectorstore.add_documents(summary_docs)

# vector store retrieves the summaries
sub_docs = retriever.vectorstore.similarity_search(
    "Professional skill Summary", k=2)

print("sub docs: ", sub_docs[0].page_content)

print("length of sub docs:\n", len(sub_docs[0].page_content))

# Whereas the retriever will return the larger source document chunks:
retrieved_docs = retriever.invoke("Professional skill summary")

print("length of retrieved docs: ", len(retrieved_docs))
print("retrieved docs: ", retrieved_docs)