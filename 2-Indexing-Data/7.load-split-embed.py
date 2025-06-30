from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

load_dotenv()

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