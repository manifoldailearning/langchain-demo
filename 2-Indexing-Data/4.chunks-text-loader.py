from langchain_community.document_loaders import TextLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

loader = TextLoader("./text_demo.txt")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=20,
    chunk_overlap=5,
    length_function=len
)

splitted_docs = splitter.split_documents(docs)
print(splitted_docs)