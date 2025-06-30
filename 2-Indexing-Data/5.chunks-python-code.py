# chunking for stateless_demo.py

from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import TextLoader

# load the python code file
loader = TextLoader("./stateless_demo.py")
docs = loader.load()

# split the documents into chunks
splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, chunk_size=100, chunk_overlap=0)

splitted_docs = splitter.split_documents(docs)
print(splitted_docs)
