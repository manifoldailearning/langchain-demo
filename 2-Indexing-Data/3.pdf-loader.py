from langchain_community.document_loaders.pdf import PyPDFLoader

loader = PyPDFLoader("./guideline-170-en.pdf")
docs = loader.load()
print(docs)