from langchain_community.document_loaders import TextLoader

loader = TextLoader("./text_demo.txt")
docs = loader.load()
print(docs)