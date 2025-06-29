from langchain_community.document_loaders.web_base import WebBaseLoader

loader = WebBaseLoader("https://en.wikipedia.org/wiki/Artificial_intelligence")

docs = loader.load()
print(docs)