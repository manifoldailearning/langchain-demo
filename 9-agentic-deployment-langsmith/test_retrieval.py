from vector_store import retrieve

query = "What is Agentic AI?"

print("Query: ", query)

results = retrieve(query)

if not results:
    print("No results found")
else:
    for i, doc in enumerate(results):
        print(f"Result {i+1}:")
        print(f"Document: {doc}")
        print("-"*100)