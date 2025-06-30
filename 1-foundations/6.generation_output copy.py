from langchain_openai.chat_models import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel,Field

load_dotenv()
# initialize the chat model
model = ChatOpenAI( model = "gpt-4.1-nano", temperature = 0.5)

# single request
single_response = model.invoke("write a short story about a cat")
print("Single Response:")
print(single_response.content)

print("\n--------------------------------")
print("Batch Resposes")
# batch requests

batch_responses = model.batch([
    "write a short story about a dog",
    "write a short story about a bird",
    "write a short story about a fish",
    "translate 'Hello, how are you?' to French",])

print("\nBatch Responses:")
for i, response in enumerate(batch_responses):
    print(f"Response {i+1}:")
    print(response.content)
print("\n--------------------------------")
print("Stream Resposes")
# stream responses
for token in model.stream("write a short story about a rabbit"):
    print(token.content, end="", flush=True)