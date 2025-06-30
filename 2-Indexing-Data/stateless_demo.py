from langchain_openai.chat_models import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel,Field

load_dotenv()
# initialize the chat model
model = ChatOpenAI( model = "gpt-4.1-nano", temperature = 0.5)

print("\n--------------------------------")
print("Batch Resposes")
# batch requests

batch_responses = model.batch([
    "Ramesh in our class has won cricket match",
    "who won the cricket match",])

print("\nBatch Responses:")    
for i, response in enumerate(batch_responses):
    print(f"Response {i+1}:")
    print(response.content)