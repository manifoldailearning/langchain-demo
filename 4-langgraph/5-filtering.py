from typing import Annotated,TypedDict
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage,filter_messages
from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import StateGraph,START,END,add_messages
from langgraph.checkpoint.memory import MemorySaver
import uuid
from dotenv import load_dotenv

load_dotenv()

sample_messages=[
    SystemMessage(content="You are a helpful assistant that can answer questions.", id = "1"),
    HumanMessage(content="Hello, how are you?", id = "2", name="bob"),
    AIMessage(content="I'm doing great, thank you! How can I help you today?", id = "3", name="alice"),
    HumanMessage(content="What is the capital of France?", id = "4", name="bob"),
    AIMessage(content="The capital of France is Paris.", id = "5", name="alice"),
    HumanMessage(content="What is the capital of India?", id = "6", name="bob"),
    AIMessage(content="The capital of India is New Delhi.", id = "7", name="alice"),
    HumanMessage(content="What is the capital of Germany?", id = "8", name="bob"),
    AIMessage(content="The capital of Germany is Berlin.", id = "9", name="alice"),
    HumanMessage(content="What is the capital of Japan?", id = "10", name="bob"),
    AIMessage(content="The capital of Japan is Tokyo.", id = "11", name="alice"),
]

output = filter_messages(sample_messages, include_types="human")
print(f"Filtered messages (Human): {output}")
print("*"*100)
# filter messages by name = "bob"
output = filter_messages(sample_messages, include_names = ["bob"])
print(f"Filtered messages with name = bob : {output}")
print("*"*100)
# print("*"*100)
# filter messages by name = "alice" - use lambda function
# output = filter_messages(sample_messages, filter_func=lambda message: message.name == "alice")
# print(f"Filtered messages (alice): {output}")
# print("*"*100)
# filter messages by name = "bob" and "alice"
# output = filter_messages(sample_messages, filter_func=lambda message: message.name in ["bob", "alice"])
# print(f"Filtered messages (bob & alice): {output}")
# print("*"*100)

#demo of exclude_types
output = filter_messages(sample_messages, exclude_types="ai")
print(f"Filtered messages (exclude_types=ai): {output}")
print("*"*100)
#demo of exclude with id
output = filter_messages(sample_messages, exclude_ids=["1", "2", "3"]) 
print(f"Filtered messages (exclude_ids=1,2,3): {output}")
print("*"*100)
#demo of exclude with name
output = filter_messages(sample_messages, exclude_names=["bob", "alice"])
print(f"Filtered messages (exclude_names=bob,alice): {output}")
print("*"*100)  
#demo of exclude with name and id
output = filter_messages(sample_messages, exclude_names=["bob", "alice"], exclude_ids=["1", "2", "3"])
print(f"Filtered messages (exclude_names=bob,alice & exclude_ids=1,2,3): {output}")