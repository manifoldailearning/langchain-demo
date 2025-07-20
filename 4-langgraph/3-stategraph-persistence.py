from typing import Annotated,TypedDict
from langchain_core.messages import HumanMessage
from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import StateGraph,START,END,add_messages
from langgraph.checkpoint.memory import MemorySaver
import uuid
from dotenv import load_dotenv

load_dotenv()

class State(TypedDict): # State is a dictionary of key-value pairs
    messages: Annotated[list,add_messages] # messages is a list, add_messages is a function
#The above tells Langgrah , whenever this state is updated, dont overwrite the messages, 
# just add to it

x: Annotated[int, "This is a number"] # demo of annotation

builder = StateGraph(State)
model = ChatOpenAI(model="gpt-4.1-nano", temperature=0.5)

def chatbot(state:State):   
    answer = model.invoke(state["messages"])
    return {"messages": [answer]}

builder.add_node("chatbot",chatbot)
# add edges
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

# add persistence
graph = builder.compile(checkpointer=MemorySaver())

# Configure the graph with a unique thread_id   
thread_id = str(uuid.uuid4())
thread1 = {"configurable": {"thread_id": thread_id}}

result_1 = graph.invoke({"messages": [("human", "Hello, my name is Nachiketh")]}, thread1)
print(result_1)
print("*"*100)
result_2 = graph.invoke({"messages": [("human", "What is my name")]}, thread1)
print(result_2)

print("*"*100)
print("Getting state")
# Get state
print("*"*100)
print(graph.get_state(thread1))
