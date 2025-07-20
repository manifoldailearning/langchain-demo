from typing import Annotated,TypedDict
from langchain_core.messages import HumanMessage
from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import StateGraph,START,END,add_messages
from langgraph.checkpoint.memory import MemorySaver

from dotenv import load_dotenv

load_dotenv()

class State(TypedDict): # State is a dictionary of key-value pairs
    messages: Annotated[list,add_messages] # messages is a list, add_messages is a function
#The above tells Langgrah , whenever this state is updated, dont overwrite the messages, 
# just add to it

x: Annotated[int, "This is a number"]

builder = StateGraph(State)
model = ChatOpenAI(model="gpt-4.1-nano", temperature=0.5)

def chatbot(state:State):   
    answer = model.invoke(state["messages"])
    return {"messages": [answer]}

builder.add_node("chatbot",chatbot)
# add edges
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile()

graph.invoke({"messages": [("human", "Hello, how are you?")]})

# Visualize the graph & save to file
# Save the graph to a file
graph.get_graph().draw_mermaid_png(output_file_path="graph.png")

# Run the graph
input = {"messages": [HumanMessage(content="Hello, how are you?")]}
for chunk in graph.stream(input):
    print(chunk)

