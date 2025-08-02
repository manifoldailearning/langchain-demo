from typing import Annotated,TypedDict
from langchain_core.messages import HumanMessage,SystemMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
import ast
import numexpr

load_dotenv()

@tool
def calculator(query:str) -> str:
    """
    A simple calculator tool. Input should be a mathematical expression.
    """
    return str(numexpr.evaluate(query))

search = DuckDuckGoSearchRun()
tools = [search,calculator]

model = ChatOpenAI(model="gpt-4.1-nano", temperature=0.1).bind_tools(tools)

class State(TypedDict):
    messages: Annotated[list,add_messages]

def model_node(state:State) -> State:
    result = model.invoke(state["messages"])
    return {"messages":result}



builder = StateGraph(State)
builder.add_node("model",model_node)
builder.add_node("tools",ToolNode(tools))
builder.add_edge(START, "model")
builder.add_conditional_edges("model",tools_condition)
builder.add_edge("tools", "model")
builder.add_edge("model", END)

graph = builder.compile()
graph.get_graph().draw_mermaid_png(output_file_path="basic-agents.png")
# input = {
#     "messages":[HumanMessage(content="How old was Albert Einstein when he died?")]
# }

input = {
    "messages":[HumanMessage(content="What will be the fixed deposit amount interest for 100000 INR for 1 year at RBI repo rate as of today?")]
}

for chunk in graph.stream(input):
    print(chunk)

# format the output in markdown
for chunk in graph.stream(input):
    print(chunk.content)