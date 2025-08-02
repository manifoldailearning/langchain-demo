from typing import Annotated,TypedDict
from langchain_core.messages import HumanMessage,SystemMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
import ast
import numexpr
from datetime import datetime
from langchain_core.tools import Tool

load_dotenv()

@tool
def date_difference(query:str) -> str:
    """
    This is a tool to calculate the difference between two dates.
    Input Format: "YYYY-MM-DD to YYYY-MM-DD"
    Example: "2025-01-01 to 2025-01-02"
    """
    try:
        start_date,end_date = query.split(" to ")
        start_date = datetime.strptime(start_date.strip(),"%Y-%m-%d")
        end_date = datetime.strptime(end_date.strip(),"%Y-%m-%d")
        delta = end_date - start_date
        years = delta.days // 365
        return f"{years} years"
    except Exception as e:
        return f"Error: {e}"

@tool
def calculator(query:str) -> str:
    """
    A simple calculator tool. Input should be a mathematical expression.
    """
    return str(numexpr.evaluate(query))

search = DuckDuckGoSearchRun()
wikipedia = WikipediaAPIWrapper()
wiki_tool = Tool.from_function(func=wikipedia.run,
                               name="wikipedia_search",
                               description="This tool is useful for retrieving the knowledge from Wikipedia articles")

tools = [search,calculator,date_difference,wiki_tool]

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
input = {
    "messages":[HumanMessage(content="How old was Albert Einstein when he died?")]
}

# input = {
#     "messages":[HumanMessage(content="What will be the fixed deposit amount interest for 100000 INR for 1 year at RBI repo rate as of today?")]
# }

for chunk in graph.stream(input):
    print(chunk)

