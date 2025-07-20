from typing import Annotated,TypedDict
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()


model_low_temp = ChatOpenAI(model="gpt-4.1-nano", temperature=0.1) # low temperature means more focused
model_high_temp = ChatOpenAI(model="gpt-4.1-nano", temperature=0.8) # high temperature means more creative

# Use case:
# Call 1 --> Generate a sql code
# Call 2 --> Explain the sql code

class State(TypedDict):
    messages: Annotated[list,add_messages] # to track the conversation history

    # input
    user_query: str
    # output
    sql_query: str
    sql_explanation: str

class Input(TypedDict):
    user_query: str

class Output(TypedDict):
    sql_query: str
    sql_explanation: str

generate_prompt = SystemMessage(content="You are a helpful assistant that can generate a sql code based on the user query.")
explain_prompt = SystemMessage(content="You are a helpful assistant that can explain the sql code in a simple way.")

def generate_sql(state: State) -> State:
    user_message = HumanMessage(content=state["user_query"])
    messages = [generate_prompt,*state["messages"],user_message] # System Prompt + Conversation History + User Message
    response = model_low_temp.invoke(messages)
    return {"sql_query": response.content, "messages": [user_message, response]}

def explain_sql(state: State) -> State:
    messages = [explain_prompt,*state["messages"]] # System Prompt + Conversation History
    response = model_high_temp.invoke(messages)
    return {"sql_explanation": response.content, "messages": [response]}

builder = StateGraph(State,input_schema=Input, output_schema=Output)

builder.add_node("generate_sql", generate_sql)
builder.add_node("explain_sql", explain_sql)
builder.add_edge(START, "generate_sql")
builder.add_edge("generate_sql", "explain_sql")
builder.add_edge("explain_sql", END)

graph = builder.compile()

# Save the graph to a file
graph.get_graph().draw_mermaid_png(output_file_path="chain_graph.png")

result = graph.invoke({"user_query": "What is the total sales of the company?"})
print(result)