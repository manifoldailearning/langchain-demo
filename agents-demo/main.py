from typing import Annotated,TypedDict
from langchain_core.messages import HumanMessage
from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import StateGraph,START,END,add_messages
from langgraph.checkpoint.memory import MemorySaver
import uuid
from dotenv import load_dotenv
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

load_dotenv()

wiki = WikipediaAPIWrapper()
wiki_tool = Tool.from_function(
    func=wiki.run,
    name="Wikipedia Search",
    description="Useful for answering general knowledge questions using Wikipedia"
)

prompt = PromptTemplate.from_template("""
You are an agent that answers questions using a tool.
Decide whether to search Wikipedia for the answer.

Question: {question}

Think step-by-step and provide the final answer below.
""")

llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.5)

llm_chain = prompt |  llm

class AgentState(dict):
    question: Annotated[list,add_messages]

def decide_wikipedia_use(state: AgentState) -> str:
    question = state["question"]
    if any(word in question for word in ["who", "when", "what", "where", "define", "describe"]):
        return "use_wikipedia"
    return "skip"

def use_wikipedia(state: AgentState) -> AgentState:
    question = state["question"]
    answer = wiki.run(question)
    state["answer"] = answer
    return state

def skip_tool(state: AgentState) -> AgentState:
    question = state["question"]
    response = llm_chain.invoke({"question": question})
    state["answer"] = response
    return state

def output_node(state: AgentState) -> AgentState:
    print("🧠 Final Answer:")
    print(state["answer"])
    return state


builder = StateGraph(AgentState)
builder.set_entry_point("decision")
builder.add_node("decision", RunnableLambda(decide_wikipedia_use))
builder.add_node("use_wikipedia", RunnableLambda(use_wikipedia))
builder.add_node("skip", RunnableLambda(skip_tool))
builder.add_node("output", RunnableLambda(output_node))

builder.add_conditional_edges("decision", decide_wikipedia_use, {
    "use_wikipedia": "use_wikipedia",
    "skip": "skip"
})
builder.add_edge("use_wikipedia", "output")
builder.add_edge("skip", "output")
builder.add_edge("output", END)

graph = builder.compile()
#download the graph
graph.get_graph().draw_mermaid_png(output_file_path="graph.png")

# Step 6: Run
if __name__ == "__main__":
    while True:
        q = input("\nAsk a question (or 'exit'): ")
        if q.lower() == "exit":
            break
        result = graph.invoke(AgentState({"question": q}))