# 1. LLM call to pick the available indexes to use, given the user query
# 2. Retrieval step that queries the chosen index for the most relevant documents for the user query
# 3. A new LLM call to generate an answer based on the retrieved documents & user query

from typing import Annotated,TypedDict, Literal
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
model_low_temp = ChatOpenAI(model="gpt-4.1-nano", temperature=0.1) # low temperature means more focused
model_high_temp = ChatOpenAI(model="gpt-4.1-nano", temperature=0.8) # high temperature means more creative


class State(TypedDict):
    messages: Annotated[list,add_messages]

    # input
    user_query: str
    # output
    domain: Literal["records","insurance","finance"]
    documents: list[Document]
    answer: str

class Input(TypedDict):
    user_query: str

class Output(TypedDict):
    documents: list[Document]
    answer: str

# Sample documents for testing

sample_docs = [
    Document(page_content="This is a sample document for the records domain", metadata={"domain": "records"}),
    Document(page_content="This is a sample document for the insurance domain", metadata={"domain": "insurance"}),
    Document(page_content="This is a sample document for the finance domain", metadata={"domain": "finance"}),
]

# Initialize the vector store
medical_records_store = InMemoryVectorStore.from_documents(sample_docs, embeddings)
medical_records_retriever = medical_records_store.as_retriever()

insurance_store = InMemoryVectorStore.from_documents(sample_docs, embeddings)
insurance_retriever = insurance_store.as_retriever()

finance_store = InMemoryVectorStore.from_documents(sample_docs, embeddings)
finance_retriever = finance_store.as_retriever()

# Router prompt
router_prompt = SystemMessage(content="""You are a helpful assistant that can route the user 
                              query to the appropriate domain. You have 3 domains to choose 
                              from: records, insurance, finance. You have to pick the domain 
                              that is most relevant to the user query. Output only the domain
                               name, no other text.""")

def router_node(state: State) -> State:
    user_message = HumanMessage(content=state["user_query"])
    messages = [router_prompt,*state["messages"],user_message]
    response = model_low_temp.invoke(messages)
    return {"domain": response.content, "messages": [user_message, response]}

def pick_retriever(state: State) -> Literal["retrieve_medical_records","retrieve_insurance","retrieve_finance"]:
    if state["domain"] == "records":
        return "retrieve_medical_records"
    elif state["domain"] == "insurance":
        return "retrieve_insurance"
    elif state["domain"] == "finance":
        return "retrieve_finance"
    else:
        raise ValueError(f"Invalid domain: {state['domain']}")
    
def retrieve_medical_records(state: State) -> State:
    documents = medical_records_retriever.invoke(state["user_query"])
    return {"documents": documents}

def retrieve_insurance(state: State) -> State:
    documents = insurance_retriever.invoke(state["user_query"])
    return {"documents": documents}

def retrieve_finance(state: State) -> State:
    documents = finance_retriever.invoke(state["user_query"])
    return {"documents": documents}

medical_records_prompt = SystemMessage(content="""You are a helpful assistant that can answer questions about medical records.
                             """)

insurance_prompt = SystemMessage(content="""You are a helpful assistant that can answer questions about insurance.
                              """)

finance_prompt = SystemMessage(content="""You are a helpful assistant that can answer questions about finance.
                              """)

def generate_answer(state: State) -> State:
    if state["domain"] == "records":
        prompt = medical_records_prompt
    elif state["domain"] == "insurance":
        prompt = insurance_prompt
    elif state["domain"] == "finance":
        prompt = finance_prompt
    else:
        raise ValueError(f"Invalid domain: {state['domain']}")

    messages = [prompt,*state["messages"], HumanMessage(f"Documents: {state['documents']}")]
    response = model_high_temp.invoke(messages)
    return {"answer": response.content, "messages": [response]}

builder = StateGraph(State,input_schema=Input, output_schema=Output)

builder.add_node("router", router_node)
builder.add_node("pick_retriever", pick_retriever)
builder.add_node("retrieve_medical_records", retrieve_medical_records)
builder.add_node("retrieve_insurance", retrieve_insurance)
builder.add_node("retrieve_finance", retrieve_finance)
builder.add_node("generate_answer", generate_answer)

builder.add_edge(START, "router")
builder.add_conditional_edges("router", pick_retriever)
builder.add_edge("retrieve_medical_records", "generate_answer")
builder.add_edge("retrieve_insurance", "generate_answer")
builder.add_edge("retrieve_finance", "generate_answer")
builder.add_edge("generate_answer", END)

graph = builder.compile()
graph.get_graph().draw_mermaid_png(output_file_path="router_graph.png")

input = {"user_query": "Am I covered for the medical expenses?"}

result = graph.invoke(input)
print(result)


