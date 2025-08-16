# LangGraph node-graph that uses your Supabase vector store + OpenAI
import os
from typing import Annotated, Dict, Any
from typing_extensions import TypedDict

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# ← your helper for Supabase vectors
from vector_store import get_vector_store

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
TOP_K_DEFAULT = int(os.environ.get("TOP_K", "4"))

# ---- State type ----
class RAGState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# ---- Init shared resources ----
vectorstore = get_vector_store()
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_DEFAULT, "filter": {"content_type": "text"}})
llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)

system_prompt = (
    "You are a strict RAG assistant. Answer ONLY using the provided context. "
    "If the answer is not in the context, say you don't know."
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Context:\n{context}\n\nQuestion: {question}")
])

def _extract_last_user(state: RAGState) -> str:
    for m in reversed(state["messages"]):
        if getattr(m, "type", None) == "human" or getattr(m, "role", None) == "user":
            return getattr(m, "content", "") or ""
    return ""

def _retrieve_ctx(q: str, k: int) -> str:
    if not q:
        return ""
    docs = retriever.get_relevant_documents(q) or []
    if isinstance(k, int) and k > 0:
        docs = docs[:k]
    return "\n\n".join(d.page_content for d in docs)

# ---- Single node that answers ----
def answer_node(state: RAGState) -> Dict[str, Any]:
    # allow overriding k via config (dev API passes it separately; we’ll default if absent)
    k = TOP_K_DEFAULT
    # Pull question
    question = _extract_last_user(state)
    context = _retrieve_ctx(question, k)
    messages = prompt.format_messages(context=context, question=question)
    out = llm.invoke(messages)
    return {"messages": [AIMessage(content=out.content)]}

def graph():
    g = StateGraph(RAGState)
    g.add_node("answer", answer_node)
    g.add_edge(START, "answer")
    g.add_edge("answer", END)
    return g.compile()
