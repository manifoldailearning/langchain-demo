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
from langgraph.types import interrupt
from langgraph.checkpoint.memory import InMemorySaver

# ← your helper for Supabase vectors
from vector_store import get_vector_store

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
TOP_K_DEFAULT = int(os.environ.get("TOP_K", "4"))

# ---- State type ----
class RAGState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    draft: str
    final: str
    sources: list

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

def _retrieve(q: str, k: int) -> str:
    if not q:
        return ""
    docs = retriever.get_relevant_documents(q) or []
    if isinstance(k, int) and k > 0:
        docs = docs[:k]
    context = "\n\n".join(d.page_content for d in docs)
    return context,docs

def draft_answer(state: RAGState) -> Dict[str, Any]:
    q = _extract_last_user(state)
    context,docs = _retrieve(q, TOP_K_DEFAULT)
    messages = prompt.format_messages(context=context, question=q)
    out = llm.invoke(messages)
    draft = out.content

    if docs:
        draft += "\n\nSources: " + " ".join(f"[{i+1}]" for i, _ in enumerate(docs))
    
    return {
        "draft": draft,
        "sources": [
            {"i": i+1, "metadata": getattr(d, "metadata", {}), "snippet": d.page_content[:240]}
            for i, d in enumerate(docs)
        ],
    }

def human_review(state: RAGState) -> Dict[str, Any]:
    """
    Pause here and ask a human to approve/edit the draft.
    - If the human replies "approve" -> we use the draft as final.
    - If the human replies with a string -> treat it as the edited final answer.
    - If the human replies with {"final": "..."} -> use that exact text.
    """
    review_request = {
        "type": "review",
        "instructions": "Reply 'approve' to accept, paste edited text to override, "
                        "or send {\"final\": \"...\"}.",
        "draft": state.get("draft", ""),
        "sources": state.get("sources", []),
    }
    human_value = interrupt(review_request)  # <-- pauses execution here until resumed
    # normalize the resume value
    if isinstance(human_value, dict) and "final" in human_value:
        final_text = str(human_value["final"])
    elif isinstance(human_value, str):
        final_text = state.get("draft", "") if human_value.strip().lower() == "approve" else human_value
    else:
        final_text = state.get("draft", "")
    return {"final": final_text}

def finalize(state: RAGState) -> Dict[str, Any]:
    return {"messages": [AIMessage(content=state["final"])]}

# ----- Build graph -----
def graph():
    g = StateGraph(RAGState)
    g.add_node("draft", draft_answer)
    g.add_node("human_review", human_review)
    g.add_node("finalize", finalize)

    g.add_edge(START, "draft")
    g.add_edge("draft", "human_review")
    g.add_edge("human_review", "finalize")
    g.add_edge("finalize", END)

    # IMPORTANT: interrupts require a checkpointer so the run can pause & resume
    memory = InMemorySaver()
    return g.compile(checkpointer=memory)
