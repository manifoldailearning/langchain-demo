from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from supabase import create_client, Client
from dotenv import load_dotenv
import os

load_dotenv()

# ── Environment ────────────────────────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
INDEX_NAME = "documents"

# ── Clients & Models ──────────────────────────────────────────────────────────
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ── Add Documents ──────────────────────────────────────────────────────────────
def add_documents(texts: list[str]):
    docs = [
        Document(page_content=text, metadata={"content_type": "text"})
        for text in texts
    ]
    # Use `from_documents()` with `client=...`
    SupabaseVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        client=supabase,
        table_name=INDEX_NAME,
        query_name="match_documents",  # Adjust if your RPC is named differently
    )

# ── Get Vector Store Instance ─────────────────────────────────────────────────
def get_vector_store() -> SupabaseVectorStore:
    return SupabaseVectorStore(
        client=supabase,
        embedding=embeddings,
        table_name=INDEX_NAME,
        query_name="match_documents",
    )

# ── Retrieval Helpers ──────────────────────────────────────────────────────────
def retrieve(query: str, k: int = 2) -> list[str]:
    vs = get_vector_store()
    results = vs.similarity_search(query, k=k, filter={"content_type": "text"})
    return [doc.page_content for doc in results]

