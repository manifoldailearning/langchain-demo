```-- (a) Enable required extensions
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS vector;

-- (b) Drop & recreate the documents table
DROP TABLE IF EXISTS public.documents;
CREATE TABLE public.documents (
  id        uuid       PRIMARY KEY DEFAULT gen_random_uuid(),
  content   text       NOT NULL,
  metadata  jsonb,
  embedding vector(1536)
);

-- (c) Create an ivfflat index for fast searches
CREATE INDEX ON public.documents
  USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);

-- (d) Create the RPC that LangChain calls
DROP FUNCTION IF EXISTS public.match_documents(jsonb, vector);
CREATE FUNCTION public.match_documents(
  filter          jsonb,
  query_embedding vector(1536)
)
RETURNS TABLE (
  id        uuid,
  content   text,
  metadata  jsonb,
  embedding vector(1536)
) AS $$
  SELECT id, content, metadata, embedding
    FROM public.documents
   WHERE (filter = '{}' OR metadata @> filter)
   ORDER BY embedding <-> query_embedding;
$$ LANGUAGE sql STABLE;
```

```
conda create --prefix ./my_env python=3.11 -y
conda activate ./my_env

OPENAI_API_KEY="<key-here>" in .env file
```

```
curl http://127.0.0.1:2024/assistants/search \
  --request POST \
  --header 'Content-Type: application/json' \
  --data '{
  "metadata": {},
  "graph_id": "retrieval_graph",
  "limit": 10,
  "offset": 0,
  "sort_by": "assistant_id",
  "sort_order": "asc"
}'

```

```
curl -X POST http://127.0.0.1:2024/runs/stream \
  -H 'Content-Type: application/json' \
  -d '{
    "assistant_id": "571ade52-f5cd-582a-89d9-d79dc861a8ba",
    "input": { "messages": [ { "role": "user", "content": "What is Agentic AI?" } ] },
    "config": { "configurable": { "k": 4 } },
    "multitask_strategy": "reject",
    "stream_mode": ["values"]
  }'

```

```
OPENAI_API_KEY=<api-key>
SUPABASE_URL=<url>
SUPABASE_SERVICE_ROLE_KEY=<>
LANGCHAIN_TRACING_V2=true
LANGSMITH_API_KEY=<langchain-api-key>
LANGCHAIN_API_KEY=<langchain-api-key>
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
```