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