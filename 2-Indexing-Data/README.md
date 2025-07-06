```
docker run \
    --name pgvector-container \
    -e POSTGRES_USER=langchain \
    -e POSTGRES_PASSWORD=langchain \
    -e POSTGRES_DB=langchain \
    -p 6024:5432 \
    -d pgvector/pgvector:pg16



postgresql+psycopg://langchain:langchain@localhost:6024/langchain


```

# Modes of Indexing
- `None` does not automatically cleanup, user has to do the manual cleaning of old content
- `Incremental` mode - deletes the previous version of the content if the content of the source document or derived document has changed
- `Full` - will additionally delete any documents not included in the documents currently being indexed.