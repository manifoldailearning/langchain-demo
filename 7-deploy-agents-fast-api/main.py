from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from agent_graph import create_agent_graph
from langgraph.graph import END
import uvicorn
from dotenv import load_dotenv
import logging 

logger = logging.getLogger(__name__)
load_dotenv()
app = FastAPI()
graph = create_agent_graph()

class Query(BaseModel):
    question: str

@app.get("/")
def get_root():
    return {"message":"LangGraph Agent Server is Running"}

@app.post("/ask")
def ask_agent(query:Query):
    logger.info(f"Received query: {query.question}")
    input_data = {"messages":[HumanMessage(content=query.question)]}

    all_messages = []   
    for chunk in graph.stream(input_data):
        logger.info(f"New chunk: {chunk}")
        all_messages.append(chunk)
    return {"messages":all_messages[-1]}

    # def event_stream():
    #     for step in graph.stream(input_data):
    #         msg_list = step.get("messages",[])
    #         for msg in msg_list:
    #            if hasattr(msg,"content"):
    #                yield msg.content + "\n"
               
    # return StreamingResponse(event_stream(),media_type="text/plain")

if __name__ == "__main__":
    uvicorn.run(app,host="0.0.0.0",port=8000)