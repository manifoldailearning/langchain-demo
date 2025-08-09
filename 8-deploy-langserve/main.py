from fastapi import FastAPI
from langserve import add_routes
from agent_graph import create_agent_graph
import uvicorn
from dotenv import load_dotenv
import logging 

logger = logging.getLogger(__name__)
load_dotenv()
app = FastAPI()
graph = create_agent_graph()



@app.get("/")
def get_root():
    return {"message":"LangGraph Agent Server is Running"}

add_routes(app,graph,path="/ask")


if __name__ == "__main__":
    uvicorn.run(app,host="0.0.0.0",port=8000)