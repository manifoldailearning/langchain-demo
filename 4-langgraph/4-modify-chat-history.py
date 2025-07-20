from typing import Annotated,TypedDict
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage,trim_messages
from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import StateGraph,START,END,add_messages
from langgraph.checkpoint.memory import MemorySaver
import uuid
from dotenv import load_dotenv

load_dotenv()

sample_messages=[
    SystemMessage(content="You are a helpful assistant that can answer questions."),
    HumanMessage(content="Hello, how are you?"),
    AIMessage(content="I'm doing great, thank you! How can I help you today?"),
    HumanMessage(content="What is the capital of France?"),
    AIMessage(content="The capital of France is Paris."),
    HumanMessage(content="What is the capital of India?"),
    AIMessage(content="The capital of India is New Delhi."),
    HumanMessage(content="What is the capital of Germany?"),
    AIMessage(content="The capital of Germany is Berlin."),
    HumanMessage(content="What is the capital of Japan?"),
    AIMessage(content="The capital of Japan is Tokyo."),
]

# create trimmer
trimmer = trim_messages(max_tokens=50,
                        strategy="last",
                        token_counter=ChatOpenAI(model="gpt-4.1-nano", temperature=0.5),
                        include_system=True,
                        allow_partial=False,
                        start_on="human")
trimmed_messages = trimmer.invoke(sample_messages)
print(trimmed_messages)