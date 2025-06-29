from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.runnables import chain

load_dotenv()

# initialize the chat model
model = ChatOpenAI( model = "gpt-4.1-nano", temperature = 0.5)
template = ChatPromptTemplate([
    ("system", "You are a helpful AI bot. Your name is {name}."),
    ("human", "Hello, how are you doing?"),
    ("ai", "I'm doing well, thanks!"),
    ("human", "{user_input}"),
])

message = {"name": "Bob",
        "user_input": "What is your name?"
    }


@chain
def chatbot(values):
    prompt = template.invoke(values)
    response = model.invoke(prompt)
    return response.content

print(chatbot.invoke(message))