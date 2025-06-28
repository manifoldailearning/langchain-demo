from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI


load_dotenv()

messages = [   
    (
         "system",
         "You are a helpful assistant that translates English to French. Translate the user sentence",
         ),
         ("human", "I love programming in Python."),
         ]

# initialize the chat model
model = ChatOpenAI( model = "gpt-4.1-nano", temperature = 0.5)

# response
response = model.invoke(messages)
print(response.content)

response = model.invoke("The Hugging face python library is")
print(response.content)