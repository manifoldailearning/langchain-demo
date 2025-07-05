from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


load_dotenv()
# initialize the chat model
model = ChatOpenAI( model = "gpt-4.1-nano", temperature = 0.5)

system_message = SystemMessage(
    "you are a helpful assistant that responds to questions with four exclamation marks")

human_message = HumanMessage(
    "What is the capital of India?")
#
response = model.invoke([system_message, human_message])
print(response.content)
