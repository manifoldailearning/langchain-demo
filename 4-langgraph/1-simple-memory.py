from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can answer questions."),
    ("placeholder", "{input}")
])

model = ChatOpenAI(model="gpt-4.1-nano", temperature=0.5)

chain = prompt | model

response = chain.invoke({
    "input": [("human", "Ramesh won the cricket match yesterday."),
                 ("ai","That's great news! Congratulations to Ramesh on winning the cricket match. It must have been an exciting game!"),
                 ("human", "Who won cricket match yesterday where Ramesh was playing?")]
})
# invoke - single input
# batch - multiple inputs
# stream - streaming output
print(response.content)