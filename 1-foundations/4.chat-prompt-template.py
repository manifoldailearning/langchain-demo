from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# initialize the chat model
model = ChatOpenAI( model = "gpt-4.1-nano", temperature = 0.5)

template = ChatPromptTemplate([
    ("system", "You are a helpful AI bot. Your name is {name}."),
    ("human", "Hello, how are you doing?"),
    ("ai", "I'm doing well, thanks!"),
    ("human", "{user_input}"),
])

prompt_value = template.invoke(
    {
        "name": "Bob",
        "user_input": "What is your name?"
    })

print(prompt_value)

template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            'Answer the question based on the context below. If the question cannot be answered using the information provided, answer with "I don\'t know".',
        ),
        ("human", "Context: {context}"),
        ("human", "Question: {question}"),
    ]
)

response = template.invoke({
    "context": "The capital of France is Paris.",
    "question": "What is the capital of France?"})

print(response) 

model_response = model.invoke(response)
print("--------------------------------")
print("Model Response:")
print("--------------------------------")
print(model_response.content)   
