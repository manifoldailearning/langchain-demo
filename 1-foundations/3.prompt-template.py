from langchain_core.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# initialize the chat model
model = ChatOpenAI( model = "gpt-4.1-nano", temperature = 0.5)
template = PromptTemplate.from_template(
    """Answer the question using the context below. if the question cannot be answered using the context, say 'I don't know'. 
    
    Context: {context} 
    
    Question: {question} 
    Answer: """)

# Example usage of creation of template
response = template.invoke({
    "context": "The capital of France is Paris.",
    "question": "What is the capital of India?"})

print(response)

model_response = model.invoke(response)
print("--------------------------------")
print("Model Response:")
print("--------------------------------")
print(model_response.content)   