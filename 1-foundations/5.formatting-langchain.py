from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel,Field

load_dotenv()

class Book(BaseModel):
    title: str = Field( description="The title of the book")
    author: str = Field( description="The author of the book")
    year: int = Field( description="The year the book was published")

# initialize the chat model
model = ChatOpenAI( model = "gpt-4.1-nano", temperature = 0.5).with_structured_output(Book)

response = model.invoke(
    "The title of the book is 'The Great Gatsby' and the author is F. Scott Fitzgerald. The book was published in 1925.")

print(response)

class MovieReview(BaseModel):
    summary: str = Field( description="Short Summary of the movie")
    review: str = Field( description="The review of the movie")
    rating: float = Field( description="The rating of the movie out of 10")

# initialize the chat model
model = ChatOpenAI( model = "gpt-4.1-nano", temperature = 0.5).with_structured_output(MovieReview)

response = model.invoke("share me the review of the movie 'Inception'")

print(response)
