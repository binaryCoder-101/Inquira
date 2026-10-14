from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv
from langsmith import traceable, Client
import os

load_dotenv()

# LangSmith Configuration
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_API_KEY"] = "true"
    os.environ.setdefault("LANGSMITH_PROJECT_NAME", "Inquira")
    print("LangSmith is configured. - Project: {os.getenv('LANGSMITH_PROJECT_NAME')}")
else:
    print("LangSmith is not configured.")


# Pydantic Schema
class QAResponse(BaseModel):
    answer: str = Field(description="Answer to the question")
    confidence: float = Field(description="Confidence level: high, medium, low")
    reasoning: str = Field(description="Reasoning behind the answer")
    follow_up_questions: List[str] = Field(
        description="A list of follow-up questions related to the topic", 
        default_factory=list
    )
    sources_needed: bool = Field(
        description="Whether sources are needed to answer the question",
        default=False
    )

# Bot implementation
class SmartQABot:
    def __init__(
    self, 
    model_name: str = "gpt-4o-mini", 
    temperature: float = 0.3,
    ):
        self.model = ChatOpenAI(
            model=model_name,
            temperature=temperature,
        ).with_structured_output(QAResponse)

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", 
                """
                You are a knowledgeable Q&A assistant.

                Your guidelines:
                1. Answer the question accurately and concisely.
                2. Be honest about uncertainity - set confidence to 'low' if unsure.
                3. Provide clear reasoning for your answers.
                4. Suggest relevant follow-up questions if applicable.
                5. Determine if external sources are needed to answer the question.
                
                Always respond with accurate, helpful information
                """
                ),
                ("human", "Question: {question}"),
            ]
        )
        
        self.chain = self.prompt | self.model

    @traceable(name="ask_question", run_type="chain")
    def ask(self, question: str) -> QAResponse:
        try:
            response = self.chain.invoke({"question": question})
            return response
        except Exception as e:
            # return a graceful error response
            return QAResponse(
                answer="I am sorry, I am not able to answer your question at the moment.",
                confidence="low",
                reasoning=f"Error: {str(e)}",
                follow_up_questions=["Could you please try again later?"],
                sources_needed=False
            )
    
    @traceable(name="ask_batch", run_type="chain")
    def ask_batch(self, questions: list[str]) -> list[QAResponse]:
        try:
            responses = self.chain.batch([{"question": question} for question in questions])
            return responses
        except Exception as e:
            # return a graceful error response
            return [QAResponse(
                answer="I am sorry, I am not able to answer your question at the moment.",
                confidence="low",
                reasoning=f"Error: {str(e)}",
                follow_up_questions=["Could you please try again later?"],
                sources_needed=False
            ) for _ in questions]

def main():
    print("Hello from inquira!")


if __name__ == "__main__":
    main()
