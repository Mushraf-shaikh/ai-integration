import os
import time
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Load environment variables
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise ValueError("MISTRAL_API_KEY not found in environment variables")

# Initialize Mistral model
llm = ChatMistralAI(api_key=api_key, model="mistral-large-latest", temperature=0)


# ---------- Classification Task ---------- #
class SentimentClassification(BaseModel):
    sentiment: str = Field(
        ..., description="Sentiment of the text (positive, negative, neutral)"
    )


class SentimentClassifier:
    def __init__(self, llm):
        self.llm = llm
        # Few-shot examples for sentiment classification
        self.few_shot_examples = [
            {
                "input": "I love the new features in this update!",
                "output": {"sentiment": "positive"},
            },
            {
                "input": "This service is terrible and keeps crashing.",
                "output": {"sentiment": "negative"},
            },
            {
                "input": "The product works as expected, nothing special.",
                "output": {"sentiment": "neutral"},
            },
        ]
        # Define few-shot prompt
        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "Text: {input}"),
                ("ai", "{output}")
            ]
        )
        self.few_shot_prompt = FewShotChatMessagePromptTemplate(
            examples=self.few_shot_examples,
            example_prompt=example_prompt,
        )
        # Define main prompt
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a sentiment classifier. Analyze the input text and classify its sentiment as positive, negative, or neutral.",
                ),
                self.few_shot_prompt,
                ("human", "Text: {input}"),
            ]
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def classify(self, text: str) -> SentimentClassification:
        try:
            chain = self.prompt | self.llm.with_structured_output(SentimentClassification)
            result = chain.invoke({"input": text})
            return result
        except Exception as e:
            print(f"Error in sentiment classification: {e}")
            return SentimentClassification(sentiment="error")


# ---------- Extraction Task ---------- #
class InfoExtraction(BaseModel):
    name: Optional[str] = Field(None, description="Name mentioned in the text")
    date: Optional[str] = Field(None, description="Date mentioned in the text")
    email: Optional[str] = Field(None, description="Email address mentioned in the text")


class InfoExtractor:
    def __init__(self, llm):
        self.llm = llm
        # Few-shot examples for extraction
        self.few_shot_examples = [
            {
                "input": "Contact Jane Smith on March 5th, 2023 at jane.smith@example.com.",
                "output": {
                    "name": "Jane Smith",
                    "date": "March 5th, 2023",
                    "email": "jane.smith@example.com",
                },
            },
            {
                "input": "No meeting scheduled today.",
                "output": {"name": None, "date": None, "email": None},
            },
        ]
        # Define few-shot prompt
        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "Text: {input}"),
                ("ai", "{output}"),
            ]
        )
        self.few_shot_prompt = FewShotChatMessagePromptTemplate(
            examples=self.few_shot_examples,
            example_prompt=example_prompt,
        )
        # Define main prompt
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an information extractor. Extract the name, date, and email from the input text. Return null for fields not present.",
                ),
                self.few_shot_prompt,
                ("human", "Text: {input}"),
            ]
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def extract(self, text: str) -> InfoExtraction:
        try:
            chain = self.prompt | self.llm.with_structured_output(InfoExtraction)
            result = chain.invoke({"input": text})
            return result
        except Exception as e:
            print(f"Error in info extraction: {e}")
            return InfoExtraction(name=None, date=None, email=None)


# ---------- Test Run ---------- #
def main():
    classifier = SentimentClassifier(llm)
    extractor = InfoExtractor(llm)

    examples = [
        "John Doe sent an email on April 10th, 2023. His email address is john.doe@example.com. He's happy about the update.",
        "I am not satisfied with your service and I want a refund.",
        "Meeting with Alice on May 1st, 2025 at alice@company.org.",
    ]

    for i, text in enumerate(examples):
        print(f"\n--- Example {i + 1} ---")
        print(f"Input: {text}")

        # Sentiment Classification
        sentiment = classifier.classify(text)
        print("Sentiment Classification:", sentiment.model_dump())

        # Info Extraction
        extracted = extractor.extract(text)
        print("Info Extraction:", extracted.model_dump())


if __name__ == "__main__":
    main()