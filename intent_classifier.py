from enum import Enum
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import config

class Intent(str, Enum):
    TOOL_USE = "tool_use"
    GENERAL_CONVERSATION = "general_conversation"

class IntentResponse(BaseModel):
    intent: Intent = Field(description="The final classification of the user's intent.")

class IntentClassifier:
    def __init__(self, model_name: str = "gemini-1.5-flash-latest"):
        self.model = ChatGoogleGenerativeAI(model=model_name, google_api_key=config.GOOGLE_API_KEY)
        self.structured_llm = self.model.with_structured_output(IntentResponse)
        self.prompt = ChatPromptTemplate.from_template(
            """
            **Role**: You are an expert at classifying a user's intent based on their message in the context of a video analysis session.

            **Context**: The user is interacting with an AI that has access to a set of forensic tools to analyze a video. Your job is to determine if the user is asking the AI to use one of these tools, or if they are just having a general conversation.

            **Classification Options**:
            1.  `tool_use`: Select this if the user is asking a question that requires running a tool on the video or its analysis. This includes asking for specific details, creating new files, or performing actions.
                -   Examples: "Show me a picture of the car at 15 seconds.", "What does the sign say?", "Who was speaking at the beginning?", "Find all the people in this shot.", "Create a PDF report."

            2.  `general_conversation`: Select this if the user is making a general statement, greeting, or asking a question that doesn't require analyzing the video file.
                -   Examples: "Hello, how are you?", "Thank you, that was helpful.", "What can you do?", "That's interesting."

            **Task**: Based on the user's message below, classify the intent into one of the available options.

            **User Message**:
            "{user_message}"
            """
        )
        self.chain = self.prompt | self.structured_llm

    def classify(self, user_message: str) -> IntentResponse:
        return self.chain.invoke({"user_message": user_message})