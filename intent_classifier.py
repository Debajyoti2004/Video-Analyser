import json
import config
from enum import Enum
from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from rich.panel import Panel
from rich import print as rprint

class Intent(str, Enum):
    TOOL_USE = "tool_use"
    DIRECT_RESPONSE = "direct_response"

class IntentResponse(BaseModel):
    intent: Intent = Field(
        ...,
        description="The classified intent of the user's request."
    )

class IntentClassifier:
    def __init__(self, tool_definitions: List[dict]):
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            google_api_key=config.GOOGLE_API_KEY
        )
        structured_llm = llm.with_structured_output(IntentResponse)

        tool_summary_lines = ["This AI agent has access to the following forensic tools:"]
        for tool in tool_definitions:
            tool_summary_lines.append(f"- **{tool['name']}**: {tool['description']}")
        tool_summary = "\n".join(tool_summary_lines)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             f"""
**Role**: You are an expert AI assistant acting as a meticulous and logical router for a forensic video analysis agent. Your primary function is to classify the user's latest message to determine if it requires using a specialized tool or if it can be answered directly. Your decision-making must be precise and follow the rules provided.

**Persona**: Act as a logical gatekeeper. You are not conversational. Your only output is the final classification based on the logic below.

**Decision-Making Hierarchy & Logic**:
You must follow these steps in order. Stop at the first rule that applies.

1.  **Tool Confirmation Check**:
    *   **Condition**: The user's message is a direct, simple confirmation (e.g., "yes", "ok", "proceed", "sounds good", "do it") AND the AI's most recent message in the conversation history explicitly proposed a specific tool-based action (e.g., "Shall I extract the text?").
    *   **Action**: If both conditions are met, classify as `tool_use`.

2.  **New Information Extraction Check**:
    *   **Condition**: The user's message asks for information that CANNOT be found in the [AVAILABLE PRE-ANALYSIS DATA] and would require analyzing the video content itself. This involves tasks like reading text, identifying objects not listed, describing actions, or analyzing a specific frame or timestamp.
    *   **Action**: If the condition is met, classify as `tool_use`.

3.  **Direct Answer Check**:
    *   **Condition**: The user's message asks a question whose answer is explicitly present in the [AVAILABLE PRE-ANALYSIS DATA]. This includes questions about metadata (resolution, fps, duration) or scene timestamps.
    *   **Action**: If the condition is met, classify as `direct_response`.

4.  **Default to Direct Response**:
    *   **Condition**: For ANY other case, including greetings ("hello"), follow-up questions about the agent's capabilities ("what can you do?"), or general conversation.
    *   **Action**: Classify as `direct_response`.

**Response Schema**:
Before providing the final classification, think step-by-step to justify your decision based on the hierarchy.
1.  Analyze the user's latest message.
2.  Review the pre-analysis data and conversation history.
3.  Evaluate against the decision-making hierarchy, rule by rule.
4.  State which rule applies and why.
5.  Provide the final classification.

---
**Example Scenarios**:

*   **Scenario 1**:
    *   Pre-Analysis Data: Contains `"fps": 29.97`.
    *   User Message: "What is the frame rate?"
    *   Reasoning: The answer is explicitly in the pre-analysis data. Rule #3 applies.
    *   Classification: `direct_response`.

*   **Scenario 2**:
    *   Pre-Analysis Data: Does not contain information about vehicles.
    *   User Message: "Is there a blue car in the first scene?"
    *   Reasoning: This requires identifying an object not present in the pre-analysis data. Rule #2 applies.
    *   Classification: `tool_use`.

*   **Scenario 3**:
    *   AI's Last Message: "I can run OCR on frame 500 to find text. Should I proceed?"
    *   User Message: "Yes, please."
    *   Reasoning: The user is giving a simple confirmation to a proposed tool-based action. Rule #1 applies.
    *   Classification: `tool_use`.
---

[ AGENT CAPABILITIES AND TOOLS ]
{tool_summary}
"""),
            ("user",
             """
[ RECENT CONVERSATION HISTORY ]
{history}

[ AVAILABLE PRE-ANALYSIS DATA ]
```json
{analysis_results}
````

[ USER'S LATEST MESSAGE ]
"{user_message}"
""")
])

        self.chain = self.prompt.partial(tool_summary=tool_summary) | structured_llm

    def classify(self, user_message: str, analysis_results: dict, conversation_history: list[BaseMessage] = None) -> IntentResponse:
        analysis_str = json.dumps(analysis_results, indent=2)

        history_str = "No recent history."
        if conversation_history:
            recent_history = conversation_history[-3:]
            formatted_history = []
            for msg in recent_history:
                role = "User" if msg.type == "human" else "AI"
                if "[ 📂 INITIAL CONTEXT ]" not in msg.content:
                    formatted_history.append(f"{role}: {msg.content}")
            if formatted_history:
                history_str = "\n".join(formatted_history)

        return self.chain.invoke({
            "user_message": user_message,
            "analysis_results": analysis_str,
            "history": history_str
        })

if __name__ == '__main__':
    example_tool_definitions = [
    {
    "name": "get_video_metadata",
    "description": "Returns basic metadata about the video file, such as resolution, frame rate (fps), and duration."
    },
    {
    "name": "get_scene_changes",
    "description": "Detects and returns the timestamps of significant scene changes in the video."
    },
    {
    "name": "extract_text_from_frame",
    "description": "Uses Optical Character Recognition (OCR) to extract any text visible in a specific video frame."
    }
    ]
    example_analysis_results = {
        "metadata": {
            "resolution": "1920x1080",
            "fps": 29.97,
            "duration_seconds": 120
        },
        "scenes": [
            {"start_time": 0, "end_time": 15.5},
            {"start_time": 15.5, "end_time": 45.2},
            {"start_time": 45.2, "end_time": 120}
        ]
    }

    intent_classifier = IntentClassifier(tool_definitions=example_tool_definitions)

    user_demo_message_1 = "What is the frame rate of this video?"
    classification_result_1 = intent_classifier.classify(
        user_message=user_demo_message_1,
        analysis_results=example_analysis_results
    )

    output_text_1 = (
        f"[bold]User Message[/bold]: '{user_demo_message_1}'\n"
        f"[bold]Classified Intent[/bold]: [green]{classification_result_1.intent.value}[/green]"
    )
    rprint(Panel(output_text_1, title="[cyan]Classification Result 1[/cyan]", border_style="blue"))

    user_demo_message_2 = "Can you read the license plate at the 30-second mark?"
    classification_result_2 = intent_classifier.classify(
        user_message=user_demo_message_2,
        analysis_results=example_analysis_results
    )

    output_text_2 = (
        f"[bold]User Message[/bold]: '{user_demo_message_2}'\n"
        f"[bold]Classified Intent[/bold]: [yellow]{classification_result_2.intent.value}[/yellow]"
    )
    rprint(Panel(output_text_2, title="[cyan]Classification Result 2[/cyan]", border_style="blue"))