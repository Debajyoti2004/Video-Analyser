from typing import TypedDict, List, Optional, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    user_command: str
    analysis_results: dict
    video_path: str
    messages: Annotated[List[BaseMessage], add_messages]
    recalled_memories: List[str]
    intent: str
    cached_plan: Optional[List[dict]]
    last_plan: Optional[List[dict]]
    image_path: Optional[str]
    structured_data: Optional[dict]