from typing import TypedDict, List, Optional
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    user_command: str
    analysis_results: dict
    messages: List[BaseMessage]
    cached_plan: Optional[List[dict]]
    last_plan: Optional[List[dict]]
    recalled_memories: List[str]
    intent: str