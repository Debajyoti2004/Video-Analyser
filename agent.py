import json
import asyncio
import cohere
from rich.console import Console
from rich.panel import Panel
from mcp import ClientSession
from mcp.client.sse import sse_client
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from state import AgentState
from tool_definations import get_tool_definitions
from intent_classifier import IntentClassifier
from graph_manager import KnowledgeGraphManager
from chroma_manager import ChromaMemoryManager
import config

PREAMBLE_OWNER = """# 👑 ROLE: AI Forensic & Intelligence Analyst (Codename: Kala-Sahayak)
You are an elite AI partner to a human operator, serving as an interactive forensics layer over a pre-analyzed video. Your mission is to use your advanced tools to extract critical, actionable intelligence from multimedia evidence with clinical precision.

# 📥 CONTEXT PROVIDED ON EACH TURN
You will always receive a complete operational context. You MUST use this context to inform every decision.
1.  **Initial Video Analysis (JSON)**: The primary evidence file with a high-level `event_log`. This is your source of truth for answering general questions and finding approximate timestamps.
2.  **User's Command**: The direct instruction from the owner.
3.  **Available Tools**: Your set of advanced actions. You MUST know what each tool does and when to use it.
4.  **Knowledge Graph Plan**: A suggestion for a plan if a similar task has been successfully completed before.
5.  **Recalled Memories**: Snippets from past conversations that might be relevant to the current request.

# 🧭 CORE WORKFLOW & DIRECTIVES
Your function is a cyclical reasoning process: **ANALYZE -> PLAN -> EXECUTE -> REPORT**.

1.  **⛓️ STRATEGIC PLANNING & EXECUTION (for `tool_use`):**
    You MUST chain tools together to fulfill complex requests.
    -   **Example Request**: "Show me a picture of the red car and where it is."
    -   **Your Internal Plan (Generated Step-by-Step):**
        1.  **Call `detect_and_track_object`** with `object_query="red car"`.
        2.  **Analyze the result**. Find a good timestamp and bounding box from the tracking data.
        3.  **Call `get_incident_snapshot`** using the chosen timestamp.
        4.  **Call `annotate_snapshot`** using the saved image path and the bounding box.
        5.  **Report the final annotated image path** and summarize the findings.

2.  **💡 FAILURE ANALYSIS & CORRECTION:** If a tool returns an error, analyze the failure, report it, and suggest a new course of action. Example: "⚠️ The query 'red sedan' returned no results. Suggestion: Broaden the search by trying `detect_and_track_object` with `object_query="car"`."

# 🛡️ GUARDRAILS & COMMUNICATION
-   **Clarification First**: If a command is ambiguous, ask for clarification.
-   **Tone**: Clinical, precise, strategic.
-   **Structure**: Use symbols: ✅, ⚠️, ❌, 💡, 📊, 🚀.
-   **Conclusion**: Always end your final response with a summary and suggest logical `Next Steps`.
"""

class VideoAnalysisAgent:
    def __init__(self, role: str = "owner", language="en-IN"):
        self.console = Console()
        self.cohere_client = cohere.Client(api_key=config.COHERE_API_KEY)
        self.knowledge_graph = KnowledgeGraphManager()
        self.chroma_manager = ChromaMemoryManager()
        self.role = role
        self.language = language
        self.preamble = PREAMBLE_OWNER
        self.active_tools = get_tool_definitions() if self.role == "owner" else []
        self.mcp_session: ClientSession = None
        self.mcp_sse_client = None
        self.intent_classifier = IntentClassifier()
        self.graph = self._build_graph()

    @classmethod
    async def create(cls, role: str = "owner", language="en-IN"):
        agent = cls(role, language)
        await agent._initialize_connection()
        return agent

    async def _initialize_connection(self):
        sse_url = f"{config.MCP_SERVER_URL}/sse"
        self.console.print(Panel(f"🔌 Connecting to MCP Tool Server via SSE at {sse_url}...", title="MCP Client"))
        try:
            self.mcp_sse_client = sse_client(sse_url)
            read_stream, write_stream = await self.mcp_sse_client.__aenter__()
            self.mcp_session = ClientSession(read_stream, write_stream)
            await self.mcp_session.initialize()
            self.console.print(Panel(f"✅ Successfully connected to: [bold green]{self.mcp_session.server_info.name}[/]", title="MCP Client"))
        except Exception as e:
            self.console.print(Panel(f"❌ Failed to connect to MCP Tool Server: {e}", title="[bold red]Connection Error[/]"))
            raise

    def _build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("load_memories", self.load_memories)
        builder.add_node("classify_intent", self.classify_intent)
        builder.add_node("find_strategic_plan", self.find_strategic_plan)
        builder.add_node("brain_adapt_plan", self.brain_adapt_plan)
        builder.add_node("brain_generate_plan", self.brain_generate_plan)
        builder.add_node("tool_node", self.tool_node)
        builder.add_node("general_conversation_node", self.general_conversation_node)
        builder.set_entry_point("load_memories")
        builder.add_edge("load_memories", "classify_intent")
        builder.add_conditional_edges("classify_intent", lambda state: "general_conversation_node" if state["intent"] == "general_conversation" else "find_strategic_plan")
        builder.add_conditional_edges("find_strategic_plan", lambda state: "brain_adapt_plan" if state.get("cached_plan") else "brain_generate_plan")
        def should_execute_tools(state):
            return "tool_node" if state["messages"][-1].tool_calls else END
        builder.add_conditional_edges("brain_adapt_plan", should_execute_tools)
        builder.add_conditional_edges("brain_generate_plan", should_execute_tools)
        builder.add_edge("tool_node", "brain_generate_plan")
        builder.add_edge("general_conversation_node", END)
        memory = MemorySaver()
        return builder.compile(checkpointer=memory)

    def _emit_status(self, message: str):
        self.console.print(f"[yellow]STATUS: {message}[/yellow]")

    def load_memories(self, state: AgentState):
        self._emit_status("🧠 Loading Memories...")
        user_query = state["user_command"]
        session_id = state["messages"][0].additional_kwargs.get("session_id", "default_session")
        allowed_speakers = ["owner", "agent"]
        recalled_memories = self.chroma_manager.retrieve_relevant_memories(
            query=user_query, session_id=session_id, allowed_speaker_types=allowed_speakers
        )
        self._emit_status(f"Recalled {len(recalled_memories)} similar memories for context.")
        return {"recalled_memories": recalled_memories or []}
    
    def classify_intent(self, state: AgentState):
        self._emit_status("🤔 Classifying user intent...")
        user_message = state["user_command"]
        if self.role != "owner" or not self.active_tools:
            self._emit_status("Intent classified as: [bold]general_conversation[/bold] (no tools available)")
            return {"intent": "general_conversation"}
        
        response = self.intent_classifier.classify(user_message)
        intent = response.intent.value
        self._emit_status(f"Intent classified as: [bold]{intent}[/bold]")
        return {"intent": intent}

    def find_strategic_plan(self, state: AgentState):
        if self.role != "owner": return {"cached_plan": None}
        self._emit_status("🔍 Searching Knowledge Graph for a successful plan...")
        cached_plan = self.knowledge_graph.find_successful_plan(state["user_command"])
        if cached_plan: self._emit_status("♻️ Found a relevant successful plan.")
        else: self._emit_status("No relevant successful plan found.")
        return {"cached_plan": cached_plan}

    def brain_adapt_plan(self, state: AgentState):
        self._emit_status("🧠⚡ Adapting cached plan...")
        message = f"Previous Plan: {json.dumps(state['cached_plan'], indent=2)}\nNew Request: \"{state['user_command']}\"\nUpdate the parameters of the previous plan to fit the new request. Output only the updated `tool_calls`."
        response = self.cohere_client.chat(message=message, model="command-r-plus", tools=[t.model_dump() for t in self.active_tools], preamble=self.preamble)
        return {"messages": state["messages"] + [AIMessage(content=response.text, tool_calls=response.tool_calls)]}

    def brain_generate_plan(self, state: AgentState):
        self._emit_status("💡 Generating new plan or continuing task...")
        contextual_message = f"**Initial Video Analysis**:\n```json\n{json.dumps(state['analysis_results'], indent=2)}\n```\n---\n**Recalled Memories**:\n{state['recalled_memories']}\n---\n**Current User Request**:\n{state['user_command']}"
        messages_for_llm = state["messages"][:-1] + [HumanMessage(content=contextual_message)]
        response = self.cohere_client.chat(model="command-r-plus", chat_history=[m.dict() for m in messages_for_llm], tools=[t.model_dump() for t in self.active_tools], preamble=self.preamble, message="")
        last_plan = [tc.dict() for tc in response.tool_calls] if response.tool_calls else None
        return {"messages": state["messages"] + [AIMessage(content=response.text, tool_calls=response.tool_calls)], "last_plan": last_plan}

    def general_conversation_node(self, state: AgentState):
        self._emit_status("💬 Handling general conversation...")
        response = self.cohere_client.chat(chat_history=[m.dict() for m in state["messages"]], preamble=self.preamble, message="")
        return {"messages": state["messages"] + [AIMessage(content=response.text)]}

    async def tool_node(self, state: AgentState):
        last_message = state["messages"][-1]
        tool_messages = []
        if not self.mcp_session: raise RuntimeError("MCP session not initialized.")
        for tool_call in last_message.tool_calls:
            self._emit_status(f"📡 Calling remote tool: {tool_call.name}...")
            try:
                result = await self.mcp_session.call(tool_call.name, **tool_call.parameters)
                tool_messages.append(ToolMessage(content=json.dumps(result), tool_call_id=tool_call.id))
                self._emit_status(f"✅ Success: {tool_call.name}")
            except Exception as e:
                error_message = f"Error calling {tool_call.name}: {e}"; self._emit_status(f"❌ {error_message}")
                tool_messages.append(ToolMessage(content=json.dumps({"error": error_message}), tool_call_id=tool_call.id))
        return {"messages": state["messages"] + tool_messages}

    async def get_agent_response(self, user_command: str, analysis_results: dict, config: RunnableConfig) -> str:
        session_id = config["configurable"]["thread_id"]
        self.chroma_manager.add_message("owner", user_command, session_id)
        
        initial_state = {
            "user_command": user_command,
            "analysis_results": analysis_results,
            "messages": [HumanMessage(content=user_command, additional_kwargs={"session_id": session_id})],
        }
        
        final_state = None
        async for chunk in self.graph.astream(initial_state, config=config):
            if END in chunk:
                final_state = chunk[END]
        
        if final_state:
            final_messages = final_state.get("messages", [])
            if final_messages:
                final_message = final_messages[-1]
                response_text = final_message.content
                self.chroma_manager.add_message("agent", response_text, session_id)
                if self.role == "owner" and final_state.get("last_plan"):
                    self.knowledge_graph.store_successful_plan(user_command, final_state["last_plan"])
                    self._emit_status("💾 Stored final successful plan in Knowledge Graph.")
                return response_text
        return "I'm sorry, I encountered an issue and couldn't generate a response."