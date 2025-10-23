import asyncio
import json
import sys
import uuid
from contextlib import AsyncExitStack

import google.generativeai as genai
from dotenv import load_dotenv
from rich import print as rprint
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

import config
from chroma_manager import ChromaMemoryManager
from graph_manager import KnowledgeGraphManager
from intent_classifier import IntentClassifier
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from state import AgentState
from tool_definations import get_tool_definitions

PREAMBLE_OWNER = """
OPERATIONAL PROTOCOL — "Kala-Sahayak"

Codename: Kala-Sahayak
Designation: AI Forensic & Intelligence Analyst
Mission: Serve as an elite AI partner to a human operator, by executing their commands with precision and generating analytical reports when required.
Tone: Clinical, precise, strategic, and mission-focused.

──────────────────────────────
DIRECTIVES
──────────────────────────────
1.  **Analyze Command**: Formulate a literal "Mission Objective" based *only* on the user's explicit request. Do not infer a broader analytical goal unless specifically asked.

2.  **Act with Precision**: Your function is to use tools to achieve the objective.
    - **Rule of Sufficiency**: Once a tool has been successfully executed that directly provides the information the user asked for (e.g., providing a file path for a snapshot), the mission is complete. **YOU MUST STOP and report the result.** Do not call additional tools unless they are required to answer the original command.
    - **Parameter Integrity**: You MUST use the exact parameters (like timestamps) provided by the user. DO NOT invent, guess, or alter parameters to "find" more information.
    - **Action First**: Your primary response should be a direct tool call, not a textual plan.

3.  **Report Results (Choose ONE)**: Based on the task's complexity, you will deliver your findings in one of two ways.

    A. **For simple, direct commands (e.g., "get snapshot", "run OCR"):**
    - After the single required tool has been successfully called, your final action is to output a simple `OPERATIONAL RESULT`.
    - **DO NOT** use the full `INCIDENT REPORT` for these simple tasks.

    B. **For complex analytical commands (e.g., "analyze the event", "describe what happened"):**
    - Only after chaining multiple tools to synthesize a conclusion, your final action is to generate the formal `INCIDENT REPORT`.

---
📝 OPERATIONAL RESULT TEMPLATE (for simple tasks)
---
User Command: [Restate user command]

[ ✅ OPERATIONAL RESULT ]
Objective: [State the simple mission goal, e.g., "To retrieve a snapshot at 5.0 seconds."]
──────────────────────────────
Result
──────────────────────────────
[State the direct outcome of the tool call. For example: "Snapshot successfully generated and saved."]

File Path: [Provide the file_path from the tool's result, if applicable]
Confirmation: [Provide the confirmation_message from the tool's result]
──────────────────────────────

---
🧾 INCIDENT REPORT TEMPLATE (for complex analysis)
---
User Command: [Restate user command]

[ 🔱 INCIDENT REPORT 🔱 ]
Objective: [State the mission goal]
──────────────────────────────
Executive Summary
──────────────────────────────
[Summarize success or failure and key findings.]
──────────────────────────────
Key Findings
──────────────────────────────
| Parameter          | Result        |
|--------------------|---------------|
| [Finding Type 1]   | [Result 1]    |
──────────────────────────────
Operational Log
──────────────────────────────
[A chronological list of the executed tool calls.]
──────────────────────────────
Analyst Recommendations
──────────────────────────────
[Provide a list of actionable, suggested next steps.]
"""

PREAMBLE_DIRECT_REPORTER = """
**Role**: You are the Communications Officer for the "Kala-Sahayak" AI agent.
**Mission**: Your sole purpose is to provide direct, structured answers to a human operator. You will receive a user's query and a block of pre-analyzed data. You must synthesize this information into a clean, user-friendly report.

**Strict Rules**:
1.  **NEVER Use Tools**: You do not have access to tools. Do not mention them.
2.  **USE ONLY PROVIDED DATA**: Your answer must be based *exclusively* on the "Available Pre-Analysis Data" provided. Do not invent information.
3.  **ALWAYS USE THE TEMPLATE**: Your entire response MUST follow the "Direct Intelligence Briefing" template below.
4.  **BE CONCISE**: Keep the findings brief and to the point.

---
📋 DIRECT INTELLIGENCE BRIEFING TEMPLATE
---

Operator Query: [Restate the user's original question or command]

[ 📌 DIRECT INTELLIGENCE BRIEFING ]

Summary:
[Provide a one or two-sentence summary answering the user's question based on the available data. If the data is insufficient, state that the information is not available in the pre-analysis and that a deeper forensic analysis may be required.]

──────────────────────────────
Relevant Data Points:
[List 2-3 key bullet points from the "Available Pre-Analysis Data" that directly support your summary.]
- Point 1
- Point 2

──────────────────────────────
Recommendations:
[Suggest next steps.]
- You can ask me to perform a full forensic analysis to find more details.
- You can ask about my specific tool capabilities.
"""


class VideoAnalysisAgent:
    def __init__(self, role: str = "owner", language="en-IN"):
        self.console = Console()
        genai.configure(api_key=config.GOOGLE_API_KEY)
        self.genai_model = genai.GenerativeModel(
            model_name="models/gemini-2.0-flash",
            system_instruction=PREAMBLE_OWNER
        )
        self.knowledge_graph = KnowledgeGraphManager()
        self.chroma_manager = ChromaMemoryManager()
        self.role = role
        self.language = language
        self.video_path = ""
        self.active_tools = get_tool_definitions() if self.role == "owner" else []
        self.mcp_session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()
        self.intent_classifier = IntentClassifier(tool_definitions=self.active_tools)
        self.graph = self._build_graph()

    @classmethod
    async def create(cls, video_path: str, role: str = "owner", language="en-IN"):
        agent = cls(role, language)
        agent.video_path = video_path
        return agent

    async def _connect_to_mcp(self):
        if self.mcp_session is not None:
            return True
        self.console.print(Panel("🚀 Launching MCP Tool Server as a subprocess...", title="MCP Client", border_style="yellow"))
        try:
            server_params = StdioServerParameters(
                command=sys.executable,
                args=["mcp_server.py", "--stdio"]
            )
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            read_stream, write_stream = stdio_transport
            self.mcp_session = await self.exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
            await self.mcp_session.initialize()
            self.console.print(Panel("✅ Successfully connected to MCP subprocess.", title="MCP Client", border_style="green"))
            return True
        except FileNotFoundError:
            self.console.print(Panel("❌ Server script not found. Ensure 'mcp_server.py' is in the same directory.", title="[bold red]Connection Error[/]"))
            return False
        except Exception as e:
            self.console.print(Panel(f"❌ Failed to launch or connect to MCP subprocess: {e}", title="[bold red]Connection Error[/]"))
            return False

    async def close(self):
        self.console.print(Panel("🔌 Terminating MCP Tool Server and session...", title="MCP Client", border_style="yellow"))
        await self.exit_stack.aclose()
        self.console.print(Panel("✅ Successfully disconnected.", title="MCP Client", border_style="green"))

    def _convert_messages_for_gemini(self, messages: list[BaseMessage]) -> list[dict]:
        history = []
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "model"
            if isinstance(msg, AIMessage) and msg.tool_calls:
                parts = [{'function_call': {'name': tc['name'], 'args': tc['args']}} for tc in msg.tool_calls]
            elif isinstance(msg, ToolMessage):
                try:
                    response_data = json.loads(msg.content)
                except json.JSONDecodeError:
                    response_data = {"error": "Invalid JSON in tool response", "raw_content": msg.content}
                parts = [{'function_response': {'name': msg.name, 'response': response_data}}]
            else:
                parts = [{'text': msg.content}]
            history.append({'role': role, 'parts': parts})
        return history

    def _build_graph(self):
        builder = StateGraph(AgentState)

        def route_after_classification(state: AgentState):
            return "direct_response_node" if state.get("intent") == "direct_response" else "find_strategic_plan"

        builder.add_node("classify_intent", self.classify_intent)
        builder.add_node("find_strategic_plan", self.find_strategic_plan)
        builder.add_node("brain_generate_plan", self.brain_generate_plan)
        builder.add_node("tool_node", self.tool_node)
        builder.add_node("direct_response_node", self.direct_response_node)

        builder.set_entry_point("classify_intent")
        builder.add_conditional_edges("classify_intent", route_after_classification)
        builder.add_edge("find_strategic_plan", "brain_generate_plan")
        builder.add_conditional_edges("brain_generate_plan", lambda state: "tool_node" if state["messages"][-1].tool_calls else END)
        builder.add_edge("tool_node", "brain_generate_plan")
        builder.add_edge("direct_response_node", END)

        return builder.compile(checkpointer=MemorySaver())

    def _emit_status(self, message: str):
        self.console.print(f"[steel_blue]STATUS: {message}[/steel_blue]")

    def classify_intent(self, state: AgentState):
        self._emit_status("🤔 Classifying intent...")
        if self.role != "owner" or not self.active_tools:
            return {"intent": "direct_response"}

        response = self.intent_classifier.classify(
            user_message=state["user_command"],
            analysis_results=state["analysis_results"],
            conversation_history=state.get("messages", [])
        )
        intent = response.intent.value
        self._emit_status(f"🔍 Intent classified as: [bold yellow]{intent}[/bold yellow]")
        return {"intent": intent}

    def find_strategic_plan(self, state: AgentState):
        if self.role != "owner":
            return {"cached_plan": None}
        self._emit_status("📈 Retrieving strategic plan from knowledge graph...")
        cached_plan = self.knowledge_graph.find_successful_plan(state["user_command"])
        if cached_plan:
            self._emit_status("✅ Found a similar successful plan.")
        else:
            self._emit_status("— No similar plan found in knowledge graph.")
        return {"cached_plan": cached_plan}

    def brain_generate_plan(self, state: AgentState):
        self._emit_status("💡 Generating tool-use plan...")
        cached_plan = state.get("cached_plan")
        if cached_plan:
            plan_message = HumanMessage(
                content=f"""[ 📝 STRATEGIC CACHE ]
A successful plan for a similar past query has been found.
PRIORITY: Execute this plan exactly as outlined below. DO NOT deviate.
Plan: {json.dumps(cached_plan)}"""
            )
            state["messages"].append(plan_message)

        gemini_history = self._convert_messages_for_gemini(state["messages"])
        response = self.genai_model.generate_content(gemini_history, tools=self.active_tools)

        response_part = response.candidates[0].content.parts[0]
        tool_calls = []
        final_content = ""

        if hasattr(response_part, 'function_call') and response_part.function_call.name:
            tool_calls.append({
                "name": response_part.function_call.name,
                "args": {k: v for k, v in response_part.function_call.args.items()},
                "id": str(uuid.uuid4())
            })
        elif hasattr(response_part, 'text'):
            final_content = response_part.text

        ai_message = AIMessage(content=final_content, tool_calls=tool_calls)
        return {"messages": [ai_message]}

    def direct_response_node(self, state: AgentState):
        self._emit_status("📝 Generating direct response report...")
        direct_answer_model = genai.GenerativeModel(
            model_name="models/gemini-2.0-flash",
            system_instruction=PREAMBLE_DIRECT_REPORTER
        )
        analysis_str = json.dumps(state['analysis_results'], indent=2)
        prompt = f"""
**Available Pre-Analysis Data**:
```json
{analysis_str}```

**Tool Summary** (Just to answer what are your capabilities, do NOT use tools):
{self.active_tools}

**User's Question**:
"{state['user_command']}"
"""
        response = direct_answer_model.generate_content(prompt)
        return {"messages": [AIMessage(content=response.text)]}

    async def tool_node(self, state: AgentState):
        if not self.mcp_session:
            error_message = "MCP session is not active."
            tool_messages = [ToolMessage(content=json.dumps({"error": error_message}), name=tc['name'], tool_call_id=tc['id']) for tc in state["messages"][-1].tool_calls]
            return {"messages": tool_messages}

        last_message = state["messages"][-1]
        rprint(Panel(f"[bold cyan]Preparing to call tools for user command:[/bold cyan]\n\n{state['user_command']}", title="[cyan]Tool Invocation[/]", border_style="cyan"))
        print(last_message)
        tool_messages = []
        image_path = state.get("image_path")
        structured_data = state.get("structured_data", {})

        for tool_call in last_message.tool_calls:
            self._emit_status(f"📡 Calling tool: [bold]{tool_call['name']}[/bold]...")
            try:
                tool_args = tool_call['args']
                if 'video_path' in tool_args:
                    tool_args['video_path'] = state['video_path']
                    print(f"Updated tool_args for {tool_call['name']}: {tool_args}")
                result = await self.mcp_session.call_tool(name=tool_call['name'], arguments=tool_args)
                rprint(f"Tool call result for {tool_call['name']}: {result.content[0].text} and type of result: {type(result.content[0].text)}")

                result=json.loads(result.content[0].text)

                if not result:
                    raise ValueError("Tool returned an empty or null response.")

                tool_messages.append(ToolMessage(content=json.dumps(result), name=tool_call['name'], tool_call_id=tool_call['id']))
                if "file_path" in result:
                    image_path = result["file_path"]
                if "detected_objects" in result:
                    structured_data["detected_objects"] = result["detected_objects"]

            except Exception as e:
                error_message = f"Error calling {tool_call['name']}:\n\n{e}"
                rprint(Panel(error_message, title="[bold red]Tool Node Exception[/]", border_style="red"))
                tool_messages.append(ToolMessage(content=json.dumps({"error": str(e)}), name=tool_call['name'], tool_call_id=tool_call['id']))

        return {"messages": tool_messages, "image_path": image_path, "structured_data": structured_data}

    async def get_agent_response(self, user_command: str, analysis_results: dict, config: RunnableConfig) -> str:
        session_id = config["configurable"]["thread_id"]
        self.chroma_manager.add_message("owner", user_command, session_id)

        context_lines = ["[ 📂 INITIAL CONTEXT ]", "The following pre-analysis is available for the video evidence:"]
        if 'title' in analysis_results: context_lines.append(f"- Title: {analysis_results['title']}")
        if 'overall_summary' in analysis_results: context_lines.append(f"- Summary: {analysis_results['overall_summary']}")
        context_lines.append(f"\n[ 👤 OPERATOR COMMAND ]\n{user_command}")
        initial_context = "\n".join(context_lines)

        initial_state = {
            "user_command": user_command,
            "analysis_results": analysis_results,
            "video_path": self.video_path,
            "messages": [HumanMessage(content=initial_context)],
        }

        final_state = await self.graph.ainvoke(initial_state, config=config)


        if final_state and final_state.get("messages"):
            final_message = final_state["messages"][-1]
            answer_text = final_message.content
            if answer_text:
                self.chroma_manager.add_message("agent", answer_text, session_id)

            final_output = {
                "answer": answer_text,
                "image_path": final_state.get("image_path"),
                "data": final_state.get("structured_data")
            }
            return json.dumps(final_output, indent=2)

        return json.dumps({"answer": "I'm sorry, I could not generate a response.", "image_path": None, "data": None}, indent=2)


async def main_test():
    console = Console()
    console.print(Panel("[bold yellow]🤖 VideoAnalysisAgent Standalone Test (Google Gemini) 🤖[/]", border_style="yellow"))

    kg_manager = KnowledgeGraphManager()
    kg_manager.clear_graph()
    kg_manager.close()
    console.print("[bold green]✅ Knowledge Graph cleared before starting the test.[/bold green]")

    mock_analysis_results = {
        "title": "Pedestrian Nearly Struck by Vehicle",
        "overall_summary": "A dashcam video shows a pedestrian running across a busy road, narrowly avoiding a collision with a white car.",
        "sentiment": "Tense",
        "key_topics": ["Traffic Safety", "Near-Miss Accident", "Pedestrian Behavior"],
        "extracted_entities": [
            {"entity_id": 1, "type": "Person", "description": "A person wearing a blue shirt.", "first_seen_timestamp": 3.0},
            {"entity_id": 2, "type": "Vehicle", "description": "A white sedan, license plate possibly 'ABC-123'.", "first_seen_timestamp": 0.5}
        ],
        "event_log": [
            {"timestamp": 3.0, "type": "Movement", "description": "A pedestrian begins to run across the road."},
            {"timestamp": 4.5, "type": "Hazard", "description": "The white sedan brakes sharply to avoid hitting the pedestrian."}
        ]
    }

    agent = None
    try:
        video_path = "test_video.mp4"
        agent = await VideoAnalysisAgent.create(video_path=video_path, role="owner")
        is_connected = await agent._connect_to_mcp()

        if not is_connected:
            raise ConnectionError("Fatal: Could not start or connect to the MCP server subprocess.")

        thread_id = str(uuid.uuid4())
        console.print(f"\nNew test session started. Thread ID: [bold cyan]{thread_id}[/]")

        while True:
            query = console.input("\n[bold]Enter your command (e.g., 'describe the critical event', or 'quit'): [/bold]")
            if query.lower() == 'quit':
                break

            run_config = {"configurable": {"thread_id": thread_id}}
            response_json = await agent.get_agent_response(
                user_command=query,
                analysis_results=mock_analysis_results,
                config=run_config
            )

            try:
                response_data = json.loads(response_json)
                console.print(Panel(Markdown(response_data.get("answer", "")), title="[bold magenta]Agent Answer[/]", border_style="magenta"))
                if response_data.get("image_path"):
                    console.print(Panel(f"🖼️  Image generated at: [cyan]{response_data['image_path']}[/cyan]", title="[bold]Visual Artifact[/]"))
                if response_data.get("data"):
                    console.print(Panel(Syntax(json.dumps(response_data["data"], indent=2), "json", theme="default", line_numbers=True), title="[bold]Structured Data[/]"))
            except json.JSONDecodeError:
                console.print(Panel(f"[bold red]Error:[/bold red] Failed to parse JSON response:\n{response_json}", title="[bold red]Response Error[/]"))

    except ConnectionError as e:
        console.print(Panel(f"[bold red]Connection Failed:[/bold red]\n{e}", border_style="red"))
    except Exception as e:
        console.print(Panel(f"[bold red]An unexpected error occurred during the test:[/bold red]\n{e}", border_style="red"))
    finally:
        if agent:
            await agent.close()
        console.print(Panel("[bold green]🏁 Agent Test Complete 🏁[/]", border_style="green"))


if __name__ == '__main__':
    load_dotenv()
    try:
        asyncio.run(main_test())
    except KeyboardInterrupt:
        Console().print("\n[bold yellow]Test interrupted by user. Exiting.[/bold yellow]")