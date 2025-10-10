import os
from typing import Optional, Literal, AsyncGenerator
from src.agent.memory import MemoryManager, MessageType
from src.agent.memory_filter import MessageFilter
from src.agent.utils import set_memory_manager
from src.agent.utils import get_tool_list, set_db_mcp
from langgraph.prebuilt import ToolNode
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from src.agent.state import IFractalState
from src.rag_service.rag import RAGService
from src.mcp.db_mcp import get_db_mcp

class CodingAgent:
    def __init__(self, llm: str, api_key: Optional[str] = None, verbose: bool = False, rag_service: RAGService | None = None, enable_db_tools: bool = False) -> None:
        self.llm_provider = llm
        self.verbose = verbose
        self.client = None
        self.model_name = None
        self.enable_db_tools = enable_db_tools

        ##############################################################
        self.db_mcp = None
        if enable_db_tools:
            try:
                self.db_mcp = get_db_mcp()
                set_db_mcp(self.db_mcp)
                if self.verbose:
                    print("Database MCP initialized")
            except ImportError:
                print("Warning: FastMCP not installed. Database tools disabled.")
                print("Install with: pip install fastmcp asyncpg aiomysql motor")
                self.enable_db_tools = False
        ##############################################################

        ##############################################################
        # binding with tools
        self.tools = get_tool_list(include_db_tools=enable_db_tools) # if we need db tools only then get them else no
        self.rag_service = rag_service 
        ##############################################################
        self.graph = None
        ##############################################################
        # memory manager
        self.memory_manager = MemoryManager(max_memory_size=100_000)
        self.message_filter = MessageFilter(self.memory_manager)

        set_memory_manager(self.memory_manager)
        ##############################################################
        # state
        self.state: Optional[IFractalState] = IFractalState(
            messages=[],
            memory_manager=self.memory_manager,
            message_filter=self.message_filter
        )
        ##############################################################
        # initialize llm
        ###############################################################
        self._initialize_llm(llm, api_key)        
        self._build_graph()
        ###############################################################


    def _initialize_llm(self, llm: str, api_key: Optional[str] = None):    
        """Initialize the LLM client with tool binding"""
        if llm == "openai":
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            self.client = init_chat_model("gpt-4o", model_provider=llm, temperature=0.8, api_key=api_key)
            self.model_name = "gpt-4o"
                
        elif llm == "claude":
            api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.")
            self.client = init_chat_model("claude-opus-20240229", model_provider="anthropic", temperature=0.8, api_key=api_key)
            self.model_name = "claude-opus-20240229"
            
        elif llm == "gemini":
            api_key = api_key or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("Gemini API key not found. Set GEMINI_API_KEY environment variable.")
            self.client = init_chat_model("gemini-2.0-flash", model_provider="google_genai", temperature=0.8, api_key=api_key)
            self.model_name = "gemini-2.0-flash"
        else:
            raise ValueError(f"Unsupported LLM provider: {llm}")

        ####################################################################################################################
         # binding with tools
        self.client_with_tools = self.client.bind_tools(self.tools)
        if self.verbose:
            db_status = f" (including {len([t for t in self.tools if 'postgres' in t.name or 'mysql' in t.name or 'mongodb' in t.name])} DB tools)" if self.enable_db_tools else ""
            print(f"Initialized {self.model_name} with {len(self.tools)} tools{db_status}")
        ####################################################################################################################

    def _build_graph(self):
        """Build the LangGraph agent workflow"""

        async def agent_node(state: IFractalState) -> IFractalState:
            """Node that calls the LLM with tools"""

            ##############################################################
            ##############################################################
            # # here we aet memory summary for context
            memory_context = ""
            if state.memory_manager:
                recent_memories = state.memory_manager.get_recent_important(limit=3)
                if recent_memories:
                    memory_context = "\n\nRecent important context:\n"
                    for entry in recent_memories:
                        memory_context += f"- {entry.content[:150]}...\n"
            ##############################################################
            ##############################################################

            db_tools_info = ""
            if self.enable_db_tools:
                db_tools_info = """
                DATABASE TOOLS:
                Connection Management:
                - connect_postgres, connect_mysql, connect_mongodb: Establish database connections
                - list_connections: View all active connections
                - disconnect_database: Close a connection

                PostgreSQL:
                - query_postgres: Execute SELECT queries
                - execute_postgres: Execute INSERT/UPDATE/DELETE queries

                MySQL:
                - query_mysql: Execute SELECT queries
                - execute_mysql: Execute INSERT/UPDATE/DELETE queries

                MongoDB:
                - query_mongodb: Query documents
                - insert_mongodb: Insert documents
                - update_mongodb: Update documents
                - delete_mongodb: Delete documents

                Always connect to the database first before executing queries!
                """
            
            # --- Start of Changed Section ---
            system_message = SystemMessage(content=f"""
            You are **Fractal**, an expert autonomous coding assistant with access to:
            - File system tools
            - RAG-powered codebase search
            - Database management tools

            You operate in **Developer Mode**, meaning:
            → You must *explain your reasoning before each major step or tool use*,
            → But still keep your output structured, efficient, and concise.

            Your mission: **Plan, execute, and complete technical tasks** (coding, database, or file ops) *end-to-end*, using the right tools at the right time.

            ---

            ### TOOL SELECTION STRATEGY

            You have access to multiple categories of tools.
            Use them intelligently, not mechanically.

            #### File System Tools
            - `read_file_tool`, `write_file_tool`, `edit_file_tool`: For creating or modifying code or data files.
            - `create_directory_tool`, `delete_file_tool`, `read_directory_tool`: For project scaffolding or navigation.
            - `search_files_tool`: Fast regex-based search across files.

            > *Explain why you're creating each directory or file before using these tools.*

            #### RAG + Code Understanding
            - `search_codebase_tool`: Semantic code search (slow). Use only for large or complex lookups spanning multiple files.
            - `search_memory_tool`: Retrieve context from previous sessions or chat history.

            > *If you use these tools, explain what you’re searching for and why.*

            #### Database Tools
            If `enable_db_tools` is true:

            {db_tools_info}

            > *Always connect first, run query or execution, then disconnect.*

            ---

            ### WORKFLOW RULES (Developer Mode)

            1.  **Understand the request**
                - Restate the user's intent in your own words.
                - Identify whether it's a *code creation*, *file editing*, *data query*, or *debug* task.

            2.  **Plan before action**
                - Outline a short 3-6 step plan.
                - Explain *which tools* you'll use and *why*.
                - Example: “I'll use `create_directory_tool` to scaffold the FastAPI project and `write_file_tool` to generate the main app.”

            3.  **Execute with transparency**
                - Before each tool call, say something like:
                > “Using `write_file_tool` to create main.py — this will define the FastAPI app and root endpoint.”
                - Then perform the actual tool call.

            4.  **Verify and summarize**
                - After completing the task, summarize in ≤4 lines:
                - What files or DB operations were created or changed
                - Whether the task is ready to run or needs user input (e.g., DB credentials)

            5.  **Tool usage guardrails**
                - Never call tools in loops unless necessary.
                - Avoid redundant tool calls.
                - Do **not** create TODO lists unless the user explicitly asks for one.
                - Do **not** create a TODO *inside* another TODO list.

            6.  **Code quality rules**
                - Generate *complete, functional code* (no `...` placeholders or `# TODO`).
                - Include imports, main app setup, and runnable entry points.
                - Prefer modular organization (e.g., `models/`, `routers/`, `auth/`, etc.) for web apps.

            7.  **Database operations**
                - Connect first, execute, and disconnect.
                - If you query data, summarize results briefly.
                - If executing mutations (INSERT/UPDATE/DELETE), confirm affected records.

            8.  **Error handling & verbosity**
                - Be concise: explain reasoning briefly but clearly.
                - Never over-justify or produce long essays.
                - Balance between *insightful reasoning* and *clean output*.

            ---

            ### 💡 EXAMPLES

            #### Example 1: "Create a FastAPI user management app with authentication"
            ✅ You should:
            - Explain you'll scaffold a full FastAPI project
            - Create directories (`models`, `schemas`, `routers`, `auth`)
            - Generate main app, models, and JWT-based auth routes
            - Summarize with “Project scaffolded and ready to run via `uvicorn main:app --reload`”

            #### Example 2: "Query the users table in Postgres"
            ✅ You should:
            - Explain that you're connecting to Postgres
            - Run `query_postgres("SELECT * FROM users LIMIT 5;")`
            - Show formatted output
            - Disconnect cleanly

            ---

            ### 🧠 MEMORY + RAG CONTEXT
            {memory_context}

            ---

            ### 🧩 SUMMARY
            You are a disciplined, expert coding agent with full autonomy.
            You:
            - Plan before acting
            - Explain your reasoning clearly
            - Use tools efficiently
            - Write complete, runnable code
            - Never spam, repeat, or half-finish tasks

            Follow these principles strictly. Prioritize correctness, completeness, and transparency.
            """)
            
            # Filter messages for LLM state (remove tool results stored in memory)
            messages = state.messages
            if len(messages) > 20:
                messages = messages[-20:]
            full_messages = [system_message] + messages
            
            if self.verbose:
                print(f"\nCalling {self.model_name}...")
            
            response = await self.client_with_tools.ainvoke(full_messages)
            
            if self.verbose and response.tool_calls: #type:ignore
                print(f"Tool calls requested: {[tc['name'] for tc in response.tool_calls]}") #type:ignore
            
            if state.memory_manager and response.content:
                state.memory_manager.add_entry(
                    MessageType.AI_RESPONSE,
                    str(response.content),
                    importance=0.7
                )
            
            return {"messages": [response]} #type:ignore
        
        def should_continue(state: IFractalState) -> Literal["tools", "end"]:
            """Determine if we should continue to tools or end"""
            messages = state.messages
            last_message = messages[-1]
            
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls: #type:ignore
                return "tools"
            return "end"

        async def tools_with_memory(state: IFractalState) -> IFractalState:
            """Execute tools and process results through memory filter"""
            tool_node = ToolNode(self.tools)
            result = await tool_node.ainvoke(state)
            
            if state.message_filter and "messages" in result:
                processed_messages = []
                for msg in result["messages"]:
                    if isinstance(msg, ToolMessage):
                        tool_name = getattr(msg, 'name', 'unknown_tool')
                        
                        filter_result = state.message_filter.process_tool_result(
                            tool_name, 
                            str(msg.content)
                        )
                        
                        if filter_result.get("stored_in_memory"):
                            summary_msg = ToolMessage(
                                content=filter_result["summary"],
                                tool_call_id=msg.tool_call_id,
                                name=tool_name
                            )
                            processed_messages.append(summary_msg)
                        else:
                            processed_messages.append(msg)
                    else:
                        processed_messages.append(msg)
                
                result["messages"] = processed_messages
            
            return result
        
# ###########################################################################################################################################################
# ###########################################################################################################################################################
# # BUILD GRAPH HERE
# ###########################################################################################################################################################
# ###########################################################################################################################################################        

        graph_builder = StateGraph(IFractalState)
        graph_builder.add_node("agent", agent_node)
        graph_builder.add_node("tools", tools_with_memory)
        graph_builder.add_edge(START, "agent")
        graph_builder.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                "end": END
            }
        )
        graph_builder.add_edge("tools", "agent")
        
        self.graph = graph_builder.compile()
    
        if self.verbose:
            print("Agent graph compiled successfully")

###########################################################################################################################################################
# ###########################################################################################################################################################
# # GRAPH COMPILED
# ###########################################################################################################################################################
# ###########################################################################################################################################################

    async def ainvoke(self, user_input: str) -> str:
        """Async version of invoke"""
        if not self.graph:
            raise RuntimeError("Agent graph not initialized")

        if self.state is None:
            raise RuntimeError("State for graph is not initialized")

        # Add user message to both state and memory
        user_msg = HumanMessage(content=user_input)
        self.state.messages.append(user_msg)
        
        if self.state.memory_manager:
            self.state.memory_manager.add_entry(
                MessageType.USER_QUERY,
                user_input,
                importance=0.9
            )
        
        final_state = await self.graph.ainvoke(self.state, {"recursion_limit": 100})
        final_message = final_state["messages"][-1]
        
        if isinstance(final_message, AIMessage):
            response = final_message.content
        else:
            response = str(final_message)
        
        if isinstance(response, list):
            response = "\n".join(str(item) for item in response)

        return response
    
    async def astream(self, user_input: str) -> AsyncGenerator[dict, None]:
        """Stream the agent's response with reasoning display"""
        if not self.graph:
            raise RuntimeError("Agent graph not initialized")
        if self.state is None:
            raise RuntimeError("State is not initialized")
        
        # Add to memory
        if self.state.memory_manager:
            self.state.memory_manager.add_entry(
                MessageType.USER_QUERY,
                user_input,
                importance=0.9
            )
        
        self.state.messages.append(HumanMessage(content=user_input))
        is_final_response = False
        tool_calls_made = set()  # Track tool calls to avoid duplicates
        
        async for event in self.graph.astream(self.state, stream_mode="updates"):
            for node_name, node_state in event.items():
                if "messages" in node_state:
                    messages = node_state["messages"]
                    
                    for message in messages:
                        # Handle tool calls from agent node
                        if isinstance(message, AIMessage) and hasattr(message, 'tool_calls') and message.tool_calls:
                            is_final_response = False
                            for tool_call in message.tool_calls:
                                tool_call_id = tool_call.get("id", "")
                                # Only yield if we haven't seen this tool call yet
                                if tool_call_id not in tool_calls_made:
                                    tool_calls_made.add(tool_call_id)
                                    yield {
                                        "type": "tool_call",
                                        "content": tool_call.get("name", "unknown"),
                                        "metadata": {
                                            "args": tool_call.get("args", {}),
                                            "id": tool_call_id
                                        }
                                    }
                        
                        # Handle tool results from tools node
                        elif isinstance(message, ToolMessage):
                            yield {
                                "type": "tool_result",
                                "content": str(message.content),
                                "metadata": {
                                    "tool_name": getattr(message, 'name', 'unknown'),
                                    "tool_call_id": getattr(message, 'tool_call_id', '')
                                }
                            }
                        
                        # Handle final AI response (no tool calls)
                        elif isinstance(message, AIMessage) and message.content:
                            has_tool_calls = hasattr(message, 'tool_calls') and message.tool_calls
                            
                            # Only treat as final response if from agent node and no tool calls
                            if not has_tool_calls and node_name == "agent":
                                if not is_final_response:
                                    is_final_response = True
                                    yield {
                                        "type": "response_start",
                                        "content": "",
                                        "metadata": {}
                                    }
                                
                                content = str(message.content)
                                for char in content:
                                    yield {
                                        "type": "response_token",
                                        "content": char,
                                        "metadata": {}
                                    }

    def get_memory_stats(self):
        """Get memory usage statistics"""
        if self.memory_manager:
            return self.memory_manager.get_memory_stats()
        return {}
    
    def get_memory(self):
        """Get conversation memory"""
        return self.state.messages if self.state else []  
    
    def clear_memory(self):
        """Clear conversation memory"""
        if self.state:
            self.state.messages = []
            self.state.todo_task_list = {}
            self.state.mcp_state = {}
            self.state.current_task = None
            self.state.error_count = 0
            
            # Also clear memory manager
            if self.memory_manager:
                self.memory_manager.memory_entries = []
                self.memory_manager.current_size = 0
            
            if self.verbose:
                print("History, state, and memory cleared")