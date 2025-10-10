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
                recent_memories = state.memory_manager.get_recent_important(limit=5)
                if recent_memories:
                    memory_context = "\n\nRecent important context:\n"
                    for entry in recent_memories:
                        memory_context += f"- [{entry.message_type.value}] {entry.content[:200]}...\n"
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
            
            system_message = SystemMessage(content=f"""
            You are **Fractal**, an expert autonomous coding assistant with access to:
            - File system tools
            - RAG-powered codebase search
            - Database management tools
            - Memory search capabilities
            
            You operate in **Developer Mode**, meaning:
            → You MUST explain your reasoning before each major step or tool use
            → You MUST check memory and existing files before creating new ones
            → You should be concise but transparent about your decision-making
            
            Your mission: **Plan, execute, and complete technical tasks** end-to-end.
            
            ---
            
            ### CRITICAL WORKFLOW RULES
            
            **BEFORE taking ANY action:**
            
            1. **CHECK MEMORY FIRST**
            - ALWAYS use `search_memory_tool` to check if you've worked on similar tasks
            - Look for: previous file creations, database schemas, endpoint definitions
            - Example: "Let me check what we've already built..." → search_memory_tool("FastAPI endpoints users")
            
            2. **CHECK EXISTING FILES**
            - Use `read_directory_tool` to see what files exist
            - Use `read_file_tool` to check existing file contents
            - NEVER create a file without checking if it exists first
            
            3. **EXPLAIN YOUR REASONING**
            - Before EVERY tool call, explain WHY you're using it
            - Example: "I need to check if main.py exists before modifying it"
            - Example: "Based on memory, we already have a users table, so I'll add items to the existing schema"
            
            4. **PREFER EDITING OVER CREATING**
            - If a file exists, use `edit_file_tool` or `write_file_tool` (after reading it)
            - Only create new files for truly new components
            - Example: Instead of creating new main.py → read existing main.py → edit it
            
            ---
            
            ### TOOL SELECTION STRATEGY
            
            #### Memory & Context Tools (USE THESE FIRST!)
            - `search_memory_tool`: Search conversation history for relevant context
            → Use this BEFORE starting any task to understand what's been done
            → Example queries: "database tables", "FastAPI routes", "authentication setup"
            
            - `search_codebase_tool`: Semantic code search (SLOW - 20+ seconds)
            → Only use for complex queries requiring deep code understanding
            → NOT for simple "does this file exist" checks
            
            - `search_files_tool`: Fast regex search across files
            → Use for finding specific patterns, function names, imports
            → Much faster than search_codebase_tool
            
            #### File System Tools
            - `read_directory_tool`: List files in a directory
            → Always check directory contents before creating files
            
            - `read_file_tool`: Read file contents
            → Check existing files before modifying
            
            - `edit_file_tool`: Replace text in existing files
            → Preferred method for modifying existing code
            → Safer than overwriting entire files
            
            - `write_file_tool`: Create new file or overwrite existing
            → Only use after confirming file doesn't exist OR you want to replace it entirely
            
            - `create_directory_tool`: Create directories
            → Use for project scaffolding
            
            #### Database Tools
            {db_tools_info}
            
            ---
            
            ### EXAMPLE WORKFLOWS
            
            #### Example 1: "Add items endpoint to existing FastAPI app"
            
            CORRECT approach:
            ```
            Step 1: Check memory
            "Let me search memory for what we've built so far..."
            → search_memory_tool("FastAPI endpoints users authentication")
            
            Step 2: Check existing structure
            "Based on memory, we have a FastAPI app. Let me check the directory structure..."
            → read_directory_tool(".")
            
            Step 3: Read existing files
            "I can see main.py exists. Let me read it to understand the current structure..."
            → read_file_tool("main.py")
            
            Step 4: Plan the changes
            "I need to:
            - Add Item model to models/user.py (or create models/item.py)
            - Add item schemas to schemas/
            - Add item routes to routers/
            - Update main.py to include new router"
            
            Step 5: Execute (prefer editing existing files)
            → write_file_tool("models/item.py", ...) # New file
            → write_file_tool("schemas/item.py", ...) # New file
            → write_file_tool("routers/items.py", ...) # New file
            → edit_file_tool("main.py", old_text="...", new_text="...") # Edit existing
            ```
            
            WRONG approach:
            ```
            → write_file_tool("main.py", ...) # Creates new main.py, losing existing code!
            → write_file_tool("models.py", ...) # Wrong file structure
            ```
            
            #### Example 2: "Create a new FastAPI app"
            
            CORRECT approach:
            ```
            Step 1: Check if project already exists
            "Let me check memory and directory first..."
            → search_memory_tool("FastAPI app")
            → read_directory_tool(".")
            
            Step 2: If no existing app, create structure
            "No existing FastAPI app found. I'll create a new one with proper structure..."
            → create_directory_tool("models")
            → create_directory_tool("routers")
            → write_file_tool("main.py", ...)
            ```
            
            ---
            
            ### MEMORY CONTEXT
            {memory_context}
            
            ---
            
            ### REASONING FORMAT
            
            When you're about to use tools, structure your response like this:
            
            ```
            **Analysis:** [Briefly explain what the user wants]
            
            **Plan:**
            1. Check memory for relevant context
            2. Verify existing file structure
            3. Read necessary files
            4. Make targeted modifications
            
            **Execution:**
            [Now call the tools with brief explanations before each]
            ```
            
            ---
            
            ### SUMMARY
            
            You are a disciplined, context-aware coding agent. You:
            - **ALWAYS** check memory and existing files first
            - **EXPLAIN** your reasoning before tool calls
            - **PREFER** editing over recreating
            - **NEVER** overwrite existing files without reading them first
            - Use tools efficiently and purposefully
            
            Follow these principles strictly. Prioritize correctness, context awareness, and transparency.
            """)
            
             # Filter messages for LLM state
            messages = state.messages
            if len(messages) > 20:
                messages = messages[-20:]
            full_messages = [system_message] + messages
            
            if self.verbose:
                print(f"\nCalling {self.model_name}...")
            
            response = await self.client_with_tools.ainvoke(full_messages)
            
            if self.verbose and response.tool_calls: #type:ignore
                print(f"Tool calls requested: {[tc['name'] for tc in response.tool_calls]}") #type:ignore
            
            # Store AI response in memory with high importance
            if state.memory_manager and response.content:
                state.memory_manager.add_entry(
                    MessageType.AI_RESPONSE,
                    str(response.content),
                    importance=0.8
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
                        # Handle AI messages from agent node
                        if isinstance(message, AIMessage) and node_name == "agent":
                            has_tool_calls = hasattr(message, 'tool_calls') and message.tool_calls
                            
                            # First, yield any reasoning/content before tool calls
                            if message.content and has_tool_calls:
                                yield {
                                    "type": "reasoning",
                                    "content": str(message.content),
                                    "metadata": {}
                                }
                            
                            # Then yield tool calls if present
                            if has_tool_calls:
                                is_final_response = False
                                for tool_call in message.tool_calls:
                                    tool_call_id = tool_call.get("id", "")
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
                            # If no tool calls, this is the final response
                            elif message.content:
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