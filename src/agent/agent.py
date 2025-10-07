# import os
# from typing import Optional, Literal
# from src.agent.utils import get_tool_list
# from langgraph.prebuilt import ToolNode
# from langchain.chat_models import init_chat_model
# from langgraph.graph import StateGraph, START, END
# from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
# from src.agent.state import IFractalState
# from src.rag_service.rag import RAGService

# class CodingAgent:
#     def __init__(self, llm: str, api_key: Optional[str] = None, verbose: bool = False, rag_service: RAGService | None = None) -> None:
#         self.llm_provider = llm
#         self.verbose = verbose
#         self.client = None
#         self.model_name = None
        
#         ##########################################################################
#         self.tools = get_tool_list()
#         self.rag_service = rag_service 
#         ##########################################################################

#         ##########################################################################
#         self.graph = None
#         self.state: Optional[IFractalState] = IFractalState(messages=[])
#         ########################################################################## 
        
#         # here we will first initialize the llm with the llm which the user has provided
#         self._initialize_llm(llm, api_key)        
#         # Build the agent graph
#         self._build_graph()

#     def _initialize_llm(self, llm: str, api_key: Optional[str] = None):    
#         """Initialize the LLM client with tool binding"""
#         if llm == "openai":
#             api_key = api_key or os.getenv("OPENAI_API_KEY")
#             if not api_key:
#                 raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
#             self.client = init_chat_model("gpt-4o", model_provider=llm, temperature=0.8, api_key=api_key)
#             self.model_name = "gpt-4o"
                
#         elif llm == "claude":
#             api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
#             if not api_key:
#                 raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.")
#             self.client = init_chat_model("claude-opus-20240229", model_provider="anthropic", temperature=0.8, api_key=api_key)
#             self.model_name = "claude-opus-20240229"
            
#         elif llm == "gemini":
#             api_key = api_key or os.getenv("GEMINI_API_KEY")
#             if not api_key:
#                 raise ValueError("Gemini API key not found. Set GEMINI_API_KEY environment variable.")
#             self.client = init_chat_model("gemini-2.0-flash", model_provider="google_vertexai", temperature=0.8, api_key=api_key)
#             self.model_name = "gemini-2.0-flash"
#         else:
#             raise ValueError(f"Unsupported LLM provider: {llm}")

#         ####################################################################################################################
#         # binding with tools
#         self.client_with_tools = self.client.bind_tools(self.tools)
#         if self.verbose:
#             print(f"Initialized {self.model_name} with {len(self.tools)} tools")
#         ####################################################################################################################
        

#     def _build_graph(self):
#         """Build the LangGraph agent workflow"""

#         async def agent_node(state: IFractalState) -> IFractalState:
#             """Node that calls the LLM with tools"""
#             messages = state.messages
            
#             system_message = SystemMessage(content="""You are Fractal, an expert coding assistant with access to file system tools and Retrieval-Augmented Generation (RAG)-powered codebase search.

#                 TOOL SELECTION STRATEGY:
#                 - For SIMPLE queries (greetings, basic questions, file operations): Use fast tools like read_file_tool, write_file_tool, search_files_tool
#                 - For COMPLEX queries (architecture questions, code understanding, cross-file analysis): Use search_codebase_tool (RAG)
                
#                 Available tools:
#                 - read_file_tool, write_file_tool, edit_file_tool: Fast file operations
#                 - read_directory_tool: Directory navigation
#                 - search_files_tool: Fast regex-based file content search
#                 - search_codebase_tool: Advanced semantic codebase search (SLOW - use only for complex queries)
#                 - create_directory_tool, delete_file_tool: File management

#                 WORKFLOW:
#                 1. For simple tasks: Use appropriate fast tools directly
#                 2. For complex tasks requiring codebase understanding: Use search_codebase_tool first
#                 3. Break down complex tasks into steps
#                 4. Execute necessary operations
#                 5. Provide clear explanations

#                 PERFORMANCE TIP: Only use search_codebase_tool when you need semantic understanding of the codebase. For specific file searches, use search_files_tool instead.""")
            
#             full_messages = [system_message] + messages
            
#             if self.verbose:
#                 print(f"\nCalling {self.model_name}...")
            
#             response = await self.client_with_tools.ainvoke(full_messages)
            
#             if self.verbose and response.tool_calls: #type:ignore
#                 print(f"Tool calls requested: {[tc['name'] for tc in response.tool_calls]}") #type:ignore
            
#             return {"messages": [response]} #type:ignore
        
#         # Define routing function
#         def should_continue(state: IFractalState) -> Literal["tools", "end"]:
#             """Determine if we should continue to tools or end"""
#             messages = state.messages
#             last_message = messages[-1]
            
#             # If there are tool calls, route to tools node
#             if hasattr(last_message, 'tool_calls') and last_message.tool_calls: #type:ignore
#                 return "tools"
#             # Otherwise, end the conversation
#             return "end"


# ###########################################################################################################################################################
# ###########################################################################################################################################################
# # BUILD GRAPH HERE
# ###########################################################################################################################################################
# ###########################################################################################################################################################
#         graph_builder = StateGraph(IFractalState)

#         graph_builder.add_node("agent", agent_node)
#         graph_builder.add_node("tools", ToolNode(self.tools))

#         graph_builder.add_edge(START, "agent")

#         graph_builder.add_conditional_edges(
#             "agent",
#             should_continue,
#             {
#                 "tools": "tools",
#                 "end": END
#             }
#         )

#         graph_builder.add_edge("tools", "agent")
        
#         self.graph = graph_builder.compile()
    
#         if self.verbose:
#             print("Agent graph compiled successfully")

# ###########################################################################################################################################################
# ###########################################################################################################################################################
# # GRAPH COMPILED
# ###########################################################################################################################################################
# ###########################################################################################################################################################
#     async def ainvoke(self, user_input: str) -> str:
#         """Async version of invoke"""
#         if not self.graph:
#             raise RuntimeError("Agent graph not initialized")

#         if self.state is None:
#             raise RuntimeError("State for graph is not initialized")

#         self.state.add_message(HumanMessage(content=user_input))

#         final_state = await self.graph.ainvoke(self.state, {"recursion_limit": 100})

#         final_message = final_state["messages"][-1]
        
#         if isinstance(final_message, AIMessage):
#             self.state.add_message(final_message)
#             response = final_message.content
#         else:
#             response = str(final_message)
        
#         # Ensure response is always a string
#         if isinstance(response, list):
#             response = "\n".join(str(item) for item in response)

#         return response
    
#     async def astream(self, user_input: str):
#         """
#         Asynchronously stream the agent's response token by token
        
#         Args:
#             user_input: The user's query or command
            
#         Yields:
#             Response chunks as they're generated
#         """
#         if not self.graph:
#             raise RuntimeError("Agent graph not initialized")

#         if self.state is None:
#             raise RuntimeError("State is not initialized")

#         self.state.messages.append(HumanMessage(content=user_input))

#         async for event in self.graph.astream(self.state, stream_mode="values"):
#             if "messages" in event and event["messages"]:
#                 last_message = event["messages"][-1]
#                 if isinstance(last_message, AIMessage) and last_message.content:
#                     yield last_message.content

#     def get_memory(self):
#         """Get conversation memory"""
#         return self.state.messages if self.state else []  
    
#     def clear_memory(self):
#         """Clear conversation memory"""
#         if self.state:
#             self.state.messages = []
#             self.state.todo_task_list = {}
#             self.state.mcp_state = {}
#             self.state.current_task = None
#             self.state.error_count = 0
#             if self.verbose:
#                 print("History and state cleared")

import os
from typing import Optional, Literal, AsyncGenerator
from src.agent.utils import get_tool_list
from langgraph.prebuilt import ToolNode
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.agent.state import IFractalState
from src.rag_service.rag import RAGService

class CodingAgent:
    def __init__(self, llm: str, api_key: Optional[str] = None, verbose: bool = False, rag_service: RAGService | None = None) -> None:
        self.llm_provider = llm
        self.verbose = verbose
        self.client = None
        self.model_name = None
        
        self.tools = get_tool_list()
        self.rag_service = rag_service 
        
        self.graph = None
        self.state: Optional[IFractalState] = IFractalState(messages=[])
        
        self._initialize_llm(llm, api_key)        
        self._build_graph()

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
            self.client = init_chat_model("gemini-2.0-flash", model_provider="google_vertexai", temperature=0.8, api_key=api_key)
            self.model_name = "gemini-2.0-flash"
        else:
            raise ValueError(f"Unsupported LLM provider: {llm}")

        self.client_with_tools = self.client.bind_tools(self.tools)
        if self.verbose:
            print(f"Initialized {self.model_name} with {len(self.tools)} tools")

    def _build_graph(self):
        """Build the LangGraph agent workflow"""

        async def agent_node(state: IFractalState) -> IFractalState:
            """Node that calls the LLM with tools"""
            messages = state.messages
            
            system_message = SystemMessage(content="""You are Fractal, an expert coding assistant with access to file system tools and Retrieval-Augmented Generation (RAG)-powered codebase search.

                TOOL SELECTION STRATEGY:
                - For SIMPLE queries (greetings, basic questions, file operations): Use fast tools like read_file_tool, write_file_tool, search_files_tool
                - For COMPLEX queries (architecture questions, code understanding, cross-file analysis): Use search_codebase_tool (RAG)
                
                Available tools:
                - read_file_tool, write_file_tool, edit_file_tool: Fast file operations
                - read_directory_tool: Directory navigation
                - search_files_tool: Fast regex-based file content search
                - search_codebase_tool: Advanced semantic codebase search (SLOW - use only for complex queries)
                - create_directory_tool, delete_file_tool: File management

                WORKFLOW:
                1. For simple tasks: Use appropriate fast tools directly
                2. For complex tasks requiring codebase understanding: Use search_codebase_tool first
                3. Break down complex tasks into steps
                4. Execute necessary operations
                5. Provide clear explanations

                PERFORMANCE TIP: Only use search_codebase_tool when you need semantic understanding of the codebase. For specific file searches, use search_files_tool instead.""")
            
            full_messages = [system_message] + messages
            
            if self.verbose:
                print(f"\nCalling {self.model_name}...")
            
            response = await self.client_with_tools.ainvoke(full_messages)
            
            if self.verbose and response.tool_calls:
                print(f"Tool calls requested: {[tc['name'] for tc in response.tool_calls]}")
            
            return {"messages": [response]}
        
        def should_continue(state: IFractalState) -> Literal["tools", "end"]:
            """Determine if we should continue to tools or end"""
            messages = state.messages
            last_message = messages[-1]
            
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"
            return "end"

        graph_builder = StateGraph(IFractalState)
        graph_builder.add_node("agent", agent_node)
        graph_builder.add_node("tools", ToolNode(self.tools))
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

    async def ainvoke(self, user_input: str) -> str:
        """Async version of invoke"""
        if not self.graph:
            raise RuntimeError("Agent graph not initialized")

        if self.state is None:
            raise RuntimeError("State for graph is not initialized")

        self.state.add_message(HumanMessage(content=user_input))
        final_state = await self.graph.ainvoke(self.state, {"recursion_limit": 100})
        final_message = final_state["messages"][-1]
        
        if isinstance(final_message, AIMessage):
            self.state.add_message(final_message)
            response = final_message.content
        else:
            response = str(final_message)
        
        if isinstance(response, list):
            response = "\n".join(str(item) for item in response)

        return response
    
    async def astream(self, user_input: str) -> AsyncGenerator[dict, None]:
        """
        Stream the agent's response with reasoning display
        
        Yields:
            dict with keys:
                - type: 'reasoning' | 'tool_call' | 'tool_result' | 'response' | 'response_token'
                - content: the actual content
                - metadata: additional info (tool name, args, etc.)
        """
        if not self.graph:
            raise RuntimeError("Agent graph not initialized")

        if self.state is None:
            raise RuntimeError("State is not initialized")

        self.state.messages.append(HumanMessage(content=user_input))

        final_response_content = ""
        is_final_response = False

        async for event in self.graph.astream(self.state, stream_mode="updates"):
            for node_name, node_state in event.items():
                if "messages" in node_state:
                    messages = node_state["messages"]
                    
                    for message in messages:
                        # Handle tool calls
                        if isinstance(message, AIMessage) and hasattr(message, 'tool_calls') and message.tool_calls:
                            is_final_response = False
                            for tool_call in message.tool_calls:
                                yield {
                                    "type": "tool_call",
                                    "content": tool_call.get("name", "unknown"),
                                    "metadata": {
                                        "args": tool_call.get("args", {}),
                                        "id": tool_call.get("id", "")
                                    }
                                }
                        
                        # Handle tool results
                        elif hasattr(message, 'content') and hasattr(message, 'name'):
                            yield {
                                "type": "tool_result",
                                "content": str(message.content),
                                "metadata": {
                                    "tool_name": getattr(message, 'name', 'unknown')
                                }
                            }
                        
                        # Handle AI response - detect if it's the final response (no tool calls)
                        elif isinstance(message, AIMessage) and message.content:
                            # Check if this is likely the final response (no tool calls in this message)
                            has_tool_calls = hasattr(message, 'tool_calls') and message.tool_calls
                            
                            if not has_tool_calls and node_name == "agent":
                                # This is the final response - stream it token by token
                                if not is_final_response:
                                    is_final_response = True
                                    yield {
                                        "type": "response_start",
                                        "content": "",
                                        "metadata": {}
                                    }
                                
                                # Stream token by token
                                content = str(message.content)
                                for char in content:
                                    yield {
                                        "type": "response_token",
                                        "content": char,
                                        "metadata": {}
                                    }

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
            if self.verbose:
                print("History and state cleared")