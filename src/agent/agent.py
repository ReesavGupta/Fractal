import os
from pydantic import BaseModel, Field
from typing import Optional, Annotated, List, Literal, Dict
from utils import get_tool_list
from memory import FractalMemory
from langgraph.prebuilt import ToolNode
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages

class IFractalState(BaseModel):
    messages: Annotated[List[BaseMessage], add_messages]
    # todo_task_list: Optional[Dict[str, bool]] = Field(default_factory=dict)
    # current_task: Optional[str] = None
    # code_context: Optional[str] = None
    # error_count: int = 0
    
    class Config:
        arbitrary_types_allowed = True

class CodingAgent:
    def __init__(self, llm: str, api_key: Optional[str] = None, verbose: bool = False) -> None:
        self.llm_provider = llm
        self.verbose = verbose
        self.client = None
        self.model_name = None
        ##########################################################################
        self.memory = FractalMemory()
        self.tools = get_tool_list()
        ##########################################################################
        self.graph = None
        self.state = None
        ########################################################################## 
        # here we will first initialize the llm with the llm which the user has provided
        self._initialize_llm(llm, api_key)
        self._build_graph()
        # Build the agent graph
        # self._build_graph()
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

        ####################################################################################################################
        # binding with tools
        self.client_with_tools = self.client.bind_tools(self.tools)
        if self.verbose:
            print(f"✓ Initialized {self.model_name} with {len(self.tools)} tools")
        ####################################################################################################################
        

    def _build_graph(self):
        """Build the LangGraph agent workflow"""

        async def agent_node(state: IFractalState) -> IFractalState:
            """Node that calls the LLM with tools"""
            messages = state.messages
            
            system_message = SystemMessage(content="""You are Fractal, an expert coding assistant with access to file system tools.
                You can read, write, edit files, navigate directories, and search through code.

                When asked to perform a task:
                1. Break down the task into steps
                2. Use available tools to gather information
                3. Execute the necessary file operations
                4. Provide clear explanations of what you're doing
                5. Show the results

                Always be thorough and explain your reasoning.""")
            
            full_messages = [system_message] + messages
            
            if self.verbose:
                print(f"\nCalling {self.model_name}...")
            
            response = await self.client_with_tools.ainvoke(full_messages)
            
            if self.verbose and response.tool_calls: #type:ignore
                print(f"Tool calls requested: {[tc['name'] for tc in response.tool_calls]}") #type:ignore
            
            return {"messages": [response]} #type:ignore
        
        # Define routing function
        def should_continue(state: IFractalState) -> Literal["tools", "end"]:
            """Determine if we should continue to tools or end"""
            messages = state.messages
            last_message = messages[-1]
            
            # If there are tool calls, route to tools node
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls: #type:ignore
                return "tools"
            # Otherwise, end the conversation
            return "end"


###########################################################################################################################################################
###########################################################################################################################################################
# BUILD GRAPH HERE
###########################################################################################################################################################
###########################################################################################################################################################
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

###########################################################################################################################################################
###########################################################################################################################################################
# GRAPH COMPILED
###########################################################################################################################################################
###########################################################################################################################################################

    def invoke(self, user_input: str) -> str:
        """
        Process user input through the agent graph
        
        Args:
            user_input: The user's query or command
            
        Returns:
            The final response from the agent
        """
        if not self.graph:
            raise RuntimeError("Agent graph not initialized")
        
        # Create initial state with user message
        # initial_state = {
        #     "messages": [HumanMessage(content=user_input)]
        # }

        initial_state = IFractalState(messages=[HumanMessage(content=user_input)])
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Processing: {user_input}")
            print(f"{'='*60}")
        
        # yaha pe invoke karenge
        final_state = self.graph.invoke(initial_state)
        
        final_message = final_state["messages"][-1]
        
        if isinstance(final_message, AIMessage):
            response = final_message.content
        else:
            response = str(final_message)
        
        # Store in memory
        self.memory.add_interaction(user_input, response) #type:ignore
        
        # Ensure response is always a string
        if isinstance(response, list):
            response = "\n".join(str(item) for item in response)
        return response
    
    async def ainvoke(self, user_input: str) -> str:
        """Async version of invoke"""
        if not self.graph:
            raise RuntimeError("Agent graph not initialized")
        
        initial_state = IFractalState(messages=[HumanMessage(content=user_input)])

        final_state = await self.graph.ainvoke(initial_state, {"recursion_limit": 100})
        final_message = final_state["messages"][-1]
        
        if isinstance(final_message, AIMessage):
            response = final_message.content
        else:
            response = str(final_message)
        
        self.memory.add_interaction(user_input, response) #type:ignore

        # Ensure response is always a string
        if isinstance(response, list):
            response = "\n".join(str(item) for item in response)
        return response
    
    def stream(self, user_input: str):
        """
        Stream the agent's response token by token
        
        Args:
            user_input: The user's query or command
            
        Yields:
            Response chunks as they're generated
        """
        if not self.graph:
            raise RuntimeError("Agent graph not initialized")
        
        initial_state = IFractalState(messages=[HumanMessage(content=user_input)])
                    
        for event in self.graph.stream(initial_state, stream_mode="values"):
            if "messages" in event and event["messages"]:
                last_message = event["messages"][-1]
                if isinstance(last_message, AIMessage) and last_message.content:
                    yield last_message.content

    def get_memory(self):
        """Get conversation memory"""
        return self.memory.get_history()
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()
        if self.verbose:
            print("✓ Memory cleared")