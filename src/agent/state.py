from datetime import datetime
from src.agent.memory import MemoryManager, MessageType
from src.agent.memory_filter import MessageFilter
from pydantic import BaseModel, Field
from typing import Optional, Annotated, List, Literal, Dict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages

class IFractalState(BaseModel):
    messages: Annotated[List[BaseMessage], add_messages]
    todo_task_list: Dict[str, bool] = Field(default_factory=dict)
    current_task: Optional[str] = None
    code_context: Optional[str] = None  # yaha pe RAG ka context aayega
    error_count: int = 0
    mcp_state: Dict[str, dict] = Field(default_factory=dict)  # stores MCP plugin states
    session_start: datetime = Field(default_factory=datetime.now)
    
    memory_manager: Optional[MemoryManager] = None
    message_filter: Optional[MessageFilter] = None
    ###############################################################
    # helper methods 
    #---------------
    def add_message(self, msg: BaseMessage):
        self.messages.append(msg)

        if self.memory_manager:
            if isinstance(msg, HumanMessage):
                self.memory_manager.add_entry(MessageType.USER_QUERY, msg.content, importance=0.9) #type:ignore
            elif isinstance(msg, AIMessage):
                self.memory_manager.add_entry(MessageType.AI_RESPONSE, msg.content, importance=0.8) #type:ignore

    def get_recent_messages(self, n: int = 10) -> List[BaseMessage]:
        return self.messages[-n:]    

    def get_memory_summary(self) -> str:
        """Get summary of important memories"""
        if not self.memory_manager:
            return "No memory available"
        
        recent_important = self.memory_manager.get_recent_important(limit=3)
        if not recent_important:
            return "No important memories"
        
        summary = "Recent important context:\n"
        for entry in recent_important:
            summary += f"- {entry.message_type.value}: {entry.content[:100]}...\n"
        
        return summary
    
    def save_history_to_db(self):
        pass
    ###############################################################
    
    class Config:
        arbitrary_types_allowed = True