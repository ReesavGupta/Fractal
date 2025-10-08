from typing import Dict, Any
from src.agent.memory import MemoryManager, MessageType
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage

class MessageFilter:
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
    
    def should_store_in_state(self, message: BaseMessage) -> bool:
        """Determine if message should be stored in LLM state"""
        if isinstance(message, HumanMessage):
            return True  # Always keep user messages in state
        
        if isinstance(message, AIMessage):
            #  Keep AI responses but truncate if too long
            if len(message.content) > 1000:
                message.content = message.content[:1000] + "... [truncated]" #type:ignore
            return True
        
        if isinstance(message, ToolMessage):
            # Don't store tool results in state - they go to memory
            return False
        
        return True
    
    def process_tool_result(self, tool_name: str, result: str) -> Dict[str, Any]:
        """Process tool result and determine importance"""
        importance = self._calculate_importance(tool_name, result)
        
        # Store in memory instead of state
        self.memory_manager.add_entry(
            message_type=MessageType.TOOL_RESULT,
            content=result,
            metadata={"tool_name": tool_name},
            importance=importance
        )
        
        # Return summary for state
        if len(result) > 200:
            return {"summary": result[:200] + "...", "stored_in_memory": True}
        return {"summary": result, "stored_in_memory": False}
    
    def _calculate_importance(self, tool_name: str, result: str) -> float:
        """Calculate importance score for tool results"""
        base_importance = 0.5
        
        # RAG results are more important
        if tool_name == "search_codebase_tool":
            base_importance = 0.8
        
        # File operations are moderately important
        elif tool_name in ["read_file_tool", "write_file_tool", "edit_file_tool"]:
            base_importance = 0.6
        
        # Directory operations are less important
        elif tool_name in ["read_directory_tool", "create_directory_tool"]:
            base_importance = 0.4
        
        # Adjust based on result length (longer = more important)
        if len(result) > 1000:
            base_importance += 0.2
        elif len(result) < 100:
            base_importance -= 0.2
        
        return min(1.0, max(0.0, base_importance))