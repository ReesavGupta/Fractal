from typing import Dict, Any
from .memory import MemoryManager, MessageType
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage

class MessageFilter:
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
    
    def should_store_in_state(self, message: BaseMessage) -> bool:
        """Determine if message should be stored in LLM state"""
        if isinstance(message, HumanMessage):
            return True  # Always keep user messages in state
        
        if isinstance(message, AIMessage):
            # Keep AI responses but truncate if too long
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
        
        # Store in memory with enhanced metadata
        metadata = {
            "tool_name": tool_name,
            "result_length": len(result),
            "truncated": len(result) > 2000
        }
        
        # Store full result in memory (up to reasonable limit)
        content_to_store = result[:5000] if len(result) > 5000 else result
        
        self.memory_manager.add_entry(
            message_type=MessageType.TOOL_RESULT,
            content=content_to_store,
            metadata=metadata,
            importance=importance
        )
        
        # Return summary for state
        if len(result) > 200:
            return {
                "summary": f"[Stored in memory] {result[:200]}...",
                "stored_in_memory": True
            }
        return {
            "summary": result,
            "stored_in_memory": False
        }
    
    def _calculate_importance(self, tool_name: str, result: str) -> float:
        """Calculate importance score for tool results"""
        base_importance = 0.5
        
        # Memory search results are very important
        if tool_name == "search_memory_tool":
            base_importance = 0.9
        
        # RAG results are very important
        elif tool_name == "search_codebase_tool":
            base_importance = 0.85
        
        # File reads are important (contain code context)
        elif tool_name == "read_file_tool":
            base_importance = 0.75
        
        # Directory reads are moderately important
        elif tool_name == "read_directory_tool":
            base_importance = 0.6
        
        # File operations are important to track
        elif tool_name in ["write_file_tool", "edit_file_tool"]:
            base_importance = 0.8
        
        # File creation/deletion is important
        elif tool_name in ["create_directory_tool", "delete_file_tool"]:
            base_importance = 0.7
        
        # Search results are important
        elif tool_name == "search_files_tool":
            base_importance = 0.65
        
        # Database operations are important
        elif any(db in tool_name for db in ["postgres", "mysql", "mongodb"]):
            if "connect" in tool_name:
                base_importance = 0.7
            elif "query" in tool_name or "execute" in tool_name:
                base_importance = 0.75
            else:
                base_importance = 0.6
        
        # Adjust based on result characteristics
        if "error" in result.lower() or "failed" in result.lower():
            base_importance += 0.1  # Errors are important to remember
        
        if len(result) > 2000:
            base_importance += 0.1  # Longer results likely contain more info
        elif len(result) < 50:
            base_importance -= 0.1  # Very short results less important
        
        return min(1.0, max(0.1, base_importance))