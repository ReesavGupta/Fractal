import json
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum
import hashlib

class MessageType(Enum):
    USER_QUERY = "user_query"
    AI_RESPONSE = "ai_response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    SYSTEM = "system"

@dataclass
class MemoryEntry:
    id: str
    timestamp: datetime
    message_type: MessageType
    content: str
    metadata: Dict[str, Any]
    importance: float  # 0.0 to 1.0
    size_bytes: int

class MemoryManager:
    def __init__(self, max_memory_size: int = 50_000, importance_threshold: float = 0.3):
        self.max_memory_size = max_memory_size
        self.importance_threshold = importance_threshold
        self.memory_entries: List[MemoryEntry] = []
        self.current_size = 0
        
    def add_entry(self, message_type: MessageType, content: str, metadata: Dict | None = None, importance: float = 0.5):
        """Add a new memory entry"""
        entry_id = hashlib.md5(f"{datetime.now()}{content}".encode()).hexdigest()[:8]
        size_bytes = len(content.encode('utf-8'))
        
        entry = MemoryEntry(
            id=entry_id,
            timestamp=datetime.now(),
            message_type=message_type,
            content=content,
            metadata=metadata or {},
            importance=importance,
            size_bytes=size_bytes
        )
        
        self.memory_entries.append(entry)
        self.current_size += size_bytes
        
        # Auto-cleanup if over threshold
        if self.current_size > self.max_memory_size:
            self._cleanup_memory()
    
    def _cleanup_memory(self):
        """Remove least important entries when memory is full"""
        # Sort by importance (ascending) then by timestamp (oldest first)
        self.memory_entries.sort(key=lambda x: (x.importance, x.timestamp))
        
        while self.current_size > self.max_memory_size * 0.8:  # Clean to 80% capacity
            if not self.memory_entries:
                break
                
            removed = self.memory_entries.pop(0)
            self.current_size -= removed.size_bytes
    
    def search_memory(self, query: str, limit: int = 10) -> List[MemoryEntry]:
        """Search memory entries by content"""
        query_lower = query.lower()
        results = []
        
        for entry in self.memory_entries:
            if query_lower in entry.content.lower():
                results.append(entry)
        
        # Sort by importance and recency
        results.sort(key=lambda x: (x.importance, x.timestamp), reverse=True)
        return results[:limit]
    
    def get_recent_important(self, limit: int = 5) -> List[MemoryEntry]:
        """Get recent important entries"""
        important = [e for e in self.memory_entries if e.importance > self.importance_threshold]
        important.sort(key=lambda x: x.timestamp, reverse=True)
        return important[:limit]
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        return {
            "total_entries": len(self.memory_entries),
            "current_size_bytes": self.current_size,
            "max_size_bytes": self.max_memory_size,
            "usage_percentage": (self.current_size / self.max_memory_size) * 100,
            "important_entries": len([e for e in self.memory_entries if e.importance > self.importance_threshold])
        }