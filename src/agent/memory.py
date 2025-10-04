from typing import List, Dict, Tuple
from datetime import datetime

class FractalMemory:
    """
    Memory management for the Fractal agent using a sliding window approach.
    Stores conversation history with timestamps and implements context management.
    """
    def __init__(self, max_history: int = 50, context_window: int = 10) -> None:
        """
        Initialize memory with configurable limits.
        
        Args:
            max_history: Maximum number of interactions to store
            context_window: Number of recent interactions to include in context
        """
    
        # https://medium.com/@gopikwork/building-agentic-memory-patterns-with-strands-and-langgraph-3cc8389b350d 
        # check the above article for some memory implementations please

        self.max_history = max_history
        self.context_window = context_window
        self.history: List[Dict] = []
        self.session_start = datetime.now()

    
    def add_interaction(self, user_input: str, agent_response: str) -> None:
        """
        Add a user-agent interaction to memory.
        
        Args:
            user_input: The user's input/query
            agent_response: The agent's response
        """
        interaction = {
            "timestamp": datetime.now(),
            "user": user_input,
            "agent": agent_response
        }
        
        self.history.append(interaction)
        
        # Maintain max_history limit using sliding window
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_recent_context(self, n: int | None = None) -> List[Dict]:
        """
        Get the n most recent interactions for context.
        
        Args:
            n: Number of recent interactions to retrieve (defaults to context_window)
            
        Returns:
            List of recent interactions
        """
        if n is None:
            n = self.context_window
        
        return self.history[-n:] if self.history else []
    
    def get_history(self) -> List[Dict]:
        """
        Get the complete conversation history.
        
        Returns:
            Complete list of all interactions
        """
        return self.history.copy()
    
    def format_context_for_llm(self, n: int | None= None) -> str:
        """
        Format recent context as a string suitable for LLM input.
        
        Args:
            n: Number of recent interactions to include
            
        Returns:
            Formatted context string
        """
        recent = self.get_recent_context(n)
        
        if not recent:
            return "No previous conversation context."
        
        formatted = ["Previous conversation:"]
        for interaction in recent:
            timestamp = interaction["timestamp"].strftime("%H:%M:%S")
            formatted.append(f"\n[{timestamp}] User: {interaction['user']}")
            formatted.append(f"[{timestamp}] Agent: {interaction['agent']}")
        
        return "\n".join(formatted)
    
    def search_history(self, query: str) -> List[Dict]:
        """
        Search through conversation history for interactions containing the query.
        
        Args:
            query: Search term
            
        Returns:
            List of matching interactions
        """
        query_lower = query.lower()
        matches = []
        
        for interaction in self.history:
            if (query_lower in interaction["user"].lower() or 
                query_lower in interaction["agent"].lower()):
                matches.append(interaction)
        
        return matches
    
    def get_session_summary(self) -> Dict:
        """
        Get a summary of the current session.
        
        Returns:
            Dictionary with session statistics
        """
        duration = datetime.now() - self.session_start
        
        return {
            "session_start": self.session_start,
            "duration": str(duration).split('.')[0],  # Remove microseconds
            "total_interactions": len(self.history),
            "context_window": self.context_window,
            "max_history": self.max_history
        }
    
    def clear(self) -> None:
        """Clear all conversation history and reset session."""
        self.history = []
        self.session_start = datetime.now()
    
    def export_history(self, filepath: str | None = None) -> str:
        """
        Export conversation history to a file or return as string.
        
        Args:
            filepath: Optional path to save the history
            
        Returns:
            Formatted history string
        """
        if not self.history:
            return "No conversation history to export."
        
        lines = [
            "=" * 60,
            f"Fractal Conversation History",
            f"Session Start: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            ""
        ]
        
        for i, interaction in enumerate(self.history, 1):
            timestamp = interaction["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            lines.extend([
                f"Interaction {i} - {timestamp}",
                "-" * 60,
                f"User: {interaction['user']}",
                "",
                f"Agent: {interaction['agent']}",
                "",
                "=" * 60,
                ""
            ])
        
        content = "\n".join(lines)
        
        if filepath:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                return f"History exported to {filepath}"
            except Exception as e:
                return f"Error exporting history: {str(e)}"
        
        return content