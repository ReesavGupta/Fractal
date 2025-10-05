from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional, Annotated, List, Literal, Dict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class IFractalState(BaseModel):
    messages: Annotated[List[BaseMessage], add_messages]
    todo_task_list: Dict[str, bool] = Field(default_factory=dict)
    current_task: Optional[str] = None
    code_context: Optional[str] = None  # yaha pe RAG ka context aayega
    error_count: int = 0
    mcp_state: Dict[str, dict] = Field(default_factory=dict)  # stores MCP plugin states
    session_start: datetime = Field(default_factory=datetime.now)

    ###############################################################
    # helper methods 
    #---------------
    def add_message(self, msg: BaseMessage):
        self.messages.append(msg)

    def get_recent_messages(self, n: int = 10) -> List[BaseMessage]:
        return self.messages[-n:]    

    def search_message(self, query: str) -> List[BaseMessage]:
        matched_messages = []
        query_lower = query.lower()
        for message in self.messages:
            if hasattr(message, 'content') and message.content is str and  query_lower in message.content.lower() :
                matched_messages.append(message)
        return matched_messages
    
    def save_history_to_db(self):
        pass
    ###############################################################
    
    class Config:
        arbitrary_types_allowed = True