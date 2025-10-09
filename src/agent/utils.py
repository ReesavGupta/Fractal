from langchain_core.tools import tool
from typing import Optional
from langsmith import traceable
from src.agent.tools import (
    read_file,
    write_file,
    read_dir,
    edit_file,
    search_fs_using_regex,
    create_directory,
    delete_file
)

_rag_service_ref = None
_memory_manager_ref = None
_db_mcp_ref = None

def set_rag_service(rag_service):
    """Set the global RAG service reference for tools to use"""
    global _rag_service_ref
    _rag_service_ref = rag_service

def set_memory_manager(memory_manager):
    """Set the global memory manager reference"""
    global _memory_manager_ref
    _memory_manager_ref = memory_manager

def set_db_mcp(db_mcp):
    """Set the global DB MCP reference"""
    global _db_mcp_ref
    _db_mcp_ref = db_mcp

def get_tool_list(include_db_tools: bool = False):
    """
    Convert tool functions to LangChain tool objects with proper schemas.
    
    Args:
        include_db_tools: If True, include database MCP tools
    """

    @tool
    @traceable('tool', name="read_file_tool")
    def read_file_tool(file_path: str) -> str:
        """
        Read and return the contents of a file.
        
        Args:
            file_path: Path to the file to read
            
        Returns:
            The contents of the file or an error message
        """
        return read_file(file_path)
    
    @tool
    @traceable('tool', name="write_file_tool")
    def write_file_tool(file_path: str, content: str) -> str:
        """
        Write content to a file. This will overwrite existing content.
        
        Args:
            file_path: Path to the file to write
            content: Content to write to the file
            
        Returns:
            Success message or error
        """
        return write_file(file_path, content)
    
    @tool
    @traceable('tool', name="read_directory_tool")
    def read_directory_tool(dir_path: str, recursive: bool = False) -> str:
        """
        List all files and directories in a directory.
        
        Args:
            dir_path: Path to the directory to read
            recursive: If True, list contents recursively
            
        Returns:
            Directory contents or error message
        """
        return read_dir(dir_path, recursive)
    
    @tool
    @traceable('tool', name="edit_file_tool") 
    def edit_file_tool(file_path: str, old_text: str, new_text: str) -> str:
        """
        Edit a file by replacing old_text with new_text.
        
        Args:
            file_path: Path to the file to edit
            old_text: Text to search for and replace
            new_text: Text to replace with
            
        Returns:
            Success message with number of replacements or error
        """
        return edit_file(old_text, new_text, file_path)
    
    @tool
    @traceable('tool', name="search_files_tool") 
    def search_files_tool(
        dir_path: str, 
        pattern: str, 
        file_extension: Optional[str] = None
    ) -> str:
        """
        FAST search for files matching a regex pattern in their content.
        Use this for simple searches instead of search_codebase_tool.
        
        Args:
            dir_path: Directory to search in
            pattern: Regex pattern to search for
            file_extension: Optional file extension filter (e.g., '.py', '.js')
            
        Returns:
            Search results showing matching files and lines
        """
        return search_fs_using_regex(dir_path, pattern, file_extension)
        
    @tool
    @traceable('tool', name="create_dir_tool")
    def create_directory_tool(dir_path: str) -> str:
        """
        Create a new directory. Creates parent directories if needed.
        
        Args:
            dir_path: Path of the directory to create
            
        Returns:
            Success message or error
        """
        return create_directory(dir_path)
    
    @tool
    @traceable('tool', name="delete_file_tool")
    def delete_file_tool(file_path: str) -> str:
        """
        Delete a file.
        
        Args:
            file_path: Path to the file to delete
            
        Returns:
            Success message or error
        """
        return delete_file(file_path)
    
    @tool
    @traceable('tool', name="search_codebase_tool")
    def search_codebase_tool(query: str) -> str:
        """
        Search the codebase using RAG for relevant code snippets and functions.
        WARNING: This tool is SLOW (20+ seconds). Only use for complex queries requiring semantic understanding.
        For simple file searches, use search_files_tool instead.
        
        Args:
            query: Natural language query about the codebase
            
        Returns:
            Relevant code snippets and explanations from the codebase
        """
        if not _rag_service_ref:
            return "Error: RAG service not available. Please initialize the agent with RAG service."
        
        try:
            results = _rag_service_ref.search(query, top_k=3)
            
            if not results:
                return f"No relevant code found for query: '{query}'"
            
            summary = _rag_service_ref.rerank(results, query)
            return summary
            
        except Exception as e:
            return f"Error searching codebase: {str(e)}"

    @tool
    @traceable('tool', name="search_memory_tool")
    def search_memory_tool(query: str) -> str:
        """
        Search through conversation memory for relevant information.
        Use this when you need to recall previous conversations or tool results.
        
        Args:
            query: Search query to find relevant memory entries
            
        Returns:
            Relevant memory entries and summaries
        """
        if not _memory_manager_ref:
            return "Error: Memory manager not available."
        
        try:
            results = _memory_manager_ref.search_memory(query, limit=5)
            
            if not results:
                return f"No relevant memory found for query: '{query}'"
            
            response = f"Memory Search Results for '{query}':\n\n"
            
            for i, entry in enumerate(results, 1):
                response += f"{i}. [{entry.message_type.value}] {entry.timestamp.strftime('%H:%M:%S')}\n"
                response += f"   Importance: {entry.importance:.2f}\n"
                response += f"   Content: {entry.content[:300]}{'...' if len(entry.content) > 300 else ''}\n\n"
            
            return response
            
        except Exception as e:
            return f"Error searching memory: {str(e)}"
    
    # Base tools
    base_tools = [
        read_file_tool,
        write_file_tool,
        read_directory_tool,
        edit_file_tool,
        search_files_tool,
        create_directory_tool,
        delete_file_tool,
        search_codebase_tool,
        search_memory_tool
    ]
    
    # Add database tools if requested and available
    if include_db_tools and _db_mcp_ref:
        try:
            # Get database tools synchronously
            db_tools = _db_mcp_ref.get_tools_sync()
            base_tools.extend(db_tools)
        except Exception as e:
            print(f"Warning: Could not load database tools: {e}")
    
    return base_tools