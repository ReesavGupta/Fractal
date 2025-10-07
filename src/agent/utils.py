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

# global reference to rag service that tools can access
_rag_service_ref = None

def set_rag_service(rag_service):
    """Set the global RAG service reference for tools to use"""
    global _rag_service_ref
    _rag_service_ref = rag_service

def get_tool_list():
    """
    Convert tool functions to LangChain tool objects with proper schemas.
    The @tool decorator automatically creates the schema from function signatures.
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
            # Reduce top_k for faster performance
            results = _rag_service_ref.search(query, top_k=3)
            
            if not results:
                return f"No relevant code found for query: '{query}'"
            
            # Skip LLM reranking for better performance - just return raw results
            response = f"RAG Search Results for '{query}':\n\n"
            response += "Relevant Code Snippets:\n"
            
            for i, doc in enumerate(results, 1):
                source = doc.metadata.get('source_file', 'unknown')
                # Truncate long content for readability
                content = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                response += f"\n{i}. From {source}:\n{content}\n"
            
            return response
            
        except Exception as e:
            return f"Error searching codebase: {str(e)}"


    return [
        read_file_tool,
        write_file_tool,
        read_directory_tool,
        edit_file_tool,
        search_files_tool,
        create_directory_tool,
        delete_file_tool,
        search_codebase_tool
    ]