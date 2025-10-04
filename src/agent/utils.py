from langchain_core.tools import tool
from typing import Optional
from tools import (
    read_file,
    write_file,
    read_dir,
    edit_file,
    search_fs_using_regex,
    create_directory,
    delete_file
)


def get_tool_list():
    """
    Convert tool functions to LangChain tool objects with proper schemas.
    The @tool decorator automatically creates the schema from function signatures.
    """
    
    @tool
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
    def search_files_tool(
        dir_path: str, 
        pattern: str, 
        file_extension: Optional[str] = None
    ) -> str:
        """
        Search for files matching a regex pattern in their content.
        
        Args:
            dir_path: Directory to search in
            pattern: Regex pattern to search for
            file_extension: Optional file extension filter (e.g., '.py', '.js')
            
        Returns:
            Search results showing matching files and lines
        """
        return search_fs_using_regex(dir_path, pattern, file_extension)
    
    @tool
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
    def delete_file_tool(file_path: str) -> str:
        """
        Delete a file.
        
        Args:
            file_path: Path to the file to delete
            
        Returns:
            Success message or error
        """
        return delete_file(file_path)
    
    return [
        read_file_tool,
        write_file_tool,
        read_directory_tool,
        edit_file_tool,
        search_files_tool,
        create_directory_tool,
        delete_file_tool
    ]