import re
import os
from pathlib import Path

def read_file(file_path: str) -> str:
    """Read and return the contents of the file specified in the file_path"""
    try:
        path = Path(file_path)

        if not path.is_file():
            return f"Error: '{file_path}' is not a file"

        with open(path, 'r', encoding="utf-8") as f:
            content = f.read()

        return f"Sucessfully read file '{file_path}':\n\n{content}"

    except UnicodeDecodeError:
        return f"Error: Could not decode the file '{file_path}' as UTF-8. It may be a binary file."
    except PermissionError:
        return f"Error: Permission denied to read the file '{file_path}'"
    except Exception as e:
        return f"Error reading the file: {str(e)}"

def write_file(file_path: str, content: str) -> str:
    """Write content to a file, creating it if it doesn't exist"""
    try:
        path = Path(file_path)

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        
        return f"Successfully wrote {len(content)} characters to '{file_path}'"
    
    except PermissionError:
        return f"Error: Permission denied to write to file '{file_path}'"
    except Exception as e:
        return f"Error writing file: {str(e)}"
    
def read_dir(dir_path: str, recursive: bool = False) -> str:
    """List all the files and directories in a directory"""
    try:
        path = Path(dir_path)
        
        if not path.exists():
            return f"Error: Directory '{dir_path}' does not exist"
        
        if not path.is_dir():
            return f"Error: '{dir_path}' is not a directory"
        
        result = [f"Contents of '{dir_path}':\n"]

        if recursive:
            for root, dirs, files in os.walk(path):
                level = root.replace(str(path), '').count(os.sep)
                indent = '  ' * level
                result.append(f"{indent}{os.path.basename(root)}/")
                sub_indent = '  ' * (level + 1)
                for file in files:
                    result.append(f"{sub_indent}{file}")
        else:
            items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
            for item in items:
                if item.is_dir():
                    result.append(f"  [DIR]  {item.name}/")
                else:
                    size = item.stat().st_size
                    result.append(f"  [FILE] {item.name} ({size} bytes)")
        
        return "\n".join(result)

    except PermissionError:
        return f"Error: Permission denied to access directory '{dir_path}'"
    except Exception as e:
        return f"Error reading directory: {str(e)}"

def edit_file(old_text: str, new_text: str, file_path: str) -> str:
    """Edit a file by replacing old_text with new_text"""
    try:
        path = Path(file_path)

        if not path.exists():
            return f"Error: File '{file_path}' does not exist"
        
        content = ""
        with open(path, "r", encoding="utf-8") as f:
            content += f.read()

        if old_text not in content:
            return f"Error: Text to replace not found in '{file_path}'"

        occurrences = content.count(old_text)
        new_content = content.replace(old_text, new_text)

        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)

        return f"Successfully replaced {occurrences} occurrence(s) in '{file_path}'"
        
    except UnicodeDecodeError:
        return f"Error: Could not decode file '{file_path}' as UTF-8"
    except PermissionError:
        return f"Error: Permission denied to edit file '{file_path}'"
    except Exception as e:
        return f"Error editing file: {str(e)}"

def search_fs_using_regex(dir_path: str, pattern: str, file_extension: str | None = None) -> str:
    """Search for files matching a regex pattern in their content"""
    try:
        path = Path(dir_path)
        
        if not path.exists():
            return f"Error: Directory '{dir_path}' does not exist"
        
        if not path.is_dir():
            return f"Error: '{dir_path}' is not a directory"
        
        regex = re.compile(pattern)
        matches = []
        
        # Define directories and files to exclude
        excluded_dirs = {
            '.git', '.svn', '.hg',  # Version control
            'node_modules', 'bower_components',  # Package managers
            '__pycache__', '.pytest_cache', '.mypy_cache',  # Python cache
            '.venv', 'venv', 'env', '.env',  # Virtual environments
            'dist', 'build', '.next', '.nuxt',  # Build outputs
            'coverage', '.nyc_output',  # Test coverage
            '.DS_Store', 'Thumbs.db',  # System files
            'logs', 'log',  # Log files
            'tmp', 'temp', '.tmp', '.temp',  # Temporary files
            'cache', '.cache',  # Cache directories
            'target',  # Rust/Java build
            '.idea', '.vscode',  # IDE files
            'vendor',  # PHP dependencies
            '.terraform',  # Terraform
            '.gradle',  # Gradle
            'bin', 'obj',  # .NET build
        }
        
        excluded_files = {
            '.env', '.env.local', '.env.production', '.env.development',
            '.gitignore', '.gitattributes', '.gitmodules',
            'package-lock.json', 'yarn.lock', 'composer.lock',
            '*.log', '*.tmp', '*.temp', '*.cache',
            '.DS_Store', 'Thumbs.db', 'desktop.ini',
            '*.pyc', '*.pyo', '*.pyd', '__pycache__',
            '*.class', '*.jar', '*.war',
            '*.exe', '*.dll', '*.so', '*.dylib',
            '*.pdf', '*.doc', '*.docx', '*.xls', '*.xlsx',
            '*.zip', '*.tar', '*.gz', '*.rar', '*.7z',
            '*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.svg',
            '*.mp3', '*.mp4', '*.avi', '*.mov', '*.wav',
        }
        
        for root, dirs, files in os.walk(path):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if d not in excluded_dirs]
            
            for file in files:
                # Skip excluded files
                if any(file.endswith(ext.replace('*', '')) for ext in excluded_files):
                    continue
                
                # Skip hidden files (starting with .)
                if file.startswith('.'):
                    continue
                
                # Filter by extension if specified
                if file_extension and not file.endswith(file_extension):
                    continue
                
                file_path = Path(root) / file
                
                # Skip if path contains excluded directories
                path_parts = file_path.parts
                if any(part in excluded_dirs for part in path_parts):
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    found_matches = regex.finditer(content)
                    match_list = list(found_matches)
                    
                    if match_list:
                        matches.append(f"\n{file_path}:")
                        for i, match in enumerate(match_list[:5], 1):  # Limit to 5 matches per file
                            line_num = content[:match.start()].count('\n') + 1
                            matches.append(f"  Line {line_num}: {match.group()}")
                        
                        if len(match_list) > 5:
                            matches.append(f"  ... and {len(match_list) - 5} more matches")
                
                except (UnicodeDecodeError, PermissionError):
                    continue  # Skip files that can't be read
        
        if not matches:
            ext_info = f" with extension '{file_extension}'" if file_extension else ""
            return f"No matches found for pattern '{pattern}' in '{dir_path}'{ext_info}"
        
        result = f"Found matches for pattern '{pattern}':" + ''.join(matches)
        return result
    
    except re.error as e:
        return f"Error: Invalid regex pattern: {str(e)}"
    except Exception as e:
        return f"Error searching files: {str(e)}"
    
def create_directory(dir_path: str) -> str:
    """Create a new directory"""
    try:
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        return f"Successfully created directory '{dir_path}'"
    except PermissionError:
        return f"Error: Permission denied to create directory '{dir_path}'"
    except Exception as e:
        return f"Error creating directory: {str(e)}"

def delete_file(file_path: str) -> str:
    """Delete a file"""
    try:
        path = Path(file_path)
        
        if not path.exists():
            return f"Error: File '{file_path}' does not exist"
        
        if not path.is_file():
            return f"Error: '{file_path}' is not a file (use appropriate tool for directories)"
        
        path.unlink()
        return f"Successfully deleted file '{file_path}'"
    
    except PermissionError:
        return f"Error: Permission denied to delete file '{file_path}'"
    except Exception as e:
        return f"Error deleting file: {str(e)}"