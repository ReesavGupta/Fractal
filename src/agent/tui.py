import os
import sys
from pathlib import Path
from prompt_toolkit.styles import Style
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.completion import WordCompleter
from src.agent.agent import CodingAgent
from dotenv import load_dotenv
from src.rag_service.rag import RAGService
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import SecretStr
from langchain.chat_models import init_chat_model
from src.agent.utils import set_rag_service
from langchain_nomic import NomicEmbeddings

load_dotenv()

google_key = os.getenv("GOOGLE_EMBEDDING_API_KEY")
nomic_key = os.getenv("NOMIC_EMBEDDING_API_KEY")

if not nomic_key:
    raise ValueError("no nomic api key set")
if not google_key:
    raise ValueError("no google api key set")


class Colors:
    """ANSI color codes"""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # Colors
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    GRAY = "\033[90m"
    
    # Backgrounds
    BG_BLUE = "\033[44m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"


class FractalAgent:
    def __init__(self) -> None:
        self.config= {
            'llm' : None,
            'verbose': False,
            'enable_db_tools': False,
            'db_connections': {},  # Store connection strings
            'api_keys': {},
            'embedding_api_keys': {} 
        }
        self.session= None 
        self.running = True
        self.agent = None
        self.rag_service = None

    def print_reasoning(self, text: str):
        """Print agent's reasoning/thinking process"""
        print(f"\n{Colors.DIM}{Colors.BLUE}💭 Thinking: {Colors.RESET}{Colors.DIM}{text}{Colors.RESET}")
    
    def print_tool_call(self, tool_name: str, args: dict):
        """Print tool call information - shows what tool is being called"""
        # Format args nicely - truncate long values
        formatted_args = []
        for k, v in args.items():
            v_str = repr(v)
            if len(v_str) > 60:
                v_str = v_str[:60] + "..."
            formatted_args.append(f"{k}={v_str}")
        
        args_str = ", ".join(formatted_args)
        print(f"{Colors.CYAN}🔧 Calling: {Colors.BOLD}{tool_name}{Colors.RESET}{Colors.CYAN}({args_str}){Colors.RESET}")
    
    def print_response_start(self):
        """Print response header"""
        print(f"\n{Colors.BOLD}{Colors.MAGENTA}🤖 Fractal:{Colors.RESET}")
    
    def setup_tui(self):
        commands = WordCompleter([
            '/llm', '/verbose', '/config', '/clear', '/help', '/quit',
            '/session', '/apikey', '/embedkey', '/reembed',
            '/dbtools', '/dbconnect', '/dblist', '/dbdisconnect',
            'openai', 'gemini', 'claude', 'postgresql', 'mysql', 'mongodb'
        ], ignore_case=True)

        style = Style.from_dict({
            'prompt': '#00ff00 bold',
            'command': '#00d7ff',
            'error': '#ff0000',
            'success': '#00ff00',
            'info': '#ffff00',
        })

        self.session = PromptSession(
            completer=commands,
            style=style,
            complete_while_typing=True,
        )

    def print_banner(self):
       banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   ███████╗██████╗  █████╗  ██████╗████████╗ █████╗ ██╗        ║
║   ██╔════╝██╔══██╗██╔══██╗██╔════╝╚══██╔══╝██╔══██╗██║        ║
║   █████╗  ██████╔╝███████║██║        ██║   ███████║██║        ║
║   ██╔══╝  ██╔══██╗██╔══██║██║        ██║   ██╔══██║██║        ║
║   ██║     ██║  ██║██║  ██║╚██████╗   ██║   ██║  ██║███████╗   ║
║   ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝   ║
║                                                               ║
║              To the infinity and beyond!                      ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
        """       
       print(banner)
       print("\nType /help for available commands or start typing your prompt\n")
        
    def print_help(self):
        """Display help information"""
        help_text = """
┌─────────────────────────────────────────────────────────────┐
│ Available Commands:                                         │
├─────────────────────────────────────────────────────────────┤
│ /llm <provider>    - Set LLM provider                       │
│                      Options: openai, gemini, claude        │
│─────────────────────────────────────────────────────────────│
│ /verbose           - Toggle verbose output                  │
│─────────────────────────────────────────────────────────────│
│ /dbtools           - Toggle database tools                  │
│─────────────────────────────────────────────────────────────│
│ /dbconnect <type> <connection_string> [alias]               │
│                    - Add database connection                │
│                      Types: postgres, mysql, mongodb        │
│─────────────────────────────────────────────────────────────│
│ /dblist            - List saved database connections        │
│─────────────────────────────────────────────────────────────│
│ /dbdisconnect <type> [alias] - Remove database connection   │
│─────────────────────────────────────────────────────────────│
│ /config            - Show current configuration             │
│─────────────────────────────────────────────────────────────│
│ /apikey <provider> <key>  - Set API key for LLM provider    │
│─────────────────────────────────────────────────────────────│
│ /embedkey <provider> <key> - Set API key for embedding      │
│─────────────────────────────────────────────────────────────│
│ /reembed           - Re-Index the project codebase          │
│─────────────────────────────────────────────────────────────│
│ /clear             - Clear screen                           │
│─────────────────────────────────────────────────────────────│
│ /help              - Show this help message                 │
│─────────────────────────────────────────────────────────────│
│ /exit or /quit     - Exit Fractal                           │
└─────────────────────────────────────────────────────────────┘
        """
        print(help_text)

    def print_config(self):
        """Display current configuration"""
        print("\n┌─────────────── Current Configuration ───────────────┐")
        print(f"│ LLM Provider: {self.config['llm'] or 'Not set':<37} │")
        print(f"│ Verbose Mode: {'Enabled' if self.config['verbose'] else 'Disabled':<37} │")
        print(f"│ Database Tools: {'Enabled' if self.config['enable_db_tools'] else 'Disabled':<35} │")
        
        db_count = len(self.config['db_connections'])
        print(f"│ DB Connections: {db_count:<36} │")
        
        agent_status = 'Initialized' if self.agent else 'Not initialized'
        print(f"│ Agent Status: {agent_status:<37} │")
        print("└─────────────────────────────────────────────────────┘\n")
    
    async def initialize_agent(self):
        """Initialize the coding agent with current configuration"""
        if not self.config['llm']:
            print("Error: No LLM provider set. Use /llm <provider> first.")
            return False
        
        try:
            print(f"\nInitializing {self.config['llm'].upper()} agent...")

            api_key = self.config['api_keys'].get(self.config['llm'])

            rag_service = None
            if not hasattr(self, 'rag_service') or not self.rag_service:
                try:
                    embed_key = self.config['embedding_api_keys'].get('gemini')
                    google_api_key = google_key or embed_key

                    if google_api_key:
                        embedding_model = NomicEmbeddings(
                            nomic_api_key=nomic_key,
                            dimensionality=768,
                            model="nomic-embed-text-v1.5"
                        )

                        temp_llm = init_chat_model(
                            "gpt-4o", 
                            model_provider="openai", 
                            temperature=0.2, 
                            api_key=os.getenv("OPENAI_API_KEY")
                        )
                        
                        project_path = os.getcwd()
                        
                        self.rag_service = RAGService(
                            llm=temp_llm,
                            embedding_model=embedding_model,
                            project_name=Path(project_path).name
                        )
                        
                        index_file = Path(project_path) / ".fractal_index.json"
                        if not index_file.exists():
                            print("First time setup: Indexing codebase...")
                            await self.rag_service.index_codebase(project_path)
                        else:
                            print("Loading existing codebase index...")
                            self.rag_service._rebuild_retrievers()
                        
                        rag_service = self.rag_service
                        print("✓ RAG service initialized successfully!")
                except Exception as e:
                    print(f"Warning: Could not initialize RAG service: {e}")
                    print("Agent will work without RAG capabilities.")
            else:
                rag_service = self.rag_service

            self.agent = CodingAgent(
                llm=self.config['llm'],
                api_key=api_key,
                verbose=self.config['verbose'],
                rag_service=rag_service,
                enable_db_tools=self.config['enable_db_tools']
            )

            if rag_service:
                set_rag_service(rag_service)

            print(f"Agent initialized successfully!")

            # Auto-connect to saved databases if DB tools are enabled
            if self.config['enable_db_tools'] and self.config['db_connections']:
                print(f"\nAuto-connecting to {len(self.config['db_connections'])} saved database(s)...")
                for conn_key, conn_string in self.config['db_connections'].items():
                    db_type, alias = conn_key.split('_', 1)
                    try:
                        # Use the agent to connect
                        if db_type == 'postgres':
                            result = await self.agent.ainvoke(f"Connect to PostgreSQL using connection string: {conn_string} with alias {alias}")
                        elif db_type == 'mysql':
                            result = await self.agent.ainvoke(f"Connect to MySQL using connection string: {conn_string} with alias {alias}")
                        elif db_type == 'mongodb':
                            result = await self.agent.ainvoke(f"Connect to MongoDB using connection string: {conn_string} with alias {alias}")
                        print(f"  • {db_type.upper()} ({alias}): Connected")
                    except Exception as e:
                        print(f"  • {db_type.upper()} ({alias}): Failed - {e}")
            
            return True

        except ValueError as e:
            print(f"Error: {str(e)}")
            print("Make sure you have set the appropriate API key environment variable.")
            return False
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return False

    async def handle_command(self, cmd):
        """Process user commands"""
        parts = cmd.strip().split()
        
        if not parts:
            return True
            
        command = parts[0].lower()

        if command in ['/quit']:
            print("\nThanks for using Fractal! See you soon in space!\n")
            return False

        elif command == '/help':
            self.print_help()

        elif command == '/clear':
            print("\033[2J\033[H", end="")

        elif command == '/config':
            self.print_config()
            
        elif command == '/verbose':
            self.config['verbose'] = not self.config['verbose']
            status = "enabled" if self.config['verbose'] else "disabled"
            print(f"Verbose mode {status}")

        elif command == '/dbtools':
            self.config['enable_db_tools'] = not self.config['enable_db_tools']
            status = "enabled" if self.config['enable_db_tools'] else "disabled"
            print(f"Database tools {status}")
            if self.agent:
                print("Note: Reinitialize agent to apply database tools changes")

        elif command == '/dbconnect':
            if len(parts) < 3:
                print("Usage: /dbconnect <type> <connection_string> [alias]")
                print("Examples:")
                print("  /dbconnect postgres postgresql://user:pass@localhost:5432/mydb prod")
                print("  /dbconnect mysql mysql://user:pass@localhost:3306/mydb")
                print("  /dbconnect mongodb mongodb://localhost:27017/mydb local")
            else:
                db_type = parts[1].lower()
                conn_string = parts[2]
                alias = parts[3] if len(parts) > 3 else "default"
                
                if db_type not in ['postgres', 'mysql', 'mongodb']:
                    print("Error: Supported types are postgres, mysql, mongodb")
                else:
                    conn_key = f"{db_type}_{alias}"
                    self.config['db_connections'][conn_key] = conn_string
                    print(f"✓ Saved {db_type.upper()} connection '{alias}'")
                    
                    if not self.config['enable_db_tools']:
                        print("Note: Database tools are disabled. Use /dbtools to enable")

        elif command == '/dblist':
            if not self.config['db_connections']:
                print("No database connections saved")
            else:
                print("\nSaved Database Connections:")
                for conn_key in self.config['db_connections'].keys():
                    db_type, alias = conn_key.split('_', 1)
                    print(f"  • {db_type.upper()} - {alias}")
                print()

        elif command == '/dbdisconnect':
            if len(parts) < 2:
                print("Usage: /dbdisconnect <type> [alias]")
            else:
                db_type = parts[1].lower()
                alias = parts[2] if len(parts) > 2 else "default"
                conn_key = f"{db_type}_{alias}"
                
                if conn_key in self.config['db_connections']:
                    del self.config['db_connections'][conn_key]
                    print(f"✓ Removed {db_type.upper()} connection '{alias}'")
                else:
                    print(f"No connection found for {db_type} with alias '{alias}'")

        elif command == '/apikey':
            if len(parts) < 3:
                print("Usage: /apikey <provider> <key>")
            else:
                provider = parts[1].lower()
                key = parts[2]
                if provider not in ['openai', 'gemini', 'claude']:
                    print("Error: Supported providers are openai, gemini, claude")
                else:
                    self.config['api_keys'][provider] = key
                    print(f"API key set for {provider}")

        elif command == '/embedkey':
            if len(parts) < 3:
                print("Usage: /embedkey <provider> <key>")
            else:
                provider = parts[1].lower()
                key = parts[2]
                self.config['embedding_api_keys'][provider] = key
                print(f"Embedding API key set for {provider}")

        elif command == '/llm':
            if len(parts) < 2:
                print("Error: Please specify an LLM provider (openai, gemini, claude)")
            elif parts[1].lower() in ['openai', 'gemini', 'claude']:
                self.config['llm'] = parts[1].lower()
                print(f"LLM provider set to: {parts[1].lower()}")
                if self.agent:
                    await self.initialize_agent()
            else:
                print("Error: Invalid LLM provider. Choose from: openai, gemini, claude")

        elif command == '/reembed':
            if not self.agent:
                print("Agent not initialized.")
            else:
                project_path = os.getcwd()
                if not hasattr(self.agent, "rag_service") or not self.agent.rag_service:
                    print("Error: RAG service not available")
                else:
                    print("Re-indexing codebase...")
                    self.agent.rag_service.reembed_changed_files(project_path)

        elif not command.startswith('/'):
            # Process user query through agent with streaming
            if not self.config['llm']:
                print("Warning: No LLM provider set. Use /llm <provider> to configure.")
                return True
            
            if not self.agent:
                if not await self.initialize_agent():
                    return True
            
            try:
                response_started = False
                
                if not self.agent:
                    raise ValueError("agent not initialized")

                async for event in self.agent.astream(cmd):
                    event_type = event.get("type")
                    content = event.get("content", "")
                    metadata = event.get("metadata", {})
                    
                    if event_type == "reasoning":
                        self.print_reasoning(content)
                    
                    elif event_type == "tool_call":
                        tool_name = content
                        args = metadata.get("args", {})
                        self.print_tool_call(tool_name, args)
                    
                    elif event_type == "tool_result":
                        # Don't display tool results anymore - only tool calls
                        pass
                    
                    elif event_type == "response_start":
                        if not response_started:
                            self.print_response_start()
                            response_started = True
                    
                    elif event_type == "response_token":
                        sys.stdout.write(f"{Colors.RESET}{content}")
                        sys.stdout.flush()
                
                if response_started:
                    print()
                
            except Exception as e:
                print(f"\n{Colors.RED}Error processing request: {str(e)}{Colors.RESET}\n")
                if self.config['verbose']:
                    import traceback
                    traceback.print_exc()

        else:
            print(f"Unknown command: {command}. Type /help for available commands.")
            
        return True

    async def run(self):
        """Main TUI loop"""
        self.setup_tui()
        self.print_banner()

        if self.session is None:
            print("Error: TUI session could not be initialized.")
            return

        try:
            ##############################################################################################################
            # periodic reembedding 
            project_path = os.getcwd()
            asyncio.create_task(periodic_reembedding(self, project_path, interval=600))
            print("Background re-embedding started (every 10 min)")
            ##############################################################################################################
        except Exception as e:
            print(f"Failed to start background re-embedding: {e}")

        try:
            while self.running:
                try:
                    user_input = await self.session.prompt_async(
                        HTML('<prompt>fractal ▶</prompt> '),
                    )
                    
                    if user_input.strip():
                        self.running = await self.handle_command(user_input)
                        
                except KeyboardInterrupt:
                    print("\nUse /exit or /quit to exit Fractal")
                    continue
                except EOFError:
                    break
                    
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            if self.config['verbose']:
                import traceback
                traceback.print_exc()
        finally:
            print()

import asyncio

async def periodic_reembedding(agent, path, interval=600):
    """Periodically re-embed changed files in the background"""
    while agent.running:
        try:
            if hasattr(agent, "rag_service") and agent.rag_service:
                # First cleanup deleted files
                agent.rag_service.cleanup_deleted_files(path)
                # Then re-embed changed files
                agent.rag_service.reembed_changed_files(path)
        except Exception as e:
            print(f"Reembedding error: {e}")
        await asyncio.sleep(interval)

        