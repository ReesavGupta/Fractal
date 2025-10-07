import os
from pathlib import Path
from prompt_toolkit.styles import Style
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.completion import WordCompleter
from src.agent.agent import CodingAgent #type:ignore
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

class FractalAgent:
    def __init__(self) -> None:
        self.config= {
            'llm' : None,
            'verbose': False,
            'mcp': [],
            'api_keys': {},
            'embedding_api_keys': {} 
        }
        self.session= None 
        self.running = True
        self.agent = None

    
    def setup_tui(self):
        commands = WordCompleter([
            '/llm', '/verbose', '/mcp', '/config', '/clear', '/help', '/quit',
            '/session', '/apikey', '/embedkey',
            'openai', 'gemini', 'claude', 'postgresql', 'mongodb'
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
|-------------------------------------------------------------|
│ /verbose           - Toggle verbose output                  │
|-------------------------------------------------------------|
│ /mcp <add|remove> <db> - Manage MCP databases               │
│                      Options: postgresql, mongodb           │
|-------------------------------------------------------------|
│ /config            - Show current configuration             │
|-------------------------------------------------------------|
│ /apikey <provider> <key>  - Set API key for LLM provider    │
|-------------------------------------------------------------|
│ /embedkey <provider> <key> - Set API key for embedding model│
|-------------------------------------------------------------|
│ /reembed           - Re-Index the project codebase          │
|-------------------------------------------------------------|
│ /clear             - Clear screen                           │
|-------------------------------------------------------------|
│ /help              - Show this help message                 │
|-------------------------------------------------------------|
│ /exit or /quit     - Exit Fractal                           │
└─────────────────────────────────────────────────────────────┘
        """
        print(help_text)

    def print_config(self):
        """Display current configuration"""
        print("\n┌─────────────── Current Configuration ───────────────┐")
        print(f"│ LLM Provider: {self.config['llm'] or 'Not set':<37} │")
        print(f"│ Verbose Mode: {'Enabled' if self.config['verbose'] else 'Disabled':<37} │")
        mcp_str = ', '.join(self.config['mcp']) if self.config['mcp'] else 'None'
        print(f"│ MCP Databases: {mcp_str:<36} │")
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

            # Initialize RAG service if not already done
            rag_service = None
            if not hasattr(self, 'rag_service') or not self.rag_service:
                try:
                    embed_key = self.config['embedding_api_keys'].get('gemini')
                    google_api_key = google_key or embed_key

                    if google_api_key:
                        # embedding_model = GoogleGenerativeAIEmbeddings(
                        #     model="gemini-embedding-001", 
                        #     google_api_key=SecretStr(google_api_key)
                        # )
                        
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
                        
                        # Check if index exists, if not do initial indexing
                        index_file = Path(project_path) / ".fractal_index.json"
                        if not index_file.exists():
                            print("First time setup: Indexing codebase...")
                            await self.rag_service.index_codebase(project_path)
                        else:
                            print("Loading existing codebase index...")
                            # Still rebuild retrievers with existing data
                            self.rag_service._rebuild_retrievers()
                        
                        rag_service = self.rag_service
                        print("✓ RAG service initialized successfully!")
                except Exception as e:
                    print(f"Warning: Could not initialize RAG service: {e}")
                    print("Agent will work without RAG capabilities.")
            else:
                rag_service = self.rag_service

            # Initialize the agent
            self.agent = CodingAgent(
                llm=self.config['llm'],
                api_key=api_key,
                verbose=self.config['verbose'],
                rag_service=rag_service
            )

            # FIXED: Set RAG service reference globally before tools are created
            if rag_service:
                set_rag_service(rag_service)

            print(f"Agent initialized successfully!")
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
            # self.print_banner()


        elif command == '/config':
            self.print_config()
            
        elif command == '/verbose':
            self.config['verbose'] = not self.config['verbose']
            if self.config['verbose']:
                status = "enabled"
            else:
                status = "disabled"
            print(f"Verbose mode {status}")

        
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
               
                if not hasattr(self.agent, "rag_service"):
                    embed_key = self.config['embedding_api_keys'].get('gemini')
                    
                    if not google_key:
                        raise ValueError()
                        
                    google_api_key = google_key or embed_key
                    
                    embedding_model = GoogleGenerativeAIEmbeddings(
                        model="gemini-embedding-001", google_api_key=SecretStr(google_api_key)
                    )
                    self.agent.rag_service = RAGService(
                        llm=self.agent.client, 
                        embedding_model=embedding_model,
                        project_name=Path(project_path).name
                    )

                if self.agent.rag_service:
                    self.agent.rag_service.reembed_changed_files(project_path)   

        elif command == '/mcp':
            if len(parts) < 3:
                print("Error: Usage: /mcp <add|remove> <postgresql|mongodb>")
            else:
                action = parts[1].lower()
                db = parts[2].lower()
                
                if db not in ['postgresql', 'mongodb']:
                    print("Error: Invalid database. Choose from: postgresql, mongodb")

                elif action == 'add':
                    if db not in self.config['mcp']:
                        self.config['mcp'].append(db)
                        print(f"Added {db} to MCP")
                    else:
                        print(f"{db} is already in MCP")
                elif action == 'remove':
                    if db in self.config['mcp']:
                        self.config['mcp'].remove(db)
                        print(f"Removed {db} from MCP")
                    else:
                        print(f"{db} is not in MCP")
                else:
                    print("Error: Invalid action. Use 'add' or 'remove'")

 
        #####################################################################
        # here if input does not start with "/" then we process here#
        ##################################################################### 
        elif not command.startswith('/'):
            # Process user query through agent
            if not self.config['llm']:
                print("Warning: No LLM provider set. Use /llm <provider> to configure.")
                return True
            
            if not self.agent:
                if not await self.initialize_agent():
                    return True
            
            # Process the query
            try:
                print(f"\nProcessing with {self.config['llm'].upper()}...\n")
                if self.agent:
                    response = await self.agent.ainvoke(cmd)
                    # response = ""
                    # async for token in self.agent.astream(cmd):
                    #     await asyncio.sleep(10)
                    #     print(str(token))
                    print(f"\n{response}\n")
            except Exception as e:
                print(f"\nError processing request: {str(e)}\n")
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
        # --------------------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------------------------
        #   here we are creating a async task so that we can periodically reembed the codebase
        # --------------------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------------------------
        try:
            project_path = os.getcwd()
            asyncio.create_task(periodic_reembedding(self, project_path, interval=600))
            print("Background re-embedding started (every 10 min)")
        except Exception as e:
            print(f"Failed to start background re-embedding: {e}")

        # --------------------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------------------------

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


# ----------------------------------------------------------------------
# periodic re-embedding background task
# ----------------------------------------------------------------------
import asyncio

async def periodic_reembedding(agent, path, interval=600):
    """Periodically re-embed changed files in the background"""
    while agent.running:
        try:
            if hasattr(agent, "rag_service"):
                agent.rag_service.reembed_changed_files(path)
        except Exception as e:
            print(f"Reembedding error: {e}")
        await asyncio.sleep(interval)