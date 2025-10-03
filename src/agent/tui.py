from pydantic import BaseModel
from prompt_toolkit.styles import Style
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.completion import WordCompleter

class FractalAgent:
    def __init__(self) -> None:
        self.config= {
            'llm' : None,
            'verbose': False,
            'mcp': []
        }
        self.session= None 
        self.running = True

    
    def setup_tui(self):
        commands = WordCompleter([
            '/llm', '/verbose', '/mcp', '/config', '/clear', '/help', '/quit',
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
│ /verbose           - Toggle verbose output                  │
│ /mcp <add|remove> <db> - Manage MCP databases               │
│                      Options: postgresql, mongodb           │
│ /config            - Show current configuration             │
│ /clear             - Clear screen                           │
│ /help              - Show this help message                 │
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
        print("└─────────────────────────────────────────────────────┘\n")

    def handle_command(self, cmd):
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
            
        elif command == '/llm':
            if len(parts) < 2:
                print("Error: Please specify an LLM provider (openai, gemini, claude)")

            elif parts[1].lower() in ['openai', 'gemini', 'claude']:
                self.config['llm'] = parts[1].lower()
                print(f"LLM provider set to: {parts[1].lower()}")
            else:
                print("Error: Invalid LLM provider. Choose from: openai, gemini, claude")
                
        elif command == '/verbose':
            self.config['verbose'] = not self.config['verbose']
            if self.config['verbose']:
                status = "enabled"
            else:
                status = "disabled"
            print(f"Verbose mode {status}")
            
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
            # things i need to do here:- 
            # check if there is a api key for the set llm or not
            # if not ask the user to input their api key and store it securely
            # use the api key for the agent invoke the agent with the user query
            if not self.config['llm']:
                print("Warning: No LLM provider set. Use /llm <provider> to configure.")
                return True
            
            print(f"\nProcessing with {self.config['llm'].upper()}...")
            if self.config['verbose']:
                print(f"Prompt: {cmd}")
                print(f"Config: {self.config}")
            print("[AI response would appear here]\n")
            
        else:
            print(f"Unknown command: {command}. Type /help for available commands.")
            
        return True

    def run(self):
        """Main TUI loop"""
        self.setup_tui()
        self.print_banner()

        if self.session is None:
            print("Error: TUI session could not be initialized.")
            return
        
        try:
            while self.running:
                try:
                    user_input = self.session.prompt(
                        HTML('<prompt>fractal ▶</prompt> '),
                    )
                    
                    if user_input.strip():
                        self.running = self.handle_command(user_input)
                        
                except KeyboardInterrupt:
                    print("\nUse /exit or /quit to exit Fractal")
                    continue
                except EOFError:
                    break
                    
        except Exception as e:
            print(f"\nAn error occurred: {e}")
        finally:
            print()