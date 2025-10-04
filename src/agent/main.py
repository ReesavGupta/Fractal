import sys
import argparse
from tui import FractalAgent

def main():
    """Main entry point for the Fractal CLI Agent"""
    
    parser = argparse.ArgumentParser(
        description="Fractal - To the infinity and beyond!",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            %(prog)s                          # Start with interactive setup
            %(prog)s --llm claude             # Start with Claude
            %(prog)s --llm openai --verbose   # Start with OpenAI in verbose mode
            
            For more information, visit: https://github.com/ReesavGupta/Fractal-CLI
        """
    )
   
    parser.add_argument(
        "--llm", 
        "-l", 
        help="LLM provider to use", 
        choices=["openai", "gemini", "claude"],
        metavar="PROVIDER"
    )
    
    parser.add_argument(
        "--verbose", 
        "-v", 
        action="store_true",
        help="Enable verbose output for debugging"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="Fractal v0.1.0"
    )
    
    args = parser.parse_args()
    
    # initialize the TUI agent
    agent = FractalAgent()
    
    # apply command-line arguments
    if args.llm:
        agent.config['llm'] = args.llm
        print(f"LLM provider preset to: {args.llm}")
    
    if args.verbose:
        agent.config['verbose'] = True
        print("Verbose mode enabled")
    
    try:
        agent.run()
    except KeyboardInterrupt:
        print("\n\nSeeYa Space Cowboy!")
        sys.exit(0)
    except Exception as e:
        print(f"\nFatal error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()