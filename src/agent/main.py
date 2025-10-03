import argparse
from tui import FractalAgent

def main():
    """main entry point for the CLI Agent"""
    parser = argparse.ArgumentParser(description="Fractal - To the infinity :D")
    
    parser.add_argument("--llm", "-l", help="LLM provider to user", choices=["openai", "gemini", "claude"])
    parser.add_argument("--verbose", "-v", help="Enable verbose output")

    args = parser.parse_args()

    agent = FractalAgent()

    if args.llm:
        agent.config['llm'] = args.llm
    if args.verbose:
        agent.config['verbose'] = True

    agent.run()

if __name__ == "__main__":
    main()