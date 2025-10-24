import asyncio
import sys
import argparse
import warnings

# Silence Pydantic deprecation warnings coming from dependencies (e.g. langsmith/langchain)
# These libraries may still call the deprecated `dict()` API on Pydantic models.
# Prefer upgrading dependencies long-term; this silences the noisy warning for now.
try:
    # pydantic v2 exposes a specific warning class in different places depending on version
    try:
        from pydantic.errors import PydanticDeprecatedSince20 as _PydWarn
    except Exception:
        from pydantic import PydanticDeprecatedSince20 as _PydWarn
except Exception:
    _PydWarn = DeprecationWarning

warnings.filterwarnings("ignore", category=_PydWarn)
# Also ignore generic deprecation-style warnings emitted from langsmith/langchain internals.
# Use multiple filters (by module and by message) to catch different warning subclasses
# emitted by dependency packages.
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"langsmith.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"langchain.*")
# Some libraries create their own PydanticDeprecatedSince20 class; match the deprecation message too
warnings.filterwarnings(
    "ignore",
    message=r".*`dict` method is deprecated; use `model_dump` instead.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*model_dump.*",
)

# As an extra safeguard, intercept the global showwarning handler and suppress
# warnings that match the Pydantic deprecation text coming from dependencies.
_orig_showwarning = warnings.showwarning
def _filtered_showwarning(message, category, filename, lineno, file=None, line=None):
    try:
        text = str(message)
        if 'dict' in text and 'model_dump' in text:
            # drop this Pydantic deprecation warning
            return
        if 'PydanticDeprecatedSince20' in getattr(category, '__name__', ''):
            return
    except Exception:
        pass
    return _orig_showwarning(message, category, filename, lineno, file=file, line=line)

warnings.showwarning = _filtered_showwarning

from .tui import FractalAgent
from dotenv import load_dotenv
# import nest_asyncio

load_dotenv()

async def main():
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
        # asyncio 
        await agent.run()
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
    asyncio.run(main())