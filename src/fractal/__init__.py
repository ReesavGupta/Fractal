"""
Fractal CLI Agent Package

An intelligent coding assistant with memory and database integration.
"""

__version__ = "0.1.0"
__author__ = "Reesav Gupta"
__email__ = "your-email@example.com"

# Main exports - lazy imports to avoid initialization errors
def get_CodingAgent():
    from .agent.agent import CodingAgent
    return CodingAgent

def get_FractalAgent():
    from .agent.tui import FractalAgent
    return FractalAgent

def get_RAGService():
    from .rag_service.rag import RAGService
    return RAGService

__all__ = [
    "get_CodingAgent",
    "get_FractalAgent", 
    "get_RAGService",
    "__version__",
    "__author__",
    "__email__"
]
