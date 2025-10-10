# Fractal CLI Agent

![Fractal Logo](public/image.png)

> **To the infinity and beyond!** 🚀

Fractal is a sophisticated CLI agent built with LangChain and LangGraph, providing comprehensive AI-driven assistance directly from your terminal. It features advanced RAG capabilities, multi-database support, intelligent memory management, and a beautiful interactive TUI.

## 🎯 Core Features

### **🤖 Multi-LLM Support**
- **OpenAI GPT-4o** - Advanced reasoning and code generation
- **Anthropic Claude Opus** - Excellent for complex analysis
- **Google Gemini 2.0 Flash** - Fast and efficient responses
- Runtime switching between providers with `/llm` command

### **🧠 Advanced RAG System**
- **Hybrid Retrieval**: Dense embeddings (Nomic) + Sparse BM25 + LLM reranking
- **Real-time Codebase Indexing**: Continuous delta re-indexing with change detection
- **Background Re-embedding**: Automatic updates every 10 minutes
- **Smart Chunking**: 800-character chunks with 150-character overlap for optimal code understanding

### **💾 Intelligent Memory Management**
- **Importance-based Storage**: Automatic cleanup with 50KB default limit
- **Context-aware Filtering**: Smart message filtering for optimal performance
- **Conversation History**: Searchable memory with relevance scoring
- **Automatic Summarization**: Large tool results stored efficiently

### **🗄️ Database Integration**
- **Multi-Database Support**: PostgreSQL, MySQL, MongoDB
- **Connection Management**: Persistent connections with aliasing
- **Full CRUD Operations**: Query, insert, update, delete with async performance
- **MCP Integration**: Model Context Protocol for seamless database operations

### **🎨 Interactive TUI**
- **Streaming Responses**: Real-time reasoning display with tool call transparency
- **Command Completion**: Intelligent autocomplete with suggestions
- **Colored Output**: Status indicators and syntax highlighting
- **Configuration Management**: Runtime settings adjustment

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Installation

```bash
# Clone the repository
git clone https://github.com/ReesavGupta/Fractal-CLI.git
cd Fractal-CLI

# Install dependencies with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Environment Setup

Create a `.env` file in the project root:

```env
# LLM API Keys
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GEMINI_API_KEY=your_google_api_key

# Embedding API Keys (Required for RAG)
GOOGLE_EMBEDDING_API_KEY=your_google_api_key
NOMIC_EMBEDDING_API_KEY=your_nomic_api_key

# Vector Database (Required for RAG)
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
```

### Usage

```bash
# Start with interactive setup
python -m src.agent.main

# Start with specific LLM provider
python -m src.agent.main --llm openai
python -m src.agent.main --llm claude
python -m src.agent.main --llm gemini

# Enable verbose mode for debugging
python -m src.agent.main --verbose

# Show version
python -m src.agent.main --version
```

### Interactive Commands

Once Fractal is running, use these commands:

```bash
# LLM Management
/llm openai          # Switch to OpenAI
/llm claude          # Switch to Claude
/llm gemini          # Switch to Gemini

# Database Operations
/dbtools             # Enable database tools
/dbconnect postgres postgresql://user:pass@localhost:5432/dbname prod
/dbconnect mysql mysql://user:pass@localhost:3306/dbname
/dbconnect mongodb mongodb://localhost:27017/mydb local
/dblist              # List saved connections
/dbdisconnect postgres prod

# Configuration
/config              # Show current settings
/verbose             # Toggle debug output
/apikey openai sk-... # Set API key
/embedkey google AI... # Set embedding key

# RAG Operations
/reembed             # Re-index codebase
/clear               # Clear screen
/help                # Show help
/quit                # Exit Fractal
```

## 🏗️ Project Architecture

```
Fractal/
├── src/
│   ├── agent/                    # Core agent implementation
│   │   ├── agent.py             # LangGraph-based CodingAgent
│   │   ├── main.py              # CLI entry point with async execution
│   │   ├── state.py             # Pydantic state management
│   │   ├── tools.py             # File system operations
│   │   ├── tui.py               # Interactive terminal interface
│   │   ├── utils.py             # Tool integration & coordination
│   │   ├── memory.py            # Conversation memory management
│   │   └── memory_filter.py     # Message filtering & optimization
│   ├── rag_service/             # Advanced RAG implementation
│   │   └── rag.py               # Hybrid retrieval with Qdrant + BM25
│   └── mcp/                     # Model Context Protocol
│       └── db_mcp.py            # Database operations (PostgreSQL, MySQL, MongoDB)
├── public/                      # Static assets
├── tests/                       # Test suite (structure ready)
├── pyproject.toml               # Dependencies & configuration
└── README.md                    # This file
```

### **Core Components**

- **`CodingAgent`**: LangGraph-based agent with tool binding and memory integration
- **`RAGService`**: Hybrid retrieval system with real-time indexing
- **`MemoryManager`**: Importance-based conversation memory with automatic cleanup
- **`DatabaseMCP`**: Multi-database support with connection management
- **`FractalAgent`**: Interactive TUI with streaming responses and command completion

## 🔧 Technical Capabilities

### **Tool Ecosystem**
- **File System Tools**: Complete CRUD operations, directory management, content editing
- **RAG Tools**: Semantic codebase search with hybrid retrieval and LLM reranking
- **Memory Tools**: Conversation history search with relevance scoring
- **Database Tools**: Full CRUD operations for PostgreSQL, MySQL, MongoDB
- **Search Tools**: Regex-based pattern matching and semantic code understanding

### **Performance Optimizations**
- **Async Architecture**: Full async/await support for optimal performance
- **Memory Efficiency**: Automatic cleanup with importance-based retention
- **Background Processing**: Non-blocking re-indexing and memory management
- **Connection Pooling**: Efficient database connection management
- **Smart Caching**: Optimized retrieval with hybrid search strategies

### **Developer Experience**
- **Streaming Responses**: Real-time reasoning display with tool call transparency
- **Error Handling**: Graceful degradation with comprehensive error messages
- **Configuration Management**: Runtime settings adjustment without restart
- **Debug Mode**: Verbose output for development and troubleshooting

## 🔮 Roadmap

### ✅ **Phase 1: Core Features (Completed)**
- **Hybrid RAG**: Dense embeddings + Sparse BM25 + LLM reranking ✅
- **Continuous Delta Re-indexing**: Real-time codebase indexing ✅
- **Multi-LLM Support**: OpenAI, Claude, Gemini integration ✅
- **Memory Management**: Importance-based conversation memory ✅

### ✅ **Phase 2: Database Integration (Completed)**
- **NoSQL Support**: MongoDB integration ✅
- **SQL Databases**: MySQL, PostgreSQL support ✅
- **MCP Integration**: Model Context Protocol implementation ✅
- **Direct Database Operations**: Full CRUD with connection management ✅

### 🚧 **Phase 3: Cloud Platform Integration (In Progress)**
- **AWS MCP**: Direct AWS resource management
- **Vercel MCP**: Deploy and manage Vercel projects
- **Multi-Cloud Support**: Unified interface for multiple cloud providers

### 🔮 **Phase 4: Advanced Features (Planned)**
- **Plugin System**: Extensible architecture for custom tools
- **Workflow Automation**: Complex multi-step task automation
- **Team Collaboration**: Shared workspaces and knowledge bases
- **Advanced Analytics**: Usage metrics and performance insights

## 🛠️ Development

### **Dependencies**
- **Python 3.11+** required
- **uv** (recommended) or pip for package management
- **Qdrant** vector database for RAG functionality
- **API Keys** for LLM providers and embedding services

### **Key Dependencies**
```toml
langchain>=0.3.27          # Core AI framework
langgraph>=0.6.8           # Agent orchestration
fastembed>=0.7.3           # Embeddings
langchain-qdrant>=0.2.1    # Vector storage
asyncpg>=0.30.0            # PostgreSQL
aiomysql>=0.2.0            # MySQL
motor>=3.7.1               # MongoDB
prompt-toolkit>=3.0.52     # TUI
rank-bm25>=0.2.2           # Keyword search
```

### Running Tests

```bash
# Run all tests (when implemented)
uv run pytest

# Run with coverage
uv run pytest --cov=src
```

### Code Quality

```bash
# Format code
uv run black src/

# Lint code
uv run flake8 src/

# Type checking
uv run mypy src/
```

## 📊 Project Status

**Current Version**: 0.1.0  
**Development Status**: Active development with core features complete  
**Architecture**: Production-ready with comprehensive tool ecosystem  
**Testing**: Basic structure in place, ready for test implementation  

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### **Areas for Contribution**
- **Testing**: Implement comprehensive test suite
- **Documentation**: Improve examples and tutorials
- **Cloud Integration**: AWS, Vercel MCP implementations
- **Plugin System**: Extensible architecture development
- **Performance**: Optimization and benchmarking

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the AI framework
- [LangGraph](https://github.com/langchain-ai/langgraph) for agent orchestration
- [Prompt Toolkit](https://github.com/prompt-toolkit/python-prompt-toolkit) for the TUI
- [Qdrant](https://github.com/qdrant/qdrant) for vector storage
- [FastMCP](https://github.com/jlowin/fastmcp) for database integration

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/ReesavGupta/Fractal/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ReesavGupta/Fractal/discussions)
- **Email**: [dev.reesav@gmail.com]

---

**Fractal** - *To the infinity and beyond!*

A sophisticated CLI agent demonstrating advanced AI engineering practices, modern Python development patterns, and thoughtful user experience design. Built with ❤️ for developers who want powerful AI assistance directly from their terminal.