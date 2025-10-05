# Fractal CLI Agent

![Fractal Logo](public/image.png)

> **To the infinity and beyond!** 🚀

Fractal is an intelligent CLI agent built with LangChain and LangGraph, designed to provide powerful AI-driven assistance directly from your terminal. Currently in active development with exciting features planned for the future.

## 🎯 Current Features

- **Multi-LLM Support**: Choose from OpenAI, Claude, or Gemini
- **Interactive TUI**: Beautiful terminal user interface
- **RAG Service**: Basic retrieval-augmented generation capabilities
- **Agent Architecture**: Built with LangGraph for complex reasoning workflows
- **Async Operations**: Full async/await support for optimal performance

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
# OpenAI
OPENAI_API_KEY=your_openai_api_key

# Anthropic Claude
ANTHROPIC_API_KEY=your_anthropic_api_key

# Google Gemini
GOOGLE_API_KEY=your_google_api_key
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

## 🏗️ Project Structure

```
Fractal/
├── src/
│   ├── agent/           # Core agent implementation
│   │   ├── agent.py     # Main agent logic
│   │   ├── main.py      # CLI entry point
│   │   ├── state.py     # Agent state management
│   │   ├── tools.py     # Agent tools and functions
│   │   ├── tui.py       # Terminal user interface
│   │   └── utils.py     # Utility functions
│   └── rag_service/     # RAG implementation
│       └── rag.py       # Retrieval-augmented generation
├── public/              # Static assets
├── tests/               # Test suite
└── pyproject.toml       # Project configuration
```

## 🔮 Roadmap

### Phase 1: Enhanced RAG (In Progress)
- **Hybrid RAG**: Combine multiple retrieval strategies
- **Continuous Delta Re-indexing**: Real-time codebase indexing
- **Advanced Vector Search**: Improved semantic search capabilities

### Phase 2: Database Integration
- **NoSQL Support**: MongoDB, DynamoDB integration
- **SQL Databases**: MySQL, PostgreSQL support
- **MCP Integration**: Model Context Protocol for database operations
- **Direct Database Operations**: Query, modify, and manage databases from CLI

### Phase 3: Cloud Platform Integration
- **AWS MCP**: Direct AWS resource management
- **Vercel MCP**: Deploy and manage Vercel projects
- **Multi-Cloud Support**: Unified interface for multiple cloud providers

### Phase 4: Advanced Features
- **Plugin System**: Extensible architecture for custom tools
- **Workflow Automation**: Complex multi-step task automation
- **Team Collaboration**: Shared workspaces and knowledge bases

## 🛠️ Development

### Running Tests

```bash
# Run all tests
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

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the AI framework
- [LangGraph](https://github.com/langchain-ai/langgraph) for agent orchestration
- [Prompt Toolkit](https://github.com/prompt-toolkit/python-prompt-toolkit) for the TUI

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/ReesavGupta/Fractal-CLI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ReesavGupta/Fractal-CLI/discussions)
- **Email**: [Your Email]

---

**Note**: This project is currently in active development. Features and APIs may change. Stay tuned for updates! 🎉