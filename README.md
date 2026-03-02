# LLM Agent Platform

A multi-agent LLM platform for test generation and verified code understanding.

## Overview

This platform provides an LLM-based multi-agent workflow for:
- **Unit Test Generation**: Generate pytest tests with verification
- **UI Test Generation**: Generate Playwright tests (Phase 3)
- **Code Explanation**: Structured explanations with complexity analysis (Phase 4)

All generated artifacts pass through verification gates before being returned:
1. Static Analysis (SAST)
2. Dependency Validation
3. Sandboxed Execution

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp .env.example .env
# Edit .env and add your Groq API key (free tier available)
```

Get your free Groq API key at: https://console.groq.com/

### 3. Build the Docker Sandbox

```bash
docker build -t llm-agent-sandbox -f docker/Dockerfile .
```

### 4. Generate Tests

```bash
# Generate unit tests for a Python file
python -m src.main generate --file path/to/your/code.py --output tests/

# With verbose output
python -m src.main generate --file path/to/your/code.py --output tests/ --verbose
```

## Architecture

```
User Request → Router Agent → Unit Test Agent → Verification Gates → Output
                                    ↑                    ↓
                                    └──── Repair Loop ←──┘
```

The platform uses a Generate-Detect-Repair (GDR) loop:
1. Generate test candidates
2. Verify in sandboxed environment
3. If verification fails, repair and retry (up to k=3 times)

## Project Structure

```
src/
├── agents/          # Agent implementations
│   ├── router.py    # Task routing
│   └── unit_test.py # Test generation
├── graph/           # LangGraph orchestration
│   └── pipeline.py  # Workflow definition
├── verification/    # Verification gates
│   └── sandbox.py   # Docker sandbox execution
└── utils/           # Utilities
    └── logging.py   # Audit logging
```

## Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `GROQ_API_KEY` | Groq API key (required) | - |
| `DEFAULT_MODEL` | LLM model to use | `llama-3.1-70b-versatile` |
| `SANDBOX_TIMEOUT` | Execution timeout (seconds) | `60` |
| `SANDBOX_MEMORY_LIMIT` | Container memory limit | `512m` |

## License

MIT
