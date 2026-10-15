# Inquira

**A simple, structured AI Q&A application powered by LangChain.**

Inquira is an AI-powered question-answering application built with LangChain, supporting multiple LLM providers through OpenAI and Anthropic APIs.

## Features

- **Multi-Provider LLM Support** — Integrates OpenAI and Anthropic
  models through their respective APIs.
- **Natural Language Question Answering** — Processes user queries
  and generates AI-powered responses.
- **Structured Outputs** — Generates consistent, schema-validated
  responses using Pydantic.
- **Confidence Scoring** — Provides LLM-generated confidence scores
  alongside answers.
- **Error Handling** — Handles API and model invocation failures.
- **LangSmith Observability** — Enables tracing and debugging
  of LLM interactions.

## Tech Stack

- **Python** — Core programming language
- **LangChain** — LLM application framework
- **Pydantic** — Structured data validation
- **LangSmith** — LLM tracing and observability

## Installation

### Prerequisites

Before getting started, ensure you have the following installed:

- Python 3.12 or higher (or the version specified in `pyproject.toml`)
- [uv](https://docs.astral.sh/uv/) — Python package and project manager
- API keys for OpenAI and/or Anthropic

### 1. Clone the Repository

```bash
git clone https://github.com/binaryCoder-101/inquira.git
cd inquira
```

### 2. Install Dependencies

Install the project dependencies using `uv`:

```bash
uv sync
```

This automatically creates a virtual environment and installs the dependencies defined in `pyproject.toml`.

### 3. Activate the Virtual Environment (Optional)

**Linux/macOS:**

```bash
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
```

> Activating the virtual environment is optional when using `uv run`, as it automatically uses the project's environment.

### 4. Configure Environment Variables

Create a `.env` file in the project root and add your API credentials:

```env
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
LANGSMITH_API_KEY=your_langsmith_api_key

# Optional defaults:
LLM_PROVIDER=openai
OPENAI_MODEL=gpt-4o-mini
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
LLM_TEMPERATURE=0.3
```

> **Security:** Never commit your `.env` file or expose your API keys publicly. Ensure `.env` is included in `.gitignore`.

## Usage

Inquira provides flexible interaction modes: an interactive terminal session with live commands, single command-line queries, batch processing, and configurable providers/models.

### 1. Interactive Mode (Default)

Launch the interactive assistant by running with no arguments:

```bash
uv run main.py
```

When the session starts, an interactive banner displays your active configuration and command cheat-sheet:

```text
============================================================================
  INQUIRA - AI-Powered Structured Q&A Assistant
============================================================================
  Active Provider : openai
  Active Model    : gpt-4o-mini
  Temperature     : 0.3
  Output Format   : Human-readable structured text
  Observability   : LangSmith Active (Inquira)
----------------------------------------------------------------------------
  HOW TO INTERACT & SWITCH CONFIGURATION:
  • Ask Question   : Type your question directly and press Enter.
  • Change Provider: /provider <openai|anthropic>  (e.g., /provider anthropic)
  • Change Model   : /model <model_name>          (e.g., /model gpt-4o)
  • Change Temp    : /temp <0.0-1.0>              (e.g., /temp 0.7)
  • Toggle JSON    : /json                        (switch Text <-> JSON output)
  • Check Settings : /info                        (view active config)
  • Full Manual    : /help                        (view detailed guide)
  • Exit Session   : exit, quit, or Ctrl+C
============================================================================
```

#### In-Session Slash Commands:
| Command | Description | Example |
|---|---|---|
| `/provider <name>` | Switch between OpenAI and Anthropic | `/provider anthropic` |
| `/model <name>` | Switch active model dynamically | `/model claude-3-5-sonnet-20241022` |
| `/temp <float>` | Adjust sampling temperature (0.0 to 1.0) | `/temp 0.7` |
| `/json` | Toggle between formatted text and raw JSON output | `/json` |
| `/info` | Display active provider, model, temperature, and tracing | `/info` |
| `/clear` | Clear the terminal screen | `/clear` |
| `/help` | Display interactive instructions and guidelines | `/help` |
| `exit` / `quit` | End the session | `exit` |

---

### 2. Single Question Mode

Ask a question directly from the command line without opening a session:

```bash
# Positional query argument:
uv run main.py "What is retrieval-augmented generation?"

# Using the explicit -q / --question flag:
uv run main.py -q "Explain the difference between SQL and NoSQL."
```

---

### 3. Multi-Provider & Custom Model Selection

Specify your LLM provider (`-p` / `--provider`), model name (`-m` / `--model`), and temperature (`-t` / `--temperature`):

```bash
# Query OpenAI with GPT-4o:
uv run main.py -p openai -m gpt-4o "Explain quantum superposition."

# Query Anthropic Claude:
uv run main.py -p anthropic -m claude-3-5-sonnet-20241022 "What is LangChain?"

# Adjust creativity with temperature (0.0 = deterministic/factual, 0.7+ = creative):
uv run main.py -p openai -m gpt-4o-mini -t 0.7 "Brainstorm 3 project ideas for AI agents."
```

#### Popular Model Identifiers:
- **OpenAI (`-p openai`)**: `gpt-4o-mini` (default), `gpt-4o`, `o3-mini`
- **Anthropic (`-p anthropic`)**: `claude-3-5-sonnet-20241022` (default), `claude-3-5-haiku-20241022`, `claude-3-haiku-20240307`

---

### 4. Batch Processing Mode

Process multiple questions in parallel using the `-b` / `--batch` flag:

```bash
uv run main.py -b "What is Python?" "What is JavaScript?" "What is Rust?"
```

---

### 5. JSON Output Mode

Format responses as machine-readable, schema-validated JSON with the `--json` flag:

```bash
uv run main.py --json "What is vector indexing?"
uv run main.py -b "What is Docker?" "What is Kubernetes?" --json
```

---

### 6. Built-in Demos

Run built-in test suites to verify functionality, edge-case error handling, and LangSmith tracing:

```bash
# Run all demo suites:
uv run main.py --demo

# Run a specific demo (qa, batch, or error):
uv run main.py --demo qa
uv run main.py --demo batch
uv run main.py --demo error
```

---

### 7. CLI Reference & Flag Summary

Print the complete in-terminal guide at any time:

```bash
# Quick options summary:
uv run main.py --help

# In-depth CLI user manual:
uv run main.py help
# or
uv run main.py --instructions
```

#### CLI Flags:
| Flag | Name | Description |
|---|---|---|
| `-h` | `--help` | Show summary help and argument synopsis |
| | `--instructions` | Print the complete in-depth user guide and manual |
| `-p` | `--provider` | Choose LLM provider: `openai` or `anthropic` (default: `$LLM_PROVIDER` or `openai`) |
| `-m` | `--model` | Specify model identifier (e.g. `gpt-4o`, `claude-3-5-sonnet-20241022`) |
| `-t` | `--temperature` | Sampling temperature between 0.0 and 2.0 (default: `0.3`) |
| `-q` | `--question` | Explicit question string |
| `-b` | `--batch` | Process multiple questions concurrently in batch |
| `-i` | `--interactive` | Force launch interactive session |
| | `--json` | Output response as formatted JSON string |
| | `--demo` | Run built-in demo test suite (`qa`, `batch`, `error`, `all`) |

---

## How It Works

Inquira follows a structured question-answering pipeline:

1. **Input** — The user submits a question via CLI argument, batch list, or interactive prompt.
2. **Configuration** — The provider, model, and temperature are resolved from CLI flags, interactive commands, or `.env` defaults.
3. **Execution** — LangChain orchestrates the request and passes the prompt to the selected LLM.
4. **Validation** — The model's response is parsed and validated against a Pydantic schema (`QAResponse`).
5. **Observability** — When LangSmith tracing is active, execution traces are logged to your project.
6. **Output** — Results are rendered either as formatted text or schema-validated JSON.

### Response Structure

Every response is validated against the following Pydantic schema:

```json
{
  "answer": "Retrieval-augmented generation (RAG) is a technique that enhances language model responses by incorporating information retrieved from external knowledge sources.",
  "confidence": "high",
  "reasoning": "RAG combines search retrieval with generative models to ground responses in factual data.",
  "follow_up_questions": [
    "What are common vector databases used in RAG?",
    "How does RAG differ from fine-tuning a model?"
  ],
  "sources_needed": false
}
```

- **answer** (`str`) — The LLM-generated answer.
- **confidence** (`str`) — Model confidence assessment (`high`, `medium`, `low`).
- **reasoning** (`str`) — Logical justification for the provided answer.
- **follow_up_questions** (`List[str]`) — Suggested follow-up topics.
- **sources_needed** (`bool`) — Whether external sources/citations are required.

