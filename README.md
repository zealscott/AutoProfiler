# Automated Profile Inference with Language Model Agents

This repository contains the implementation of **AutoProfiler** on the synthetic dataset, a novel multi-agent system that performs automated profile inference using Large Language Models (LLMs).

## Overview

AutoProfiler combines:
- **Multi-agent coordination** for systematic information extraction
- **Retrieval-Augmented Generation (RAG)** for context-aware reasoning (optional)
- **Iterative refinement** through agent interactions
- **Confidence-based filtering** for reliable attribute inference

The system can infer various personal attributes including age, education, location, occupation, relationship status, income level, and more from user-generated text content.

## Architecture

### Multi-Agent System Design

The system employs four specialized agents that work in coordination:

1. **Tagger Agent** — Preprocesses user history by tagging personal attribute types in each comment (runs automatically before profiling).
2. **Retriever Agent** — Gathers relevant user history and context via semantic search and chronological retrieval.
3. **Profiler Agent** (Strategist + Extractor) — Coordinates the attack plan, analyzes user content, and performs iterative attribute inference with confidence scoring.
4. **Summarizer Agent** — Validates, deduplicates, and consolidates inferred attributes; generates natural language summaries.

### Project Structure

```
AutoProfiler/
├── agents/           # Multi-agent implementation
│   ├── tagger.py     # Attribute tagging agent (preprocessing)
│   ├── retriever.py  # Information retrieval agent
│   ├── profiler.py   # Core plan/inference agent
│   ├── summarizer.py # Result validation agent
│   └── evaluator.py  # PII evaluation agent
├── config/           # Model configurations (YAML)
├── core/             # Core framework (LLM client, memory, parser, toolkit)
├── dataset/          # SynthPAI dataset and results
├── functions/        # Local & web retrieval functions
├── prompts/          # Agent-specific prompt templates
├── util/             # Data loading and utility functions
├── tagging.py        # Standalone tagging script (also used by main.py)
├── init_agents.py    # Agent initialization
└── main.py           # Main execution script
```

### Dataset Structure

```
dataset/
├── synthpai/           # User history files
│   ├── User1/          # Individual user directories
│   │   ├── history_1.txt
│   │   ├── history_2.txt
│   │   └── ...
│   └── User2/
├── tag/                # Tagged data for RAG (per-user)
├── vdb/                # Vector DB indices (auto-generated)
├── ground_truth.json   # Ground truth annotations
└── {model}/            # Inference results by model
    ├── pii/            # Extracted personal information (JSON)
    └── summary/        # Natural language summaries
```

## Installation

### Prerequisites

- Python 3.8+
- An API key for at least one LLM provider (see below)

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd AutoProfiler

# Install dependencies
pip install -r requirements.txt
```

## Configuration

### Supported Models

AutoProfiler uses [LiteLLM](https://docs.litellm.ai/) as a unified LLM interface, so it supports any LiteLLM-compatible provider. Three configurations are provided out of the box:

| Config File | Model | Embedding Model | Notes |
|---|---|---|---|
| `config/gemini-2.5-flash.yaml` | `gemini/gemini-2.5-flash` | `gemini/gemini-embedding-001` | **Default.** Free tier available, single API key |
| `config/gpt-4o.yaml` | `gpt-4o` | `openai/text-embedding-3-small` | Requires OpenAI API key |
| `config/claude-sonnet.yaml` | `claude-sonnet-4-20250514` | `openai/text-embedding-3-small` | Requires Anthropic key + OpenAI key (for embeddings) |
| `config/gemini-2.0-flash.yaml` | `gemini/gemini-2.0-flash` | `gemini/text-embedding-001` | Legacy Gemini config |

API keys are configured directly in the YAML config files under `config/`. Open the config file for your chosen model and replace the `api_key` placeholder with your actual key. For Claude Sonnet, you also need to set `embedding_api_key` (an OpenAI key) since Anthropic has no embedding API.

To add a new model, create a YAML file in `config/` following the same format. See [LiteLLM docs](https://docs.litellm.ai/docs/providers) for supported model identifiers.

## Usage

```bash
python main.py -m <model-config> -u <target-user>
```

- `-m, --llm_model`: Config name (filename in `config/` without `.yaml`), e.g. `gpt-4o`, `gemini-2.0-flash`, `claude-sonnet`
- `-u, --target_user`: Target user directory name from `dataset/synthpai/`, e.g. `AdorableAardvark`

The pipeline automatically runs **tagging → RAG index build → profiling**. Tagging is skipped for files that already have tag output in `dataset/tag/{user}/`.

You can also run tagging independently:

```bash
python tagging.py -m <model-config> -u <target-user>
```

### Examples

```bash
# Run with default model (Gemini 2.5 Flash)
python main.py -u AdorableAardvark

# Run with GPT-4o
python main.py -m gpt-4o -u AdorableAardvark

# Run with Claude Sonnet
python main.py -m claude-sonnet -u AdorableAardvark

# Run tagging only
python tagging.py -m gpt-4o -u AdorableAardvark
```

### Output

Results are saved to:
- `dataset/<model>/pii/<user>.json` — Inferred attributes as JSON (type, confidence, evidence, guess)
- `dataset/<model>/summary/<user>.txt` — Natural language summary of the inferred profile

If inference is incomplete, the user is logged to `incomplete_<model>.txt`.

## Ethical Considerations

This research is conducted for **academic purposes only**. The system is designed to:
- **Raise awareness** about privacy risks in online content
- **Improve privacy protection** mechanisms of LLMs
- **Advance the field** of privacy and machine learning

**Disclaimer**: This tool should not be used for unauthorized personal information extraction or privacy violations. This research is for academic purposes only. Users are responsible for complying with all applicable laws and ethical guidelines.
