# Automated Profile Inference with Language Model Agents

This repository contains the implementation of **AutoProfiler** on the synthetic dataset, a novel multi-agent system that performs automated profile inference using Large Language Models (LLMs). 

## 🎯 Overview

AutoProfiler combines:
- **Multi-agent coordination** for systematic information extraction
- **Retrieval-Augmented Generation (RAG)** for context-aware reasoning (optional)
- **Iterative refinement** through agent interactions
- **Confidence-based filtering** for reliable attribute inference

The system can infer various personal attributes including age, education, location, occupation, relationship status, income level, and more from user-generated text content.

## 🏗️ Architecture

### Multi-Agent System Design

The system employs four specialized agents that work in coordination:

1. **Retriever Agent**
   - Responsible for gathering relevant user history and context
   - Implements semantic search and chronological retrieval
   - Provides targeted information to the Extractor agent

2. **Strategist/Extractor Agent**
   - Coordinate the attack plan
   - Core reasoning agent that analyzes user content
   - Performs iterative attribute inference with confidence scoring

3. **Summarizer Agent**
   - Validates and consolidates inferred attributes
   - Removes duplicates and conflicting information
   - Generates natural language summaries of results

Note that in implementation we combine Strategist and Extractor as Profiler.

### Key Components

```
AutoProfiler/
├── agents/           # Multi-agent implementation
│   ├── retriever.py  # Information retrieval agent
│   ├── profiler.py   # Core plan/inference agent
│   └── summarizer.py # Result validation agent
├── config/           # Model and API configurations
├── dataset/          # SynthPAI dataset and results
├── functions/        # Local retrieval functions
├── prompts/          # Agent-specific prompts
├── util/             # Data loading and utility functions
└── main.py          # Main execution script
```

## 📊 Dataset

### Dataset Structure

```
dataset/
├── synthpai/           # User history files
│   ├── User1/         # Individual user directories
│   └── User2/
├── ground_truth.json  # Ground truth annotations
└── {model}/           # Inference results by model
    ├── pii/           # Extracted personal information
    └── summary/       # Natural language summaries
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- OpenAI API key (or compatible LLM provider)
- Bing Search API key (for web search functionality)

### Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API keys**:
   - Edit `config/gpt-4.json` with your LLM API credentials
   - Edit `config/web_api.py` with your search API keys

## 💻 Usage

### Basic Usage

Run the inference attack on a target user:

```bash
python main.py -m gpt-4 -u UserName
```

### Parameters

- `-m, --llm_model`: LLM model to use (e.g., gpt-4, claude, gemini)
- `-u, --target_user`: Target user identifier from SynthPAI dataset


## 🔒 Ethical Considerations

This research is conducted for **academic purposes only**. The system is designed to:
- **Raise awareness** about privacy risks in online content
- **Improve privacy protection** mechanisms of LLMs
- **Advance the field** of privacy and machine learning


**Disclaimer**: This tool should not be used for unauthorized personal information extraction or privacy violations. This research is for academic purposes only. Users are responsible for complying with all applicable laws and ethical guidelines.

