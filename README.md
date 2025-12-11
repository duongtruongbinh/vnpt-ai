# VNPT AI RAG Pipeline

Pipeline designed for the VNPT AI Hackathon (Track 2).

This project implements a modular, model-agnostic workflow using **LangGraph** to intelligently route questions, execute Python code for complex reasoning, and retrieve knowledge from a persistent vector store.

## Key Features

- **Workflow**:
  - Utilizes a **Router Node** to classify questions into distinct domains: *Math/Logic*, *Knowledge (History/Culture/Law)*, *Reading Comprehension*, or *Toxic*.
  - Routes each question to the most specialized solver for optimal accuracy.

- **Safety & Policy Compliance**:
  - **Toxic Content Detection**: Sensitive or policy-violating content is identified immediately at the routing stage.
  - **Fast-Track Refusal**: Automatically selects the appropriate refusal option without invoking heavy reasoning models, saving cost and time.

- **Program-Aided Language Models (PAL)**:
  - **Code Agent**: Solves math and logic problems by generating and executing Python code via a local REPL, rather than relying solely on LLM hallucination.
  - **Self-Correction Loop**: The agent iteratively executes code, captures output, and if an error occurs, attempts to correct its own code (up to 5 retry steps).

- **Robust Checkpointing & Resumability**:
  - **Real-time Saving**: Every processed question is immediately saved to `inference_log.jsonl`.
  - **Seamless Resume**: If the pipeline is interrupted (e.g., system crash, power loss), simply re-running the command will skip processed questions and continue exactly where it left off.

- **Smart Rate Limit Handling**:
  - **Auto-Detection**: Automatically detects API quota limits (HTTP 429/401 errors).
  - **Graceful Shutdown**: Upon hitting a limit, the pipeline safely stops execution, consolidates all logs, and generates an emergency CSV submission file to ensure no progress is lost.

- **Multi-Source Ingestion**:
  - **Firecrawl Integration**: Capability to crawl single pages, full domains, or perform topic-based searches.
  - **Universal Document Support**: Ingests JSON, PDF, DOCX, and TXT files directly into the Qdrant Vector DB.
  - **Advanced Normalization**: Automatic Unicode normalization and whitespace cleaning for Vietnamese text.

## Architecture

The pipeline is orchestrated by a **LangGraph StateGraph**:

```mermaid
graph TD
    Start([Input Question]) --> RouterNode{Router Node<br/>Small Model}
    
    RouterNode -- "Math/Logic" --> LogicSolver[Logic Solver - Code Agent<br/>Large Model]
    RouterNode -- "History/Culture/Law" --> KnowledgeRAG[Knowledge RAG - Retrieval<br/>Large Model]
    RouterNode -- "Reading/General" --> DirectAnswer[Direct Answer - Zero-shot<br/>Large Model]
    RouterNode -- "Toxic/Sensitive" --> End([Final Answer<br/>Refusal Option])
    
    subgraph "Knowledge Processing"
        KnowledgeRAG <--> VectorDB[(Qdrant Local Disk)]
        VectorDB <..- IngestionScript[Ingestion Logic]
    end
    
    subgraph "Logic Processing"
        LogicSolver <--> PythonREPL[Python Interpreter<br/>Iterative Execution]
    end
    
    LogicSolver --> End
    KnowledgeRAG --> End
    DirectAnswer --> End
````

## Tech Stack

| Component | Implementation |
| :--- | :--- |
| **Orchestration** | LangGraph, LangChain |
| **Package Manager** | uv |
| **Vector DB** | Qdrant (Local Persistence) |
| **Embedding** | VNPT API / BKAI Vietnamese Bi-encoder |
| **Web Crawler** | Firecrawl API |
| **Doc Parser** | pypdf, python-docx |
| **Code Execution** | LangChain Experimental PythonREPL |
| **Models** | Local HuggingFace or VNPT API (configurable via `.env`) |

## Quick Start

### Prerequisites

  - Python ≥3.12
  - [uv](https://github.com/astral-sh/uv) (Recommended for fast dependency management)
  - CUDA-capable GPU (Recommended if running local models)

### Installation

1.  **Clone the repository**

    ```bash
    git clone https://github.com/duongtruongbinh/vnpt-ai
    cd vnpt-ai
    ```

2.  **Install dependencies**

    ```bash
    uv sync
    ```

3.  **Configure Environment**
    Create a `.env` file in the root directory:

    ```env
    # --- Model Selection ---
    USE_VNPT_API=False

    # --- Local Models (Used if USE_VNPT_API=False) ---
    LLM_MODEL_SMALL=/path/to/your/small/model
    LLM_MODEL_LARGE=/path/to/your/large/model
    EMBEDDING_MODEL=bkai-foundation-models/vietnamese-bi-encoder

    # --- VNPT API Config (Used if USE_VNPT_API=True) ---
    VNPT_LARGE_AUTHORIZATION=Bearer <your_token>
    VNPT_LARGE_TOKEN_ID=<your_token_id>
    VNPT_LARGE_TOKEN_KEY=<your_token_key>

    VNPT_SMALL_AUTHORIZATION=Bearer <your_token>
    VNPT_SMALL_TOKEN_ID=<your_token_id>
    VNPT_SMALL_TOKEN_KEY=<your_token_key>

    VNPT_EMBEDDING_AUTHORIZATION=Bearer <your_token>
    VNPT_EMBEDDING_TOKEN_ID=<your_token_id>
    VNPT_EMBEDDING_TOKEN_KEY=<your_token_key>
    ```

### Usage

#### 1\. Data Collection & Ingestion (Optional)

Expand your knowledge base by crawling websites or adding local documents.

```bash
# Crawl a website
uv run python scripts/crawl.py --url https://example.com --mode links --topic "keyword"

# Ingest data into Vector DB
uv run python scripts/ingest.py data/crawled/*.json --append
```

#### 2\. Run the Pipeline

**Option A: Local Development (Resumable)**
Uses `main.py`. Best for testing and processing large datasets over time.

```bash
uv run python main.py
```

**Option B: Docker / Deployment**
Uses `app.py`. Designed for the competition submission environment.

```bash
uv run python app.py
```

### Handling API Limits & Resuming

This pipeline is designed to be **fault-tolerant**:

1.  **Automatic Save**: If the VNPT API returns a Rate Limit error (429/401), the program will:

      * Log the error.
      * Stop processing new questions immediately.
      * Consolidate all successful answers into `submission_emergency.csv`.
      * Exit safely.

2.  **How to Resume**:

      * Wait for your API quota to reset (or switch tokens in `.env`).
      * Run the same command again (`uv run python main.py`).
      * The system detects the existing `inference_log.jsonl`, calculates which questions are missing, and **only processes the remaining questions**.

## Project Structure

```
vnpt-ai/
├── data/                 
│   ├── qdrant_storage/   # Persistent Vector DB
│   ├── crawled/          # Crawled raw data
│   ├── documents/        # PDF/DOCX source files
│   ├── val.json          # Validation dataset
│   └── test.json         # Test dataset
├── output/               # Results and logs (inference_log.jsonl stored here)
├── src/
│   ├── nodes/            # Logic for Router, RAG, Logic, Direct nodes
│   ├── utils/            # Utilities (Checkpointing, Ingestion, LLM)
│   ├── pipeline.py       # Core execution logic & Resume handling
│   └── graph.py          # LangGraph workflow definition
├── app.py                # Deployment entry point
├── main.py               # Development entry point
└── pyproject.toml        # Dependencies
```

## Input/Output Format

### Input (JSON - for `main.py`)

```json
[
  {
    "qid": "Q001",
    "question": "Câu hỏi ở đây?",
    "choices": ["Đáp án A", "Đáp án B", "Đáp án C", "Đáp án D"],
    "answer": "A"
  }
]
```

### Input (CSV - for `app.py`)
Columns: `qid`, `question`, `choice_a`, `choice_b`, `choice_c`, `choice_d` (or a `choices` column).


### Output (CSV)

```csv
qid,answer
Q001,A
Q002,B
```