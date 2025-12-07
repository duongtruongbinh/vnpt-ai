# VNPT AI RAG Pipeline

Agentic RAG Pipeline designed for the VNPT AI Hackathon (Track 2).

This project implements a modular, model-agnostic workflow using **LangGraph** to intelligently route questions, execute Python code for complex reasoning, and retrieve knowledge from a persistent vector store.

## 🚀 Key Features

- **Agentic Workflow**: Utilizes a **Router Node** to classify questions into distinct domains (Math, Knowledge, Direct Comprehension, or Toxic) and routes them to specialized solvers.
- **🛡️ Safety First**: Toxic or policy-violating content is detected immediately in the router node, automatically selecting refusal options without invoking heavy reasoning models.
- **Program-Aided Language Models (PAL)**:
  - Solves math and logic problems by generating and executing Python code via a local REPL.
  - **Self-Correction Loop**: Iteratively executes code, captures output, and corrects errors (up to 5 retry steps).
- **🔄 Resumable Inference**: Built-in checkpointing system (`inference_log.jsonl`) allows the pipeline to resume from where it left off in case of crashes or API rate limits.
- **Multi-Source Ingestion & Crawling**:
  - **Firecrawl Integration**: Crawl websites (single page, full domain, or topic search).
  - **Document Support**: Ingest JSON, PDF, DOCX, and TXT files into the Vector DB.
  - **Text Normalization**: Automatic Unicode normalization and whitespace cleaning.
- **Hybrid Model Selection**:
  - Supports both **Local HuggingFace** models and **VNPT API** models.
  - Granular credential configuration for Large, Small, and Embedding models.
- **Quota Optimization**:
  - **Tiered Modeling**: Lightweight "Small" models for routing, "Large" models for deep reasoning/RAG.
  - **Local Vector Store**: Qdrant runs locally with disk persistence to prevent redundant re-embedding.

## 🏗️ Architecture

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

## 🛠️ Tech Stack

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

## ⚡ Quick Start

### Prerequisites

  - Python ≥3.12
  - [uv](https://github.com/astral-sh/uv) (Recommended for fast dependency management)
  - CUDA-capable GPU (Recommended if running local models)

### Installation

1.  **Clone the repository**

    ```bash
    git clone [https://github.com/duongtruongbinh/vnpt-ai](https://github.com/duongtruongbinh/vnpt-ai)
    cd vnpt-ai
    ```

2.  **Install dependencies**

    ```bash
    uv sync
    ```

3.  **Configure Environment**
    Create a `.env` file in the root directory. You can configure different credentials for the Large, Small, and Embedding models:

    ```env
    # --- Model Selection ---
    # Set to True to use VNPT API, False for local HuggingFace models
    USE_VNPT_API=False

    # --- Local Models (Used if USE_VNPT_API=False) ---
    LLM_MODEL_SMALL=/path/to/your/small/model
    LLM_MODEL_LARGE=/path/to/your/large/model
    EMBEDDING_MODEL=bkai-foundation-models/vietnamese-bi-encoder

    # --- VNPT API Config (Used if USE_VNPT_API=True) ---
    # Large Model Credentials
    VNPT_LARGE_AUTHORIZATION=Bearer <your_token>
    VNPT_LARGE_TOKEN_ID=<your_token_id>
    VNPT_LARGE_TOKEN_KEY=<your_token_key>
    VNPT_LARGE_ENDPOINT=[https://api.idg.vnpt.vn/data-service/v1/chat/completions/vnptai-hackathon-large](https://api.idg.vnpt.vn/data-service/v1/chat/completions/vnptai-hackathon-large)

    # Small Model Credentials
    VNPT_SMALL_AUTHORIZATION=Bearer <your_token>
    VNPT_SMALL_TOKEN_ID=<your_token_id>
    VNPT_SMALL_TOKEN_KEY=<your_token_key>
    VNPT_SMALL_ENDPOINT=[https://api.idg.vnpt.vn/data-service/v1/chat/completions/vnptai-hackathon-small](https://api.idg.vnpt.vn/data-service/v1/chat/completions/vnptai-hackathon-small)

    # Embedding Model Credentials
    VNPT_EMBEDDING_AUTHORIZATION=Bearer <your_token>
    VNPT_EMBEDDING_TOKEN_ID=<your_token_id>
    VNPT_EMBEDDING_TOKEN_KEY=<your_token_key>
    VNPT_EMBEDDING_ENDPOINT=[https://api.idg.vnpt.vn/data-service/vnptai-hackathon-embedding](https://api.idg.vnpt.vn/data-service/vnptai-hackathon-embedding)

    # --- Optional: Web Crawling ---
    FIRECRAWL_API_KEY=your_firecrawl_key
    ```

### Usage

#### 1\. Data Collection & Ingestion (Optional)

Expand your knowledge base by crawling websites or adding local documents.

  * **Crawl Data**:

    ```bash
    # Crawl a website filtering by topic keywords (results saved to data/crawled/)
    uv run python scripts/crawl.py --url [https://example.com](https://example.com) --mode links --topic "keyword1,keyword2"
    ```

  * **Ingest Data**:
    Load data into the Qdrant vector store (`data/qdrant_storage`).

    ```bash
    # Ingest crawled JSON files
    uv run python scripts/ingest.py data/crawled/*.json --append

    # Ingest a directory of documents (PDF, DOCX, TXT)
    uv run python scripts/ingest.py --dir data/documents --append
    ```

#### 2\. Run the Pipeline

There are two modes to run the pipeline:

**Option A: Local Development (Resumable)**
Uses `main.py`. Supports checkpointing to `output/inference_log.jsonl`. If stopped, it resumes processing from the last specific Question ID.

  * **Input:** Looks for `val.json` or `test.json` in `data/`.
  * **Command:**
    ```bash
    uv run python main.py
    ```

**Option B: Docker / Deployment (Simple)**
Uses `app.py`. Designed for the competition submission environment.

  * **Input:** Looks for `public_test.csv` or `private_test.csv` in `data/`.
  * **Command:**
    ```bash
    uv run python app.py
    ```

## 📂 Project Structure

```
vnpt-ai/
├── data/                 
│   ├── qdrant_storage/   # Persistent Vector DB
│   ├── crawled/          # Crawled raw data
│   ├── documents/        # PDF/DOCX source files
│   ├── val.json          # Validation dataset
│   └── test.json         # Test dataset
├── docker/               # Docker configuration
├── output/               # Results and logs
├── scripts/
│   ├── crawl.py          # CLI: Web crawler
│   └── ingest.py         # CLI: Vector ingestion
├── src/
│   ├── data_processing/  # Loaders, Formatting, Answer Extraction
│   ├── nodes/            # LangGraph Nodes (Router, RAG, Logic, Direct)
│   ├── utils/            # Utilities (LLM, Embeddings, Checkpointing)
│   ├── config.py         # Configuration settings
│   ├── graph.py          # Workflow definition
│   ├── pipeline.py       # Core execution logic
│   ├── prompts.py        # System prompts
│   └── state.py          # Graph state schema
├── app.py                # Deployment entry point (CSV input)
├── main.py               # Dev entry point (JSON input + Resume)
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