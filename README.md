# 🧠 Agent Cookbook

This reposiory is a collection of personal experiments, prototypes, and working notebooks focused on document intelligence — especially **PDF understanding**, **retrieval-augmented generation (RAG)**, and **structured responses** for real-world documents.

This space combines tools like **Docling**, **LlamaIndex**, and **HuggingFace embeddings** to explore workflows around:

- Chunking and parsing complex PDFs
- Summarizing tables and text
- Embedding metadata-rich nodes
- Building flexible and queryable RAG pipelines
- Returning structured answers via agents

> ⚠️ This repository is not intended as a standalone library or package. It is a working lab for experimentation and concept validation.

---

## 📂 Repository Overview

```markdown

agent\_cookbook/
    ├── environment.yml                # Conda setup (recommended)
    ├── requirements.txt              # pip fallback (if needed)
    └── notebook/
        ├── 01\_agent\_chunk\_rag.ipynb 
        ├── 02\_docling\_chunking.ipynb
        ├── 03\_document\_workflow\.ipynb 
        ├── 04\_example\_agent.ipynb
        ├── 05\_RAG\_Docling.ipynb
        ├── 06\_structured\_agent.ipynb 
        ├── 07\_test\_pdf\_parsers.ipynb
        ├── zz\_poc\_sec\_downloader.ipynb
        └── src/
            ├── converter.py                   
            ├── pdf-rag-pipeline.py            
            ├── updated-pdf-rag-pipeline.py   
            ├── RAGWorkflow\.py                 
            └── StructuredResponse.py          

````

---

## 🧪 Setup (Optional)

Although this isn't a packaged project, if you want to run the notebooks:

```bash
# Option 1: Conda (recommended)
conda env create -f environment.yml
conda activate llm_agent

# Option 2: pip
pip install -r requirements.txt
````

---

## 🔍 What’s Inside

### ✅ `converter.py`

* Docling-based PDF parser
* Chunking logic with table filtering
* Embedding nodes with rich metadata
* Summarization via Ollama

### 🧩 `pdf-rag-pipeline.py` and `updated-pdf-rag-pipeline.py`

* RAG pipelines from scratch
* One uses standard LlamaIndex `VectorStoreIndex`
* The other builds parent-child node graphs with `RecursiveRetriever`

### 📦 `StructuredResponse.py`

* Pydantic schema for **invoice-style responses**:
  * `short_answer` (YES/NO/N/A)
  * full explanation and raw LLM response

### ⚙️ `RAGWorkflow.py`

* Event-based RAG architecture with:

  * async retrieval
  * LLM-based reranking
  * `CompactAndRefine` summarization

---

## 📓 Notebook Highlights

| Notebook                    | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `01_agent_chunk_rag.ipynb` | Builds a chunking and retrieval pipeline using agent-inspired RAG patterns. |
| `02_docling_chunking.ipynb`| Explores a chunking strategy combining layout, semantics, and content type using Docling. |
| `03_document_workflow.ipynb`| Walks through a full pipeline for processing and embedding documents using the converter module. |
| `04_example_agent.ipynb`   | Quick prototype of an agent with minimal orchestration for LLM prompting.   |
| `05_RAG_Docling.ipynb`     | Integrates Docling and RAG pipelines; builds embedding and retrieval index. |
| `06_structured_agent.ipynb`| Demonstrates returning structured responses using Pydantic models and agent prompts. |
| `07_test_pdf_parsers.ipynb`| Compares different PDF extraction strategies to evaluate fidelity and completeness. |

---

## 📌 Design Philosophy

* **Notebooks first**: All workflows begin in `*.ipynb` for fast iteration
* **No assumptions about runtime**: Mixes CPU, MPS, and GPU settings
* **Tinker-friendly**: Change models, chunkers, and prompts easily
* **Real-world inspired**: Optimized for invoice-like documents and semi-structured layouts

---

## 🛠 Future Ideas

This repo is still exploratory, but potential next steps include:

* Autonomous document agents
* JSON schema validation in responses
* Embedding quality evaluation tooling

---

## 📝 License

MIT License – Feel free to reuse and adapt.

---

## 🙋 About

This is a personal project exploring semantic document processing. Feedback, ideas, and collaborations are welcome!
