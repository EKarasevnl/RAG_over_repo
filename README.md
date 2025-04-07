# RAG System

This project implements LLM Listwise Reranker for **CodeRAG**.

## Setup

- The project requires **Python 3.11.0**.

- Install requirements.

 ```bash
pip install -r requirements.txt
```

## Execution

### Build BM25 index and compute sentence embeddings for the files in the repository and save them.

```bash
python main.py --setup <GitHub repository URL>
```

### Query the repository for relevant files.

```bash
python main.py --question "Question?"
```

### Evaluate the system on the dataset.

```bash
python main.py --evaluate <path-to-dataset-file>
```

## Project Structure

- main.py: Entry point for the command-line interface.

- RAGSystem.py: Contains the main RAG system logic (repository cloning, file processing, indexing, and querying).

- utils.py: Contains utility functions like evaluate for evaluating the system.

- requirements.txt: A list of required dependencies.