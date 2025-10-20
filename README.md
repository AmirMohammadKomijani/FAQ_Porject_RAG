# FAQ Project - Persian RAG System Benchmark

A comprehensive **Retrieval-Augmented Generation (RAG)** system for Persian language question-answering tasks. This project benchmarks different embedding models and vector stores for retrieval performance and evaluates generation quality using BLEU and ROUGE metrics.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## 🎯 Overview

This project implements a complete RAG pipeline for Persian language FAQ systems with:
- **3 Persian embedding models** for semantic search
- **3 FAISS vector stores** for efficient retrieval
- **LLM-based generation** using OpenAI's GPT-4o
- **Comprehensive evaluation** of both retrieval and generation components
- **Rate limiting** to handle API constraints gracefully

## 📊 Dataset

This project uses the **Synthetic Persian Chatbot RAG Topics Retrieval** dataset from MCINext:

🔗 **Dataset Link:** [https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-rag-topics-retrieval](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-rag-topics-retrieval)

### Dataset Details:
- **Language:** Persian (Farsi)
- **Type:** Synthetic FAQ dataset for RAG tasks
- **Format:** CSV
- **Contains:**
  - `corpus.csv` - Knowledge base documents with IDs and text
  - `test_enriched.csv` - Test queries with ground truth corpus IDs
- **Use Case:** Training and evaluating Persian language question-answering systems

### How to Use the Dataset:
1. Download from HuggingFace: [MCINext/synthetic-persian-chatbot-rag-topics-retrieval](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-rag-topics-retrieval)
2. Extract `corpus.csv` and `test_enriched.csv`
3. Place them in the project root directory
4. Run `data_preparation.py` to process the data

### Citation:
```bibtex
@dataset{mcinext_persian_rag,
  title={Synthetic Persian Chatbot RAG Topics Retrieval},
  author={MCINext},
  year={2024},
  url={https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-rag-topics-retrieval}
}






Input Query
    ↓
[Embedding Model] → [Vector Store (FAISS)]
    ↓
[Retriever] → Top-K Documents (k=3)
    ↓
[Context + Query] → [LLM (GPT-4o)]
    ↓
Generated Answer
    ↓
[BLEU & ROUGE Evaluation]


git clone https://github.com/AmirMohammadKomijani/FAQ_Porject_RAG.git
cd FAQ_Porject_RAG


# Using venv
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate


pip install -r requirements.txt

# Create .env file in project root
echo "OPENAI_API_KEY=your_api_key_here" > .env

pip install faiss-gpu

FAQ_Porject_RAG/
├── data_preparation.py           # Data loading and Document conversion
│   ├── Load corpus.csv
│   ├── Load test_enriched.csv
│   └── Convert to LangChain Document objects
│
├── embedding_models.py           # 3 Persian embedding models setup
│   ├── HooshvareLab/bert-fa-base-uncased
│   ├── HooshvareLab/sentence-bert-fa-base-stsb
│   └── HooshvareLab/bert-fa-zeroshot-clf-base
│
├── vector_stores.py              # FAISS vector store creation
│   ├── FAISS with model 1
│   ├── FAISS with model 2
│   ├── FAISS with model 3
│   └── Create retrievers
│
├── generation.py                 # RAG chain setup with LLM
│   ├── Define Persian prompt template
│   ├── Initialize GPT-4o
│   ├── Create QA chain
│   └── Create RAG chains
│
├── evaluate_retriever.py         # Retriever evaluation metrics
│   ├── Precision calculation
│   ├── Recall calculation
│   ├── F1-Score calculation
│   └── Detailed output per query
│
├── evaluate_generator.py         # Generator evaluation metrics
│   ├── BLEU score calculation
│   ├── ROUGE score calculation
│   ├── Rate limiting with delays
│   └── Break after N requests
│
├── main.py                       # Main execution script
│   └── Orchestrates all components
│
├── corpus.csv                    # Knowledge base documents
│   ├── _id: Document ID
│   ├── text: Document content
│   └── title: Document title
│
├── test_enriched.csv             # Test queries dataset
│   ├── query_text: User question
│   ├── corpus-id: Ground truth document ID
│   └── answer: Expected answer
│
├── requirements.txt              # Python dependencies
├── .env                          # Environment variables (not in repo)
├── .gitignore                    # Git ignore rules
└── README.md                     # This file


python main.py
