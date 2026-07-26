# EDUBREIF AI

An intelligent Document Intelligence Assistant built with Streamlit — upload documents and get automatic RAG-based Q&A, NLP insights, study assets, research extraction, and exportable reports using the free Groq API.


## Features

- **📄 Multi-Format Upload** — PDF, TXT, CSV, MD, JSON files supported
- **🧠 RAG-Based Q&A** — Ask questions from your documents with grounded, cited answers
- **📊 Document Dashboard** — Auto-classification, health scores, knowledge graphs
- **📚 Learning Assistant** — Flashcards, revision notes, mind maps, quizzes, study coach
- **🔬 Research Assistant** — Abstract extraction, paper comparison, citation checker, timeline
- **📈 Analytics** — Sentiment analysis, keyword extraction, entity recognition, readability
- **🤖 Smart AI** — Confidence meter, hallucination detection, AI critic, follow-up questions
- **💾 Export** — Markdown, HTML, DOCX, PPT, PDF report generation

## Quick Start

1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Get a free Groq API key at [console.groq.com](https://console.groq.com)
3. Run the app:
   ```bash
   streamlit run app.py
   ```
4. Enter your API key in the sidebar → Upload documents → Click **Process**

## Architecture

```
Streamlit UI
  ├── FileHandler → Reads PDF/TXT/CSV/MD/JSON → Normalized chunks
  ├── RAGPipeline → Chunk → Embed (sentence-transformers) → FAISS index → Retrieve → Groq LLM → Answer
  ├── NLPAnalyzer → NLTK (VADER sentiment, TF-IDF keywords, readability, extractive summary)
  └── DocumentIntelligenceAnalyzer → Dashboard, Learning, Research, Analytics, Smart AI
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| LLM | Groq (Llama3 8B / 70B) |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` |
| Vector Database | FAISS (CPU) |
| NLP | NLTK, VADER Sentiment |
| PDF | pdfplumber |
| Reports | fpdf2, Custom DOCX/PPTX build |

## Use Cases

- 📝 **Query project logs** — "What are the main risks?"
- 📋 **NLP on meeting notes** — Sentiment, keywords, action items
- 📑 **Daily reports** — Summarize and export from multiple documents
- 🎓 **Study assistant** — Generate flashcards, quizzes, revision plans
- 🔍 **Research papers** — Extract abstracts, compare papers, check citations

## Tests

```bash
pytest tests/ -v
```


## License

MIT License. Author: Sejal

