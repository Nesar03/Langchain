# 📌 RAG Chatbot with LangChain


---

## Pipeline Overview

```
PDF  →  PyPDFLoader  →  RecursiveCharacterTextSplitter
     →  HuggingFace/OpenAI Embeddings  →  FAISS vector index
     →  RetrievalQA chain  →  Groq / OpenAI / Gemini LLM
```

---

## 1 — Setup

```bash
pip install -r requirements.txt
```

### Free (no OpenAI credits needed)
| Component  | Option |
|------------|--------|
| Embeddings | `HuggingFaceEmbeddings` (sentence-transformers, runs locally) |
| LLM        | Groq free tier (`llama3-8b-8192`) **or** Google Gemini free tier |

Get a free Groq key → https://console.groq.com  
Get a free Gemini key → https://aistudio.google.com

---

## 2 — Run the Streamlit UI

```bash
streamlit run app.py
```

1. Select your **embedding provider** and **LLM provider** in the sidebar.
2. Paste the relevant API key.
3. Upload any PDF.
4. Ask questions — the bot answers only from the document.

---

## 3 — Run as CLI

```bash
# Using Groq (default)
export GROQ_API_KEY=your_key
python rag_cli.py --pdf lecture_notes.pdf --embed huggingface --llm groq

# Using OpenAI
export OPENAI_API_KEY=your_key
python rag_cli.py --pdf lecture_notes.pdf --embed openai --llm openai
```

---

## 4 — Test (5 sample questions)

After loading a PDF run these to verify grounded answers:

1. "What is the main topic of this document?"
2. "List the key concepts mentioned."
3. "What does the document say about [specific term]?"
4. "Who are the authors or contributors?"
5. "Summarise the conclusion."

If the answer isn't in the document the bot replies: **"Not found in document."**

---

## Project Structure

```
rag_chatbot/
├── app.py           # Streamlit UI + RAG pipeline
├── rag_cli.py       # CLI version (no Streamlit)
├── requirements.txt
└── README.md
```

---

## Submission Checklist

- [ ] GitHub repo with this code
- [ ] Short demo video / screenshots of answers
- [ ] README explaining the pipeline (this file ✅)
