"""
RAG Chatbot with LangChain + FAISS
Supports: OpenAI, HuggingFace (local), Groq, Google Gemini
"""

import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ── Embedding options ──────────────────────────────────────────────────────────
def get_embeddings(provider: str = "huggingface"):
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings()
    else:  # default: free local HuggingFace
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ── LLM options ────────────────────────────────────────────────────────────────
def get_llm(provider: str = "groq"):
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    elif provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model="llama3-8b-8192", temperature=0)
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")

# ── RAG pipeline ───────────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """Use ONLY the context below to answer the question.
If the answer is not in the context, reply exactly: "Not found in document."

Context:
{context}

Question: {question}

Answer:"""

def build_qa_chain(pdf_path: str, embed_provider: str, llm_provider: str):
    # 1. Load
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 2. Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    # 3. Embed & Store
    embeddings = get_embeddings(embed_provider)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 4. Retrieve + Generate
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )
    llm = get_llm(llm_provider)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return qa_chain

# ── Streamlit UI ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Chatbot", page_icon="📌")
st.title("📌 RAG Chatbot — Ask Your PDF")

with st.sidebar:
    st.header("⚙️ Configuration")
    embed_provider = st.selectbox("Embedding provider", ["huggingface", "openai"])
    llm_provider   = st.selectbox("LLM provider",       ["groq", "openai", "gemini"])

    if embed_provider == "openai" or llm_provider == "openai":
        os.environ["OPENAI_API_KEY"] = st.text_input("OpenAI API key", type="password")
    if llm_provider == "groq":
        os.environ["GROQ_API_KEY"]   = st.text_input("Groq API key",   type="password")
    if llm_provider == "gemini":
        os.environ["GOOGLE_API_KEY"] = st.text_input("Google API key", type="password")

uploaded = st.file_uploader("Upload a PDF", type="pdf")

if uploaded:
    tmp_path = f"/tmp/{uploaded.name}"
    with open(tmp_path, "wb") as f:
        f.write(uploaded.read())

    if "qa_chain" not in st.session_state or st.session_state.get("pdf_name") != uploaded.name:
        with st.spinner("Building index…"):
            st.session_state.qa_chain = build_qa_chain(tmp_path, embed_provider, llm_provider)
            st.session_state.pdf_name = uploaded.name
        st.success("Ready! Ask anything about the document.")

    question = st.text_input("Your question")
    if st.button("Ask") and question:
        with st.spinner("Thinking…"):
            result = st.session_state.qa_chain.invoke({"query": question})
        st.markdown(f"**Answer:** {result['result']}")
        with st.expander("Source chunks"):
            for doc in result["source_documents"]:
                st.write(f"Page {doc.metadata.get('page', '?')}: {doc.page_content[:300]}…")
else:
    st.info("Upload a PDF to get started.")
