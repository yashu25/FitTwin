"""
FitTwin - Single-file Streamlit app (Digital Twin AI Coach)

Features:
- Chat UI to ask questions about Nutrition, Workout, Biomechanics, Physio
- Upload your personal knowledge files (PDF / txt / md) to build the RAG knowledge base
- Optional live web search (Serper.dev / SerpAPI) to fetch latest info beyond LLM cutoff
- RAG pipeline using LangChain + FAISS
- Simple orchestration (MCP-like): combine local RAG context + web search snippets before calling LLM

HOW TO DEPLOY (summary):
1. Create a new GitHub repo and add this file (FitTwin_streamlit_app.py) + requirements.txt (see comment below)
2. Sign in to Streamlit Cloud and create a new app from the repo -> deploy
3. Set secrets in Streamlit Cloud: OPENAI_API_KEY, SERPER_API_KEY (optional)

Requirements (example requirements.txt):
streamlit
openai
langchain
faiss-cpu
tqdm
pypdf
python-multipart
serpapi

Note: If "faiss-cpu" is not available on the host, use "chromadb" as vectorstore or Pinecone.

This file intentionally keeps the UI minimal and polished for use as an Insta-bio link and resume demo.
"""

import streamlit as st
from typing import List, Optional
import os
import tempfile
import json
from typing import List
from langchain_core.documents import Document




# --- Third-party libraries used by the app ---
# langchain, openai, faiss, serpapi, pypdf

try:
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.docstore.document import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.llms import OpenAI
    from langchain.chains import RetrievalQA
except Exception as e:
    st.warning("Some dependencies are missing. See the top comments in the file for requirements.\nIf deploying to Streamlit Cloud, add a requirements.txt with langchain, openai, faiss-cpu, pypdf, serpapi, streamlit")
    # We still proceed, but functions will check imports

# ----------------------- Helper utilities -----------------------

def set_page_style():
    st.set_page_config(page_title="FitTwin — Digital Twin AI Coach", page_icon="💪", layout="centered")
    st.markdown("""
    <style>
    .stApp { background: linear-gradient(180deg,#0f172a, #001219); color: #e6eef8 }
    .sidebar .stButton>button { background-color:#0ea5a4 }
    .chat-bubble-user{ background:#047857; color:white; padding:10px; border-radius:10px; margin:6px 0}
    .chat-bubble-assistant{ background:#1e293b; color:#e6eef8; padding:10px; border-radius:10px; margin:6px 0}
    </style>
    """, unsafe_allow_html=True)


# ----------------------- Simple Web Search (SerpAPI) -----------------------

def serpapi_search(query: str, serpapi_key: str, num_results: int = 3) -> List[str]:
    """Use SerpAPI to fetch top search snippets. Requires SERPAPI key set as environment or provided."""
    try:
        from serpapi import GoogleSearch
    except Exception as e:
        st.error("serpapi library not installed. Add 'serpapi' to requirements.txt to use live web search.")
        return []

    params = {
        "q": query,
        "engine": "google",
        "num": num_results,
        "api_key": serpapi_key,
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    snippets = []
    if "organic_results" in results:
        for r in results["organic_results"][:num_results]:
            title = r.get("title", "")
            snippet = r.get("snippet", "")
            link = r.get("link", "")
            combined = f"{title}\n{snippet}\n{link}"
            snippets.append(combined)
    return snippets


# ----------------------- Knowledge base utilities -----------------------

@st.cache_resource
def init_embeddings():
    # Requires OPENAI_API_KEY set in environment or streamlit secrets
    return OpenAIEmbeddings()


def load_uploaded_files(files) -> List[Document]:
    """Extract text from uploaded files (PDF/txt/md) and return LangChain Documents"""
    docs = []
    for uploaded_file in files:
        fname = uploaded_file.name
        suffix = fname.split('.')[-1].lower()
        content = ""
        if suffix in ("pdf",):
            try:
                from pypdf import PdfReader
                pdf = PdfReader(uploaded_file)
                for p in pdf.pages:
                    text = p.extract_text() or ""
                    content += text + "\n"
            except Exception as e:
                try:
                    content = uploaded_file.getvalue().decode("utf-8")
                except Exception:
                    content = ""
        else:
            # txt, md, csv etc.
            try:
                content = uploaded_file.getvalue().decode("utf-8")
            except Exception:
                content = ""
        if content.strip():
            docs.append(Document(page_content=content, metadata={"source": fname}))
    return docs


@st.cache_resource
def build_vectorstore(documents: List[Document], embeddings) -> Optional[FAISS]:
    if not documents:
        return None
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    splits = []
    for d in documents:
        splits += text_splitter.split_text(d.page_content)
    split_docs = [Document(page_content=t, metadata={"source": d.metadata.get("source", "uploaded")}) for d in [Document(page_content=s) for s in splits]]
    try:
        store = FAISS.from_documents(split_docs, embeddings)
        return store
    except Exception as e:
        st.error(f"Failed to create FAISS vectorstore: {e}")
        return None


# ----------------------- Orchestration: MCP-lite -----------------------

def assemble_context(query: str, vectorstore: Optional[FAISS], embeddings, use_web: bool, serpapi_key: str) -> List[str]:
    """Return list of context snippets from (1) local vectorstore and (2) web search snippets as needed"""
    contexts = []
    # 1) Local retrieval
    if vectorstore is not None:
        try:
            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
            docs = retriever.get_relevant_documents(query)
            for d in docs:
                contexts.append(f"SOURCE: {d.metadata.get('source','uploaded')}\n{d.page_content[:800]}")
        except Exception as e:
            st.warning(f"Local retrieval failed: {e}")

    # 2) Optional web
    if use_web and serpapi_key:
        snippets = serpapi_search(query, serpapi_key, num_results=3)
        for s in snippets:
            contexts.append(f"WEB_SNIPPET:\n{s}")

    return contexts


# ----------------------- Answer generator -----------------------

def generate_answer(query: str, contexts: List[str], openai_api_key: str) -> str:
    """Calls the LLM with the assembled context to generate a final answer."""
    # We use LangChain OpenAI LLM wrapper if available; otherwise fallback to direct OpenAI call
    prompt = "You are FitTwin — a friendly, accurate fitness coach. Use the provided context snippets and answer concisely. If no context is available, answer from general fitness knowledge, and ask for clarification when needed."
    if contexts:
        prompt += "\n\nCONTEXT: \n" + "\n\n---\n\n".join(contexts[:6])
    prompt += f"\n\nUSER QUERY: {query}\n\nAnswer:" 

    try:
        llm = OpenAI(temperature=0.2, max_tokens=600)
        res = llm(prompt)
        return res
    except Exception as e:
        # fallback direct openai
        try:
            import openai
            openai.api_key = openai_api_key
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini" if False else "gpt-4o-mini" ,
                messages=[{"role":"system","content": "You are FitTwin — a friendly, accurate fitness coach."},
                          {"role":"user","content": prompt}],
                temperature=0.2,
                max_tokens=600,
            )
            return resp["choices"][0]["message"]["content"]
        except Exception as e2:
            return f"LLM call failed: {e2}"


# ----------------------- Streamlit UI -----------------------

def main():
    set_page_style()

    st.sidebar.title("FitTwin — Settings")
    openai_key = st.sidebar.text_input("OpenAI API Key", type="password", value=os.environ.get("OPENAI_API_KEY", ""))
    serpapi_key = st.sidebar.text_input("SerpAPI Key (optional, for live web search)", type="password", value=os.environ.get("SERPER_API_KEY", os.environ.get("SERPAPI_API_KEY", "")))
    use_web = st.sidebar.checkbox("Enable live web search (use SerpAPI)", value=False)

    st.sidebar.markdown("\n---\n**Upload your personal knowledge base (PDF / txt / md)**")
    uploaded_files = st.sidebar.file_uploader("Upload files to build FitTwin's personal dataset", accept_multiple_files=True)
    build_btn = st.sidebar.button("Build Knowledge Base")

    st.title("💪 FitTwin — Your AI Fitness Coach")
    st.markdown("Ask FitTwin anything about **nutrition, workout programming, biomechanics, or mobility & physio**.\n\nLink this page in your Instagram bio and in your resume as a working demo.")

    # Persona card
    col1, col2 = st.columns([1,4])
    with col1:
        st.image("https://images.unsplash.com/photo-1554284126-aa88f22d8d0d?auto=format&fit=crop&w=200&q=60", width=120)
    with col2:
        st.markdown("**Yash's Digital Twin** — evidence-based, friendly, and focused on sustainable progress.")

    # Chat area
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    # Build KB when requested
    if build_btn:
        if not uploaded_files:
            st.sidebar.error("Upload at least one file before building the knowledge base.")
        else:
            with st.spinner("Building knowledge base — extracting text and embeddings..."):
                docs = load_uploaded_files(uploaded_files)
                embeddings = init_embeddings()
                store = build_vectorstore(docs, embeddings)
                if store:
                    st.session_state.vectorstore = store
                    st.sidebar.success("Knowledge base built — ready for RAG retrieval!")
                else:
                    st.sidebar.error("Failed to build vectorstore. Check logs in the app.")

    # Input form
    with st.form(key="query_form"):
        user_query = st.text_input("Ask FitTwin:", placeholder="e.g., How many grams of protein should I eat to preserve muscle on a 2000 kcal diet?", key="query_input")
        submitted = st.form_submit_button("Ask")

    if submitted and user_query.strip():
        st.session_state.messages.append({"role": "user", "content": user_query})
        # Assemble context
        embeddings = None
        try:
            embeddings = init_embeddings()
        except Exception:
            pass
        contexts = assemble_context(user_query, st.session_state.vectorstore, embeddings, use_web, serpapi_key if use_web else None)
        with st.spinner("Generating answer..."):
            answer = generate_answer(user_query, contexts, openai_key)
        st.session_state.messages.append({"role": "assistant", "content": answer})

    # Display conversation
    for m in st.session_state.messages:
        if m["role"] == "user":
            st.markdown(f"<div class=\"chat-bubble-user\">**You:** {m['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class=\"chat-bubble-assistant\">**FitTwin:** {m['content']}</div>", unsafe_allow_html=True)

    # Footer / deploy instructions
    st.markdown("---")
    st.markdown("**Deploy & Insta-bio:** Add this app's public URL to your Instagram bio. For a cleaner short-link use a custom domain or Linktree." )
    st.markdown("**Resume line:** Built `FitTwin` — an AI digital twin for fitness (nutrition, programming, biomechanics, physio) using RAG + live web retrieval + LLMs — deployed as a public Streamlit app.")


if __name__ == '__main__':
    main()

