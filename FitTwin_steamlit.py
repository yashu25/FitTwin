import streamlit as st
from langchain.llms import OpenAI, HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from PIL import Image
from io import BytesIO
import requests

# -----------------------------------
# PAGE SETTINGS
# -----------------------------------
st.set_page_config(page_title="FitTwin v3 💪", layout="wide", page_icon="🏋️")

# -----------------------------------
# CSS: Gradient Banner + Themes
# -----------------------------------
st.markdown("""
<style>
body {
    transition: all 0.3s ease-in-out;
}
[data-testid="stAppViewContainer"] {
    background-color: var(--bg-color);
    color: var(--text-color);
}

.light-mode {
    --bg-color: #F9FAFB;
    --text-color: #111;
}
.dark-mode {
    --bg-color: #0E1117;
    --text-color: #FAFAFA;
}
.header {
    text-align: center;
    padding: 2rem 0;
    background: linear-gradient(270deg, #00FFAA, #007BFF, #FF00AA);
    background-size: 600% 600%;
    animation: gradientShift 8s ease infinite;
    border-radius: 15px;
    margin-bottom: 1rem;
}
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.chat-bubble {
    background: #202833;
    color: white;
    padding: 1rem;
    border-radius: 12px;
    margin: 10px 0;
}
.light-mode .chat-bubble {
    background: #E5E7EB;
    color: black;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------
# SIDEBAR SETTINGS
# -----------------------------------
theme = st.sidebar.radio("🌓 Theme", ["Light", "Dark"])
theme_class = "dark-mode" if theme == "Dark" else "light-mode"
st.markdown(f"<body class='{theme_class}'>", unsafe_allow_html=True)

st.sidebar.title("⚙️ Configuration")
openai_key = st.sidebar.text_input("🔑 OpenAI API Key", type="password")
hf_key = st.sidebar.text_input("🤗 HuggingFace API Key", type="password")
uploaded_files = st.sidebar.file_uploader("📂 Upload your notes (PDF / txt)", accept_multiple_files=True)
st.sidebar.markdown("---")
st.sidebar.markdown("<small>© 2025 FitTwin | Made for Yash 💪</small>", unsafe_allow_html=True)

# -----------------------------------
# HEADER SECTION
# -----------------------------------
st.markdown("""
<div class='header'>
    <h1 style='color:white;'>🏋️ FitTwin v3 — Your AI Fitness Twin</h1>
    <p>Ask anything about <b>Nutrition, Workout, Mobility, or Physio</b>.</p>
    <p style='font-size:15px;'>Multimodal. Smart. Always Available.</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------
# MODEL SETUP
# -----------------------------------
if not (openai_key or hf_key):
    st.warning("⚠️ Please enter at least one API key to start.")
    st.stop()

def load_uploaded_files(files):
    docs = []
    for f in files:
        text = f.read().decode("utf-8", errors="ignore")
        docs.append(Document(page_content=text))
    return docs

qa_chain = None
if uploaded_files and openai_key:
    docs = load_uploaded_files(uploaded_files)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    db = FAISS.from_documents(chunks, embeddings)
    retriever = db.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_key), retriever=retriever)
    st.success("✅ Your fitness notes are now connected to FitTwin!")

# -----------------------------------
# CORE LOGIC (LLM FALLBACK)
# -----------------------------------
def fit_twin_answer(query):
    try:
        if qa_chain:
            return qa_chain.run(query)
        elif openai_key:
            llm = OpenAI(openai_api_key=openai_key, temperature=0.6)
            return llm(f"You are FitTwin, a friendly AI fitness coach. {query}")
        else:
            raise Exception("Primary model unavailable")
    except Exception:
        if hf_key:
            hf_llm = HuggingFaceHub(
                repo_id="tiiuae/falcon-7b-instruct",
                huggingfacehub_api_token=hf_key,
                model_kwargs={"temperature": 0.6, "max_new_tokens": 512}
            )
            return hf_llm(f"You are FitTwin, a fitness coach. {query}")
        return "⚠️ Both models unavailable."

def generate_image(prompt):
    try:
        headers = {"Authorization": f"Bearer {hf_key}"}
        payload = {"inputs": prompt}
        response = requests.post(
            "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2",
            headers=headers, json=payload)
        return Image.open(BytesIO(response.content))
    except Exception:
        return None

# -----------------------------------
# CHAT UI
# -----------------------------------
query = st.text_input("💬 Ask FitTwin:")
gen_image = st.checkbox("🖼️ Generate Visual Demo")

if st.button("Ask"):
    if query.strip():
        with st.spinner("💭 Thinking like your FitTwin..."):
            answer = fit_twin_answer(query)
            st.markdown(f"<div class='chat-bubble'>💡 {answer}</div>", unsafe_allow_html=True)
            if gen_image and hf_key:
                img = generate_image(f"fitness demonstration: {query}")
                if img:
                    st.image(img, caption="AI-Generated Demo", use_container_width=True)
                else:
                    st.warning("⚠️ Couldn’t generate image.")
    else:
        st.warning("Enter a question to start.")

st.markdown("<hr><div style='text-align:center;'>💫 Built with ❤️ for Yash — FitTwin 2025</div>", unsafe_allow_html=True)
