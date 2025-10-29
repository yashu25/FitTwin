import streamlit as st
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
import openai

# -------------------- ⚙️ PAGE SETUP --------------------
st.set_page_config(page_title="FitTwin AI", page_icon="💪", layout="wide")

st.markdown("""
    <style>
    .main {background-color:#0e1117;color:#FAFAFA;}
    .stTextInput, .stTextArea, .stButton>button {border-radius:12px;}
    </style>
""", unsafe_allow_html=True)

st.title("💪 FitTwin — Your AI Fitness Coach")

# -------------------- 🌗 THEME TOGGLE --------------------
theme = st.toggle("🌙 Dark / ☀️ Light Mode", value=True)
if theme:
    st.markdown("<style>body{background-color:#0e1117;color:white}</style>", unsafe_allow_html=True)
else:
    st.markdown("<style>body{background-color:#FFFFFF;color:black}</style>", unsafe_allow_html=True)

# -------------------- 🔑 API KEYS --------------------
with st.sidebar:
    st.header("🔐 API Keys")
    openai_key = st.text_input("Enter OpenAI API Key (optional)", type="password")
    hf_key = st.text_input("Enter Hugging Face API Token", type="password")
    st.markdown("---")
    st.write("💬 Ask about: Nutrition, Workout, Biomechanics, Mobility, Physio stretches")

# -------------------- 🧠 HUGGINGFACE MODEL --------------------
@st.cache_resource
def load_hf_model():
    generator = pipeline(
        "text-generation",
        model="tiiuae/falcon-7b-instruct",
        tokenizer="tiiuae/falcon-7b-instruct",
        max_new_tokens=512,
        temperature=0.6
    )
    return HuggingFacePipeline(pipeline=generator)

hf_llm = load_hf_model()

# -------------------- 🧬 MULTIMODAL FUNCTION --------------------
def fit_twin_answer(query, image=None):
    if image:
        # Future multimodal handling (e.g., image captioning / pose check)
        return "📸 Multimodal analysis coming soon — please describe your image for now!"
    else:
        try:
            return hf_llm(f"You are FitTwin, a certified coach in nutrition, biomechanics and mobility.\nUser query: {query}")
        except Exception as e:
            return f"⚠️ Model Error: {e}"

# -------------------- 💬 CHAT UI --------------------
st.markdown("### 🤖 Chat with FitTwin")
query = st.text_area("Ask FitTwin anything:", placeholder="e.g., Best pre-workout meal for fat loss?")
image = st.file_uploader("Optional: Upload an image (for future posture analysis)", type=["jpg","png"])

if st.button("💪 Get Answer"):
    if not query.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Thinking..."):
            answer = fit_twin_answer(query, image)
            st.success("✅ Answer:")
            st.write(answer)
