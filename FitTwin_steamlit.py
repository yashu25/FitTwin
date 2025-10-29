import streamlit as st
import requests

# -------------------- ⚙️ PAGE SETUP --------------------
st.set_page_config(page_title="FitTwin AI", page_icon="💪", layout="wide")

st.title("💪 FitTwin — Your AI Fitness Coach")

# -------------------- 🌗 THEME TOGGLE --------------------
theme = st.toggle("🌙 Dark / ☀️ Light Mode", value=True)
if theme:
    st.markdown("<style>body{background-color:#0e1117;color:white}</style>", unsafe_allow_html=True)
else:
    st.markdown("<style>body{background-color:#FFFFFF;color:black}</style>", unsafe_allow_html=True)

# -------------------- 🔑 API KEYS --------------------
with st.sidebar:
    st.header("🔐 API Key (Hugging Face)")
    hf_key = st.text_input("Enter your Hugging Face Access Token", type="password")
    st.markdown("---")
    st.write("💬 Ask about: Nutrition, Workout, Biomechanics, Mobility, Physio stretches")

# -------------------- 🤖 FUNCTION --------------------
def fit_twin_answer(query, hf_key):
    if not hf_key:
        return "⚠️ Please enter your Hugging Face API Token in the sidebar."

    headers = {"Authorization": f"Bearer {hf_key}"}
    API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
    payload = {"inputs": f"You are FitTwin, a certified coach in nutrition, biomechanics and mobility.\nUser query: {query}"}

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            data = response.json()
            return data[0]["generated_text"]
        else:
            return f"⚠️ API Error: {response.text}"
    except Exception as e:
        return f"⚠️ Request Error: {e}"

# -------------------- 💬 CHAT UI --------------------
st.markdown("### 🤖 Chat with FitTwin")
query = st.text_area("Ask FitTwin anything:", placeholder="e.g., Best post-workout protein meal?")
image = st.file_uploader("Optional: Upload an image (future posture analysis)", type=["jpg","png"])

if st.button("💪 Get Answer"):
    if not query.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Thinking..."):
            answer = fit_twin_answer(query, hf_key)
            st.success("✅ Answer:")
            st.write(answer)
