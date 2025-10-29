import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document

st.set_page_config(page_title="FitTwin - Your AI Fitness Twin 💪", layout="wide")

# --- Theme Toggle ---
theme = st.sidebar.radio("🌓 Theme", ["Dark", "Light"])
if theme == "Dark":
    st.markdown("<style>body {background-color: #0E1117; color: #FAFAFA;}</style>", unsafe_allow_html=True)
else:
    st.markdown("<style>body {background-color: #FAFAFA; color: #0E1117;}</style>", unsafe_allow_html=True)

# --- Header ---
st.markdown(f"""
    <div style="text-align:center; padding:20px;">
        <h1 style="color:#00FFAA;">🏋️ FitTwin - Your AI Fitness Coach</h1>
        <p style="font-size:17px;">Ask anything about <b>nutrition, workouts, mobility, biomechanics, or physio</b>.</p>
        <p style="font-size:15px;">Link this page in your Instagram bio as your own digital AI coach 👇</p>
    </div>
""", unsafe_allow_html=True)

# --- Sidebar Settings ---
st.sidebar.title("⚙️ Configuration")
api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
uploaded_files = st.sidebar.file_uploader("Upload your fitness notes / PDFs", accept_multiple_files=True)
st.sidebar.markdown("---")
st.sidebar.markdown("<small>© 2025 FitTwin | Made for Yash 💪</small>", unsafe_allow_html=True)

# --- Stop if API key missing ---
if not api_key:
    st.warning("Please enter your API key to continue.")
    st.stop()

# --- Initialize Models ---
llm = OpenAI(temperature=0.6, openai_api_key=api_key)
embeddings = OpenAIEmbeddings(openai_api_key=api_key)

# --- Load Uploaded Files ---
def load_uploaded_files(files):
    docs = []
    for file in files:
        text = file.read().decode("utf-8", errors="ignore")
        docs.append(Document(page_content=text))
    return docs

if uploaded_files:
    data = load_uploaded_files(uploaded_files)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(data)
    db = FAISS.from_documents(chunks, embeddings)
    retriever = db.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    st.success("✅ Knowledge base updated with your files.")
else:
    qa_chain = None
    st.info("No files uploaded — FitTwin is in general coaching mode 🧠")

# --- Chat Interface ---
query = st.text_input("💬 Ask FitTwin anything:")

if st.button("Ask"):
    if query.strip():
        with st.spinner("Analyzing..."):
            if qa_chain:
                answer = qa_chain.run(query)
            else:
                prompt = f"You are FitTwin, a friendly and evidence-based AI fitness coach. {query}"
                answer = llm(prompt)
        st.markdown(f"**Answer:** {answer}")
    else:
        st.warning("Enter a valid question first.")

# --- Footer ---
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center; color:gray;'>Built with ❤️ using LangChain + Streamlit</div>",
    unsafe_allow_html=True,
)
