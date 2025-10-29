import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document

st.set_page_config(page_title="FitTwin - Your AI Fitness Twin 💪", layout="wide")
st.markdown("<h1 style='text-align:center; color:#00FFAA;'>🏋️ FitTwin - Your AI Fitness Twin</h1>", unsafe_allow_html=True)

st.sidebar.title("⚙️ Configuration")
api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
st.sidebar.markdown("---")
uploaded_files = st.sidebar.file_uploader("Upload your fitness notes / PDFs", accept_multiple_files=True)

if not api_key:
    st.warning("Please enter your API key to continue.")
    st.stop()

llm = OpenAI(temperature=0.6, openai_api_key=api_key)
embeddings = OpenAIEmbeddings(openai_api_key=api_key)

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
    st.success("Knowledge base updated with your files 💾")
else:
    qa_chain = None

st.markdown("### 🤖 Ask FitTwin Anything (Workout, Nutrition, Mobility, Biomechanics)")
query = st.text_input("Enter your question")

if st.button("Ask"):
    if not qa_chain:
        st.warning("Upload files first or provide a context.")
    elif query.strip():
        with st.spinner("Analyzing..."):
            answer = qa_chain.run(query)
        st.markdown(f"**Answer:** {answer}")
    else:
        st.warning("Enter a valid question.")

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray;'>© 2025 FitTwin | Powered by LangChain + Streamlit</div>",
    unsafe_allow_html=True,
)
