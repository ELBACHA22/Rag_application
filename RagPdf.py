import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import os
from huggingface_hub import login
login(token="Your huggingface token")
# Define the list of available LLMs
list_llm = ["meta-llama/Llama-2-7b-chat-hf", "tiiuae/falcon-7b-instruct", "mistralai/Mistral-7B-Instruct-v0.2"]
list_llm_simple = [os.path.basename(llm) for llm in list_llm]

# Hugging Face API Token (optional if needed for downloading models)
api_token = os.getenv("HF_TOKEN")

# Function to process and split documents
def load_doc(file_paths):
    loaders = [PyPDFLoader(x) for x in file_paths]
    pages = []
    for loader in loaders:
        pages.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=64
    )
    doc_splits = text_splitter.split_documents(pages)
    return doc_splits

# Function to create FAISS vector store from document splits
def create_db(splits):
    embeddings = HuggingFaceEmbeddings()
    vectordb = FAISS.from_documents(splits, embeddings)
    return vectordb

# Initialize the QA chain with Hugging Face model and tokenizer
def initialize_qa_chain(llm_model, temperature, max_tokens, top_k, vector_db):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(llm_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(llm_model, trust_remote_code=True)
    
    # Create a HuggingFace pipeline
    hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)  # Use device=0 for GPU
    
    # Define the LLM
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # Memory setup
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True
    )

    # Initialize the retriever from FAISS vector store
    retriever = vector_db.as_retriever()

    # Initialize the ConversationalRetrievalChain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        chain_type="stuff",
        memory=memory,
        return_source_documents=True,
        verbose=False,
    )
    
    return qa_chain

# Streamlit interface
def main():
    st.title("RAG PDF Chatbot")
    st.sidebar.title("Configuration")

    # PDF Upload
    st.sidebar.subheader("Step 1: Upload PDFs")
    uploaded_files = st.sidebar.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        file_paths = [uploaded_file.name for uploaded_file in uploaded_files]
        for file in uploaded_files:
            with open(file.name, "wb") as f:
                f.write(file.getbuffer())
        st.sidebar.success(f"Uploaded {len(uploaded_files)} file(s).")

    # Choose LLM
    st.sidebar.subheader("Step 2: Select LLM")
    selected_llm = st.sidebar.radio("Choose a model", list_llm_simple, index=0)
    selected_llm_full = list_llm[list_llm_simple.index(selected_llm)]

    # Hyperparameters
    st.sidebar.subheader("Step 3: Set Hyperparameters")
    temperature = st.sidebar.slider("Temperature", 0.01, 1.0, 0.5, step=0.1)
    max_tokens = st.sidebar.slider("Max Tokens", 128, 4096, 1024, step=128)
    top_k = st.sidebar.slider("Top-K Sampling", 1, 10, 3, step=1)

    # Initialize database and LLM
    if st.sidebar.button("Initialize Chatbot"):
        with st.spinner("Processing documents..."):
            # Load and process PDFs
            doc_splits = load_doc(file_paths)

            # Create FAISS vector store using the new function
            vector_db = create_db(doc_splits)
            st.success("FAISS database created or loaded.")

        with st.spinner("Initializing LLM..."):
            qa_chain = initialize_qa_chain(selected_llm_full, temperature, max_tokens, top_k, vector_db)
            st.success("Chatbot initialized. Ready to chat!")
            st.session_state["qa_chain"] = qa_chain

    # Chat Interface
    if "qa_chain" in st.session_state:
        st.subheader("Chat with your document")
        message = st.text_input("Ask a question:", placeholder="Type your query here...")
        if message:
            qa_chain = st.session_state["qa_chain"]
            with st.spinner("Generating response..."):
                response = qa_chain.invoke({"question": message})
                answer = response["answer"]
                sources = response.get("source_documents", [])

                # Display the answer
                st.write(f"**Answer:** {answer}")

                # Display sources
                if sources:
                    st.markdown("### Sources:")
                    for i, source in enumerate(sources[:3], start=1):
                        st.markdown(f"**Source {i}:** {source.page_content.strip()}")
                        st.markdown(f"*Page {source.metadata['page'] + 1}*")

if __name__ == "__main__":
    main()
