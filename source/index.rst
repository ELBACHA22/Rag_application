.. PDF_rag documentation master file, created by
   sphinx-quickstart on Sun Dec  1 20:20:17 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PDF_rag documentation
=====================


Overview
============
This application allows users to upload PDF documents and interact with them using a Retrieval-Augmented Generation (RAG) model. The user can ask questions related to the content of the uploaded PDFs, and the system retrieves relevant information, generating responses based on both the document contents and the selected language model (LLM).

The application uses Langchain for document processing, vector database management (FAISS), conversational memory, and integration with Hugging Face’s language models

Installation
============

To install the RAG PDF Chatbot, follow these steps:

1. Clone the repository:

https://github.com/ELBACHA22/Rag_application




2. Install the required dependencies:

pip install -r requirements1.txt


3. Set up your Hugging Face API token:

export HF_TOKEN=your_huggingface_token


Usage
============
To run the application, follow these steps:


1. Run the application:

streamlit run RagPdf.py


2.Upload PDFs:

In the sidebar, the user can upload multiple PDF files.
These PDFs are then processed and split into smaller chunks for better efficiency during retrieval.

3.Select LLM:

The user selects an LLM (e.g., meta-llama/Llama-2-7b-chat-hf, tiiuae/falcon-7b-instruct, or mistralai/Mistral-7B-Instruct-v0.2) that will be used for generating responses.
The list of available models is predefined, and the user can choose which one to use for the conversation.

4.Set Hyperparameters:

Users can adjust the following hyperparameters:
Temperature: Controls the randomness of the model's output.
Max Tokens: Specifies the maximum number of tokens the model will generate.
Top-K Sampling: Determines how many top predictions the model considers when generating a response.

5.Initialize Chatbot:

After uploading the PDFs and configuring the settings, the user can click "Initialize Chatbot."
The system processes the uploaded PDFs, splits them into manageable chunks using RecursiveCharacterTextSplitter, and creates a FAISS vector store to store the document embeddings.

6.Chat with the Document:

Once initialized, users can interact with the document using a chat interface where they can ask questions.
The application will retrieve relevant sections from the PDFs using the FAISS vector database and generate an answer using the selected LLM.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

