__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import pickle
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_cohere import ChatCohere
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.memory import ConversationBufferMemory

from streamlit import logger
import sqlite3

app_logger = logger.get_logger ('SMI_APP' )
app_logger.info(f"Sqlite version: {sglite3.sqlite_version}")
app_logger.info(f"Sys version: {sys.version}")

# Suppress warnings
def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Initialize Streamlit app
st.title("Legal Chatbot")

# Initialize session state for storing chat history and app components
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'qa' not in st.session_state:
    st.session_state.qa = None

# Initialization progress bar
progress_bar = st.progress(0)

# Function to load the vector store and create the QA chain
def initialize_app():
    # Load Embedding Model
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    st.toast('Embedding model loaded')
    progress_bar.progress(20)

    if os.path.exists('chroma_db'):
        # Load vector store
        vectorstore = Chroma(embedding_function=embeddings, persist_directory="chroma_db")
        st.toast('Vector store loaded')
        progress_bar.progress(80)
    else:
        st.toast('No Chroma vector store found. Please upload documents to create a new vector store.')

        upload_choice = st.radio("Choose the file type you want to upload:", ('Pickle file', 'PDF files'))
        all_documents = []

        if upload_choice == 'Pickle file':
            cache_file = st.file_uploader("Upload a pickle file", type=['pkl'], accept_multiple_files=False)
            if cache_file is not None:
                all_documents = pickle.load(cache_file)
                st.toast('Loaded documents from cache')
                progress_bar.progress(50)

        elif upload_choice == 'PDF files':
            pdf_files = st.file_uploader("Upload PDF documents", type=['pdf'], accept_multiple_files=True)
            if pdf_files:
                progress = 20
                for i, pdf_file in enumerate(pdf_files):
                    loader = PyPDFLoader(pdf_file)
                    documents = loader.load()
                    all_documents.extend(documents)
                    progress += int(25 / len(pdf_files))
                    progress_bar.progress(progress)
                st.toast(f'Processed {len(pdf_files)} PDFs.')
                
            # Cache documents into a pickle file for future use
            if all_documents:
                with open('document_cache.pkl', 'wb') as f:
                    pickle.dump(all_documents, f)
                progress_bar.progress(50)
                st.toast('Processed documents and saved to cache')

        # Split the documents into chunks
        if all_documents:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            texts = text_splitter.split_documents(all_documents)
            st.toast('Documents chunked')
            progress_bar.progress(65)

            # Create vector store
            vectorstore = Chroma.from_documents(texts, embeddings, persist_directory="chroma_db")
            st.toast('Vector store created')
            progress_bar.progress(80)
        else:
            st.warning("No documents were processed. Please upload valid files.")
            return None

    # Create LLM Model
    llm = ChatCohere(model="command-r-08-2024", temperature=0.5)
    st.toast('LLM loaded')
    progress_bar.progress(85)

    # Create prompt template
    prompt_template = """
        You are an AI lawyer specializing in Indian law.
        Your role is to provide clear, concise, and accurate legal advice based solely on the information from the provided documents and prior conversations with the user.
        You must always respond as a legal expert and avoid disclaiming your expertise.
        If an answer is unknown, simply state that and refrain from speculation.
        Cite relevant law sections, acts, or provisions in your response.
        Note: The developer has provided the legal documents, not the user.

    Previous conversations:
    {history}

    Document context:
    {context}

    Question: {question}
    """
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["history", "context", "question"]
    )
    progress_bar.progress(90)

    # Create the QA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={
            "prompt": prompt,
            "memory": ConversationBufferMemory(memory_key="history", input_key="question")
        },
        return_source_documents=False
    )
    st.toast('QA Chain created')
    progress_bar.progress(100)
    progress_bar.empty()
    return qa

# Run initialization once
if st.session_state.qa is None:
    st.session_state.qa = initialize_app()

def fnAsk():
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user input
    if user_query := st.chat_input("Ask a legal question"):
        # Append user's question to the chat history
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Process the query using the QA chain
        result = st.session_state.qa.invoke(user_query)
        bot_response = result["result"]

        # Append bot's response to the chat history
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        with st.chat_message("assistant"):
            st.markdown(bot_response)

if __name__ == "__main__":
    fnAsk()