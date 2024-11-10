# Comment when running locally
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
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import sqlite3

def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

load_dotenv()

st.set_page_config(page_title="Legal AI Catbot")
st.title("Legal Chatbot")

# Initialize session state for storing chat history and app components
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'llm_chain' not in st.session_state:
    st.session_state.llm_chain = None

def initialize_app(user_query=None):
    progress_bar = st.progress(0)

    # Load Embedding Model
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    progress_bar.progress(20)

    if os.path.exists('chroma_db'):
        # Load vector store
        vectorstore = Chroma(embedding_function=embeddings, persist_directory="chroma_db")
        progress_bar.progress(80)
    else:
        upload_choice = st.radio("Choose the file type you want to upload:", ('Pickle file', 'PDF files'))
        all_documents = []

        if upload_choice == 'Pickle file':
            cache_file = st.file_uploader("Upload a pickle file", type=['pkl'], accept_multiple_files=False)
            if cache_file is not None:
                all_documents = pickle.load(cache_file)
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
                
            # Cache documents into a pickle file for future use
            if all_documents:
                with open('document_cache.pkl', 'wb') as f:
                    pickle.dump(all_documents, f)
                progress_bar.progress(50)

        # Split the documents into chunks
        if all_documents:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            texts = text_splitter.split_documents(all_documents)
            progress_bar.progress(65)

            # Create vector store
            vectorstore = Chroma.from_documents(texts, embeddings, persist_directory="chroma_db")
            progress_bar.progress(80)
        else:
            st.warning("No documents were processed. Please upload valid files.")
            return None

    # Create LLM Model
    llm = ChatCohere(model="command-r-08-2024", temperature=0.5)
    progress_bar.progress(85)

    # Create prompt template
    system_prompt = """
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
    """

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    retriever = vectorstore.as_retriever()
    if user_query:
        relevant_docs = retriever.invoke(user_query)
        context_documents_str = "\n\n".join(doc.page_content for doc in relevant_docs)
    else:
        context_documents_str = ""

    qa_prompt = qa_prompt.partial(
        history=st.session_state.messages,
        context=context_documents_str
    )
    progress_bar.progress(90)

    llm_chain = { "input": RunnablePassthrough() } | qa_prompt | llm

    progress_bar.progress(100)
    progress_bar.empty()
    return llm_chain

def fnAsk():
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"].content)

    # Get user input
    if st.session_state.get('is_processing', False):
        st.chat_input("Waiting for response...", disabled=True)
    else:
        if user_query := st.chat_input("Ask a legal question"):
            st.session_state.messages.append({"role": "user", "content": HumanMessage(content=user_query)})
            with st.chat_message("user"):
                st.markdown(user_query)
            st.session_state.is_processing = True

            st.session_state.llm_chain = initialize_app(user_query)
            result = st.session_state.llm_chain.invoke(user_query)
            bot_response = result.content

            st.session_state.messages.append({"role": "assistant", "content": AIMessage(content=bot_response)})
            with st.chat_message("assistant"):
                st.markdown(bot_response)
            st.session_state.is_processing = False

if __name__ == "__main__":
    fnAsk()