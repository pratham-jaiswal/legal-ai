def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
# from langchain_ollama import ChatOllama
from langchain_cohere import ChatCohere
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.memory import ConversationBufferMemory
import pickle
from flask import Flask, request, render_template, jsonify
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Load Embedding Model
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
print('embedding model loaded')

if os.path.exists('chroma_db'): # Load vector store
    vectorstore = Chroma(embedding_function=embeddings, persist_directory="chroma_db")
    print('Vector store loaded')
else: # No Vector store found, create one
    # Load documents from cache or read them
    cache_file = 'cache.pkl'
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            all_documents = pickle.load(f)
        print("Loaded documents from cache.")
    else:
        all_documents = []
        directory = './context/'
        for filename in os.listdir(directory):
            if filename.endswith('.pdf'):
                filepath = os.path.join(directory, filename)
                loader = PyPDFLoader(filepath)
                documents = loader.load()
                all_documents.extend(documents)

        with open(cache_file, 'wb') as f:
            pickle.dump(all_documents, f)
        print("Processed documents and saved to cache.")

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(all_documents)
    print('Document chunked')

    # Create vector store
    vectorstore = Chroma.from_documents(texts, embeddings, persist_directory="chroma_db")
    print('Vector store created')


# Create Model
# llm = ChatOllama(model="phi3.5", temperature=0.5, repeat_penalty=1.5, verbose=False)
llm = ChatCohere(model="command-r-08-2024", temperature=0.5)

prompt_template = """
    You are an AI lawyer specializing in Indian law.\
    Your role is to provide clear, concise, and accurate legal advice based solely on the information from the provided documents and prior conversations with the user.\
    You must always respond as a legal expert and avoid disclaiming your expertise.\
    If an answer is unknown, simply state that and refrain from speculation.\
    Cite relevant law sections, acts, or provisions in your response.\
    Note: The developer has provided the legal documents, not the user.\

Previous conversations:
{history}

Document context:
{context}

Question: {question}
"""

prompt = PromptTemplate(
    template=prompt_template, input_variables=["history", "context", "question"]
)

# Create QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={
        "prompt": prompt,
        "memory": ConversationBufferMemory(
            memory_key="history",
            input_key="question"),
    },
    return_source_documents=False
)

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question', '')

    if question.lower() in ["quit", "exit", "bye"]:
        return jsonify({"response": "Goodbye!"})

    # Run the QA chain with the input query
    result = qa.invoke(question)
    response = result["result"]
    
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)