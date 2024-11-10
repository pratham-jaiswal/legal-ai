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
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
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
history = []

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    query = data.get('question', '')

    if query.lower() in ["quit", "exit", "bye"]:
        return jsonify({"response": "Goodbye!"})

    history.append({"role": "user", "content": HumanMessage(content=query)})

    if query:
        relevant_docs = retriever.invoke(query)
        context_documents_str = "\n\n".join(doc.page_content for doc in relevant_docs)
    else:
        context_documents_str = ""

    qa_prompt_local  = qa_prompt.partial(
        history=history,
        context=context_documents_str
    )

    llm_chain = { "input": RunnablePassthrough() } | qa_prompt_local  | llm

    result = llm_chain.invoke(query)

    history.append({"role": "assistant", "content": AIMessage(content=result.content)})
    
    return jsonify({"response": result.content})

if __name__ == '__main__':
    app.run(debug=True)