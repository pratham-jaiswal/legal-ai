# Legal AI - Legal QA System

This is a legal question-answering system that utilizes LangChain to process and query legal documents. The system leverages Large Language Models, Natural Language Processing and Embeddings to provide accurate legal advice based on the provided documents.

> [!NOTE]
> The retrieval-augmented generation (RAG) model utilized is tailored to the Indian legal context.

## Context Documents Used:
- The Bharatiya Nyaya Sanhita, 2023 - *./context/250883_english_01042024.pdf*
- The Constitution of India, 2024 - *./context/20240716890312078.pdf*
- Indian Polity by M. Laxmikanth 7th Edition

> [!NOTE]
> The document "Indian Polity by M. Laxmikanth, 7th Edition" is referenced for context but is not included in this repository. To recreate the Embedding Vector DB and use this file, you need to acquire it separately and place it in the `./context/` directory. Alternatively, you can use the provided `./chroma_db/`, which includes this document.

## Getting Started

1. Clone the repository
    ```sh
    git clone https://github.com/pratham-jaiswal/legal-ai.git
    ```
2. Install the required libraries
    ```sh
    pip install -r requirements.txt
    ```
3. Download and install [Ollama](https://ollama.com/download)
4. Run Ollama
    ```sh
    ollama serve
    ```
5. Pull the required model(s) from Ollama (here `phi3.5`). Refer to [Ollama library](https://ollama.com/library) for more models.
    ```sh
    ollama pull phi3.5
    ```
6. Run the `main.ipynb` file.

## Changes you can make

1. Use a different LLM model. Ollama offers [multiple models](https://ollama.com/library) which you can pull and use. For example
    ```sh
    ollama pull llama3.1:8b
    ```

    ```py
    llm = Ollama(model="llama3.1:8b")
    ```

    You can also use OpenAI, IBM WAtsonX or any other LLM - Ollama is just free and can be pulled to be used locally.

> [!NOTE]
> You should have at least 8 GB of RAM available to run the 7B models, 16 GB to run the 13B models, and 32 GB to run the 33B models. [Source](https://github.com/ollama/ollama?tab=readme-ov-file#model-library).</br>
> Phi3.5 is a 3B model, and I'm on a 8GB RAM i5 9th gen 4 year old laptop, and it takes about 10 mintues to create a vector db with around 4.8 chunks of context data, and 2-5 mins to answer a query.

2. Use different chunking methods <sup>[1](https://python.langchain.com/v0.2/docs/how_to/#text-splitters)</sup> <sup>[2](https://blog.lancedb.com/chunking-techniques-with-langchain-and-llamaindex/)</sup>, or parameters. For example,
    ```py
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=70)
    ```
> [!NOTE]
> The smaller the chunk_size, the more number of chunks which means it will take more time to complete the process.

3. Use different embedding models provided by [sentence-transformers](https://huggingface.co/sentence-transformers#models), [Ollama](https://ollama.com/blog/embedding-models), OpenAI, and more. For example,
    ```py
    embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
    ```

> [!NOTE]
> The size of embedding model, and your hardware specification will affect the time it will take to create the Vector DB.

4. Use a different Vector Store such as [Chroma](https://python.langchain.com/v0.2/docs/integrations/vectorstores/chroma/), [Faiss](https://python.langchain.com/v0.2/docs/integrations/vectorstores/faiss/), and [more](https://python.langchain.com/v0.2/docs/integrations/vectorstores/).

5. Manipulate the prompt to refine or change the responses.

6. Replace the documents in `./context/` with those relevant to your legal department or country.

## Contributing

Feel free to submit issues or pull requests if you have suggestions or improvements.

## License

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/pratham-jaiswal/legal-ai/blob/main/LICENSE) file for details.
