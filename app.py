import shutil
import os
from flask import Flask, request, render_template
from langchain_community.llms import Ollama
from langchain_chroma import Chroma  # Updated import for Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader, WebBaseLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate

app = Flask(__name__)

folder_path = "db"

cached_llm = Ollama(model="mistral")  # or another smaller model you have

embedding = FastEmbedEmbeddings()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512, chunk_overlap=80, length_function=len, is_separator_regex=False
)

raw_prompt = PromptTemplate.from_template(
    """<s>[INST] You are a technical assistant. Answer ONLY using the provided context below. 
If the answer is not in the context, reply: "I could not find the answer in the provided document."
Do NOT use any outside knowledge. 
[/INST]</s>
[INST] {input}
Context: {context}
Answer:
[/INST]
"""
)


@app.route("/", methods=["GET", "POST"])
def home():
    # Home page: handles PDF upload and URL input
    if request.method == "POST":
        pdf_file = request.files.get("file")
        web_url = request.form.get("web_url")
        status = None
        if pdf_file or web_url:
            # --- Clear previous vector store ---
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
        if pdf_file:
            # --- PDF Upload and Chunking ---
            file_name = pdf_file.filename
            save_file = f"pdf/{file_name}"
            pdf_file.save(save_file)
            loader = PDFPlumberLoader(save_file)
            docs = loader.load_and_split()
            # --- Splitting PDF into Chunks ---
            chunks = text_splitter.split_documents(docs)
            # --- Embedding and Storing Chunks ---
            vector_store = Chroma.from_documents(
                documents=chunks, embedding=embedding, persist_directory=folder_path
            )
            status = f"PDF '{file_name}' uploaded and processed."
        elif web_url:
            # --- Web URL Scraping and Chunking ---
            loader = WebBaseLoader(web_url)
            docs = loader.load()
            # --- Splitting Web Content into Chunks ---
            chunks = text_splitter.split_documents(docs)
            # --- Embedding and Storing Chunks ---
            vector_store = Chroma.from_documents(
                documents=chunks, embedding=embedding, persist_directory=folder_path
            )
            status = f"Web page '{web_url}' scraped and processed."
        return render_template("index.html", status=status)
    return render_template("index.html", status=None)


@app.route("/ask_pdf", methods=["POST"])
def askPDFPost():
    try:
        print("Post /ask_pdf called")
        json_content = request.json
        query = json_content.get("query")

        print(f"query: {query}")

        print("Loading vector store")
        vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)

        print("Creating chain")
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 1,  # Reduce k to 1 for speed
                "score_threshold": 0.1,
            },
        )

        document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
        chain = create_retrieval_chain(retriever, document_chain)

        result = chain.invoke({"input": query})

        # Limit context size for LLM
        if "context" in result:
            max_chars = 1000
            total = 0
            filtered_context = []
            for doc in result["context"]:
                if total < max_chars:
                    filtered_context.append(doc)
                    total += len(doc.page_content)
            result["context"] = filtered_context

        print(result)

        sources = []
        if "context" in result:
            for doc in result["context"]:
                sources.append(
                    {"source": doc.metadata.get("source", "N/A"), "page_content": doc.page_content}
                )
        else:
            print("No context found in result.")

        response_answer = {"answer": result.get("answer", "No answer found."), "sources": sources}
        return response_answer
    except Exception as e:
        print(f"Error in /ask_pdf: {e}")
        return {"error": str(e)}, 500


@app.route("/pdf", methods=["POST"])
def pdfPost():
    # --- Clear previous vector store ---
    import shutil, os
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    file = request.files["file"]
    file_name = file.filename
    save_file = "pdf/" + file_name
    file.save(save_file)
    print(f"filename: {file_name}")

    loader = PDFPlumberLoader(save_file)
    docs = loader.load_and_split()
    print(f"docs len={len(docs)}")

    chunks = text_splitter.split_documents(docs)
    print(f"chunks len={len(chunks)}")

    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory=folder_path
    )

    response = {
        "status": "Successfully Uploaded",
        "filename": file_name,
        "doc_len": len(docs),
        "chunks": len(chunks),
    }
    return response


@app.route("/ask", methods=["POST"])
def ask():
    # --- RAG Question Answering ---
    data = request.json
    query = data.get("query")
    print(f"Received query: {query}")
    # --- Load Vector Store ---
    vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)
    # --- Create Retriever ---
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 1, "score_threshold": 0.1},  # Reduce k to 1 for speed
    )
    # --- Create Chain ---
    document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)
    # --- Get Answer ---
    result = chain.invoke({"input": query})
    # Limit context size for LLM
    if "context" in result:
        max_chars = 1000
        total = 0
        filtered_context = []
        for doc in result["context"]:
            if total < max_chars:
                filtered_context.append(doc)
                total += len(doc.page_content)
        result["context"] = filtered_context
    print(f"Chain result: {result}")
    sources = []
    if "context" in result:
        print(f"Context length: {len(result['context'])}")
        for doc in result["context"]:
            print(f"Context chunk: {doc.page_content[:200]}...")  # Print first 200 chars
            sources.append({"source": doc.metadata.get("source", "N/A"), "page_content": doc.page_content})
    else:
        print("No context found in result.")
    print(f"Returning answer: {result.get('answer', 'No answer found.')}")
    response_answer = {"answer": result.get("answer", "No answer found."), "sources": sources}
    return response_answer





def start_app():
    app.run(host="0.0.0.0", port=8080, debug=False)


if __name__ == "__main__":
    start_app()
