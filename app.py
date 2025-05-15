import os
import shutil
from flask import Flask, request, render_template

# Set USER_AGENT to avoid warning (optional, but recommended)
os.environ.setdefault("USER_AGENT", "RAG-Doc-Assistant/1.0")

# Use the new Ollama import (requires: pip install -U langchain-ollama)
try:
    from langchain_ollama import OllamaLLM
    ollama_class = OllamaLLM
except ImportError:
    from langchain_community.llms import Ollama
    ollama_class = Ollama

# Use the new Chroma import (requires: pip install -U langchain-chroma)
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader, WebBaseLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate

app = Flask(__name__)

folder_path = "db"
pdf_dir = "pdf"

if not os.path.exists(pdf_dir):
    os.makedirs(pdf_dir)

# Use the correct Ollama class
cached_llm = ollama_class(model="mistral")
embedding = FastEmbedEmbeddings()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
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
    status = None
    if request.method == "POST":
        pdf_file = request.files.get("file")
        web_url = request.form.get("web_url")
        # Clear previous vector store
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        if pdf_file:
            file_name = pdf_file.filename
            save_file = os.path.join(pdf_dir, file_name)
            pdf_file.save(save_file)
            loader = PDFPlumberLoader(save_file)
            docs = loader.load_and_split()
            chunks = text_splitter.split_documents(docs)
            if not chunks or all(not c.page_content.strip() for c in chunks):
                status = f"❌ Could not extract any text from PDF '{file_name}'. Please check your file."
                return render_template("index.html", status=status)
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embedding,
                persist_directory=folder_path
            )
            status = f"✅ PDF '{file_name}' uploaded and processed successfully. Now you can ask questions about your document below."
        elif web_url:
            loader = WebBaseLoader(web_url)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = web_url
            chunks = text_splitter.split_documents(docs)
            if not chunks or all(not c.page_content.strip() for c in chunks):
                status = f"❌ Could not extract any text from web page '{web_url}'. Please check the URL."
                return render_template("index.html", status=status)
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embedding,
                persist_directory=folder_path
            )
            status = f"✅ Web page '{web_url}' scraped and processed successfully. Now you can ask questions about this content below."
    return render_template("index.html", status=status)

@app.route("/pdf", methods=["POST"])
def pdfPost():
    file = request.files["file"]
    file_name = file.filename
    save_file = os.path.join(pdf_dir, file_name)
    file.save(save_file)
    loader = PDFPlumberLoader(save_file)
    docs = loader.load_and_split()
    chunks = text_splitter.split_documents(docs)
    if not chunks or all(not c.page_content.strip() for c in chunks):
        return {
            "status": "❌ Could not extract any text from PDF.",
            "filename": file_name,
            "doc_len": len(docs),
            "chunks": 0,
        }
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

@app.route("/ask_pdf", methods=["POST"])
def askPDFPost():
    json_content = request.json
    query = json_content.get("query")
    vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 1,  # Changed from 20 to 1 for single best result
            "score_threshold": 0.1,
        },
    )
    document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)
    result = chain.invoke({"input": query})
    sources = []
    for doc in result.get("context", []):
        sources.append(
            {"source": doc.metadata.get("source", "N/A"), "page_content": doc.page_content}
        )
    response_answer = {"answer": result.get("answer", "No answer found."), "sources": sources}
    return response_answer

# Optional: For direct LLM chat (not RAG)
@app.route("/ai", methods=["POST"])
def aiPost():
    json_content = request.json
    query = json_content.get("query")
    response = cached_llm.invoke(query)
    response_answer = {"answer": response}
    return response_answer

def start_app():
    app.run(host="0.0.0.0", port=8080, debug=False)

if __name__ == "__main__":
    start_app()
