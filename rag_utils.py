import os
from typing import List

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from pypdf import PdfReader


def extract_text_from_pdf(file) -> str:
    """Extract text from a PDF file-like object.

    This utility uses pypdf.PdfReader to extract text from each page and
    concatenates page text using newlines. If extraction fails, an empty
    string is returned so the caller can decide how to proceed.
    """
    try:
        reader = PdfReader(file)
        text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
        return "\n".join(text)
    except Exception:
        # Return empty string on any extraction error; upstream code will
        # skip documents that yield no text.
        return ""


def load_documents_from_uploads(uploaded_files) -> List[Document]:
    """Convert uploaded Streamlit files into a list of LangChain Documents.

    Supported formats: .txt and .pdf. Text files are decoded (utf-8 with
    a latin-1 fallback). PDFs are processed with pypdf. Each document is
    split into fixed-size overlapping chunks using RecursiveCharacterTextSplitter
    so that retrieval returns concise, focused snippets.

    Returns:
        List[Document]: Each Document contains a chunk of text and metadata
        with the original filename and chunk index.
    """
    docs: List[Document] = []
    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        content = ""

        # Read file contents by type with robust decoding for text files.
        if filename.lower().endswith(".txt"):
            try:
                content = uploaded_file.read().decode("utf-8")
            except Exception:
                # Some text files may use legacy encodings; fall back to latin-1.
                content = uploaded_file.read().decode("latin-1")
        elif filename.lower().endswith(".pdf"):
            # Extract text from PDF pages
            content = extract_text_from_pdf(uploaded_file)
        else:
            # Unsupported file type; skip
            continue

        if not content:
            # Skip empty or unparseable files
            continue

        # Split into chunks for embedding and retrieval. The chosen
        # chunk_size and overlap provide a balance between context and
        # retrieval granularity; adjust if you need smaller or larger chunks.
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " "],
        )
        chunks = splitter.split_text(content)
        for i, chunk in enumerate(chunks):
            metadata = {"source": filename, "chunk": i}
            docs.append(Document(page_content=chunk, metadata=metadata))

    return docs


def create_vectorstore_from_docs(
    docs: List[Document],
    persist_directory: str = "data/chroma_db",
    openai_api_key: str | None = None,
    embedding_model: str = "openai.text-embedding-3-small",
) -> Chroma:
    """Create a Chroma vectorstore from Documents and persist it.

    Args:
        docs: List of LangChain Document objects (typically produced by
            load_documents_from_uploads).
        persist_directory: Directory where Chroma will persist its data.
        openai_api_key: Optional OpenAI API key used by OpenAIEmbeddings.
        embedding_model: Embedding model name to use. The function will
            attempt the requested model and fall back to
            `openai.text-embedding-3-small` if necessary.

    Returns:
        Chroma: the instantiated (and persisted) vectorstore.
    """
    if not docs:
        raise ValueError("No documents to index")

    # Create OpenAIEmbeddings; some accounts or endpoints may not allow
    # certain embedding models, hence the guarded attempt with a fallback.
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model=embedding_model)
    except Exception:
        fallback = "openai.text-embedding-3-small"
        if embedding_model != fallback:
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model=fallback)
        else:
            # If fallback fails as well, propagate the exception to the caller.
            raise

    # Ensure persist directory exists and build the Chroma vectorstore
    os.makedirs(persist_directory, exist_ok=True)
    vectordb = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_directory)
    # Persisting is handled by Chroma; call persist for compatibility with
    # older versions of the library (no-op on newer releases).
    vectordb.persist()
    return vectordb


def get_conversational_chain(
    vectordb: Chroma, openai_api_key: str | None = None, model_name: str = "gpt-3.5-turbo"
) -> ConversationalRetrievalChain:
    """Construct a ConversationalRetrievalChain for retrieval-augmented answers.

    Notes:
        - The chain is created without attaching LangChain memory to avoid
          ambiguity around which output key to store (the chain returns both
          `answer` and `source_documents`). Instead, the application manages
          chat history in Streamlit's session state and passes it in at call time.
        - The retriever's `k` parameter controls how many top chunks are
          retrieved for each question; it is set to 4 here but can be tuned.
    """
    llm = ChatOpenAI(model_name=model_name, temperature=0, openai_api_key=openai_api_key)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        output_key="answer",
    )
    return qa_chain
