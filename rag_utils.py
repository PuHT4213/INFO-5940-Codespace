import os
from typing import List

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain_core.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferMemory

from pypdf import PdfReader


def extract_text_from_pdf(file) -> str:
    """Extract text from a PDF file-like object. Returns empty string on error."""
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
    """Convert uploaded .txt/.pdf files into chunked LangChain Documents."""
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
    """Create and return a persisted Chroma vectorstore from docs."""
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
    vectordb: Chroma, openai_api_key: str | None = None, model_name: str = "gpt-3.5-turbo", use_memory: bool = True
) -> ConversationalRetrievalChain:
    """Create a ConversationalRetrievalChain with optional memory."""
    llm = ChatOpenAI(model_name=model_name, temperature=0, openai_api_key=openai_api_key)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    memory = None
    if use_memory:
        # Set up conversation memory to retain chat history
        memory = ConversationBufferMemory(
            memory_key="chat_history",
                return_messages=True,
            input_key="question",
            output_key="answer",
        )

    # Use condense prompt and a QA prompt that include chat_history
    condense_prompt = CONDENSE_QUESTION_PROMPT

    qa_template = (
        "You are an assistant that answers questions based on provided context and the recent chat history."
        "\n\nChat history:\n{chat_history}\n\nContext:\n{context}\n\nQuestion: {question}\nHelpful Answer:"
    )
    qa_prompt = PromptTemplate(template=qa_template, input_variables=["context", "question", "chat_history"])

    # Pass our custom prompts into the chain factory so both question generation
    # and final answering will see the chat history.
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        condense_question_prompt=condense_prompt,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        return_source_documents=True,
        output_key="answer",
        memory=memory,
    )
    return qa_chain
