import streamlit as st
import os
from typing import List

from rag_utils import load_documents_from_uploads, create_vectorstore_from_docs, get_conversational_chain


st.set_page_config(page_title="RAG Chat (INFO-5940)", layout="wide")

st.title("Retrieval-Augmented Generation (RAG) Chat")
st.markdown("Upload .txt or .pdf files, build a vector index, and ask questions grounded in your documents.")

# Sidebar configuration: API key, persistence path, and model selection
with st.sidebar:
    st.header("Configuration")
    # Enter your OpenAI API key (or set OPENAI_API_KEY in the environment).
    api_key = st.text_input("OpenAI API Key", type="password", value=os.environ.get("OPENAI_API_KEY", ""))
    # Where Chroma will persist the vector DB
    persist_directory = st.text_input("Chroma persist directory", value="data/chroma_db")
    # Model selection for generation (display labels correspond to OpenAI model ids)
    model_name = st.selectbox("Model", options=["gpt-4.1", "gpt-4o", "gpt-5"], index=0)
    # Embedding model selection (the code prefixes with 'openai.' when needed)
    embedding_model = st.selectbox("Embedding model", options=["text-embedding-3-small", "text-embedding-3-large"], index=0)
    st.markdown("---")
    st.markdown("Notes: If you prefer to use an environment variable, set OPENAI_API_KEY. ")
    st.markdown("Make sure you have embedding and chat access.")

# File uploader
uploaded_files = st.file_uploader("Upload .txt or .pdf files", type=["txt", "pdf"], accept_multiple_files=True)

# Initialize session state keys used by the app
if "vectordb_initialized" not in st.session_state:
    st.session_state.vectordb_initialized = False
if "vectordb_path" not in st.session_state:
    st.session_state.vectordb_path = persist_directory
if "chain" not in st.session_state:
    st.session_state.chain = None
if "chat_history" not in st.session_state:
    # chat_history stores sequential Q/A pairs for display and to pass to the chain
    st.session_state.chat_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# Load and index documents
if uploaded_files:
    st.sidebar.success(f"{len(uploaded_files)} file(s) selected")
    # Show uploaded file names
    st.write("Uploaded files:")
    for f in uploaded_files:
        st.write(f.name)

    if st.button("Parse & Index Documents"):
        with st.spinner("Parsing files and building vector store (this may take a minute)..."):
            try:
                docs = load_documents_from_uploads(uploaded_files)
                if not docs:
                    st.error("No text extracted from uploaded files.")
                else:
                    st.session_state.uploaded_files = [f.name for f in uploaded_files]

                    # Ensure model names use the OpenAI identifier prefix where
                    # appropriate. The UI exposes short names for convenience;
                    # here we add the 'openai.' prefix if the user did not.
                    if not embedding_model.startswith("openai."):
                        embedding_model = f"openai.{embedding_model}"
                    if not model_name.startswith("openai."):
                        model_name = f"openai.{model_name}"

                    # Build the vector store (embeddings + Chroma) and create
                    # the conversational retrieval chain for Q/A.
                    vectordb = create_vectorstore_from_docs(
                        docs,
                        persist_directory=persist_directory,
                        openai_api_key=api_key or None,
                        embedding_model=embedding_model,
                    )
                    st.session_state.vectordb_initialized = True
                    st.session_state.vectordb_path = persist_directory

                    chain = get_conversational_chain(
                        vectordb, openai_api_key=api_key or None, model_name=model_name
                    )
                    st.session_state.chain = chain
                    st.success("Indexing complete. You can now ask questions in the chat below.")
            except Exception as e:
                # Surface any errors during parsing/indexing to the user in the UI
                st.exception(e)

# Chat interface
st.markdown("---")
st.header("Chat")

if not st.session_state.vectordb_initialized:
    st.info("Upload files and click 'Parse & Index Documents' to start.")
else:
    if st.session_state.chain is None:
        st.error("Chain initialization failed. Re-index the documents.")
    else:
        # Render the full conversation history stored in session state so
        # that previous user and assistant turns persist across reruns.
        # The app stores history as a list of dicts for display:
        #   [{'user': ..., 'assistant': ...}, ...]
        # But the history may also be in tuple form (user, assistant).
        history = st.session_state.get("chat_history", [])
        if history:
            for turn in history:
                try:
                    if isinstance(turn, dict):
                        user_msg = turn.get("user", "")
                        assistant_msg = turn.get("assistant", "")
                    elif isinstance(turn, (list, tuple)) and len(turn) >= 2:
                        user_msg, assistant_msg = turn[0], turn[1]
                    else:
                        # Fallback: attempt attribute access for message-like
                        user_msg = getattr(turn, "user", str(turn))
                        assistant_msg = getattr(turn, "assistant", "")

                    if user_msg:
                        st.chat_message("user").write(user_msg)
                    if assistant_msg:
                        st.chat_message("assistant").write(assistant_msg)
                except Exception:
                    # If any single turn fails to render, skip it and
                    # continue rendering the rest of the history.
                    continue

        user_question = st.chat_input("Ask a question about the uploaded documents")
        if user_question:
            # Display the user's message in the chat UI
            st.chat_message("user").write(user_question)
            with st.spinner("Generating answer..."):
                try:
                    # The chain is stateless regarding memory, so we pass the
                    # application-managed chat history explicitly. LangChain's
                    # conversational chains expect the history in one of a few
                    # formats (commonly a list of (user, assistant) tuples or
                    # a list of Message objects). During the app flow we store
                    # a user-friendly list of dicts for display:
                    #   [{'user': ..., 'assistant': ...}, ...]
                    # Convert that dict-list to the tuple-list format when
                    # calling the chain to avoid "Unsupported chat history"
                    # errors.
                    raw_history = st.session_state.get("chat_history", [])
                    if raw_history and isinstance(raw_history, list) and isinstance(raw_history[0], dict):
                        formatted_chat_history = [
                            (turn.get("user", ""), turn.get("assistant", "")) for turn in raw_history
                        ]
                    else:
                        # Already in an acceptable format or empty
                        formatted_chat_history = raw_history

                    response = st.session_state.chain(
                        {
                            "question": user_question,
                            "chat_history": formatted_chat_history,
                        }
                    )

                    if isinstance(response, dict):
                        answer = response.get("answer") or response.get("result") or str(response)
                        source_docs = response.get("source_documents") or []
                    else:
                        answer = str(response)
                        source_docs = []

                    # Display the assistant's response
                    st.chat_message("assistant").write(answer)

                    # If the chain returned source documents, show concise snippets
                    if source_docs:
                        st.markdown("**Source documents / snippets used:**")
                        for doc in source_docs:
                            # doc may be a LangChain Document or a dict-like object
                            src = getattr(doc, "metadata", None) or (
                                doc.get("metadata") if isinstance(doc, dict) else None
                            )
                            content = getattr(doc, "page_content", None) or (
                                doc.get("page_content") if isinstance(doc, dict) else str(doc)
                            )
                            src_label = src.get("source") if src and isinstance(src, dict) and "source" in src else str(src)
                            st.write(f"- Source: {src_label}")
                            snippet = content[:400].replace("\n", " ")
                            st.caption(snippet + ("..." if len(content) > 400 else ""))

                    # Store the Q/A in session state for future turns and display
                    st.session_state.chat_history.append({"user": user_question, "assistant": answer})

                except Exception as e:
                    st.exception(e)