import streamlit as st
import os
from typing import List

from rag_utils import load_documents_from_uploads, create_vectorstore_from_docs, get_conversational_chain

# Use BaseMessage objects for chat history to match LangChain expectations
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage


st.set_page_config(page_title="RAG Chat (INFO-5940)", layout="wide")

st.title("Retrieval-Augmented Generation (RAG) Chat")
st.markdown("Upload .txt or .pdf files, build a vector index, and ask questions grounded in your documents.")

# Sidebar: API key, persistence path and model selection
with st.sidebar:
    st.header("Configuration")
    # OpenAI API key (or set OPENAI_API_KEY in env)
    api_key = st.text_input("OpenAI API Key", type="password", value=os.environ.get("OPENAI_API_KEY", ""))
    # Chroma persist directory
    persist_directory = st.text_input("Chroma persist directory", value="data/chroma_db")
    # Model selection
    model_name = st.selectbox("Model", options=["gpt-4.1", "gpt-4o", "gpt-5"], index=0)
    # Embedding model
    embedding_model = st.selectbox("Embedding model", options=["text-embedding-3-small", "text-embedding-3-large"], index=0)
    # Reset chat history if needed
    if st.button("Reset chat history"):
        st.session_state.chat_history = []
        st.success("Chat history cleared")
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
    # store conversation as BaseMessage list
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

                    # Ensure model names use OpenAI identifier prefix
                    if not embedding_model.startswith("openai."):
                        embedding_model = f"openai.{embedding_model}"
                    if not model_name.startswith("openai."):
                        model_name = f"openai.{model_name}"

                    # Build vector store and create the conversational chain
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
                    # Mark memory sync flag so we can populate chain.memory once
                    # from any existing session_state.chat_history on first use.
                    st.session_state.memory_synced = False
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
    # Render conversation history
        history = st.session_state.get("chat_history", [])
        if history:
            for msg in history:
                try:
                    # Render BaseMessage or legacy formats
                    if hasattr(msg, "content") and hasattr(msg, "type"):
                        role = getattr(msg, "type", "human")
                        # map language model message types to Streamlit chat roles
                        if role in ("human", "user"):
                            st.chat_message("user").write(msg.content)
                        else:
                            st.chat_message("assistant").write(msg.content)
                    elif isinstance(msg, dict):
                        # legacy dict
                        user_msg = msg.get("user", "")
                        assistant_msg = msg.get("assistant", "")
                        if user_msg:
                            st.chat_message("user").write(user_msg)
                        if assistant_msg:
                            st.chat_message("assistant").write(assistant_msg)
                    elif isinstance(msg, (list, tuple)) and len(msg) >= 2:
                        # legacy tuple (user, assistant)
                        user_msg, assistant_msg = msg[0], msg[1]
                        if user_msg:
                            st.chat_message("user").write(user_msg)
                        if assistant_msg:
                            st.chat_message("assistant").write(assistant_msg)
                    else:
                        # Fallback: string or unknown object
                        text = str(msg)
                        st.chat_message("assistant").write(text)
                except Exception:
                    # If any single turn fails to render, skip it and continue
                    continue

        user_question = st.chat_input("Ask a question about the uploaded documents")
        if user_question:
            # show user's message
            st.chat_message("user").write(user_question)
            with st.spinner("Generating answer..."):
                try:
                    # Prepare chat history in the format expected by the chain
                    raw_history = st.session_state.get("chat_history", [])

                    def normalize_chat_history_to_messages(ch) -> List[BaseMessage]:
                        """Convert stored history into a list of HumanMessage/AIMessage."""
                        out: List[BaseMessage] = []
                        if not ch:
                            return out

                        # If a single string, attempt to split into human/assistant lines
                        if isinstance(ch, str):
                            parts = [p.strip() for p in ch.split("\n") if p.strip()]
                            # If formatted with Human:/Assistant: markers, parse them
                            if parts and (parts[0].startswith("Human:") or parts[0].startswith("Assistant:")):
                                i = 0
                                while i < len(parts):
                                    line = parts[i]
                                    if line.startswith("Human:"):
                                        human = line[len("Human:"):].strip()
                                        out.append(HumanMessage(content=human))
                                        i += 1
                                    elif line.startswith("Assistant:"):
                                        ai = line[len("Assistant:"):].strip()
                                        out.append(AIMessage(content=ai))
                                        i += 1
                                    else:
                                        out.append(HumanMessage(content=line))
                                        i += 1
                                return out
                            # fallback: treat the whole string as a single human message
                            return [HumanMessage(content=ch)]

                        # If it's a list, normalize each element
                        if isinstance(ch, list):
                            for item in ch:
                                if isinstance(item, tuple) and len(item) >= 2:
                                    human, ai = item[0], item[1]
                                    out.append(HumanMessage(content=str(human)))
                                    # append assistant only if present
                                    if ai is not None and str(ai) != "":
                                        out.append(AIMessage(content=str(ai)))
                                elif isinstance(item, dict):
                                    out.append(HumanMessage(content=str(item.get("user", ""))))
                                    assistant = item.get("assistant", "")
                                    if assistant:
                                        out.append(AIMessage(content=str(assistant)))
                                elif hasattr(item, "content") and hasattr(item, "type"):
                                    # Already a BaseMessage-like object
                                    out.append(item)
                                elif isinstance(item, str):
                                    # try parse simple Human/Assistant markers
                                    parts = [p.strip() for p in item.split("\n") if p.strip()]
                                    if len(parts) >= 2 and (parts[0].startswith("Human:") or parts[1].startswith("Assistant:")):
                                        h = parts[0][len("Human:"):].strip() if parts[0].startswith("Human:") else parts[0]
                                        a = parts[1][len("Assistant:"):].strip() if parts[1].startswith("Assistant:") else parts[1]
                                        out.append(HumanMessage(content=h))
                                        out.append(AIMessage(content=a))
                                    else:
                                        out.append(HumanMessage(content=item))
                                else:
                                    out.append(HumanMessage(content=str(item)))
                            return out

                        # Fallback: coerce into a single human message
                        return [HumanMessage(content=str(ch))]

                    print("DEBUG: raw_history type before normalize:", type(raw_history))
                    print("DEBUG: raw_history preview:", raw_history if isinstance(raw_history, (list, str)) else str(raw_history)[:200])
                    formatted_chat_history = normalize_chat_history_to_messages(raw_history)

                    # # print chain prompt (dump prompt templates / input vars)
                    # qg = getattr(st.session_state.chain, "question_generator", None)
                    # if qg:
                    #     llm_chain = getattr(qg, "llm_chain", None)
                    #     prompt_obj = getattr(llm_chain, "prompt", None)
                    #     print("Prompt used by the chain's question generator:")
                    #     if prompt_obj:
                    #         template = getattr(prompt_obj, "template", "")
                    #         print(template)

                    # Debug prints (message types/preview)
                    print("DEBUG: calling chain with chat_history messages type:", type(formatted_chat_history))
                    print("DEBUG: chat_history messages preview:", [(type(m), getattr(m, 'content', None)) for m in formatted_chat_history][:10])

                    # Update session history to normalized messages
                    st.session_state.chat_history = formatted_chat_history

                    # Sync messages into chain memory once
                    chain_obj = st.session_state.chain
                    try:
                        if getattr(chain_obj, "memory", None) and not st.session_state.get("memory_synced", False):
                            chat_mem = getattr(chain_obj.memory, "chat_memory", None)
                            if chat_mem and hasattr(chat_mem, "add_messages"):
                                print("SYNC: adding messages to chain.memory.chat_memory")
                                chat_mem.add_messages(st.session_state.chat_history)
                                st.session_state.memory_synced = True
                    except Exception as _e:
                        print("SYNC error:", _e)

                    # Call chain using memory-managed history
                    response = st.session_state.chain.invoke({"question": user_question})
                    print("question:", user_question)  # Debug log
                    print("chat_history:", formatted_chat_history)  # Debug log

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

                    # Append question/answer as messages
                    st.session_state.chat_history.append(HumanMessage(content=str(user_question)))
                    st.session_state.chat_history.append(AIMessage(content=str(answer)))

                except Exception as e:
                    st.exception(e)