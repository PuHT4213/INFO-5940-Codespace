2025-10-26: RAG Streamlit app implementation and configuration changes

Files added/modified:

- `main.py`: Streamlit app — handles file upload (txt/pdf), indexing, and conversational UI.
- `rag_utils.py`: document parsing, chunking, vectorstore creation (Chroma), and chain wiring.
- `requirements.txt`: updated to include `chromadb`.
- `scripts/check_models.py`: script to list models available to the provided OpenAI API key.
- `scripts/models_available.json`: model list output produced during development.
- `README.md`: usage and design notes.

GenAI usage and rationale

- This project was implemented and debugged with the assistance of an AI coding assistant. The assistant helped to:
- help write code comments and the README file (due to my limited English proficiency);
- suggest variable renaming to improve code readability;
- debug and analyze errors encountered during development;
- provide code suggestions and improvements for the parsing and framework modules;
- assist with resolving runtime issues (e.g., embedding model errors, memory or output conflicts);
- support other development tasks such as testing and optimization.

