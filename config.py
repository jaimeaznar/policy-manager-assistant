"""
Configuration for the Policy Manager Assistant.
All tuneable parameters are centralized here for easy experimentation and discussion.
"""
import os

# Mode Selection 
# "openai"  → Requires OPENAI_API_KEY, best quality, costs money
# "local"   → Free, no API key needed, runs on CPU, slightly lower quality

MODE = os.getenv("ASSISTANT_MODE", "local")  

# LLM Configuration (used when MODE="openai")
OPENAI_LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1024

# Local LLM settings (used when MODE="local")
OLLAMA_MODEL = "llama3.2"
OLLAMA_BASE_URL = "http://localhost:11434"

# Embedding Configuration 
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# Local embeddings (MODE="local")
LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunking Configuration 
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
SEPARATORS = ["\n\n", "\n", ". ", " "]

# Retrieval Configuration 
RETRIEVAL_K = 4
RETRIEVAL_SEARCH_TYPE = "mmr"
RETRIEVAL_MMR_FETCH_K = 10
RETRIEVAL_MMR_LAMBDA = 0.7

#  Vector Store Configuration
CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLLECTION_NAME = "ServiZurich_policies"

#  Conversation Memory 
MEMORY_MAX_EXCHANGES = 5

#  Document Paths
SAMPLE_DOCS_DIR = "./sample_docs"

#  System Prompt 
SYSTEM_PROMPT = """You are a ServiZurich Insurance policy assistant. Your role is to help \
customers understand their insurance policies, coverage, exclusions, and renewal processes.

RULES:
1. Answer questions using ONLY the provided context from policy documents.
2. If the context does not contain the answer, say: "I don't have that information in \
the available policy documents. Please contact your Zurich agent for assistance."
3. Always cite which document and section your answer comes from using brackets, e.g., \
[Home Policy - Coverage Section].
4. Never make up coverage details, limits, or exclusions that are not in the documents.
5. If asked about something outside insurance policy management, politely redirect.

RESPONSE FORMAT:
- Keep answers to 2-4 sentences for simple factual questions.
- For process questions (e.g., "how do I file a claim?"), use numbered steps. \
Keep each step to one sentence.
- Always lead with the direct answer FIRST, then add detail only if necessary.
- Do NOT repeat the question. Do NOT add disclaimers or filler phrases like \
"Great question!" or "Based on the provided context...".
- If listing amounts or limits, use a short inline format, not a table.

CONTEXT FROM POLICY DOCUMENTS:
{context}
"""

# API Key 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
