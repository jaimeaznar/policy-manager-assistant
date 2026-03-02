"""
Ingestion Pipeline

Supported formats: .txt, .pdf, .docx
"""
import os
import re
import logging
from pathlib import Path

from langchain_community.document_loaders import (
    TextLoader,
    PyMuPDFLoader,
    Docx2txtLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_embeddings():
    """
    Factory function: returns the right embedding model based on config.MODE.
    """
    if config.MODE == "openai":
        from langchain_openai import OpenAIEmbeddings
        logger.info(f"Using OpenAI embeddings: {config.OPENAI_EMBEDDING_MODEL}")
        return OpenAIEmbeddings(
            model=config.OPENAI_EMBEDDING_MODEL,
            openai_api_key=config.OPENAI_API_KEY,
        )
    else:
        from langchain_huggingface import HuggingFaceEmbeddings
        logger.info(f"Using local embeddings: {config.LOCAL_EMBEDDING_MODEL}")
        return HuggingFaceEmbeddings(
            model_name=config.LOCAL_EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
        )

#  File Loaders 
LOADER_MAP = {
    ".txt": TextLoader,
    ".pdf": PyMuPDFLoader,
    ".docx": Docx2txtLoader,
}


def load_documents(docs_dir: str) -> list:
    """
    Load all supported documents from a directory.
    Returns a list of LangChain Document objects.
    """
    docs = []
    docs_path = Path(docs_dir)

    if not docs_path.exists():
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")

    for file_path in docs_path.iterdir():
        ext = file_path.suffix.lower()
        if ext not in LOADER_MAP:
            logger.warning(f"Skipping unsupported file: {file_path.name}")
            continue

        logger.info(f"Loading: {file_path.name}")
        loader = LOADER_MAP[ext](str(file_path))
        file_docs = loader.load()

        # Attach source filename as metadata
        for doc in file_docs:
            doc.metadata["source_file"] = file_path.name
        docs.extend(file_docs)

    logger.info(f"Loaded {len(docs)} document(s) from {docs_dir}")
    return docs


def detect_policy_type(filename: str) -> str:
    """
    Infer the policy type from the filename.
    """
    filename_lower = filename.lower()
    if "home" in filename_lower:
        return "home"
    elif "auto" in filename_lower or "car" in filename_lower or "motor" in filename_lower:
        return "auto"
    elif "health" in filename_lower or "medical" in filename_lower:
        return "health"
    elif "life" in filename_lower:
        return "life"
    return "general"


def detect_section(text: str) -> str:
    """
    Heuristically detect the section of a policy document a chunk belongs to.   
    """
    section_patterns = {
        "coverage": r"(?i)(coverage|covered\s+perils|covered\s+events|benefits)",
        "exclusions": r"(?i)(exclusion|not\s+covered|excluded)",
        "deductibles": r"(?i)(deductible|excess|co-?pay)",
        "claims": r"(?i)(claims?\s+process|filing\s+a\s+claim|reimbursement)",
        "renewal": r"(?i)(renewal|policy\s+changes|cancellation|adding|upgrading)",
        "mental_health": r"(?i)(mental\s+health|therapy|psychiatric|substance)",
        "dental": r"(?i)(dental|orthodontic|implant)",
        "no_claims_bonus": r"(?i)(no.?claims\s+bonus|ncb|safe\s+driving)",
    }

    # Check which sections are referenced in this chunk
    detected = []
    for section_name, pattern in section_patterns.items():
        if re.search(pattern, text):
            detected.append(section_name)

    return detected[0] if detected else "general"


def chunk_documents(documents: list) -> list:
    """
    Split documents into chunks with metadata enrichment.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=config.SEPARATORS,
        length_function=len,  # Character-based; token-based would need tiktoken
    )

    chunks = splitter.split_documents(documents)

    # Enrich each chunk with structured metadata
    for chunk in chunks:
        source = chunk.metadata.get("source_file", "unknown")
        chunk.metadata["policy_type"] = detect_policy_type(source)
        chunk.metadata["section"] = detect_section(chunk.page_content)

    logger.info(f"Created {len(chunks)} chunks from {len(documents)} document(s)")

    # Log chunk distribution for debugging
    section_counts = {}
    for chunk in chunks:
        section = chunk.metadata["section"]
        section_counts[section] = section_counts.get(section, 0) + 1
    logger.info(f"Chunk distribution by section: {section_counts}")

    return chunks


def create_vector_store(chunks: list) -> Chroma:
    """
    Embed chunks and store them in ChromaDB.
    """
    embeddings = get_embeddings()

    # Delete existing collection to avoid stale data on re-ingestion
    if os.path.exists(config.CHROMA_PERSIST_DIR):
        import shutil
        shutil.rmtree(config.CHROMA_PERSIST_DIR)
        logger.info("Cleared existing vector store")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=config.CHROMA_COLLECTION_NAME,
        persist_directory=config.CHROMA_PERSIST_DIR,
    )

    logger.info(
        f"Vector store created with {vector_store._collection.count()} vectors "
        f"in '{config.CHROMA_PERSIST_DIR}'"
    )
    return vector_store


def run_ingestion(docs_dir: str = None) -> Chroma:
    """
    Full ingestion pipeline: Load → Chunk → Embed → Store.
    Returns the ChromaDB vector store instance.
    """
    docs_dir = docs_dir or config.SAMPLE_DOCS_DIR

    logger.info("=" * 60)
    logger.info("Starting document ingestion pipeline")
    logger.info("=" * 60)

    # Step 1: Load raw documents
    documents = load_documents(docs_dir)

    # Step 2: Chunk with metadata enrichment
    chunks = chunk_documents(documents)

    # Step 3: Embed and store
    vector_store = create_vector_store(chunks)

    logger.info("=" * 60)
    logger.info("Ingestion complete!")
    logger.info("=" * 60)

    return vector_store


if __name__ == "__main__":
    run_ingestion()
