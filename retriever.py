"""
Retrieval Module

Handles the runtime query → retrieve → answer pipeline.
"""
import logging
from typing import Optional

from langchain_chroma import Chroma
from langchain.schema import HumanMessage, SystemMessage, AIMessage

import config
from ingest import get_embeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_llm():
    """
    Factory function: returns the right LLM based on config.MODE.
    """
    if config.MODE == "openai":
        from langchain_openai import ChatOpenAI
        logger.info(f"Using OpenAI LLM: {config.OPENAI_LLM_MODEL}")
        return ChatOpenAI(
            model=config.OPENAI_LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
            openai_api_key=config.OPENAI_API_KEY,
        )
    else:
        from langchain_ollama import ChatOllama
        logger.info(f"Using Ollama LLM: {config.OLLAMA_MODEL}")
        return ChatOllama(
            model=config.OLLAMA_MODEL,
            temperature=config.LLM_TEMPERATURE,
            num_predict=config.LLM_MAX_TOKENS,
            base_url=config.OLLAMA_BASE_URL,
        )


class PolicyRetriever:
    """
    Orchestrates retrieval-augmented generation for policy questions.
    """

    def __init__(self, persist_dir: str = None):
        """Initialize the retriever with vector store and LLM connections."""
        persist_dir = persist_dir or config.CHROMA_PERSIST_DIR

        # Load existing vector store (uses same embedding provider as ingestion)
        self.embeddings = get_embeddings()

        self.vector_store = Chroma(
            collection_name=config.CHROMA_COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=persist_dir,
        )

        # Initialize LLM (OpenAI or Ollama based on config.MODE)
        self.llm = get_llm()

        # Conversation memory: list of (user_query, assistant_response) tuples
        self.memory: list[tuple[str, str]] = []

        logger.info(
            f"PolicyRetriever initialized | "
            f"Mode: {config.MODE} | "
            f"Vectors: {self.vector_store._collection.count()}"
        )

    def _detect_policy_filter(self, query: str) -> Optional[dict]:
        """
        Detect if the query targets a specific policy type.
        """
        query_lower = query.lower()

        policy_keywords = {
            "home": ["home", "house", "property", "dwelling", "flood", "fire"],
            "auto": ["auto", "car", "vehicle", "driving", "collision", "motor", "windshield"],
            "health": ["health", "medical", "dental", "hospital", "doctor", "prescription",
                       "therapy", "maternity", "surgery", "mental"],
        }

        for policy_type, keywords in policy_keywords.items():
            if any(kw in query_lower for kw in keywords):
                logger.info(f"Detected policy filter: {policy_type}")
                return {"policy_type": policy_type}

        # No specific policy detected — search across all
        logger.info("No policy filter detected — searching all documents")
        return None

    def _retrieve_context(self, query: str, filter_dict: Optional[dict] = None) -> list:
        """
        Retrieve relevant chunks using MMR (Maximum Marginal Relevance).

        Returns the raw Document objects so we can extract both content and metadata.
        """
        search_kwargs = {
            "k": config.RETRIEVAL_K,
            "fetch_k": config.RETRIEVAL_MMR_FETCH_K,
            "lambda_mult": config.RETRIEVAL_MMR_LAMBDA,
        }

        if filter_dict:
            search_kwargs["filter"] = filter_dict

        retriever = self.vector_store.as_retriever(
            search_type=config.RETRIEVAL_SEARCH_TYPE,
            search_kwargs=search_kwargs,
        )

        docs = retriever.invoke(query)
        logger.info(f"Retrieved {len(docs)} chunks for query: '{query[:50]}...'")
        for i, doc in enumerate(docs):
            logger.info(
                f"  Chunk {i+1}: [{doc.metadata.get('source_file')}] "
                f"section={doc.metadata.get('section')} "
                f"({len(doc.page_content)} chars)"
            )

        return docs

    def _format_context(self, docs: list) -> str:
        """
        Format retrieved documents into a structured context string for the LLM.
        """
        if not docs:
            return "No relevant policy documents found."

        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source_file", "Unknown")
            section = doc.metadata.get("section", "general")
            policy_type = doc.metadata.get("policy_type", "unknown")

            # Clean source name for citation
            source_label = source.replace("_", " ").replace(".txt", "").replace(".pdf", "").title()

            context_parts.append(
                f"[Source {i}: {source_label} - {section.replace('_', ' ').title()} Section]\n"
                f"{doc.page_content}\n"
            )

        return "\n---\n".join(context_parts)

    def _build_messages(self, query: str, context: str) -> list:
        """
        Assemble the full message list: system prompt + memory + current query.

        """
        messages = []

        # System prompt with injected context
        system_content = config.SYSTEM_PROMPT.format(context=context)
        messages.append(SystemMessage(content=system_content))

        # Conversation memory (last N exchanges)
        recent_memory = self.memory[-config.MEMORY_MAX_EXCHANGES:]
        for user_msg, assistant_msg in recent_memory:
            messages.append(HumanMessage(content=user_msg))
            messages.append(AIMessage(content=assistant_msg))

        # Current query
        messages.append(HumanMessage(content=query))

        return messages

    def query(self, user_query: str) -> dict:
        """
        Main entry point: process a user query and return a response.

        Returns a dict with:
        - answer: The LLM's response text
        - sources: List of source documents used
        - metadata: Retrieval metadata for debugging/display
        """
        # Step 1: Detect if we should filter by policy type
        filter_dict = self._detect_policy_filter(user_query)

        # Step 2: Retrieve relevant chunks
        docs = self._retrieve_context(user_query, filter_dict)

        # Step 3: Format context for the LLM
        context = self._format_context(docs)

        # Step 4: Build message list with memory
        messages = self._build_messages(user_query, context)

        # Step 5: Call the LLM
        logger.info("Calling LLM...")
        response = self.llm.invoke(messages)
        answer = response.content

        # Step 6: Update conversation memory
        self.memory.append((user_query, answer))

        # Step 7: Build source citations
        sources = []
        seen = set()
        for doc in docs:
            source_key = f"{doc.metadata.get('source_file')}:{doc.metadata.get('section')}"
            if source_key not in seen:
                seen.add(source_key)
                sources.append({
                    "file": doc.metadata.get("source_file", "Unknown"),
                    "policy_type": doc.metadata.get("policy_type", "unknown"),
                    "section": doc.metadata.get("section", "general"),
                })

        return {
            "answer": answer,
            "sources": sources,
            "metadata": {
                "chunks_retrieved": len(docs),
                "filter_applied": filter_dict,
                "memory_length": len(self.memory),
            },
        }

    def clear_memory(self):
        """Reset conversation memory."""
        self.memory.clear()
        logger.info("Conversation memory cleared")
