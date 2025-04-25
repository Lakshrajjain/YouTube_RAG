"""
Module for managing embeddings and vector storage.
"""
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import faiss
import numpy as np
try:
    # Try to import from langchain_ollama (new package)
    from langchain_ollama import OllamaEmbeddings
except ImportError:
    # Fall back to langchain_community (deprecated)
    from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from sqlitedict import SqliteDict

import sys
import os
# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config.config import EMBEDDING_SETTINGS, EMBEDDINGS_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingManager:
    """
    Class for managing embeddings and vector storage.
    """
    def __init__(self,
                 model_name: str = EMBEDDING_SETTINGS["model_name"],
                 embedding_dim: int = EMBEDDING_SETTINGS["embedding_dim"],
                 cache_dir: str = EMBEDDING_SETTINGS["cache_dir"]):
        """
        Initialize the EmbeddingManager.

        Args:
            model_name: Name of the Ollama model for embeddings
            embedding_dim: Dimension of the embeddings
            cache_dir: Directory to cache embeddings
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize embedding model
        self.embedding_model = OllamaEmbeddings(
            model=model_name,
            show_progress=True
        )

        # Initialize embedding cache
        self.embedding_cache = SqliteDict(
            filename=str(self.cache_dir / "embedding_cache.sqlite"),
            autocommit=True
        )

        logger.info(f"Initialized EmbeddingManager with model {model_name}")

    def _get_cache_key(self, text: str) -> str:
        """
        Generate a cache key for a text.

        Args:
            text: Text to generate cache key for

        Returns:
            str: Cache key
        """
        import hashlib
        # Use a hash of the text and model name as the cache key
        return f"{self.model_name}_{hashlib.md5(text.encode()).hexdigest()}"

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of texts, using cache when available.
        Optimized with batch processing for faster retrieval.

        Args:
            texts: List of texts to embed

        Returns:
            List[List[float]]: List of embeddings
        """
        if not texts:
            return []

        # Initialize result array with None placeholders
        result_embeddings = [None] * len(texts)
        texts_to_embed = []
        indices_to_embed = []

        # Check cache for each text
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if cache_key in self.embedding_cache:
                result_embeddings[i] = self.embedding_cache[cache_key]
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)

        # If there are texts not in cache, embed them in batches
        if texts_to_embed:
            logger.info(f"Embedding {len(texts_to_embed)} texts")

            # Determine optimal batch size (adjust based on your model and hardware)
            batch_size = min(32, len(texts_to_embed))

            # Process in batches for better performance
            for batch_start in range(0, len(texts_to_embed), batch_size):
                batch_end = min(batch_start + batch_size, len(texts_to_embed))
                batch_texts = texts_to_embed[batch_start:batch_end]
                batch_indices = indices_to_embed[batch_start:batch_end]

                # Get embeddings for the batch
                try:
                    batch_embeddings = self.embedding_model.embed_documents(batch_texts)

                    # Update cache and result array
                    for j, (idx, embedding) in enumerate(zip(batch_indices, batch_embeddings)):
                        cache_key = self._get_cache_key(texts[idx])
                        self.embedding_cache[cache_key] = embedding
                        result_embeddings[idx] = embedding

                except Exception as e:
                    logger.error(f"Error embedding batch: {str(e)}")
                    # Fall back to individual embedding if batch fails
                    for j, idx in enumerate(batch_indices):
                        try:
                            embedding = self.embedding_model.embed_query(texts[idx])
                            cache_key = self._get_cache_key(texts[idx])
                            self.embedding_cache[cache_key] = embedding
                            result_embeddings[idx] = embedding
                        except Exception as e2:
                            logger.error(f"Error embedding individual text: {str(e2)}")
                            # Use a zero vector as fallback
                            result_embeddings[idx] = [0.0] * self.embedding_dim

        # Check if any embeddings are still None (should not happen, but just in case)
        for i, embedding in enumerate(result_embeddings):
            if embedding is None:
                logger.warning(f"Missing embedding for text at index {i}, using fallback")
                result_embeddings[i] = [0.0] * self.embedding_dim

        return result_embeddings

    def create_vector_store(self, documents: List[Document], video_id: str) -> FAISS:
        """
        Create a FAISS vector store from documents.

        Args:
            documents: List of documents to index
            video_id: YouTube video ID

        Returns:
            FAISS: FAISS vector store
        """
        try:
            # Check if we have documents
            if not documents:
                logger.warning(f"No documents provided for video {video_id}")
                raise ValueError(f"No documents to index for video {video_id}")

            # Extract texts from documents
            texts = [doc.page_content for doc in documents]

            # Check if we have text content
            if not any(texts):
                logger.warning(f"Empty document content for video {video_id}")
                raise ValueError(f"Documents have no content for video {video_id}")

            # Create a simple in-memory vector store
            logger.info(f"Creating simple vector store for {len(texts)} documents")

            # Create a simple dictionary-based vector store
            # This is a fallback that doesn't require FAISS
            from langchain_community.vectorstores import DocArrayInMemorySearch

            try:
                # Try to create a DocArrayInMemorySearch vector store
                vector_store = DocArrayInMemorySearch.from_documents(
                    documents,
                    self.embedding_model
                )

                # Save vector store (custom implementation for DocArrayInMemorySearch)
                self._save_vector_store_docarray(vector_store, documents, video_id)

                logger.info(f"Successfully created DocArrayInMemorySearch vector store for video {video_id}")
                return vector_store

            except Exception as e:
                logger.warning(f"Error creating DocArrayInMemorySearch vector store: {str(e)}")

                # Try to create a FAISS vector store as fallback
                try:
                    logger.info("Trying to create FAISS vector store as fallback")
                    vector_store = FAISS.from_documents(
                        documents,
                        self.embedding_model
                    )

                    # Save the FAISS vector store
                    self._save_vector_store(vector_store, video_id)

                    logger.info(f"Successfully created FAISS vector store for video {video_id}")
                    return vector_store

                except Exception as e2:
                    logger.error(f"Error creating FAISS vector store: {str(e2)}")

                    # Create a very simple custom retriever as a last resort
                    from langchain_core.retrievers import BaseRetriever

                    class SimpleRetriever(BaseRetriever):
                        def __init__(self, documents):
                            super().__init__()
                            self.documents = documents
                            # Add necessary attributes that might be accessed
                            self.top_k = 5  # Default value
                            self.mmr_lambda = 0.5
                            self.rerank_top_n = 10
                            self.hybrid_alpha = 0.5
                            self.is_simple_store = True

                        def _get_relevant_documents(self, query):
                            # Just return all documents as a fallback
                            return self.documents[:self.top_k] if len(self.documents) > self.top_k else self.documents

                        async def _aget_relevant_documents(self, query):
                            # Async version just calls the sync version
                            return self._get_relevant_documents(query)

                    # Just return the documents directly
                    logger.info(f"Using direct document access for video {video_id}")
                    return documents

        except Exception as e:
            logger.error(f"Error creating vector store for video {video_id}: {str(e)}")
            raise ValueError(f"Failed to create vector store: {str(e)}")

    def _save_vector_store_docarray(self, vector_store, documents, video_id):
        """
        Save a DocArrayInMemorySearch vector store to disk.

        Args:
            vector_store: DocArrayInMemorySearch vector store
            documents: List of documents
            video_id: YouTube video ID
        """
        try:
            # Create directory if it doesn't exist
            save_dir = self.cache_dir / f"{video_id}_docarray"
            save_dir.mkdir(parents=True, exist_ok=True)

            # Save documents
            with open(save_dir / "documents.pkl", "wb") as f:
                pickle.dump(documents, f)

            logger.info(f"Saved documents for video {video_id}")
        except Exception as e:
            logger.error(f"Error saving DocArrayInMemorySearch vector store: {str(e)}")

    def _save_vector_store(self, vector_store: FAISS, video_id: str) -> None:
        """
        Save a FAISS vector store to disk.

        Args:
            vector_store: FAISS vector store
            video_id: YouTube video ID
        """
        save_path = self.cache_dir / f"{video_id}_faiss"
        vector_store.save_local(str(save_path))
        logger.info(f"Saved vector store for video {video_id} to {save_path}")

    def load_vector_store(self, video_id: str) -> Optional[Any]:
        """
        Load a vector store from disk.

        Args:
            video_id: YouTube video ID

        Returns:
            Optional[Any]: Vector store, or None if not found
        """
        # Try to load FAISS vector store
        faiss_path = self.cache_dir / f"{video_id}_faiss"
        if faiss_path.exists():
            try:
                vector_store = FAISS.load_local(
                    str(faiss_path),
                    self.embedding_model
                )
                logger.info(f"Loaded FAISS vector store for video {video_id} from {faiss_path}")
                return vector_store
            except Exception as e:
                logger.warning(f"Error loading FAISS vector store: {str(e)}")

        # Try to load DocArrayInMemorySearch vector store
        docarray_path = self.cache_dir / f"{video_id}_docarray"
        if docarray_path.exists() and (docarray_path / "documents.pkl").exists():
            try:
                # Load documents
                with open(docarray_path / "documents.pkl", "rb") as f:
                    documents = pickle.load(f)

                # Create a new DocArrayInMemorySearch vector store
                from langchain_community.vectorstores import DocArrayInMemorySearch
                vector_store = DocArrayInMemorySearch.from_documents(
                    documents,
                    self.embedding_model
                )

                logger.info(f"Loaded DocArrayInMemorySearch vector store for video {video_id} from {docarray_path}")
                return vector_store
            except Exception as e:
                logger.warning(f"Error loading DocArrayInMemorySearch vector store: {str(e)}")

        # If no vector store is found, return None
        logger.warning(f"No vector store found for video {video_id}")
        return None
