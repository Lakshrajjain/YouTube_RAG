"""
Main RAG pipeline module that ties everything together.
"""
import logging
import time
from typing import Dict, List, Optional, Union, Any, Tuple

from langchain_core.documents import Document

import sys
import os
# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.core.transcript_extractor import TranscriptExtractor
from src.core.document_processor import DocumentProcessor
from src.core.embedding_manager import EmbeddingManager
from src.core.retriever import EnhancedRetriever
from src.core.generator import AnswerGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    Main RAG pipeline that ties together all components.
    """
    def __init__(self):
        """
        Initialize the RAG pipeline.
        """
        self.transcript_extractor = TranscriptExtractor()
        self.document_processor = DocumentProcessor()
        self.embedding_manager = EmbeddingManager()
        self.answer_generator = AnswerGenerator()

        # Cache for video metadata
        self.video_metadata = {}

        # Cache for query results
        self.query_cache = {}

        # Cache for document retrieval results
        self.retrieval_cache = {}

        # Cache for processed transcripts
        self.transcript_cache = {}

        logger.info("Initialized RAG pipeline")

    def process_video(self, youtube_url: str, force_reprocess: bool = False) -> Dict:
        """
        Process a YouTube video through the pipeline.

        Args:
            youtube_url: URL of the YouTube video
            force_reprocess: Whether to force reprocessing even if cached

        Returns:
            Dict: Video metadata
        """
        start_time = time.time()

        try:
            # Extract video ID
            video_id = self.transcript_extractor._extract_video_id(youtube_url)

            # Check if video is already processed
            if video_id in self.video_metadata and not force_reprocess:
                logger.info(f"Using cached processing for video {video_id}")
                return self.video_metadata[video_id]

            # Check if transcript is in cache
            if video_id in self.transcript_cache:
                logger.info(f"Using cached transcript for video {video_id}")
                transcript = self.transcript_cache[video_id]["transcript"]
                used_asr = self.transcript_cache[video_id]["used_asr"]
            else:
                # Extract transcript
                logger.info(f"Extracting transcript for video {video_id}")
                transcript, used_asr = self.transcript_extractor.get_transcript(youtube_url)
                # Cache the transcript
                self.transcript_cache[video_id] = {
                    "transcript": transcript,
                    "used_asr": used_asr
                }

            # Process transcript into documents
            logger.info(f"Processing transcript for video {video_id}")
            documents = self.document_processor.process_transcript(video_id, transcript)

            # We don't need a retriever anymore, just use the documents directly
            logger.info(f"Using direct document access for video {video_id}")

            # Store metadata
            metadata = {
                "video_id": video_id,
                "youtube_url": youtube_url,
                "transcript_length": len(transcript),
                "used_asr": used_asr,
                "num_chunks": len(documents),
                "documents": documents,  # Store the documents directly
                "processing_time": time.time() - start_time
            }

            self.video_metadata[video_id] = metadata

            logger.info(f"Processed video {video_id} in {metadata['processing_time']:.2f} seconds")
            return metadata

        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            # Return a minimal metadata object
            metadata = {
                "video_id": youtube_url.split("v=")[1] if "v=" in youtube_url else "unknown",
                "youtube_url": youtube_url,
                "error": str(e),
                "documents": [],  # Empty documents list
                "processing_time": time.time() - start_time
            }
            return metadata

    def query_video(self, youtube_url: str, query: str) -> Dict:
        """
        Query a YouTube video.

        Args:
            youtube_url: URL of the YouTube video
            query: Query string

        Returns:
            Dict: Query result
        """
        start_time = time.time()

        # Create a cache key from the video URL and query
        cache_key = f"{youtube_url}:{query}"

        # Check if the result is already in the cache
        if cache_key in self.query_cache:
            logger.info(f"Using cached result for query: {query}")
            cached_result = self.query_cache[cache_key]
            # Update the processing time to indicate it's from cache
            cached_result["processing_time"] = 0.0
            cached_result["from_cache"] = True
            return cached_result

        try:
            # Process video if not already processed
            metadata = self.process_video(youtube_url)
            video_id = metadata["video_id"]

            # Get all documents directly from the metadata
            documents = metadata.get("documents", [])

            if not documents:
                # If no documents, extract them again
                transcript = self.transcript_extractor.extract_transcript(youtube_url)
                documents = self.document_processor.process_transcript(video_id, transcript)

            # Create a retrieval cache key
            retrieval_cache_key = f"{video_id}:{query}"

            # Check if retrieval result is in cache
            if retrieval_cache_key in self.retrieval_cache:
                logger.info(f"Using cached retrieval result for query: {query}")
                relevant_docs = self.retrieval_cache[retrieval_cache_key]
            else:
                # Try to use hybrid retrieval (combining semantic and keyword-based)
                try:
                    relevant_docs = self._hybrid_retrieval(query, documents, top_k=5)
                except Exception as e:
                    logger.warning(f"Hybrid retrieval failed: {str(e)}. Trying BM25 retrieval.")
                    try:
                        relevant_docs = self._bm25_retrieval(query, documents, top_k=5)
                    except Exception as e2:
                        logger.warning(f"BM25 retrieval failed: {str(e2)}. Falling back to keyword-based retrieval.")
                        relevant_docs = self._keyword_based_retrieval(query, documents, top_k=5)

                # Cache the retrieval result
                self.retrieval_cache[retrieval_cache_key] = relevant_docs

            # Generate answer
            logger.info(f"Generating answer for query: {query}")
            answer = self.answer_generator.generate_answer(query, relevant_docs)

            # Prepare result
            result = {
                "video_id": video_id,
                "youtube_url": youtube_url,
                "query": query,
                "answer": answer,
                "sources": [
                    {
                        "content": doc.page_content,
                        "timestamp": doc.metadata.get("timestamp", 0),
                        "timestamp_str": doc.metadata.get("timestamp_str", "00:00"),
                        "source_url": doc.metadata.get("source", "")
                    }
                    for doc in relevant_docs
                ],
                "processing_time": time.time() - start_time,
                "from_cache": False
            }

            # Store the result in the cache
            self.query_cache[cache_key] = result

            logger.info(f"Processed query in {time.time() - start_time:.2f} seconds")
            return result

        except Exception as e:
            logger.error(f"Error in query_video: {str(e)}")
            # Return an error result
            result = {
                "video_id": youtube_url.split("v=")[1] if "v=" in youtube_url else "unknown",
                "youtube_url": youtube_url,
                "query": query,
                "answer": f"Sorry, I encountered an error while processing your query: {str(e)}. Please try a different query or video.",
                "sources": [],
                "processing_time": time.time() - start_time,
                "error": str(e)
            }
            return result

    def _keyword_based_retrieval(self, query: str, documents: List, top_k: int = 5) -> List:
        """
        Simple keyword-based retrieval method.

        Args:
            query: Query string
            documents: List of documents
            top_k: Number of documents to return

        Returns:
            List of relevant documents
        """
        if not documents:
            return []

        # Preprocess query
        query = query.lower()
        query_terms = set(query.split())

        # Score documents based on keyword matching
        scored_docs = []
        for doc in documents:
            content = doc.page_content.lower()
            score = 0

            # Count matching terms
            for term in query_terms:
                if term in content:
                    score += 1

            # Bonus for exact phrase match
            if query in content:
                score += 3

            scored_docs.append((doc, score))

        # Sort by score (descending)
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Return top_k documents
        return [doc for doc, _ in scored_docs[:top_k]]

    def _bm25_retrieval(self, query: str, documents: List, top_k: int = 5) -> List:
        """
        BM25 retrieval method.

        Args:
            query: Query string
            documents: List of documents
            top_k: Number of documents to return

        Returns:
            List of relevant documents
        """
        try:
            # Import BM25 implementation
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank_bm25 not installed. Please install with: pip install rank-bm25")
            # Fall back to keyword-based retrieval
            return self._keyword_based_retrieval(query, documents, top_k)

        if not documents:
            return []

        # Preprocess documents
        tokenized_docs = []
        for doc in documents:
            # Simple tokenization by splitting on whitespace
            tokens = doc.page_content.lower().split()
            tokenized_docs.append(tokens)

        # Create BM25 index
        bm25 = BM25Okapi(tokenized_docs)

        # Tokenize query
        tokenized_query = query.lower().split()

        # Get BM25 scores
        scores = bm25.get_scores(tokenized_query)

        # Create document-score pairs
        scored_docs = list(zip(documents, scores))

        # Sort by score (descending)
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Return top_k documents
        return [doc for doc, _ in scored_docs[:top_k]]

    def _hybrid_retrieval(self, query: str, documents: List, top_k: int = 5) -> List:
        """
        Hybrid retrieval method combining semantic search and keyword-based search.

        Args:
            query: Query string
            documents: List of documents
            top_k: Number of documents to return

        Returns:
            List of relevant documents
        """
        if not documents:
            return []

        # Get semantic search results
        try:
            semantic_docs = self._semantic_retrieval(query, documents, top_k=top_k)
        except Exception as e:
            logger.warning(f"Semantic retrieval failed: {str(e)}. Using only keyword-based retrieval.")
            semantic_docs = []

        # Get keyword-based results
        keyword_docs = self._keyword_based_retrieval(query, documents, top_k=top_k)

        # If semantic search failed, just return keyword results
        if not semantic_docs:
            return keyword_docs

        # Combine results with deduplication
        seen_docs = set()
        combined_docs = []

        # First add semantic results (higher priority)
        for doc in semantic_docs:
            doc_id = hash(doc.page_content)
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                combined_docs.append(doc)

        # Then add keyword results
        for doc in keyword_docs:
            doc_id = hash(doc.page_content)
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                combined_docs.append(doc)

        # Return top_k documents
        return combined_docs[:top_k]

    def _semantic_retrieval(self, query: str, documents: List, top_k: int = 5) -> List:
        """
        Semantic retrieval method using embeddings.

        Args:
            query: Query string
            documents: List of documents
            top_k: Number of documents to return

        Returns:
            List of relevant documents
        """
        if not documents:
            return []

        # Check if documents is a vector store
        if hasattr(documents, 'similarity_search'):
            # It's a vector store, use its similarity search
            return documents.similarity_search(query, k=top_k)

        # If it's a list of documents, create a temporary vector store
        try:
            from langchain_community.vectorstores import DocArrayInMemorySearch

            # Create a temporary vector store
            vector_store = DocArrayInMemorySearch.from_documents(
                documents,
                self.embedding_manager.embedding_model
            )

            # Perform similarity search
            return vector_store.similarity_search(query, k=top_k)
        except Exception as e:
            logger.warning(f"Error in semantic retrieval: {str(e)}")
            # Fall back to keyword-based retrieval
            return self._keyword_based_retrieval(query, documents, top_k=top_k)