"""
Module for retrieving relevant documents from the vector store.
"""
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

import sys
import os
# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config.config import RETRIEVAL_SETTINGS, GENERATION_SETTINGS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedRetriever(BaseRetriever):
    """
    Enhanced retriever with query rewriting, hybrid search, and reranking.
    """
    def __init__(self,
                 vector_store: Any,
                 top_k: int = RETRIEVAL_SETTINGS["top_k"],
                 mmr_lambda: float = RETRIEVAL_SETTINGS["mmr_lambda"],
                 rerank_top_n: int = RETRIEVAL_SETTINGS["rerank_top_n"],
                 hybrid_alpha: float = RETRIEVAL_SETTINGS["hybrid_alpha"],
                 model_name: str = GENERATION_SETTINGS["model_name"]):
        """
        Initialize the EnhancedRetriever.

        Args:
            vector_store: Vector store (FAISS, DocArrayInMemorySearch, or SimpleVectorStore)
            top_k: Number of documents to retrieve
            mmr_lambda: MMR diversity parameter (0 = max diversity, 1 = max relevance)
            rerank_top_n: Number of documents to rerank
            hybrid_alpha: Weight for hybrid search (0 = BM25 only, 1 = vector only)
            model_name: Name of the Ollama model for query rewriting
        """
        super().__init__()
        # Store the vector store as a class attribute
        self._vector_store = vector_store
        # Store all parameters as instance variables
        self._top_k = top_k
        self._mmr_lambda = mmr_lambda
        self._rerank_top_n = rerank_top_n
        self._hybrid_alpha = hybrid_alpha
        self._model_name = model_name

        # Check if this is a SimpleVectorStore (our fallback implementation)
        self.is_simple_store = hasattr(vector_store, 'documents') and not hasattr(vector_store, 'docstore')

        # If it's a simple store, we don't need to initialize BM25 or other advanced features
        if not self.is_simple_store:
            try:
                # Initialize BM25 index
                self._initialize_bm25()

                # Initialize cross-encoder for reranking
                self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

                # Initialize query rewriter
                self.query_rewriter = self._create_query_rewriter()

                logger.info(f"Initialized EnhancedRetriever with top_k={top_k}, mmr_lambda={mmr_lambda}")
            except Exception as e:
                logger.warning(f"Failed to initialize advanced retrieval features: {str(e)}")
                self.is_simple_store = True  # Fall back to simple retrieval
        else:
            logger.info("Using simple retrieval (no BM25, reranking, or query rewriting)")

    # Define properties to access attributes safely
    @property
    def vector_store(self):
        return self._vector_store

    @property
    def top_k(self):
        return self._top_k

    @property
    def mmr_lambda(self):
        return self._mmr_lambda

    @property
    def rerank_top_n(self):
        return self._rerank_top_n

    @property
    def hybrid_alpha(self):
        return self._hybrid_alpha

    @property
    def model_name(self):
        return self._model_name

    def _initialize_bm25(self) -> None:
        """
        Initialize BM25 index from documents in the vector store.
        """
        try:
            # Get all documents from the vector store
            if hasattr(self.vector_store, 'docstore'):
                documents = self.vector_store.docstore.values()

                # Tokenize documents for BM25
                tokenized_docs = []
                for doc in documents:
                    tokenized_docs.append(doc.page_content.lower().split())

                # Create BM25 index
                self.bm25 = BM25Okapi(tokenized_docs)

                # Create mapping from BM25 index to document ID
                self.bm25_to_doc_id = {i: doc_id for i, doc_id in enumerate(self.vector_store.docstore.keys())}
            else:
                # If the vector store doesn't have a docstore, we can't initialize BM25
                raise ValueError("Vector store doesn't have a docstore, can't initialize BM25")
        except Exception as e:
            logger.warning(f"Failed to initialize BM25 index: {str(e)}")
            # Mark as simple store to use fallback retrieval
            self.is_simple_store = True

    def _create_query_rewriter(self) -> Any:
        """
        Create a query rewriter using ChatOllama.

        Returns:
            Any: Query rewriter chain
        """
        chat_model = ChatOllama(model=self.model_name, temperature=0.3)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that generates multiple search queries based on a user's original query.
            Your goal is to create 3 alternative phrasings or perspectives on the original query to improve search results.

            For example:
            Original: "What is nuclear fusion?"
            Alternative queries:
            1. "Explain nuclear fusion process"
            2. "How does nuclear fusion work"
            3. "Nuclear fusion definition and mechanism"

            Return ONLY the alternative queries as a numbered list, without any additional text."""),
            ("human", "{query}")
        ])

        return prompt | chat_model | StrOutputParser()

    def _rewrite_query(self, query: str) -> List[str]:
        """
        Rewrite a query into multiple alternative queries.

        Args:
            query: Original query

        Returns:
            List[str]: List of rewritten queries
        """
        try:
            rewritten = self.query_rewriter.invoke({"query": query})

            # Parse the numbered list
            queries = []
            for line in rewritten.strip().split('\n'):
                # Remove numbering and whitespace
                line = line.strip()
                if line and (line[0].isdigit() or line[0] in ['•', '-', '*']):
                    # Remove the numbering/bullet and any following characters like ., :, etc.
                    clean_line = line[1:].strip()
                    if clean_line[0] in ['.', ':', ')', ']']:
                        clean_line = clean_line[1:].strip()
                    queries.append(clean_line)

            # If parsing failed, just use the original query
            if not queries:
                queries = [query]

            # Add the original query if it's not already included
            if query not in queries:
                queries.append(query)

            return queries
        except Exception as e:
            logger.warning(f"Error rewriting query: {str(e)}")
            return [query]

    def _hybrid_search(self, query: str, alpha: float = 0.5) -> List[Document]:
        """
        Perform hybrid search combining vector search and BM25.

        Args:
            query: Query string
            alpha: Weight for hybrid search (0 = BM25 only, 1 = vector only)

        Returns:
            List[Document]: List of retrieved documents
        """
        # If this is a simple store, just return all documents
        if self.is_simple_store:
            logger.info("Using simple retrieval for hybrid search")
            try:
                # For our custom SimpleVectorStore
                if hasattr(self.vector_store, 'documents'):
                    return self.vector_store.documents[:self.top_k]
                # For other vector stores that might have a simple retrieval method
                elif hasattr(self.vector_store, 'as_retriever'):
                    retriever = self.vector_store.as_retriever()
                    return retriever._get_relevant_documents(query)
                else:
                    logger.warning("Unknown vector store type, returning empty results")
                    return []
            except Exception as e:
                logger.error(f"Error in simple retrieval for hybrid search: {str(e)}")
                return []

        try:
            # Get vector search results
            vector_results = self.vector_store.similarity_search(
                query=query,
                k=self.rerank_top_n,
                fetch_k=self.rerank_top_n * 2
            )
            vector_doc_ids = [doc.metadata.get("chunk_id") for doc in vector_results]

            # Get BM25 results
            tokenized_query = query.lower().split()
            bm25_scores = self.bm25.get_scores(tokenized_query)

            # Sort by BM25 score
            bm25_doc_indices = np.argsort(bm25_scores)[::-1][:self.rerank_top_n * 2]
            bm25_doc_ids = [self.bm25_to_doc_id[idx] for idx in bm25_doc_indices]

            # Combine results with weighted scores
            combined_doc_ids = set(vector_doc_ids + bm25_doc_ids)

            # Get documents
            combined_docs = [self.vector_store.docstore.get(doc_id) for doc_id in combined_doc_ids if doc_id in self.vector_store.docstore]

            return combined_docs
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            # Fall back to simple retrieval
            self.is_simple_store = True
            return self._hybrid_search(query, alpha)

    def _rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Rerank documents using a cross-encoder model.

        Args:
            query: Query string
            documents: List of documents to rerank

        Returns:
            List[Document]: List of reranked documents
        """
        # If this is a simple store or no documents, just return the documents
        if self.is_simple_store or not documents:
            return documents[:self.top_k]

        try:
            # Prepare document-query pairs for the cross-encoder
            pairs = [(query, doc.page_content) for doc in documents]

            # Get scores from cross-encoder
            scores = self.cross_encoder.predict(pairs)

            # Sort documents by score
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            # Return reranked documents
            return [doc for doc, _ in scored_docs[:self.top_k]]
        except Exception as e:
            logger.error(f"Error in reranking: {str(e)}")
            # Just return the documents without reranking
            return documents[:self.top_k]

    def _get_relevant_docs(self, query: str) -> List[Document]:
        """
        Get relevant documents for a query.

        Args:
            query: Query string

        Returns:
            List[Document]: List of relevant documents
        """
        # If this is a simple store, just return all documents
        if self.is_simple_store:
            logger.info("Using simple retrieval for query")
            try:
                # For our custom SimpleVectorStore
                if hasattr(self.vector_store, 'documents'):
                    return self.vector_store.documents[:self.top_k]
                # For other vector stores that might have a simple retrieval method
                elif hasattr(self.vector_store, 'as_retriever'):
                    retriever = self.vector_store.as_retriever()
                    return retriever._get_relevant_documents(query)
                else:
                    logger.warning("Unknown vector store type, returning empty results")
                    return []
            except AttributeError as e:
                # Handle the case where vector_store is not available
                logger.error(f"AttributeError in simple retrieval: {str(e)}")
                # Return empty list as fallback
                return []
            except Exception as e:
                logger.error(f"Error in simple retrieval: {str(e)}")
                return []

        try:
            # Rewrite query
            queries = self._rewrite_query(query)
            logger.info(f"Rewritten queries: {queries}")

            # Get results for each query
            all_docs = []
            for q in queries:
                # Perform hybrid search
                docs = self._hybrid_search(q, alpha=self.hybrid_alpha)
                all_docs.extend(docs)

            # Remove duplicates
            unique_docs = []
            seen_ids = set()
            for doc in all_docs:
                doc_id = doc.metadata.get("chunk_id")
                if doc_id not in seen_ids:
                    unique_docs.append(doc)
                    seen_ids.add(doc_id)

            # Rerank documents
            reranked_docs = self._rerank_documents(query, unique_docs)

            # Apply MMR for diversity if needed
            if self.mmr_lambda < 1.0 and len(reranked_docs) > self.top_k:
                # Get embeddings for documents
                doc_embeddings = [
                    self.vector_store.embedding_function.embed_documents([doc.page_content])[0]
                    for doc in reranked_docs[:self.rerank_top_n]
                ]

                # Apply MMR
                selected_indices = self._mmr(doc_embeddings, self.mmr_lambda, self.top_k)
                mmr_docs = [reranked_docs[i] for i in selected_indices]
                return mmr_docs

            return reranked_docs[:self.top_k]
        except Exception as e:
            logger.error(f"Error in advanced retrieval: {str(e)}")
            # Fall back to simple retrieval
            self.is_simple_store = True
            return self._get_relevant_docs(query)

    def _mmr(self, doc_embeddings: List[List[float]], lambda_param: float, k: int) -> List[int]:
        """
        Apply Maximum Marginal Relevance to select diverse documents.

        Args:
            doc_embeddings: List of document embeddings
            lambda_param: MMR lambda parameter (0 = max diversity, 1 = max relevance)
            k: Number of documents to select

        Returns:
            List[int]: Indices of selected documents
        """
        # Convert to numpy arrays
        embeddings = np.array(doc_embeddings)

        # Initialize
        selected_indices = [0]  # Start with the highest-ranked document
        remaining_indices = list(range(1, len(embeddings)))

        # Select k-1 more documents
        for _ in range(min(k - 1, len(embeddings) - 1)):
            # Calculate MMR scores
            mmr_scores = []

            for idx in remaining_indices:
                # Calculate similarity to query (already sorted by relevance)
                sim_query = 1.0 - (len(remaining_indices) / len(embeddings))

                # Calculate similarity to already selected documents
                sim_docs = max([
                    np.dot(embeddings[idx], embeddings[sel_idx])
                    for sel_idx in selected_indices
                ]) if selected_indices else 0

                # Calculate MMR score
                mmr_score = lambda_param * sim_query - (1 - lambda_param) * sim_docs
                mmr_scores.append((idx, mmr_score))

            # Select document with highest MMR score
            next_idx, _ = max(mmr_scores, key=lambda x: x[1])
            selected_indices.append(next_idx)
            remaining_indices.remove(next_idx)

        return selected_indices

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """
        Async version of _get_relevant_documents.

        Args:
            query: Query string

        Returns:
            List[Document]: List of relevant documents
        """
        # For now, just call the sync version
        return self._get_relevant_docs(query)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Get relevant documents for a query (required by BaseRetriever).

        Args:
            query: Query string

        Returns:
            List[Document]: List of relevant documents
        """
        return self._get_relevant_docs(query)
