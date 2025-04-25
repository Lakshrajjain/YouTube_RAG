"""
Module for processing documents (transcripts) into chunks for indexing.
"""
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import sys
import os
# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config.config import DOCUMENT_SETTINGS, PROCESSED_DATA_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Class for processing documents (transcripts) into chunks for indexing.
    """
    def __init__(self,
                 chunk_size: int = DOCUMENT_SETTINGS["chunk_size"],
                 chunk_overlap: int = DOCUMENT_SETTINGS["chunk_overlap"],
                 min_chunk_size: int = DOCUMENT_SETTINGS["min_chunk_size"],
                 max_chunk_size: int = DOCUMENT_SETTINGS["max_chunk_size"]):
        """
        Initialize the DocumentProcessor.

        Args:
            chunk_size: Default chunk size
            chunk_overlap: Default chunk overlap
            min_chunk_size: Minimum chunk size
            max_chunk_size: Maximum chunk size
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def _estimate_content_density(self, text: str) -> float:
        """
        Estimate the content density of text to determine optimal chunk size.

        Args:
            text: Text to analyze

        Returns:
            float: Content density score (0-1)
        """
        # Calculate various metrics to estimate content density

        # 1. Average word length
        words = re.findall(r'\b\w+\b', text)
        avg_word_length = sum(len(word) for word in words) / max(1, len(words))

        # 2. Ratio of special characters and numbers
        special_chars = re.findall(r'[^\w\s]', text)
        numbers = re.findall(r'\d', text)
        special_ratio = (len(special_chars) + len(numbers)) / max(1, len(text))

        # 3. Sentence complexity (average sentence length)
        sentences = re.split(r'[.!?]+', text)
        avg_sentence_length = sum(len(sentence.split()) for sentence in sentences) / max(1, len(sentences))

        # Normalize metrics and combine into a density score
        norm_word_length = min(1.0, avg_word_length / 10)  # Normalize to 0-1
        norm_sentence_length = min(1.0, avg_sentence_length / 30)  # Normalize to 0-1

        # Higher density = longer words, more complex sentences, fewer special chars
        density = (0.4 * norm_word_length + 0.4 * norm_sentence_length + 0.2 * (1 - special_ratio))

        return min(1.0, max(0.0, density))

    def _get_dynamic_chunk_size(self, text: str) -> int:
        """
        Determine the optimal chunk size based on content density.

        Args:
            text: Text to analyze

        Returns:
            int: Optimal chunk size
        """
        density = self._estimate_content_density(text)

        # Scale chunk size based on density
        # Higher density = smaller chunks (more information per token)
        # Lower density = larger chunks (less information per token)
        chunk_size = self.min_chunk_size + (1 - density) * (self.max_chunk_size - self.min_chunk_size)

        return int(chunk_size)

    def process_transcript(self,
                          video_id: str,
                          transcript: List[Dict],
                          metadata: Optional[Dict] = None) -> List[Document]:
        """
        Process a transcript into chunks for indexing.

        Args:
            video_id: YouTube video ID
            transcript: List of transcript segments
            metadata: Additional metadata to include with each document

        Returns:
            List[Document]: List of processed documents
        """
        # Combine transcript segments into a single text
        full_text = " ".join([segment["text"] for segment in transcript])

        # Determine optimal chunk size based on content density
        dynamic_chunk_size = self._get_dynamic_chunk_size(full_text)
        logger.info(f"Using dynamic chunk size of {dynamic_chunk_size} for video {video_id}")

        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=dynamic_chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
            length_function=len
        )

        # Split text into chunks
        chunks = text_splitter.split_text(full_text)

        # Create documents with metadata
        documents = []

        # Create a mapping of text positions to timestamps
        text_position = 0
        position_to_timestamp = {}

        for segment in transcript:
            segment_text = segment["text"]
            segment_start = segment["start"]
            position_to_timestamp[text_position] = segment_start
            text_position += len(segment_text) + 1  # +1 for the space

        # Find the closest timestamp for each chunk
        for i, chunk in enumerate(chunks):
            # Find the position of this chunk in the full text
            chunk_start = full_text.find(chunk[:50])  # Use the first 50 chars to locate

            # Find the closest timestamp
            closest_pos = max([pos for pos in position_to_timestamp.keys() if pos <= chunk_start], default=0)
            timestamp = position_to_timestamp.get(closest_pos, 0)

            # Format timestamp as MM:SS
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            timestamp_str = f"{minutes:02d}:{seconds:02d}"

            # Create document with metadata
            doc_metadata = {
                "video_id": video_id,
                "chunk_id": i,
                "timestamp": timestamp,
                "timestamp_str": timestamp_str,
                "source": f"https://www.youtube.com/watch?v={video_id}&t={int(timestamp)}s"
            }

            # Add additional metadata if provided
            if metadata:
                doc_metadata.update(metadata)

            documents.append(Document(page_content=chunk, metadata=doc_metadata))

        # Save processed documents
        self._save_processed_documents(video_id, documents)

        return documents

    def _save_processed_documents(self, video_id: str, documents: List[Document]) -> None:
        """
        Save processed documents to disk.

        Args:
            video_id: YouTube video ID
            documents: List of processed documents
        """
        # Create directory if it doesn't exist
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Save documents
        docs_data = [{
            "page_content": doc.page_content,
            "metadata": doc.metadata
        } for doc in documents]

        with open(PROCESSED_DATA_DIR / f"{video_id}.json", "w") as f:
            json.dump(docs_data, f)

    def load_processed_documents(self, video_id: str) -> List[Document]:
        """
        Load processed documents from disk.

        Args:
            video_id: YouTube video ID

        Returns:
            List[Document]: List of processed documents
        """
        file_path = PROCESSED_DATA_DIR / f"{video_id}.json"

        if not file_path.exists():
            logger.warning(f"No processed documents found for video {video_id}")
            return []

        with open(file_path, "r") as f:
            docs_data = json.load(f)

        documents = [
            Document(page_content=doc["page_content"], metadata=doc["metadata"])
            for doc in docs_data
        ]

        return documents
