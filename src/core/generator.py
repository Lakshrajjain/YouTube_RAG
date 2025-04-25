"""
Module for generating answers from retrieved documents.
"""
import logging
import re
from typing import Dict, List, Optional, Union, Any

try:
    # Try to import from langchain_ollama (new package)
    from langchain_ollama import ChatOllama
except ImportError:
    # Fall back to langchain_community (deprecated)
    from langchain_community.chat_models import ChatOllama
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import sys
import os
# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config.config import GENERATION_SETTINGS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnswerGenerator:
    """
    Class for generating answers from retrieved documents.
    """
    def __init__(self,
                 model_name: str = GENERATION_SETTINGS["model_name"],
                 temperature: float = GENERATION_SETTINGS["temperature"],
                 max_tokens: int = GENERATION_SETTINGS["max_tokens"],
                 top_p: float = GENERATION_SETTINGS["top_p"]):
        """
        Initialize the AnswerGenerator.

        Args:
            model_name: Name of the Ollama model for generation
            temperature: Temperature for generation
            max_tokens: Maximum number of tokens to generate
            top_p: Top-p sampling parameter
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

        # Initialize chat model
        self.chat_model = ChatOllama(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )

        logger.info(f"Initialized AnswerGenerator with model {model_name}")

    def _detect_query_type(self, query: str) -> str:
        """
        Detect the type of query to adapt the prompt.

        Args:
            query: Query string

        Returns:
            str: Query type (factual, summarization, exploratory, etc.)
        """
        # Check for summarization queries
        summarization_patterns = [
            r"summarize", r"summary", r"overview", r"brief", r"recap",
            r"what is the video about", r"main points", r"key points"
        ]
        for pattern in summarization_patterns:
            if re.search(pattern, query.lower()):
                return "summarization"

        # Check for exploratory queries
        exploratory_patterns = [
            r"explain", r"how", r"why", r"what is", r"describe",
            r"elaborate", r"tell me about"
        ]
        for pattern in exploratory_patterns:
            if re.search(pattern, query.lower()):
                return "exploratory"

        # Check for comparison queries
        comparison_patterns = [
            r"compare", r"difference", r"similarities", r"versus", r"vs",
            r"better", r"worse", r"pros and cons"
        ]
        for pattern in comparison_patterns:
            if re.search(pattern, query.lower()):
                return "comparison"

        # Default to factual
        return "factual"

    def _create_prompt(self, query_type: str) -> ChatPromptTemplate:
        """
        Create a prompt template based on the query type.

        Args:
            query_type: Type of query

        Returns:
            ChatPromptTemplate: Prompt template
        """
        # Base system message
        system_message = """You are a helpful assistant that answers questions about YouTube videos based on their transcripts.
        Your answers should be based ONLY on the provided context from the video transcript.
        If the answer is not in the context, say "I don't have enough information from the video to answer this question."

        For each piece of information you include in your answer, cite the source using the timestamp in square brackets like this: [MM:SS].
        For example: "The speaker explains the concept of nuclear fusion [05:23] and then discusses its applications [08:45]."

        Keep your answers concise, accurate, and directly related to the question."""

        # Adapt system message based on query type
        if query_type == "summarization":
            system_message += """

            For summarization requests:
            - Provide a concise summary of the main points from the video
            - Organize the summary in a logical flow
            - Include timestamps for each main point
            - Keep the summary to 3-5 key points unless asked for more detail"""

        elif query_type == "exploratory":
            system_message += """

            For explanatory questions:
            - Provide a detailed explanation of the concept
            - Include any definitions, examples, or analogies mentioned in the video
            - Organize your explanation in a logical sequence
            - Make sure to cite specific timestamps for key explanations"""

        elif query_type == "comparison":
            system_message += """

            For comparison questions:
            - Clearly identify the items being compared
            - Structure your answer with clear categories of comparison
            - Present both similarities and differences
            - Use a balanced approach that gives fair treatment to all sides
            - Cite timestamps for each comparison point"""

        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", "Context from the video transcript:\n\n{context}\n\nQuestion: {query}")
        ])

        return prompt

    def _format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into context for the prompt.

        Args:
            documents: List of retrieved documents

        Returns:
            str: Formatted context
        """
        context_parts = []

        for i, doc in enumerate(documents):
            # Get timestamp
            timestamp_str = doc.metadata.get("timestamp_str", "00:00")

            # Format document with timestamp
            context_parts.append(f"[{timestamp_str}] {doc.page_content}")

        return "\n\n".join(context_parts)

    def generate_answer(self, query: str, documents: List[Document]) -> str:
        """
        Generate an answer for a query based on retrieved documents.

        Args:
            query: Query string
            documents: List of retrieved documents

        Returns:
            str: Generated answer
        """
        if not documents:
            return "I don't have any information from the video to answer this question."

        # Detect query type
        query_type = self._detect_query_type(query)
        logger.info(f"Detected query type: {query_type}")

        # Create prompt
        prompt = self._create_prompt(query_type)

        # Format context
        context = self._format_context(documents)

        # Create chain
        chain = prompt | self.chat_model | StrOutputParser()

        # Generate answer
        answer = chain.invoke({
            "query": query,
            "context": context
        })

        return answer
