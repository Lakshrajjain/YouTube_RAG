# Advanced YouTube RAG Pipeline

A comprehensive Retrieval-Augmented Generation (RAG) system for processing YouTube video transcripts, built with open-source tools to ensure scalability, robustness, and uniqueness.

![YouTube RAG Pipeline](https://img.shields.io/badge/YouTube-RAG_Pipeline-red)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95.0+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.22.0+-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Features

### Core RAG Pipeline

- **Transcript Extraction**: Extract transcripts from YouTube videos with fallback to ASR using Whisper
- **Advanced Indexing**: Dynamic chunk sizing based on content density, FAISS vector storage, and persistent embedding caching
- **Enhanced Retrieval**: Query rewriting, hybrid retrieval (FAISS + BM25), Maximum Marginal Relevance, and cross-encoder reranking
- **Augmented Generation**: Dynamic prompts based on query type, inline citations with timestamps, and context-aware answers
- **Multimodal Processing**: Process video keyframes alongside transcripts (planned feature)

### Advanced Features

- **Multi-Video Analysis**: Compare information across multiple videos
- **Cross-Video Search**: Search for information across multiple videos at once
- **Video Comparison**: Compare how different videos discuss the same topics
- **Automatic Summarization**: Generate different types of video summaries (default, detailed, bullet points, chapters)
- **Custom Embedding Models**: Use different embedding models (HuggingFace, OpenAI, Ollama)
- **Real-Time Collaboration**: Collaborate with others in analyzing videos

### User Interfaces

- **Web Interface**: Interactive Streamlit app with modern design, real-time feedback, and export options
- **Chrome Extension**: Browser extension for real-time querying while watching YouTube videos
- **API**: FastAPI backend for scalable processing with async endpoints
- **Interactive Video Player**: Watch videos with synchronized transcript
- **Interactive Timeline**: Visual timeline of sources in the video
- **Text Analysis**: Word frequency analysis and key metrics for answers
- **Dark/Light Mode**: Toggle between dark and light themes

### Performance Optimizations

- **Batch Processing**: Efficient batch processing for embeddings
- **Distributed Processing**: Handle multiple videos in parallel
- **Caching Mechanisms**: Cache transcripts, embeddings, and query results
- **Collaboration Optimization**: Optimized for large numbers of users

## Prerequisites

1. **Python 3.9+**: The project requires Python 3.9 or higher.
2. **Ollama**: You need to install Ollama to run the LLM locally. Follow the instructions at [https://ollama.ai/](https://ollama.ai/).
3. **Chrome Browser**: Required for the Chrome extension.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/RAG_YT.git
   cd RAG_YT
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:

   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - Unix/MacOS:
     ```bash
     source venv/bin/activate
     ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Install Ollama and download the llama3.2 model:
   ```bash
   # Install Ollama from https://ollama.ai/
   ollama pull llama3.2:latest
   ```

## Running the Application

The easiest way to run the application is using the provided `run.py` script:

```bash
# Run both API and UI
python run.py

# Run only the API
python run.py --api-only

# Run only the UI
python run.py --ui-only

# Build the Chrome extension
python run.py --build-extension
```

### Manual Startup

If you prefer to start components manually:

#### API Server

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

#### Streamlit UI

```bash
streamlit run src/ui/app.py
```

#### Chrome Extension

1. Build the extension:

   ```bash
   cd src/chrome_extension
   ./build.sh
   ```

2. Install in Chrome:
   - Open Chrome and navigate to `chrome://extensions/`
   - Enable "Developer Mode" in the top-right corner
   - Click "Load unpacked" and select the `src/chrome_extension/dist` directory

## Usage Guide

### Web Interface

1. Open the Streamlit UI at [http://localhost:8501](http://localhost:8501)
2. Enter a YouTube URL in the sidebar
3. Click "Process Video" to extract and index the transcript
4. Once processing is complete, ask questions in the main panel
5. View answers with timestamp citations
6. Export results as PDF or JSON
7. Toggle between dark and light mode using the theme switch in the sidebar

### Multi-Video Analysis

1. Navigate to the "Multi-Video Analysis" page
2. Select multiple videos from the sidebar
3. Use cross-video query to ask questions across all selected videos
4. Compare videos on specific topics
5. View visual comparisons and analysis results

### Video Summarization

1. Navigate to the "Video Summarization" page
2. Enter a YouTube URL or select a previously processed video
3. Select a summary type (default, detailed, bullet points, chapters)
4. Click "Generate Summary" to create the summary
5. View the summary and its sources
6. Export the summary in various formats (PDF, JSON, text)

### Collaboration

1. Navigate to the "Collaboration" page
2. Create a new session or join an existing one
3. Share the session ID with others to collaborate
4. Share videos and queries with other users
5. Chat with other users in real-time
6. Add annotations to videos and comments on queries

### Custom Embeddings

1. Navigate to the "Custom Embeddings" page
2. Select an embedding model from the available options
3. Add your own custom models if needed
4. Compare different models on a sample query
5. View performance metrics and visualizations

### Chrome Extension

1. Navigate to any YouTube video
2. Click the YouTube RAG Assistant icon in the Chrome toolbar
3. Click "Open Assistant" to show the widget
4. Click "Process Video" to extract and index the transcript
5. Once processing is complete, ask questions in the widget
6. Click on timestamp citations to jump to specific parts of the video

### API Endpoints

#### Basic Endpoints

- `GET /`: API status
- `POST /process-video`: Process a YouTube video
- `GET /processing-status/{video_id}`: Check processing status
- `POST /query`: Query a processed video
- `GET /videos`: List all processed videos

#### Advanced Endpoints

- `POST /cross-video-query`: Query across multiple videos
- `POST /compare-videos`: Compare multiple videos on specific topics
- `POST /summarize`: Generate a summary of a video

#### Collaboration Endpoints

- `POST /collaboration/sessions`: Create a new session
- `GET /collaboration/sessions`: List all sessions
- `GET /collaboration/sessions/{session_id}`: Get session information
- `POST /collaboration/sessions/{session_id}/join`: Join a session
- `POST /collaboration/sessions/{session_id}/messages`: Add a message
- `POST /collaboration/sessions/{session_id}/videos`: Share a video
- `POST /collaboration/sessions/{session_id}/queries`: Share a query

## Project Structure

```
.
├── config/                 # Configuration files
├── data/                   # Data storage
│   ├── raw/                # Raw transcripts
│   ├── processed/          # Processed documents
│   └── embeddings/         # Cached embeddings
├── logs/                   # Log files
├── src/                    # Source code
│   ├── core/               # Core RAG pipeline
│   │   ├── transcript_extractor.py  # YouTube transcript extraction
│   │   ├── document_processor.py    # Document chunking and processing
│   │   ├── embedding_manager.py     # Embedding and vector storage
│   │   ├── custom_embeddings.py     # Custom embedding models
│   │   ├── retriever.py             # Enhanced retrieval
│   │   ├── generator.py             # Answer generation
│   │   ├── rag_pipeline.py          # Main pipeline
│   │   ├── distributed_processing.py # Distributed processing
│   │   └── collaboration.py         # Collaboration system
│   ├── api/                # FastAPI backend
│   │   └── main.py         # API endpoints
│   ├── ui/                 # Streamlit interface
│   │   ├── app.py          # Main Streamlit app
│   │   └── pages/          # Streamlit pages
│   │       ├── 01_Multi_Video_Analysis.py  # Multi-video analysis page
│   │       ├── 02_Video_Summarization.py   # Video summarization page
│   │       ├── 03_Collaboration.py         # Collaboration page
│   │       └── 04_Custom_Embeddings.py     # Custom embeddings page
│   └── chrome_extension/   # Chrome extension
│       ├── manifest.json   # Extension manifest
│       ├── content.js      # Content script
│       ├── content.css     # Content styles
│       ├── popup.html      # Popup UI
│       ├── popup.js        # Popup script
│       └── background.js   # Background script
├── tests/                  # Test suite
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   └── load/               # Load tests
├── run.py                  # Main runner script
├── run_tests.py            # Test runner script
├── run_coverage.py         # Coverage report generator
├── requirements.txt        # Dependencies
└── README.md               # Documentation
```

## Testing

The project includes a comprehensive test suite to ensure code quality and reliability.

### Running Tests

1. Run all tests:

   ```bash
   python run_tests.py --all
   ```

2. Run specific test types:

   ```bash
   python run_tests.py --unit      # Run unit tests
   python run_tests.py --integration  # Run integration tests
   python run_tests.py --load      # Run load tests
   ```

3. Generate a test coverage report:
   ```bash
   python run_coverage.py
   ```

### Test Types

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test interactions between components
- **Load Tests**: Test performance under load

## Performance Optimizations

The application includes several performance optimizations:

- **Batch Processing**: Efficient batch processing for embeddings
- **Distributed Processing**: Handle multiple videos in parallel
- **Caching Mechanisms**: Cache transcripts, embeddings, and query results
- **Collaboration Optimization**: Optimized for large numbers of users

## Troubleshooting

- **API Connection Issues**: Ensure the API server is running at http://localhost:8000
- **Ollama Issues**: Make sure Ollama is running and the llama3.2 model is downloaded
- **Chrome Extension Not Working**: Check the browser console for errors and ensure the API URL is correct in the extension settings
- **Embedding Model Issues**: If using custom embedding models, ensure the required packages are installed
- **Collaboration Issues**: Check that all users are connected to the same session ID
- **Performance Issues**: Try clearing the cache or using a different embedding model

## License

MIT
