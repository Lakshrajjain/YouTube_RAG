"""
FastAPI backend for the YouTube RAG Pipeline.
"""
import logging
from typing import Dict, List, Optional, Union, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl

import sys
import os
# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config.config import API_SETTINGS
from src.core.rag_pipeline import RAGPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="YouTube RAG API",
    description="API for the YouTube RAG Pipeline",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG pipeline
rag_pipeline = RAGPipeline()

# Background processing tasks
processing_videos = {}

# Pydantic models
class VideoRequest(BaseModel):
    youtube_url: str
    force_reprocess: bool = False

class QueryRequest(BaseModel):
    youtube_url: str
    query: str

class ProcessingStatus(BaseModel):
    video_id: str
    status: str
    progress: float
    message: Optional[str] = None

# Background task for processing videos
def process_video_task(youtube_url: str, force_reprocess: bool = False):
    try:
        video_id = rag_pipeline.transcript_extractor._extract_video_id(youtube_url)
        processing_videos[video_id] = {"status": "processing", "progress": 0.0}

        # Extract transcript
        processing_videos[video_id] = {"status": "processing", "progress": 0.2, "message": "Extracting transcript"}
        transcript, used_asr = rag_pipeline.transcript_extractor.get_transcript(youtube_url, force_reprocess)

        # Process transcript
        processing_videos[video_id] = {"status": "processing", "progress": 0.4, "message": "Processing transcript"}
        documents = rag_pipeline.document_processor.process_transcript(video_id, transcript)

        # Create vector store
        processing_videos[video_id] = {"status": "processing", "progress": 0.7, "message": "Creating vector store"}
        vector_store = rag_pipeline.embedding_manager.create_vector_store(documents, video_id)

        # Complete
        processing_videos[video_id] = {"status": "completed", "progress": 1.0, "message": "Processing complete"}

    except Exception as e:
        error_message = str(e)
        logger.error(f"Error processing video: {error_message}")

        # Provide more user-friendly error messages for common issues
        if not error_message:
            user_message = (
                "An unknown error occurred during processing. This might be related to missing dependencies "
                "or issues with the vector store creation. Try with a different video."
            )
        elif "ffmpeg" in error_message.lower():
            user_message = (
                "FFmpeg is required but not found. FFmpeg is needed to process audio from YouTube videos. "
                "Please install FFmpeg on your system and make sure it's in your PATH."
            )
        elif "yt_dlp" in error_message.lower():
            user_message = "Error with YouTube downloader. Please make sure yt-dlp is installed correctly."
        elif "whisper" in error_message.lower():
            user_message = "Error with the Whisper speech recognition model. Please check your installation."
        elif "faiss" in error_message.lower() or "index" in error_message.lower():
            user_message = "Error creating vector index. This might be due to an issue with the transcript or embeddings."
        else:
            user_message = error_message if error_message else "An unknown error occurred during processing."

        processing_videos[video_id] = {"status": "error", "progress": 0.0, "message": user_message}

# API endpoints
@app.get("/")
async def root():
    return {"message": "YouTube RAG API"}

@app.post("/process-video", response_model=ProcessingStatus)
async def process_video(request: VideoRequest, background_tasks: BackgroundTasks):
    try:
        video_id = rag_pipeline.transcript_extractor._extract_video_id(request.youtube_url)

        # Check if video is already being processed
        if video_id in processing_videos and processing_videos[video_id]["status"] == "processing":
            return ProcessingStatus(
                video_id=video_id,
                status=processing_videos[video_id]["status"],
                progress=processing_videos[video_id]["progress"],
                message=processing_videos[video_id].get("message")
            )

        # Check if video is already processed and not forcing reprocess
        if video_id in rag_pipeline.video_metadata and not request.force_reprocess:
            return ProcessingStatus(
                video_id=video_id,
                status="completed",
                progress=1.0,
                message="Video already processed"
            )

        # Start background processing
        background_tasks.add_task(process_video_task, request.youtube_url, request.force_reprocess)

        # Initialize processing status
        processing_videos[video_id] = {"status": "processing", "progress": 0.0, "message": "Starting processing"}

        return ProcessingStatus(
            video_id=video_id,
            status="processing",
            progress=0.0,
            message="Started processing"
        )

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/processing-status/{video_id}", response_model=ProcessingStatus)
async def get_processing_status(video_id: str):
    if video_id in processing_videos:
        return ProcessingStatus(
            video_id=video_id,
            status=processing_videos[video_id]["status"],
            progress=processing_videos[video_id]["progress"],
            message=processing_videos[video_id].get("message")
        )

    # Check if video is already processed
    if video_id in rag_pipeline.video_metadata:
        return ProcessingStatus(
            video_id=video_id,
            status="completed",
            progress=1.0,
            message="Video already processed"
        )

    raise HTTPException(status_code=404, detail=f"Video {video_id} not found")

@app.post("/query")
async def query_video(request: QueryRequest):
    try:
        # Process video if not already processed
        video_id = rag_pipeline.transcript_extractor._extract_video_id(request.youtube_url)

        # Check if video is still being processed
        if video_id in processing_videos and processing_videos[video_id]["status"] == "processing":
            raise HTTPException(
                status_code=400,
                detail=f"Video {video_id} is still being processed. Current progress: {processing_videos[video_id]['progress']:.0%}"
            )

        # Query video
        result = rag_pipeline.query_video(request.youtube_url, request.query)
        return result

    except Exception as e:
        logger.error(f"Error querying video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/videos")
async def list_videos():
    videos = []

    # Add processed videos
    for video_id, metadata in rag_pipeline.video_metadata.items():
        videos.append({
            "video_id": video_id,
            "youtube_url": metadata["youtube_url"],
            "status": "completed",
            "transcript_length": metadata["transcript_length"],
            "num_chunks": metadata["num_chunks"],
            "used_asr": metadata["used_asr"]
        })

    # Add videos being processed
    for video_id, status in processing_videos.items():
        if video_id not in rag_pipeline.video_metadata:
            videos.append({
                "video_id": video_id,
                "status": status["status"],
                "progress": status["progress"]
            })

    return {"videos": videos}

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=API_SETTINGS["host"],
        port=API_SETTINGS["port"],
        workers=API_SETTINGS["workers"]
    )
