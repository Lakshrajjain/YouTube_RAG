"""
Module for extracting transcripts from YouTube videos.
"""
import os
import re
import json
import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import whisper
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

import sys
import os
# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config.config import TRANSCRIPT_SETTINGS, RAW_DATA_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TranscriptExtractor:
    """
    Class for extracting transcripts from YouTube videos with fallback to ASR.
    """
    def __init__(self,
                 languages: List[str] = TRANSCRIPT_SETTINGS["languages"],
                 fallback_to_asr: bool = TRANSCRIPT_SETTINGS["fallback_to_asr"],
                 asr_model: str = TRANSCRIPT_SETTINGS["asr_model"]):
        """
        Initialize the TranscriptExtractor.

        Args:
            languages: List of preferred languages for transcripts
            fallback_to_asr: Whether to use ASR if no transcript is available
            asr_model: Whisper model size (tiny, base, small, medium, large)
        """
        self.languages = languages
        self.fallback_to_asr = fallback_to_asr
        self.asr_model = asr_model
        self.whisper_model = None  # Lazy loading

    def _extract_video_id(self, youtube_url: str) -> str:
        """
        Extract the video ID from a YouTube URL.

        Args:
            youtube_url: URL of the YouTube video

        Returns:
            str: YouTube video ID
        """
        # Regular expression to match YouTube video IDs
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]{11})',
            r'(?:youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})',
            r'(?:youtube\.com\/v\/)([a-zA-Z0-9_-]{11})'
        ]

        for pattern in patterns:
            match = re.search(pattern, youtube_url)
            if match:
                return match.group(1)

        raise ValueError(f"Could not extract video ID from URL: {youtube_url}")

    def _get_transcript_from_api(self, video_id: str) -> List[Dict]:
        """
        Get transcript from YouTube Transcript API.

        Args:
            video_id: YouTube video ID

        Returns:
            List[Dict]: List of transcript segments

        Raises:
            TranscriptsDisabled: If transcripts are disabled for the video
            NoTranscriptFound: If no transcript is found for the video
        """
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

            # Try to get transcript in preferred language
            for language in self.languages:
                try:
                    transcript = transcript_list.find_transcript([language])
                    # Convert to standard Python dictionaries to ensure JSON serializability
                    fetched_transcript = transcript.fetch()

                    # Process the fetched transcript
                    processed_transcript = []
                    for item in fetched_transcript:
                        # Handle different types of transcript items
                        if isinstance(item, dict):
                            # Extract text
                            if "text" in item:
                                text = str(item["text"])
                            else:
                                text = ""

                            # Extract start time
                            if "start" in item:
                                try:
                                    start = float(item["start"])
                                except (ValueError, TypeError):
                                    start = 0.0
                            else:
                                start = 0.0

                            # Extract duration
                            if "duration" in item:
                                try:
                                    duration = float(item["duration"])
                                except (ValueError, TypeError):
                                    duration = 0.0
                            else:
                                duration = 0.0
                        else:
                            # Try to access attributes if it's an object
                            try:
                                text = str(getattr(item, "text", ""))
                            except:
                                text = ""

                            try:
                                start = float(getattr(item, "start", 0))
                            except (ValueError, TypeError):
                                start = 0.0

                            try:
                                duration = float(getattr(item, "duration", 0))
                            except (ValueError, TypeError):
                                duration = 0.0

                        processed_transcript.append({
                            "text": text,
                            "start": start,
                            "duration": duration
                        })

                    return processed_transcript
                except Exception as e:
                    logger.warning(f"Failed to get transcript in language {language}: {str(e)}")
                    continue

            # If no preferred language is found, get the first available
            try:
                transcript = transcript_list.find_transcript([])
                fetched_transcript = transcript.fetch()

                # Process the fetched transcript
                processed_transcript = []
                for item in fetched_transcript:
                    # Handle different types of transcript items
                    if isinstance(item, dict):
                        # Extract text
                        if "text" in item:
                            text = str(item["text"])
                        else:
                            text = ""

                        # Extract start time
                        if "start" in item:
                            try:
                                start = float(item["start"])
                            except (ValueError, TypeError):
                                start = 0.0
                        else:
                            start = 0.0

                        # Extract duration
                        if "duration" in item:
                            try:
                                duration = float(item["duration"])
                            except (ValueError, TypeError):
                                duration = 0.0
                        else:
                            duration = 0.0
                    else:
                        # Try to access attributes if it's an object
                        try:
                            text = str(getattr(item, "text", ""))
                        except:
                            text = ""

                        try:
                            start = float(getattr(item, "start", 0))
                        except (ValueError, TypeError):
                            start = 0.0

                        try:
                            duration = float(getattr(item, "duration", 0))
                        except (ValueError, TypeError):
                            duration = 0.0

                    processed_transcript.append({
                        "text": text,
                        "start": start,
                        "duration": duration
                    })

                return processed_transcript
            except Exception as e:
                logger.warning(f"Failed to get transcript: {str(e)}")
                raise

        except (TranscriptsDisabled, NoTranscriptFound) as e:
            logger.warning(f"No transcript found via API for video {video_id}: {str(e)}")
            raise

    def _get_transcript_from_asr(self, video_id: str) -> List[Dict]:
        """
        Get transcript using Whisper ASR.

        Args:
            video_id: YouTube video ID

        Returns:
            List[Dict]: List of transcript segments
        """
        import yt_dlp
        import tempfile
        import subprocess

        logger.info(f"Using ASR to transcribe video {video_id}")

        # Check if ffmpeg is installed
        try:
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        except FileNotFoundError:
            error_msg = (
                "ffmpeg is not installed or not in PATH. ffmpeg is required for audio processing. "
                "Please install ffmpeg using one of the following methods:\n"
                "- Ubuntu/Debian: sudo apt-get install ffmpeg\n"
                "- macOS: brew install ffmpeg\n"
                "- Windows: Download from https://ffmpeg.org/download.html\n"
                "After installing, make sure ffmpeg is in your PATH."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Lazy load the Whisper model
        if self.whisper_model is None:
            logger.info(f"Loading Whisper model: {self.asr_model}")
            self.whisper_model = whisper.load_model(self.asr_model)

        # Download audio using yt-dlp
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_path = temp_file.name

        # First try to download directly as mp3
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': temp_path,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': True,
        }

        try:
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([f'https://www.youtube.com/watch?v={video_id}'])
            except Exception as e:
                logger.warning(f"Error downloading audio with FFmpeg postprocessing: {str(e)}")

                # If FFmpeg fails, try downloading just the audio without postprocessing
                logger.info("Trying alternative download method without FFmpeg postprocessing")
                alt_ydl_opts = {
                    'format': 'bestaudio/best',
                    'outtmpl': temp_path,
                    'quiet': True,
                }

                with yt_dlp.YoutubeDL(alt_ydl_opts) as ydl:
                    ydl.download([f'https://www.youtube.com/watch?v={video_id}'])

            # Transcribe with Whisper
            logger.info(f"Transcribing audio file: {temp_path}")
            result = self.whisper_model.transcribe(temp_path)

            # Convert to the same format as YouTube Transcript API
            transcript = []
            for segment in result["segments"]:
                transcript.append({
                    "text": segment["text"],
                    "start": segment["start"],
                    "duration": segment["end"] - segment["start"]
                })

            return transcript

        except Exception as e:
            logger.error(f"Error in ASR transcription: {str(e)}")
            # Create a simple transcript with the error message
            error_transcript = [{
                "text": f"Error transcribing video. Please try another video or install ffmpeg. Error: {str(e)}",
                "start": 0,
                "duration": 10
            }]
            return error_transcript

        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def get_transcript(self, youtube_url: str, force_asr: bool = False) -> Tuple[List[Dict], bool]:
        """
        Get transcript for a YouTube video.

        Args:
            youtube_url: URL of the YouTube video
            force_asr: Whether to force using ASR even if API transcript is available

        Returns:
            Tuple[List[Dict], bool]: Tuple of (transcript, used_asr)
        """
        video_id = self._extract_video_id(youtube_url)
        used_asr = False

        # Check if transcript is already cached
        cache_path = RAW_DATA_DIR / f"{video_id}.json"
        if cache_path.exists() and not force_asr:
            logger.info(f"Loading cached transcript for video {video_id}")
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    return data["transcript"], data["used_asr"]
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Error loading cached transcript: {str(e)}. Will fetch fresh transcript.")
                # If there's an error with the cached file, delete it and continue
                try:
                    os.remove(cache_path)
                except:
                    pass

        # Try to get transcript from API first
        if not force_asr:
            try:
                transcript = self._get_transcript_from_api(video_id)
                logger.info(f"Successfully retrieved transcript from API for video {video_id}")
            except (TranscriptsDisabled, NoTranscriptFound):
                if self.fallback_to_asr:
                    transcript = self._get_transcript_from_asr(video_id)
                    used_asr = True
                else:
                    raise ValueError(
                        f"No transcript found for video {video_id} and ASR fallback is disabled. "
                        f"Either enable ASR fallback in the configuration or try a video with available transcripts."
                    )
        else:
            transcript = self._get_transcript_from_asr(video_id)
            used_asr = True

        # Cache the transcript
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        try:
            # Ensure transcript is JSON serializable
            serializable_transcript = []
            for segment in transcript:
                # Handle different types of transcript segments
                if isinstance(segment, dict):
                    # Extract values safely
                    text = str(segment.get("text", "")) if segment.get("text") is not None else ""

                    # Handle different formats for start time
                    start = segment.get("start")
                    if start is not None:
                        try:
                            start = float(start)
                        except (ValueError, TypeError):
                            start = 0.0
                    else:
                        start = 0.0

                    # Handle different formats for duration
                    duration = segment.get("duration")
                    if duration is not None:
                        try:
                            duration = float(duration)
                        except (ValueError, TypeError):
                            duration = 0.0
                    else:
                        duration = 0.0

                    serializable_segment = {
                        "text": text,
                        "start": start,
                        "duration": duration
                    }
                else:
                    # If segment is not a dict, create a default segment
                    serializable_segment = {
                        "text": str(segment) if segment is not None else "",
                        "start": 0.0,
                        "duration": 0.0
                    }

                serializable_transcript.append(serializable_segment)

            # Test JSON serialization before writing to file
            json_data = json.dumps({
                "video_id": video_id,
                "transcript": serializable_transcript,
                "used_asr": used_asr
            }, ensure_ascii=False)

            # If serialization succeeded, write to file
            with open(cache_path, 'w') as f:
                f.write(json_data)

            # Update transcript with serializable version
            transcript = serializable_transcript
            logger.info(f"Successfully cached transcript for video {video_id}")
        except Exception as e:
            logger.error(f"Error caching transcript: {str(e)}")
            # Try to remove the potentially corrupted cache file
            try:
                if os.path.exists(cache_path):
                    os.remove(cache_path)
            except:
                pass
            # Continue without caching

        return transcript, used_asr

    def get_transcript_text(self, youtube_url: str, force_asr: bool = False) -> str:
        """
        Get transcript text for a YouTube video.

        Args:
            youtube_url: URL of the YouTube video
            force_asr: Whether to force using ASR even if API transcript is available

        Returns:
            str: Full transcript text
        """
        transcript, _ = self.get_transcript(youtube_url, force_asr)
        return " ".join([segment["text"] for segment in transcript])

    def get_transcript_with_timestamps(self, youtube_url: str, force_asr: bool = False) -> List[Dict]:
        """
        Get transcript with timestamps for a YouTube video.

        Args:
            youtube_url: URL of the YouTube video
            force_asr: Whether to force using ASR even if API transcript is available

        Returns:
            List[Dict]: List of transcript segments with timestamps
        """
        transcript, _ = self.get_transcript(youtube_url, force_asr)
        return transcript
