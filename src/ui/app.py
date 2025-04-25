"""
Streamlit UI for the YouTube RAG Pipeline.
"""
import json
import logging
import re
import time
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Optional, Union, Any

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from fpdf import FPDF
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.card import card
from streamlit_extras.colored_header import colored_header
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.stateful_button import button
from streamlit_extras.stylable_container import stylable_container

import sys
import os
# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config.config import UI_SETTINGS, API_SETTINGS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API URL
API_URL = f"http://{API_SETTINGS['host']}:{API_SETTINGS['port']}"

# Page config
st.set_page_config(
    page_title=UI_SETTINGS["page_title"],
    page_icon=UI_SETTINGS["page_icon"],
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "current_video_id" not in st.session_state:
    st.session_state.current_video_id = None
if "current_video_url" not in st.session_state:
    st.session_state.current_video_url = None
if "theme" not in st.session_state:
    st.session_state.theme = UI_SETTINGS["theme"]

# Helper functions
def extract_video_id(youtube_url: str) -> str:
    """
    Extract the video ID from a YouTube URL.

    Args:
        youtube_url: URL of the YouTube video

    Returns:
        str: YouTube video ID
    """
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

def get_youtube_thumbnail(video_id: str) -> str:
    """
    Get the thumbnail URL for a YouTube video.

    Args:
        video_id: YouTube video ID

    Returns:
        str: Thumbnail URL
    """
    return f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"

def format_timestamp_link(video_id: str, timestamp: int) -> str:
    """
    Format a timestamp link for a YouTube video.

    Args:
        video_id: YouTube video ID
        timestamp: Timestamp in seconds

    Returns:
        str: Formatted timestamp link
    """
    minutes = int(timestamp // 60)
    seconds = int(timestamp % 60)
    timestamp_str = f"{minutes:02d}:{seconds:02d}"
    return f"[{timestamp_str}](https://www.youtube.com/watch?v={video_id}&t={int(timestamp)}s)"

def process_video(youtube_url: str, force_reprocess: bool = False) -> Dict:
    """
    Process a YouTube video.

    Args:
        youtube_url: URL of the YouTube video
        force_reprocess: Whether to force reprocessing

    Returns:
        Dict: Processing status
    """
    try:
        response = requests.post(
            f"{API_URL}/process-video",
            json={"youtube_url": youtube_url, "force_reprocess": force_reprocess}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        return {"status": "error", "message": str(e)}

def get_processing_status(video_id: str) -> Dict:
    """
    Get the processing status for a video.

    Args:
        video_id: YouTube video ID

    Returns:
        Dict: Processing status
    """
    try:
        response = requests.get(f"{API_URL}/processing-status/{video_id}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return {"status": "not_found", "progress": 0.0}
        else:
            st.error(f"Error getting processing status: {str(e)}")
            return {"status": "error", "message": str(e)}
    except Exception as e:
        st.error(f"Error getting processing status: {str(e)}")
        return {"status": "error", "message": str(e)}

def query_video(youtube_url: str, query: str) -> Dict:
    """
    Query a YouTube video.

    Args:
        youtube_url: URL of the YouTube video
        query: Query string

    Returns:
        Dict: Query result
    """
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={"youtube_url": youtube_url, "query": query}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error querying video: {str(e)}")
        return {"status": "error", "message": str(e)}

def list_videos() -> List[Dict]:
    """
    List all processed videos.

    Returns:
        List[Dict]: List of videos
    """
    try:
        response = requests.get(f"{API_URL}/videos")
        response.raise_for_status()
        return response.json()["videos"]
    except Exception as e:
        st.error(f"Error listing videos: {str(e)}")
        return []

def export_to_pdf(result: Dict) -> BytesIO:
    """
    Export a query result to PDF.

    Args:
        result: Query result

    Returns:
        BytesIO: PDF file
    """
    pdf = FPDF()
    pdf.add_page()

    # Set font
    pdf.set_font("Arial", "B", 16)

    # Title
    pdf.cell(0, 10, "YouTube RAG Query Result", ln=True, align="C")
    pdf.ln(10)

    # Video info
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, f"Video: https://www.youtube.com/watch?v={result['video_id']}", ln=True)
    pdf.ln(5)

    # Query
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Query:", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, result["query"])
    pdf.ln(5)

    # Answer
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Answer:", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, result["answer"])
    pdf.ln(5)

    # Sources
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Sources:", ln=True)
    pdf.set_font("Arial", "", 10)

    for i, source in enumerate(result["sources"]):
        pdf.set_font("Arial", "B", 10)
        pdf.cell(0, 10, f"Source {i+1} - Timestamp: {source['timestamp_str']}", ln=True)
        pdf.set_font("Arial", "", 10)
        pdf.multi_cell(0, 10, source["content"])
        pdf.ln(5)

    # Export to BytesIO
    pdf_output = BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)

    return pdf_output

def export_to_json(result: Dict) -> BytesIO:
    """
    Export a query result to JSON.

    Args:
        result: Query result

    Returns:
        BytesIO: JSON file
    """
    json_output = BytesIO()
    json.dump(result, json_output, indent=2)
    json_output.seek(0)

    return json_output

# UI Components
def render_sidebar():
    """Render the sidebar."""
    with st.sidebar:
        st.title("YouTube RAG")

        # Theme toggle
        theme_col1, theme_col2 = st.columns([1, 3])
        with theme_col1:
            st.write("Theme:")
        with theme_col2:
            if st.toggle("Dark Mode", st.session_state.theme == "dark"):
                st.session_state.theme = "dark"
            else:
                st.session_state.theme = "light"

        st.divider()

        # Video input
        st.subheader("Enter YouTube URL")
        youtube_url = st.text_input("YouTube URL", key="youtube_url_input")

        if youtube_url:
            try:
                video_id = extract_video_id(youtube_url)
                st.session_state.current_video_id = video_id
                st.session_state.current_video_url = youtube_url

                # Show thumbnail
                st.image(get_youtube_thumbnail(video_id), use_container_width=True)

                # Process button
                if st.button("Process Video"):
                    with st.spinner("Processing video..."):
                        status = process_video(youtube_url)
                        st.session_state.processing_status = status
            except ValueError as e:
                st.error(str(e))

        st.divider()

        # History
        st.subheader("Query History")
        if not st.session_state.history:
            st.info("No queries yet")
        else:
            for i, item in enumerate(reversed(st.session_state.history[-5:])):
                with st.expander(f"{i+1}. {item['query'][:30]}..."):
                    st.write(f"**Video:** {item['video_id']}")
                    st.write(f"**Query:** {item['query']}")
                    st.write(f"**Time:** {item['timestamp']}")
                    if st.button("Rerun", key=f"rerun_{i}"):
                        st.session_state.current_video_url = item["youtube_url"]
                        st.session_state.current_video_id = item["video_id"]
                        st.session_state.query_input = item["query"]
                        st.rerun()

def render_video_processing_status():
    """Render the video processing status."""
    if not st.session_state.current_video_id:
        return

    video_id = st.session_state.current_video_id

    # Check processing status
    status = get_processing_status(video_id)

    if status["status"] == "processing":
        progress_text = status.get("message", "Processing video...")
        st.progress(status["progress"], text=progress_text)

        # Auto-refresh
        time.sleep(2)
        st.rerun()
    elif status["status"] == "completed":
        st.success("Video processed successfully!")
    elif status["status"] == "error":
        error_message = status.get('message', 'Unknown error')
        st.error(f"Error processing video: {error_message}")

        # Provide additional guidance for common errors
        if "ffmpeg" in error_message.lower():
            st.warning("""
            **FFmpeg is required but not installed.** You have two options:

            1. **Install FFmpeg** on your system:
               - Ubuntu/Debian: `sudo apt-get install ffmpeg`
               - macOS: `brew install ffmpeg`
               - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

            2. **Use videos with available transcripts**:
               - Try videos from channels that provide transcripts (like TED Talks)
               - The current configuration is set to only use YouTube's official transcripts
            """)
        elif "transcript" in error_message.lower() and "not found" in error_message.lower():
            st.warning("""
            **No transcript found for this video.** Try one of these options:

            1. **Try a different video** that has transcripts available
            2. **Enable ASR fallback** in the configuration (requires FFmpeg)
            """)

    elif status["status"] == "not_found":
        st.warning("Video not processed yet. Click 'Process Video' in the sidebar to start processing.")

def render_query_interface():
    """Render the query interface."""
    if not st.session_state.current_video_id or not st.session_state.current_video_url:
        st.info("Enter a YouTube URL in the sidebar to get started")
        return

    video_id = st.session_state.current_video_id
    youtube_url = st.session_state.current_video_url

    # Check if video is processed
    status = get_processing_status(video_id)
    if status["status"] != "completed":
        render_video_processing_status()
        return

    # Query input
    st.subheader("Ask a question about the video")

    # Suggested questions
    st.markdown("### Suggested Questions")
    suggested_questions = [
        "What is the main topic of this video?",
        "Can you summarize the key points?",
        "Who is the host of this video?",
        "What are the most important insights from this video?",
        "What examples are mentioned in the video?"
    ]

    # Display suggested questions as buttons
    cols = st.columns(2)
    for i, question in enumerate(suggested_questions):
        if cols[i % 2].button(question, key=f"suggested_{i}"):
            # Set the question in the text input
            st.session_state.query_input = question
            # Force a rerun to update the UI
            st.rerun()

    # Query input with the suggested question if selected
    query = st.text_input("Your question", key="query_input")

    # Add a submit button to make it more explicit
    submit_button = st.button("Submit Question")

    # Process query when submit button is clicked or Enter is pressed in the text input
    if submit_button or (query and st.session_state.get('last_query') != query):
        if query:  # Make sure query is not empty
            st.session_state['last_query'] = query  # Store the query to prevent duplicate processing

            with st.spinner("Generating answer..."):
                try:
                    result = query_video(youtube_url, query)

                    if "status" in result and result["status"] == "error":
                        st.error(f"Error: {result.get('message', 'Unknown error')}")
                        return

                    # Add to history with the answer
                    st.session_state.history.append({
                        "video_id": video_id,
                        "youtube_url": youtube_url,
                        "query": query,
                        "answer": result.get("answer", ""),
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })

                    # Display result
                    render_query_result(result)
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
                    st.info("Please try a different question or check the API logs for more details.")

def render_query_result(result: Dict):
    """
    Render a query result.

    Args:
        result: Query result
    """
    # Create tabs for different answer views
    answer_tab, analysis_tab = st.tabs(["Answer", "Analysis"])

    with answer_tab:
        # Answer card
        with stylable_container(
            key="answer_container",
            css_styles="""
                {
                    border-radius: 10px;
                    padding: 20px;
                    background-color: rgba(100, 100, 100, 0.1);
                }
            """
        ):
            st.markdown("### Answer")
            st.markdown(result["answer"])

    with analysis_tab:
        # Add text analysis visualization
        st.markdown("### Answer Analysis")

        # Word count analysis
        answer_text = result["answer"]
        word_count = len(answer_text.split())

        # Create a simple word frequency analysis
        words = re.findall(r'\b\w+\b', answer_text.lower())
        word_freq = {}

        # Skip common stop words
        stop_words = set(['the', 'and', 'is', 'in', 'to', 'of', 'a', 'for', 'that', 'this', 'it', 'with', 'as', 'are', 'be'])

        for word in words:
            if word not in stop_words and len(word) > 2:
                if word in word_freq:
                    word_freq[word] += 1
                else:
                    word_freq[word] = 1

        # Get top words
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]

        # Create a bar chart of top words
        if top_words:
            word_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])

            # Create a bar chart
            fig = px.bar(
                word_df,
                x='Word',
                y='Frequency',
                title=f"Top Words in Answer (Total Words: {word_count})",
                color='Frequency',
                color_continuous_scale='Viridis'
            )

            # Update layout
            fig.update_layout(
                xaxis_title="Word",
                yaxis_title="Frequency",
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )

            # Display the chart
            st.plotly_chart(fig, use_container_width=True)

            # Add sentiment analysis
            st.markdown("#### Key Insights")

            # Display key statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Word Count", word_count)
            with col2:
                st.metric("Most Common Word", top_words[0][0] if top_words else "N/A")
            with col3:
                st.metric("Unique Words", len(word_freq))

    # Show processing time and cache status
    if "from_cache" in result and result["from_cache"]:
        st.success("⚡ Result retrieved from cache")
    else:
        st.info(f"Processing time: {result['processing_time']:.2f} seconds")

    # Export options
    col1, col2 = st.columns(2)
    with col1:
        pdf_data = export_to_pdf(result)
        st.download_button(
            label="Export as PDF",
            data=pdf_data,
            file_name=f"rag_result_{result['video_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf"
        )
    with col2:
        json_data = export_to_json(result)
        st.download_button(
            label="Export as JSON",
            data=json_data,
            file_name=f"rag_result_{result['video_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

    # Add video player with synchronized transcript
    st.markdown("### Video Player with Synchronized Transcript")

    # Create tabs for different views
    video_tab, timeline_tab, sources_tab = st.tabs(["Video Player", "Timeline", "Sources"])

    with video_tab:
        # Create a 2-column layout
        video_col, transcript_col = st.columns([3, 2])

        with video_col:
            # Embed YouTube video player
            video_id = result['video_id']
            st.components.v1.iframe(
                f"https://www.youtube.com/embed/{video_id}?enablejsapi=1",
                height=315,
                scrolling=False
            )

            # Add a note about timestamps
            st.info("Click on a timestamp in the transcript to jump to that point in the video.")

        with transcript_col:
            # Display sources as a transcript with timestamps
            st.markdown("#### Transcript Highlights")

            # Sort sources by timestamp
            sorted_sources = sorted(result["sources"], key=lambda x: x["timestamp"])

            # Display sources with clickable timestamps
            for i, source in enumerate(sorted_sources):
                timestamp_link = format_timestamp_link(result['video_id'], source['timestamp'])
                st.markdown(f"**{timestamp_link}**: {source['content'][:100]}...")
                st.markdown("---")

    with timeline_tab:
        # Create a visualization of the sources
        if len(result["sources"]) > 0:
            # Create a timeline visualization
            timeline_data = []
            for i, source in enumerate(result["sources"]):
                timeline_data.append({
                    "index": i + 1,
                    "timestamp": source["timestamp"],
                    "timestamp_str": source["timestamp_str"],
                    "content_preview": source["content"][:50] + "..." if len(source["content"]) > 50 else source["content"]
                })

            # Create a DataFrame for the timeline
            timeline_df = pd.DataFrame(timeline_data)

            # Create a timeline visualization using Plotly
            if not timeline_df.empty:
                fig = px.scatter(
                    timeline_df,
                    x="timestamp",
                    y=["Source"] * len(timeline_df),
                    text="timestamp_str",
                    size=[10] * len(timeline_df),
                    color_discrete_sequence=["#4CAF50"],
                    title="Timeline of Sources in Video"
                )

                # Update layout for better appearance
                fig.update_layout(
                    xaxis_title="Time in Video (seconds)",
                    yaxis_title="",
                    yaxis_showticklabels=False,
                    height=300,
                    margin=dict(l=20, r=20, t=40, b=20),
                    hovermode="closest"
                )

                # Add hover information
                fig.update_traces(
                    hovertemplate="<b>Time:</b> %{text}<br><b>Content:</b> %{customdata}",
                    customdata=timeline_df["content_preview"]
                )

                # Display the timeline
                st.plotly_chart(fig, use_container_width=True)

                # Add interactive features
                st.markdown("""
                **Interactive Timeline:**
                - Hover over points to see content previews
                - Click and drag to zoom in on specific parts of the video
                - Double-click to reset the view
                """)

    with sources_tab:
        # Display sources in expandable sections
        for i, source in enumerate(result["sources"]):
            with st.expander(f"Source {i+1} - {source['timestamp_str']}"):
                st.markdown(f"**Timestamp:** {format_timestamp_link(result['video_id'], source['timestamp'])}")
                st.markdown(source["content"])

def render_query_history():
    """Render the query history."""
    st.subheader("Query History")

    if not st.session_state.history:
        st.info("No queries yet")
        return

    # Display history in reverse chronological order
    for i, item in enumerate(reversed(st.session_state.history)):
        with st.expander(f"{item['timestamp']} - {item['query']}"):
            st.markdown(f"**Video ID:** {item['video_id']}")
            st.markdown(f"**Query:** {item['query']}")
            if 'answer' in item:
                st.markdown(f"**Answer:** {item['answer']}")

            # Add a button to rerun the query
            if st.button("Ask Again", key=f"rerun_{i}_{item['timestamp'].replace(' ', '_').replace(':', '')}"):
                # Set the query in the text input
                st.session_state.query_input = item['query']
                # Force a rerun to update the UI
                st.rerun()

def render_video_list():
    """Render the list of processed videos."""
    st.subheader("Processed Videos")

    videos = list_videos()

    if not videos:
        st.info("No videos processed yet")
        return

    # Filter completed videos
    completed_videos = [v for v in videos if v["status"] == "completed"]

    if not completed_videos:
        st.info("No completed videos yet")
        return

    # Create a dataframe
    df = pd.DataFrame(completed_videos)

    # Ensure all required columns exist
    required_columns = ["video_id", "transcript_length", "num_chunks", "used_asr"]
    for col in required_columns:
        if col not in df.columns:
            df[col] = "N/A"

    # Select only the columns that exist
    display_columns = [col for col in required_columns if col in df.columns]

    # Display as a table
    st.dataframe(
        df[display_columns],
        column_config={
            "video_id": "Video ID",
            "transcript_length": "Transcript Length",
            "num_chunks": "Number of Chunks",
            "used_asr": "Used ASR"
        },
        use_container_width=True
    )

def main():
    """Main function."""
    # Apply theme and responsive design
    if st.session_state.theme == "dark":
        st.markdown("""
        <style>
        :root {
            --primary-color: #4CAF50;
            --background-color: #0E1117;
            --secondary-background-color: #262730;
            --text-color: #FAFAFA;
            --font: 'Source Sans Pro', sans-serif;
        }

        /* Responsive design for mobile devices */
        @media (max-width: 768px) {
            .stButton button {
                width: 100%;
                margin-bottom: 10px;
            }

            .stExpander {
                margin-bottom: 10px;
            }

            /* Make text more readable on small screens */
            p, li, .stMarkdown {
                font-size: 16px !important;
                line-height: 1.6 !important;
            }

            /* Adjust header sizes */
            h1 {
                font-size: 24px !important;
            }

            h2 {
                font-size: 20px !important;
            }

            h3 {
                font-size: 18px !important;
            }
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        :root {
            --primary-color: #4CAF50;
            --background-color: #FFFFFF;
            --secondary-background-color: #F0F2F6;
            --text-color: #31333F;
            --font: 'Source Sans Pro', sans-serif;
        }

        /* Responsive design for mobile devices */
        @media (max-width: 768px) {
            .stButton button {
                width: 100%;
                margin-bottom: 10px;
            }

            .stExpander {
                margin-bottom: 10px;
            }

            /* Make text more readable on small screens */
            p, li, .stMarkdown {
                font-size: 16px !important;
                line-height: 1.6 !important;
            }

            /* Adjust header sizes */
            h1 {
                font-size: 24px !important;
            }

            h2 {
                font-size: 20px !important;
            }

            h3 {
                font-size: 18px !important;
            }
        }
        </style>
        """, unsafe_allow_html=True)

    # Render sidebar
    render_sidebar()

    # Main content
    st.title("YouTube RAG")
    st.markdown("""
    This application allows you to ask questions about YouTube videos using Retrieval-Augmented Generation (RAG).

    1. Enter a YouTube URL in the sidebar
    2. Process the video
    3. Ask questions about the video content
    """)

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Query", "History", "Videos"])

    with tab1:
        render_query_interface()

    with tab2:
        render_query_history()

    with tab3:
        render_video_list()

if __name__ == "__main__":
    main()
