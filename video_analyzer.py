import os
import re
import json
import asyncio
import torch
from moviepy import VideoFileClip
import google.generativeai as genai
from transformers import pipeline
import config
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional

class Event(BaseModel):
    timestamp: float = Field(description="The exact time in seconds (float) where the event begins.")
    type: str = Field(description="The category of the event. Choose from: 'Action', 'Movement', 'Vehicle', 'Hazard', 'Notable Moment', 'Dialogue'.")
    description: str = Field(description="A clinical, one-sentence description of the specific event.")

class ExtractedEntity(BaseModel):
    entity_id: int = Field(description="A unique integer ID for this entity (e.g., 1, 2).")
    type: str = Field(description="The type of entity (e.g., 'Vehicle', 'Person', 'Object').")
    description: str = Field(description="A detailed description of the entity, including its appearance (e.g., 'blue sedan', 'person in red jacket'). If any text like a license plate is visible on the entity, include it here.")
    first_seen_timestamp: float = Field(description="The timestamp when this entity first appears or becomes relevant.")

class VideoAnalysisResult(BaseModel):
    title: str = Field(description="A concise, impactful, newspaper-style headline summarizing the video's core event.")
    overall_summary: str = Field(description="A comprehensive, narrative paragraph describing the video's context, the sequence of events, and the final outcome, integrating key visual and auditory information. Write this as if you are giving a formal briefing.")
    sentiment: str = Field(description="The single most descriptive emotional tone of the video (e.g., 'Calm', 'Tense', 'Chaotic', 'Joyful').")
    key_topics: List[str] = Field(description="A list of 3-5 primary keywords or themes (e.g., 'Traffic Incident', 'Public Safety', 'Product Demonstration').")
    extracted_entities: List[ExtractedEntity] = Field(description="A list of key entities (vehicles, people, important objects) involved in the main events. Assign a unique ID to each.")
    event_log: List[Event] = Field(description="A detailed log of timestamped events.")

class VideoAnalyzer:
    def __init__(self, console):
        self.console = console
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.console.print(f"INFO: Initializing local Whisper model on: [bold yellow]{self.device.upper()}[/bold yellow]")
        self.transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=self.device)
        
        if not config.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is not set.")
        genai.configure(api_key=config.GOOGLE_API_KEY)
        self.gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")
        self.output_parser = PydanticOutputParser(pydantic_object=VideoAnalysisResult)

    async def analyze(self, video_path: str, update_callback):
        update_callback(0, "Starting initial high-level analysis...")
        transcript_data = await self._extract_and_transcribe_audio(video_path, update_callback)
        gemini_analysis_data = await self._analyze_video_with_gemini(video_path, transcript_data, update_callback)
        update_callback(100, "Initial analysis complete!")
        return gemini_analysis_data

    async def _extract_and_transcribe_audio(self, video_path, cb):
        def _blocking_extract():
            with VideoFileClip(video_path) as video:
                if not video.audio: return None
                fname = f"{os.path.splitext(os.path.basename(video_path))[0]}.wav"
                apath = os.path.join(config.TEMP_AUDIO_DIR, fname)
                video.audio.write_audiofile(apath, codec='pcm_s16le', logger=None)
                return apath
        cb(5, "Extracting audio...")
        audio_path = await asyncio.to_thread(_blocking_extract)
        if audio_path and os.path.exists(audio_path):
            cb(15, "Transcribing full audio with Whisper...")
            transcription = await asyncio.to_thread(self.transcriber, audio_path, chunk_length_s=30, return_timestamps=True)
            os.remove(audio_path)
            return transcription.get('chunks', [])
        return []

    async def _analyze_video_with_gemini(self, video_path, transcript_data, cb):
        cb(25, "Uploading video to Gemini...")
        video_file = await asyncio.to_thread(genai.upload_file, path=video_path)
        while video_file.state.name == "PROCESSING":
            await asyncio.sleep(5)
            cb(40, "Processing video...")
            video_file = await asyncio.to_thread(genai.get_file, video_file.name)
        if video_file.state.name == "FAILED":
            raise ValueError("Google AI File API failed to process video.")
        
        cb(60, "Generating summary with Gemini...")
        prompt = self._get_gemini_analysis_prompt(transcript_data)
        
        try:
            response = await self.gemini_model.generate_content_async([prompt, video_file])
            parsed_output = self.output_parser.parse(response.text)
            return parsed_output.model_dump()
        except Exception as e:
            self.console.print(f"[bold red]Error parsing Gemini response: {e}[/bold red]")
            raise ValueError("Failed to parse JSON response from Gemini after processing.")
        finally:
            await asyncio.to_thread(genai.delete_file, video_file.name)

    def _get_gemini_analysis_prompt(self, transcript_chunks):
        full_transcript = " ".join([chunk['text'] for chunk in transcript_chunks])
        format_instructions = self.output_parser.get_format_instructions()
        return f"""**Role**: Elite AI Multimedia Triage Analyst 'Observer'. Your mission is to perform a high-level forensic analysis of the provided video and its audio transcript. Your goal is to identify key events and, most importantly, to proactively extract details about the primary entities involved. Your analysis will serve as the foundational intelligence briefing for a tool-using AI agent.

**Instructions**:
1.  **Synthesize All Data**: Watch the entire video and read the provided audio transcript. Your summary must integrate both visual events and spoken dialogue.
2.  **Identify Primary Entities**: Pinpoint the most significant actors in the video (e.g., the car that caused an accident, the person who was the focus of an event). Assign a unique integer ID to each.
3.  **Extract Critical Details**: For each primary entity, describe it in detail. **Crucially, if you can clearly read any text on an entity (like a license plate on a car or a name on a shirt), you MUST include that text in its description.**
4.  **Log Key Events**: Create a chronological log of the most important moments.
5.  **Structure the Output**: Generate a single, flawless JSON object that strictly adheres to the provided schema instructions. Do not add any text, explanations, or markdown formatting outside of the JSON object itself.

**Full Audio Transcript**:
---
{full_transcript}
---

{format_instructions}
"""