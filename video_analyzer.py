import os
import cv2
import torch
import numpy as np
from moviepy import VideoFileClip
from transformers import pipeline
from dotenv import load_dotenv
import google.generativeai as genai
import json
import time

class VideoAnalyzer:
    def __init__(self, video_path, temp_audio_dir):
        self.video_path = video_path
        self.temp_audio_dir = temp_audio_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"INFO: Initializing local Whisper model on: {self.device.upper()}")
        self.transcriber = pipeline(
            "automatic-speech-recognition", model="openai/whisper-base", device=self.device
        )
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set.")
        genai.configure(api_key=api_key)
        self.gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")

    def analyze(self, update_callback):
        update_callback(0, "Starting analysis...")
        
        transcript_data = self._extract_and_transcribe_audio(update_callback)
        gemini_analysis_data = self._analyze_video_with_gemini(update_callback)
        
        update_callback(95, "Merging analysis results...")
        final_results = self._merge_results(gemini_analysis_data, transcript_data)
        
        update_callback(100, "Analysis complete!")
        return final_results

    def _extract_and_transcribe_audio(self, update_callback):
        audio_path = None
        try:
            update_callback(5, "Extracting audio from video...")
            with VideoFileClip(self.video_path) as video_clip:
                if video_clip.audio:
                    audio_filename = f"{os.path.splitext(os.path.basename(self.video_path))[0]}.wav"
                    audio_path = os.path.join(self.temp_audio_dir, audio_filename)
                    video_clip.audio.write_audiofile(audio_path, codec='pcm_s16le', logger=None)
            
            if audio_path:
                update_callback(15, "Transcribing audio with Whisper...")
                transcription = self.transcriber(audio_path, chunk_length_s=30, return_timestamps=True)
                os.remove(audio_path)
                return transcription.get('chunks', [])
            return []
        except Exception as e:
            print(f"Audio processing failed: {e}")
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
            return []

    def _analyze_video_with_gemini(self, update_callback):
        try:
            update_callback(25, "Uploading video to Google AI...")
            video_file = genai.upload_file(path=self.video_path)
            
            while video_file.state.name == "PROCESSING":
                time.sleep(5)
                update_callback(40, "Processing video on Google's servers...")
                video_file = genai.get_file(video_file.name)

            if video_file.state.name == "FAILED":
                raise ValueError("Google AI File API failed to process the video.")

            update_callback(60, "Analyzing video with Gemini 1.5 Pro...")
            prompt = self._get_gemini_analysis_prompt()
            response = self.gemini_model.generate_content([prompt, video_file])
            
            genai.delete_file(video_file.name)
            
            return self._parse_gemini_response(response.text)
        except Exception as e:
            print(f"Gemini analysis failed: {e}")
            return {}

    def _get_gemini_analysis_prompt(self):
        return """
         **Role**: You are a world-class multimedia analyst AI. Your mission is to perform a deep, comprehensive analysis of the provided video file and structure your findings as a flawless JSON object.

         **Instructions**:
        1.  **Holistic Review**: Watch the entire video from beginning to end. Pay attention to visuals, spoken words (if any), sounds, and the overall narrative.
        2.  **Identify Key Elements**: Pinpoint all significant events, actions, objects, and spoken dialogue.
        3.  **Structure the Output**: Generate a single JSON object that strictly adheres to the schema below. Do not add any text or explanations outside of the JSON structure.

        **JSON Schema**:
        ```json
        {
          "title": "A catchy, newspaper-style headline that captures the single most important event or theme of the video.",
          "overall_summary": "A detailed, narrative paragraph describing the video's content, context, and the sequence of events. Write this as if you are a documentary narrator, telling the story of the video.",
          "key_topics": [
            "A list of 3-5 primary keywords or themes present in the video (e.g., 'Urban Driving', 'Pedestrian Safety', 'Traffic Flow')."
          ],
          "detailed_log": [
            {
              "timestamp": "The exact time in seconds (float, e.g., 3.14) where the event begins.",
              "type": "The category of the event. Choose from: '🎬 Action', '🚶‍♂️ Movement', '🚗 Vehicle', '⚠️ Hazard', '⭐ Notable Moment'.",
              "description": "A concise, one-sentence description of the specific event at this timestamp."
            }
          ]
        }
        ```
        **Your output must be ONLY the JSON object inside the ```json ... ``` tags.**
        """

    def _parse_gemini_response(self, response_text):
        try:
            match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if match:
                json_str = match.group(1)
                return json.loads(json_str)
            else:
                return {"summary": "Gemini response was not in the expected JSON format."}
        except json.JSONDecodeError:
            print("Failed to parse JSON from Gemini response.")
            return {"summary": response_text}

    def _merge_results(self, gemini_data, whisper_data):
        if not gemini_data:
            gemini_data = { "title": "Analysis Failed", "overall_summary": "Could not analyze video.", "key_topics": [], "detailed_log": [] }
            
        if whisper_data:
            for chunk in whisper_data:
                start_time, end_time = chunk['timestamp']
                gemini_data["detailed_log"].append({
                    "timestamp": start_time,
                    "type": "🎤 Dialogue",
                    "description": chunk['text'].strip()
                })
        
        if "detailed_log" in gemini_data:
            gemini_data["detailed_log"].sort(key=lambda x: x.get('timestamp', 0))

        return gemini_data

if __name__ == '__main__':
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    import re
    
    load_dotenv()
    console = Console()
    console.print(Panel("[bold magenta]🎬 Gemini-Native Video Analyzer Test 🎬[/bold magenta]", border_style="green"))
    test_video_path = 'test_video.mp4'
    temp_audio_folder = 'temp_audio'
    
    if not os.path.exists(test_video_path):
        console.print(Panel(f"[bold red]Error:[/bold red] Test video not found at '[cyan]{test_video_path}[/cyan]'", border_style="red"))
    else:
        if not os.path.exists(temp_audio_folder): os.makedirs(temp_audio_folder)
        
        console.print("\n[bold blue]Step 1: Initializing Analyzer...[/bold blue]")
        analyzer = VideoAnalyzer(video_path=test_video_path, temp_audio_dir=temp_audio_folder)
        console.print("[green]✅ Initialization complete.[/green]")

        console.print("\n[bold blue]Step 2: Starting Full Video Analysis...[/bold blue]")
        
        def test_callback(p, s): console.print(f"  [yellow]Progress: {int(p)}% - {s}[/yellow]")
        
        analysis_results = analyzer.analyze(test_callback)
        
        console.print("\n[bold blue]Step 3: Analysis Complete. Results:[/bold blue]")
        json_str = json.dumps(analysis_results, indent=2)
        syntax = Syntax(json_str, "json", theme="monokai", line_numbers=True)
        console.print(Panel(syntax, title="[bold green]📊 Final Analysis Output (JSON)[/bold green]", border_style="green"))

    console.print(Panel("[bold magenta]🏁 Test complete 🏁[/bold magenta]", border_style="green"))
