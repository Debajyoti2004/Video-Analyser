import os
import cv2
import torch
from moviepy import VideoFileClip
from transformers import pipeline
from PIL import Image
import google.generativeai as genai
from ultralytics import YOLO

class VideoAnalyzer:
    def __init__(self, video_path, temp_audio_dir):
        self.video_path = video_path
        self.temp_audio_dir = temp_audio_dir
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set.")
        genai.configure(api_key=api_key)
        
        self.vision_model = genai.GenerativeModel('gemini-pro-vision')
        self.detection_model = YOLO('yolov8n.pt')
        
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.transcriber = pipeline(
            "automatic-speech-recognition", model="openai/whisper-base", device=device
        )

    def analyze(self, update_callback, analysis_type='full'):
        transcript = []
        scenes = []
        
        do_audio = analysis_type in ['full', 'audio_only']
        do_vision = analysis_type in ['full', 'vision_only']

        if do_audio:
            update_callback(0, "Extracting audio...")
            audio_path = self._extract_audio()
            
            update_callback(15, "Transcribing audio...")
            transcript = self._transcribe_audio(audio_path)
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
        
        if do_vision:
            start_progress = 30 if do_audio else 0
            update_callback(start_progress, "Analyzing visual scenes...")
            scenes = self._analyze_visual_scenes(update_callback, start_progress)
        
        update_callback(100, "Analysis complete!")
        return {"transcript": transcript, "scenes": scenes}

    def _extract_audio(self):
        audio_path = None
        try:
            with VideoFileClip(self.video_path) as video_clip:
                if video_clip.audio:
                    audio_filename = f"{os.path.splitext(os.path.basename(self.video_path))[0]}.wav"
                    audio_path = os.path.join(self.temp_audio_dir, audio_filename)
                    video_clip.audio.write_audiofile(audio_path, codec='pcm_s16le')
        except Exception as e:
            print(f"Could not extract audio: {e}")
        return audio_path

    def _transcribe_audio(self, audio_path):
        if not audio_path:
            return []
        try:
            transcription = self.transcriber(audio_path, chunk_length_s=30, return_timestamps=True)
            return transcription.get('chunks', [])
        except Exception as e:
            print(f"Audio transcription failed: {e}")
            return []

    def _analyze_visual_scenes(self, update_callback, start_progress=30):
        scenes = []
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        key_frame_indices = self._detect_key_frames(cap)
        cap.release()
        
        end_progress = 100
        progress_range = end_progress - start_progress

        for idx, frame_index in enumerate(key_frame_indices):
            progress = start_progress + (idx / len(key_frame_indices) * progress_range) if key_frame_indices else 100
            update_callback(progress, f"Describing & detecting in scene {idx + 1}/{len(key_frame_indices)}")

            temp_cap = cv2.VideoCapture(self.video_path)
            temp_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = temp_cap.read()
            if ret:
                timestamp = frame_index / fps
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                description = self._get_gemini_description(pil_image, timestamp)
                detected_objects = self._get_yolo_detections(frame)
                
                scenes.append({
                    "timestamp": timestamp,
                    "description": description,
                    "objects": detected_objects
                })
            temp_cap.release()
            
        return scenes

    def _detect_key_frames(self, cap):
        key_frame_indices = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        prev_hist = None
        scene_detection_threshold = 0.6
        frame_interval = max(1, int(cap.get(cv2.CAP_PROP_FPS)))

        for i in range(0, total_frames, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret: break
            
            hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
            cv2.normalize(hist, hist)
            
            if prev_hist is not None:
                correlation = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                if correlation < scene_detection_threshold:
                    key_frame_indices.append(i)
            elif i == 0:
                key_frame_indices.append(i)
            prev_hist = hist
            
        return key_frame_indices

    def _get_gemini_description(self, pil_image, timestamp):
        try:
            response = self.vision_model.generate_content(
                ["Describe this scene in detail. What objects are present, what are they doing, and what is the overall environment?", pil_image]
            )
            return response.text
        except Exception as e:
            print(f"Gemini Vision API call failed at {timestamp:.2f}s: {e}")
            return "Visual analysis for this scene failed."

    def _get_yolo_detections(self, frame):
        objects = []
        results = self.detection_model(frame, verbose=False)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
                label = self.detection_model.names[int(box.cls)]
                confidence = float(box.conf)
                if confidence > 0.4:
                    objects.append({
                        "label": label,
                        "box": [x1, y1, x2 - x1, y2 - y1],
                        "confidence": confidence
                    })
        return objects