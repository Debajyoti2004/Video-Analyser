import os
import re
import json
import google.generativeai as genai

class ConversationalAgent:
    def __init__(self, analysis_manager):
        self.analysis_manager = analysis_manager
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set.")
        genai.configure(api_key=api_key)
        self.chat_model = genai.GenerativeModel('gemini-pro')

    def ask(self, query, video_id):
        analysis_data = self.analysis_manager.get_results(video_id)
        context = self._build_context_from_data(analysis_data)
        
        prompt = f"""
        🎬 **Role**: You are a world-class AI Video Analysis Agent. Your purpose is to provide precise, factual answers about a video's content based *exclusively* on the timed context provided. You can also identify specific objects to be highlighted.

        ---
        📚 **Video Context** 📚
        Here is the chronological log of events, combining audio transcripts, visual scene descriptions, and a list of detected objects with their bounding box coordinates.
        {context}
        ---

        ❓ **User's Query** ❓
        "{query}"

        ---
        ⚙️ **Your Reasoning Process** ⚙️
        1.  **Analyze Query**: Understand the user's intent. Are they asking to find, identify, or point out a specific object or person?
        2.  **Scan Context**: Search the video context for events related to the query.
        3.  **Synthesize Answer**: Formulate a concise answer based *only* on the facts in the context.
        4.  **Identify Timestamp & Objects**: If the user asks to see something (e.g., "point out the bicycle"), find the scene where it appears. From that scene's `[Full Object List]`, find the object entry for "bicycle" and extract its `box` coordinates.
        5.  **Format Output**: Construct the final JSON output. The `highlight_objects` field should contain a list of all objects that should be visually highlighted.

        ---
        📝 **Output Format** 📝
        You MUST respond in a strict JSON format. Do not add any text outside of the JSON block.

        ```json
        {{
          "answer": "Your concise, factual answer goes here.",
          "timestamp": "S.SS",
          "found_in_context": true,
          "highlight_objects": [
            {{
              "label": "object_label",
              "box": [x, y, width, height]
            }}
          ]
        }}
        ```

        - **`answer`**: The natural language response.
        - **`timestamp`**: The most relevant timestamp as a float (e.g., 123.45). Use `null` if not applicable.
        - **`found_in_context`**: `true` or `false`.
        - **`highlight_objects`**: A list of objects to draw boxes around. If the user doesn't ask to see anything, this should be an empty list `[]`.
        """
        
        try:
            response = self.chat_model.generate_content(prompt)
            return self._parse_llm_response(response.text)
        except Exception as e:
            print(f"Error querying Gemini Pro API: {e}")
            return {'text': "I'm sorry, an error occurred while processing your request."}

    def _build_context_from_data(self, data):
        context_entries = []
        if data.get('transcript'):
            for chunk in data['transcript']:
                start_time = chunk.get('timestamp', [None])[0]
                if start_time is not None:
                    context_entries.append({'time': start_time, 'content': f"Audio 🗣️: \"{chunk['text'].strip()}\""})
        
        if data.get('scenes'):
            for scene in data['scenes']:
                detected_objects_str = ", ".join([obj['label'] for obj in scene.get('objects', [])])
                scene_content = (
                    f"Visual 🖼️: {scene['description'].strip()} "
                    f"[Detected Objects: {detected_objects_str if detected_objects_str else 'None'}] "
                    f"[Full Object List: {scene.get('objects', [])}]"
                )
                context_entries.append({'time': scene['timestamp'], 'content': scene_content})
        
        sorted_entries = sorted(context_entries, key=lambda x: x['time'])
        return "\n".join([f"Time {entry['time']:.2f}s - {entry['content']}" for entry in sorted_entries])

    def _parse_llm_response(self, text):
        try:
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                data = json.loads(json_str)
                
                return {
                    "text": data.get("answer", "No answer provided."),
                    "timestamp": data.get("timestamp"),
                    "highlight_objects": data.get("highlight_objects", [])
                }
            else:
                return {"text": text, "highlight_objects": []}
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"Failed to parse LLM JSON response: {e}\nRaw response: {text}")
            return {"text": text, "highlight_objects": []}