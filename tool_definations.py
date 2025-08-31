from cohere.types import Tool
from typing import List

def get_tool_definitions() -> List[Tool]:
    return [
        Tool(
            name="get_snapshot_for_analysis",
            description="Extracts a single, high-quality frame from a video at a precise timestamp. This is the essential first step for any visual analysis task like object detection or OCR.",
            parameter_definitions={
                "video_path": {
                    "description": "The path to the source video file (e.g., 'test_video.mp4').",
                    "type": "string",
                    "required": True
                },
                "timestamp": {
                    "description": "The exact time in seconds from which to extract the frame, often identified from the initial event log.",
                    "type": "number",
                    "required": True
                }
            }
        ),
        Tool(
            name="detect_and_classify_objects",
            description="Analyzes a snapshot to find and classify all visible objects. It can optionally filter for a specific object.",
            parameter_definitions={
                "snapshot_path": {
                    "description": "The file path of the snapshot image to analyze.",
                    "type": "string",
                    "required": True
                },
                "object_query": {
                    "description": "A specific object to search for (e.g., 'car', 'person', 'traffic light'). If omitted, all objects are returned.",
                    "type": "string",
                    "required": False
                }
            }
        ),
        Tool(
            name="perform_ocr_on_snapshot",
            description="Performs Optical Character Recognition (OCR) on a snapshot to read and return all visible text.",
            parameter_definitions={
                "snapshot_path": {
                    "description": "The file path of the snapshot image from which to read text. This must be generated first.",
                    "type": "string",
                    "required": True
                }
            }
        ),
        Tool(
            name="transcribe_audio_from_clip",
            description="Extracts and transcribes audio from a small segment of the video around a specific timestamp. Ideal for isolating specific dialogue.",
            parameter_definitions={
                "video_path": {
                    "description": "The path to the source video file (e.g., 'test_video.mp4').",
                    "type": "string",
                    "required": True
                },
                "timestamp": {
                    "description": "The central timestamp of the event to analyze.",
                    "type": "number",
                    "required": True
                },
                "search_radius_seconds": {
                    "description": "The number of seconds before and after the timestamp to include in the audio clip. Defaults to 3.",
                    "type": "integer",
                    "required": False
                }
            }
        ),
        Tool(
            name="create_annotated_visual_evidence",
            description="Draws labeled boxes on a snapshot to create a final piece of visual evidence. Use this AFTER other tools to present findings.",
            parameter_definitions={
                "image_path": {
                    "description": "The file path of the source snapshot image to draw on.",
                    "type": "string",
                    "required": True
                },
                "annotations": {
                    "description": "A list of annotation objects, where each object has a 'box' ([x1, y1, x2, y2]) and a 'label' (string).",
                    "type": "array",
                    "required": True
                }
            }
        ),
        Tool(
            name="generate_report",
            description="Creates a formal report document (PDF or CSV) based on the initial high-level video analysis data.",
            parameter_definitions={
                "report_data": {
                    "description": "The full JSON object from the initial `VideoAnalyzer` containing the title, summary, event_log, etc.",
                    "type": "object",
                    "required": True
                },
                "format": {
                    "description": "The desired output format, either 'pdf' or 'csv'.",
                    "type": "string",
                    "required": True
                }
            }
        )
    ]