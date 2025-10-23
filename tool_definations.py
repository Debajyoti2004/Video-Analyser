from typing import List, Dict, Any

def get_tool_definitions() -> List[Dict[str, Any]]:
    return [
        {
            "name": "get_snapshot_for_analysis",
            "description": "Extracts a single, high-quality frame from a video at a precise timestamp. This is the essential first step for any visual analysis task like object detection or OCR.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "video_path": {
                        "type": "STRING",
                        "description": "The path to the source video file (e.g., 'test_video.mp4')."
                    },
                    "timestamp": {
                        "type": "NUMBER",
                        "description": "The exact time in seconds from which to extract the frame, often identified from the initial event log."
                    }
                },
                "required": ["video_path", "timestamp"]
            }
        },
        {
            "name": "detect_and_classify_objects",
            "description": "Analyzes a snapshot to find and classify all visible objects. It can optionally filter for a specific object.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "snapshot_path": {
                        "type": "STRING",
                        "description": "The file path of the snapshot image to analyze."
                    },
                    "object_query": {
                        "type": "STRING",
                        "description": "A specific object to search for (e.g., 'car', 'person', 'traffic light'). If omitted, all objects are returned."
                    }
                },
                "required": ["snapshot_path"]
            }
        },
        {
            "name": "perform_ocr_on_snapshot",
            "description": "Performs Optical Character Recognition (OCR) on a snapshot to read and return all visible text.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "snapshot_path": {
                        "type": "STRING",
                        "description": "The file path of the snapshot image from which to read text. This must be generated first."
                    }
                },
                "required": ["snapshot_path"]
            }
        },
        {
            "name": "transcribe_audio_from_clip",
            "description": "Extracts and transcribes audio from a small segment of the video around a specific timestamp. Ideal for isolating specific dialogue.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "video_path": {
                        "type": "STRING",
                        "description": "The path to the source video file (e.g., 'test_video.mp4')."
                    },
                    "timestamp": {
                        "type": "NUMBER",
                        "description": "The central timestamp of the event to analyze."
                    },
                    "search_radius_seconds": {
                        "type": "INTEGER",
                        "description": "The number of seconds before and after the timestamp to include in the audio clip. Defaults to 3."
                    }
                },
                "required": ["video_path", "timestamp"]
            }
        },
        {
            "name": "create_annotated_visual_evidence",
            "description": "Draws labeled boxes on a snapshot to create a final piece of visual evidence. Use this AFTER other tools to present findings.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "image_path": {
                        "type": "STRING",
                        "description": "The file path of the source snapshot image to draw on."
                    },
                    "annotations": {
                        "type": "ARRAY",
                        "description": "A list of annotation objects, where each object has a 'box' ([x1, y1, x2, y2]) and a 'label' (string).",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "box": {
                                    "type": "ARRAY",
                                    "description": "The coordinates of the bounding box [x1, y1, x2, y2].",
                                    "items": {
                                        "type": "NUMBER"
                                    }
                                },
                                "label": {
                                    "type": "STRING",
                                    "description": "The text label for the annotation."
                                }
                            },
                             "required": ["box", "label"]
                        }
                    }
                },
                "required": ["image_path", "annotations"]
            }
        },
        {
            "name": "generate_report",
            "description": "Creates a formal report document (PDF or CSV) based on the initial high-level video analysis data.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "report_data": {
                        "type": "OBJECT",
                        "description": "The full JSON object from the initial `VideoAnalyzer` containing the title, summary, event_log, etc."
                    },
                    "format": {
                        "type": "STRING",
                        "description": "The desired output format, either 'pdf' or 'csv'."
                    }
                },
                "required": ["report_data", "format"]
            }
        }
    ]