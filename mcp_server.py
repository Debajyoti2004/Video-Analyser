import os
import cv2
import csv
import asyncio
from moviepy import VideoFileClip
from mcp.server.fastmcp import FastMCP
from rich.console import Console
from rich.panel import Panel
from ultralytics import YOLO
import easyocr
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from transformers import pipeline
import config

console = Console()
mcp = FastMCP(
    name="kala-sahayak-tool-server",
    host=config.MCP_SERVER_HOST,
    port=config.MCP_SERVER_PORT
)

yolo_model = YOLO('yolov8n.pt')
ocr_reader = easyocr.Reader(['en'])
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base")


@mcp.tool()
async def get_snapshot_for_analysis(video_path: str, timestamp: float) -> dict:
    def _blocking_snapshot():
        os.makedirs(config.SNAPSHOTS_DIR, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): raise IOError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = cap.read()
        cap.release()
        if not success: raise ValueError(f"Failed to retrieve frame at {timestamp}s")
        snapshot_path = os.path.join(config.SNAPSHOTS_DIR, f"snapshot_at_{timestamp:.2f}s.jpg")
        cv2.imwrite(snapshot_path, frame)
        return {"file_path": snapshot_path, "confirmation_message": f"Snapshot saved at {snapshot_path}"}
    return await asyncio.to_thread(_blocking_snapshot)


@mcp.tool()
async def detect_and_classify_objects(snapshot_path: str, object_query: str = None) -> dict:
    def _blocking_detection():
        img = cv2.imread(snapshot_path)
        if img is None: raise FileNotFoundError(snapshot_path)
        results = yolo_model(img, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                class_name = yolo_model.names[int(box.cls[0])]
                if object_query is None or object_query.lower() in class_name.lower():
                    coords = [int(c) for c in box.xyxy[0]]
                    detections.append({"object_name": class_name, "box": coords})
        return {"object_count": len(detections), "detected_objects": detections}
    return await asyncio.to_thread(_blocking_detection)


@mcp.tool()
async def perform_ocr_on_snapshot(snapshot_path: str) -> dict:
    def _blocking_ocr():
        image = cv2.imread(snapshot_path)
        if image is None: raise FileNotFoundError(snapshot_path)
        results = ocr_reader.readtext(image)
        text_blocks = [{"box": [[int(p[0]), int(p[1])] for p in result[0]], "text": result[1]} for result in results]
        return {"text_blocks_found": len(text_blocks), "extracted_text": text_blocks}
    return await asyncio.to_thread(_blocking_ocr)


@mcp.tool()
async def transcribe_audio_from_clip(video_path: str, timestamp: float, search_radius_seconds: int = 3) -> dict:
    def _blocking_transcription():
        os.makedirs(config.TEMP_AUDIO_DIR, exist_ok=True)
        start_time = max(0, timestamp - search_radius_seconds)
        end_time = timestamp + search_radius_seconds
        temp_audio_path = None
        try:
            with VideoFileClip(video_path) as video:
                if not video.audio: return {"transcribed_text": "No audio track found."}
                if end_time > video.duration: end_time = video.duration
                clip = video.subclip(start_time, end_time)
                temp_audio_path = os.path.join(config.TEMP_AUDIO_DIR, f"temp_clip_{start_time:.2f}_{end_time:.2f}.wav")
                clip.audio.write_audiofile(temp_audio_path, codec='pcm_s16le', logger=None)
            result = transcriber(temp_audio_path)
            return {"transcribed_text": result['text'].strip() if result['text'] else "No speech detected."}
        finally:
            if temp_audio_path and os.path.exists(temp_audio_path): os.remove(temp_audio_path)
    return await asyncio.to_thread(_blocking_transcription)


@mcp.tool()
async def create_annotated_visual_evidence(image_path: str, annotations: list) -> dict:
    def _blocking_annotate():
        img = cv2.imread(image_path)
        if img is None: raise FileNotFoundError(image_path)
        for ann in annotations:
            box, label = ann['box'], ann['label']
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (36, 255, 12), 2)
            cv2.putText(img, label, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        output_path = f"{os.path.splitext(image_path)[0]}_annotated.jpg"
        cv2.imwrite(output_path, img)
        return {"file_path": output_path, "confirmation_message": f"Annotated image saved at {output_path}"}
    return await asyncio.to_thread(_blocking_annotate)


@mcp.tool()
async def generate_report(report_data: dict, format: str) -> dict:
    def _blocking_report_generation():
        os.makedirs(config.REPORTS_DIR, exist_ok=True)
        file_path = ""
        if format.lower() == "pdf":
            output_path = os.path.join(config.REPORTS_DIR, "analysis_report.pdf")
            doc = SimpleDocTemplate(output_path, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
            styles = getSampleStyleSheet()
            styles.add(ParagraphStyle(name="CustomHeading", parent=styles["Heading2"], spaceAfter=10, textColor=colors.HexColor("#2E4053")))
            story = []

            story.append(Paragraph(report_data.get("title", "Video Report"), styles["Title"]))
            story.append(Spacer(1, 20))
            story.append(Paragraph("Overall Summary", styles["CustomHeading"]))
            story.append(Paragraph(report_data.get("overall_summary", "N/A"), styles["Normal"]))
            story.append(Spacer(1, 20))

            story.append(Paragraph("Key Topics", styles["CustomHeading"]))
            key_topics_data = [[f"• {topic}"] for topic in report_data.get("key_topics", [])]
            if key_topics_data:
                table = Table(key_topics_data, colWidths=[450])
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0,0), (-1,-1), colors.whitesmoke),
                    ("TEXTCOLOR",(0,0),(-1,-1),colors.black),
                    ("FONTNAME",(0,0),(-1,-1),"Helvetica"),
                    ("FONTSIZE",(0,0),(-1,-1),11),
                    ("ALIGN",(0,0),(-1,-1),"LEFT"),
                    ("BOTTOMPADDING",(0,0),(-1,-1),6)
                ]))
                story.append(table)
            story.append(Spacer(1, 20))

            story.append(Paragraph("Event Log", styles["CustomHeading"]))
            event_data = [["Timestamp","Type","Description"]]
            for item in report_data.get("event_log", []):
                event_data.append([str(item.get("timestamp","")), item.get("type",""), item.get("description","")])
            event_table = Table(event_data, colWidths=[80,100,300])
            event_table.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1F618D")),
                ("TEXTCOLOR",(0,0),(-1,0),colors.whitesmoke),
                ("ALIGN",(0,0),(-1,-1),"LEFT"),
                ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
                ("FONTSIZE",(0,0),(-1,-1),10),
                ("BOTTOMPADDING",(0,0),(-1,0),8),
                ("BACKGROUND",(0,1),(-1,-1),colors.HexColor("#EBF5FB")),
                ("GRID",(0,0),(-1,-1),0.5,colors.grey)
            ]))
            story.append(event_table)
            story.append(PageBreak())
            doc.build(story)
            file_path = output_path

        elif format.lower() == "csv":
            output_path = os.path.join(config.REPORTS_DIR, "event_log.csv")
            with open(output_path,"w",newline="",encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["timestamp","type","description"])
                writer.writeheader()
                writer.writerows(report_data.get("event_log",[]))
            file_path = output_path

        else:
            raise ValueError("Unsupported format. Choose 'pdf' or 'csv'.")

        return {"file_path": file_path, "confirmation_message": f"Report saved at {file_path}"}

    return await asyncio.to_thread(_blocking_report_generation)


def run():
    console.print(Panel("[bold green]🚀 Kala-Sahayak MCP Tool Server Online[/]", title="🖥️ Server Status", border_style="green"))
    console.print(f"🌍 Listening on [bold cyan]{config.MCP_SERVER_URL}[/bold cyan]")
    console.print(f"👉 To inspect tools, run: [bold]mcp inspector {config.MCP_SERVER_URL}[/bold]")
    mcp.run(transport="sse")


if __name__ == "__main__":
    run()
