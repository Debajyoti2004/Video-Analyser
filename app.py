import os
import uuid
import threading
import asyncio
import json
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from rich.console import Console

from video_analyzer import VideoAnalyzer
from agent import VideoAnalysisAgent

load_dotenv()

class AnalysisManager:
    def __init__(self):
        self.tasks = {}
        self._lock = threading.Lock()

    def set_status(self, video_id, status):
        with self._lock:
            if video_id in self.tasks:
                self.tasks[video_id]['status'] = status

    def get_status(self, video_id):
        with self._lock:
            return self.tasks.get(video_id, {}).get('status')

    def store_results(self, video_id, results):
        with self._lock:
            if video_id in self.tasks:
                self.tasks[video_id]['results'] = results

    def get_results(self, video_id):
        with self._lock:
            return self.tasks.get(video_id, {}).get('results')
            
    def create_task(self, video_id, video_path):
        with self._lock:
            self.tasks[video_id] = {
                'status': 'pending',
                'video_path': video_path,
                'results': None
            }

    def get_video_path(self, video_id):
        with self._lock:
            return self.tasks.get(video_id, {}).get('video_path')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'mov', 'avi', 'webm', 'mkv'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

analysis_manager = AnalysisManager()
agent_sessions = {}
console = Console()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

async def run_async_analysis(video_id, video_path):
    try:
        analyzer = VideoAnalyzer(console=console)
        
        def update_callback(progress, status_message):
            analysis_manager.set_status(video_id, f"processing: {int(progress)}% ({status_message})")
        
        results = await analyzer.analyze(video_path, update_callback)
        analysis_manager.store_results(video_id, results)
        analysis_manager.set_status(video_id, 'completed')
    except Exception as e:
        console.print(f"[bold red]Error during analysis for {video_id}: {e}[/bold red]")
        analysis_manager.set_status(video_id, f'failed: {str(e)}')

def run_analysis_in_thread(video_id, video_path):
    asyncio.run(run_async_analysis(video_id, video_path))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type or no file selected'}), 400

    video_id = str(uuid.uuid4())
    filename = secure_filename(file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{video_id}_{filename}")
    file.save(video_path)

    analysis_manager.create_task(video_id, video_path)
    
    analysis_thread = threading.Thread(target=run_analysis_in_thread, args=(video_id, video_path))
    analysis_thread.start()

    return jsonify({
        'message': 'Upload successful. Analysis is starting.',
        'video_id': video_id
    })

@app.route('/status/<video_id>')
def get_status(video_id):
    status = analysis_manager.get_status(video_id)
    return jsonify({'status': status or 'not_found'})

@app.route('/analysis_data/<video_id>')
def get_analysis_data(video_id):
    results = analysis_manager.get_results(video_id)
    if results:
        return jsonify(results)
    return jsonify({"error": "Analysis data not found or not yet complete"}), 404

@app.route('/chat', methods=['POST'])
def chat_handler():
    data = request.get_json()
    video_id = data.get('video_id')
    user_message = data.get('message')
    session_id = data.get('session_id', str(uuid.uuid4()))

    if not all([video_id, user_message]):
        return jsonify({'error': 'Missing video_id or message'}), 400
    
    if analysis_manager.get_status(video_id) != 'completed':
        return jsonify({'answer': 'The video is still being analyzed. Please wait until analysis is complete.'})

    response_data = asyncio.run(run_async_chat(video_id, user_message, session_id))
    return jsonify(response_data)

async def run_async_chat(video_id, user_message, session_id):
    if video_id not in agent_sessions:
        console.print(f"Creating new agent session for video_id: {video_id}")
        video_path = analysis_manager.get_video_path(video_id)
        if not video_path:
            return {"error": "Could not find video path for this ID."}

        agent: VideoAnalysisAgent = await VideoAnalysisAgent.create(video_path=video_path, role="owner")
        await agent._connect_to_mcp()
        agent_sessions[video_id] = agent
    
    agent = agent_sessions[video_id]
    analysis_results = analysis_manager.get_results(video_id)
    
    run_config = {"configurable": {"thread_id": session_id}}
    
    response_json_str = await agent.get_agent_response(
        user_command=user_message,
        analysis_results=analysis_results,
        config=run_config
    )
    return json.loads(response_json_str)

@app.route('/uploads/<path:filename>')
def serve_video(filename):
    return send_from_directory(os.getcwd(), filename)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)