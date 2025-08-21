import os
import uuid
import threading
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from analysis_manager import AnalysisManager
from video_analyzer import VideoAnalyzer
from agent import ConversationalAgent

load_dotenv() 

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['TEMP_AUDIO_FOLDER'] = 'temp_audio'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'mov', 'avi', 'webm', 'mkv'}
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024 

for folder in [app.config['UPLOAD_FOLDER'], app.config['TEMP_AUDIO_FOLDER']]:
    if not os.path.exists(folder):
        os.makedirs(folder)

analysis_manager = AnalysisManager()
agent = ConversationalAgent(analysis_manager)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def run_analysis_task(video_id, video_path, analysis_type):
    try:
        analyzer = VideoAnalyzer(video_path, app.config['TEMP_AUDIO_FOLDER'])
        
        def update_callback(progress, status_message):
            analysis_manager.set_status(video_id, f"processing: {int(progress)}% ({status_message})")
        
        results = analyzer.analyze(update_callback, analysis_type)
        analysis_manager.store_results(video_id, results)
        analysis_manager.set_status(video_id, 'completed')
    except Exception as e:
        print(f"Unhandled exception in analysis thread for {video_id}: {e}")
        analysis_manager.set_status(video_id, f'failed: An internal error occurred.')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    analysis_type = request.form.get('analysis_type', 'full')
    
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type or no file selected'}), 400

    video_id = str(uuid.uuid4())
    filename = secure_filename(file.filename)
    extension = os.path.splitext(filename)[1]
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{video_id}{extension}")
    file.save(video_path)

    analysis_thread = threading.Thread(target=run_analysis_task, args=(video_id, video_path, analysis_type))
    analysis_thread.start()

    return jsonify({
        'message': 'Upload successful. Analysis is starting.',
        'video_id': video_id,
        'video_url': f'/uploads/{os.path.basename(video_path)}'
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
    return jsonify({"error": "Analysis data not found"}), 404

@app.route('/chat', methods=['POST'])
def chat_handler():
    data = request.get_json()
    video_id = data.get('video_id')
    user_message = data.get('message')

    if not all([video_id, user_message]):
        return jsonify({'error': 'Missing video_id or message'}), 400
    
    if analysis_manager.get_status(video_id) != 'completed':
        return jsonify({'response': {'text': 'The video is still being analyzed. Please wait.'}})

    response = agent.ask(user_message, video_id)
    return jsonify({'response': response})

@app.route('/uploads/<filename>')
def serve_video(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)