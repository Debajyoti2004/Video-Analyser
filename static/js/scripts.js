document.addEventListener('DOMContentLoaded', () => {
    const uploadView = document.getElementById('upload-view');
    const analysisView = document.getElementById('analysis-view');
    const dropZone = document.querySelector('.file-drop-zone');
    const fileInput = document.getElementById('video-upload-input');
    const browseBtn = document.querySelector('.browse-btn');
    const videoPlayer = document.getElementById('video-player');
    const overlayCanvas = document.getElementById('overlay-canvas');
    const chatWindow = document.getElementById('chat-window');
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-chat-btn');
    const statusBar = document.getElementById('status-bar');
    const progressBar = document.getElementById('progress-bar');
    const statusText = document.getElementById('status-text');
    const zoomInBtn = document.getElementById('zoom-in-btn');
    const zoomOutBtn = document.getElementById('zoom-out-btn');
    const zoomResetBtn = document.getElementById('zoom-reset-btn');
    const eventsTimeline = document.getElementById('events-timeline');
    const suggestedPromptsContainer = document.getElementById('suggested-prompts');
    const exportBtn = document.getElementById('export-btn');

    let currentVideoId = null;
    let fullAnalysisData = null;
    let zoomState = { scale: 1, x: 0, y: 0 };
    let isPanning = false;
    let panStart = { x: 0, y: 0 };

    const eventListeners = {
        attachAll: () => {
            dropZone.addEventListener('click', () => fileInput.click());
            browseBtn.addEventListener('click', () => fileInput.click());
            fileInput.addEventListener('change', (e) => ui.handleFiles(e.target.files));
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => dropZone.addEventListener(eventName, eventHandlers.preventDefaults, false));
            ['dragenter', 'dragover'].forEach(e => dropZone.addEventListener(e, () => dropZone.classList.add('drag-over'), false));
            ['dragleave', 'drop'].forEach(e => dropZone.addEventListener(e, () => dropZone.classList.remove('drag-over'), false));
            dropZone.addEventListener('drop', (e) => ui.handleFiles(e.dataTransfer.files), false);
            sendBtn.addEventListener('click', api.sendChatMessage);
            chatInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') api.sendChatMessage(); });
            zoomInBtn.addEventListener('click', () => ui.zoom.zoomIn());
            zoomOutBtn.addEventListener('click', () => ui.zoom.zoomOut());
            zoomResetBtn.addEventListener('click', () => ui.zoom.reset());
            videoPlayer.parentElement.addEventListener('mousedown', eventHandlers.onPanStart);
            videoPlayer.parentElement.addEventListener('mousemove', eventHandlers.onPanMove);
            window.addEventListener('mouseup', eventHandlers.onPanEnd);
            window.addEventListener('resize', ui.drawBoundingBoxes);
            suggestedPromptsContainer.addEventListener('click', eventHandlers.onSuggestedPromptClick);
            eventsTimeline.addEventListener('click', eventHandlers.onTimelineItemClick);
            exportBtn.addEventListener('click', utils.exportAnalysis);
        }
    };

    const eventHandlers = {
        preventDefaults: (e) => { e.preventDefault(); e.stopPropagation(); },
        onPanStart: (e) => { if (zoomState.scale > 1) { isPanning = true; panStart.x = e.clientX - zoomState.x; panStart.y = e.clientY - zoomState.y; } },
        onPanMove: (e) => { if (isPanning) { zoomState.x = e.clientX - panStart.x; zoomState.y = e.clientY - panStart.y; ui.zoom.apply(); } },
        onPanEnd: () => { isPanning = false; },
        onSuggestedPromptClick: (e) => {
            if (e.target.classList.contains('prompt-btn')) {
                const prompt = e.target.dataset.prompt;
                chatInput.value = prompt;
                api.sendChatMessage();
            }
        },
        onTimelineItemClick: (e) => {
            const item = e.target.closest('.event-item');
            if (item) {
                const timestamp = parseFloat(item.dataset.timestamp);
                videoPlayer.currentTime = timestamp;
                videoPlayer.play();
            }
        }
    };

    const ui = {
        handleFiles: (files) => {
            const file = files[0];
            if (!file || !file.type.startsWith('video/')) { return alert('Please select a valid video file.'); }
            ui.transitionToAnalysisView();
            const analysisType = document.querySelector('input[name="analysis_type"]:checked').value;
            api.uploadFile(file, analysisType);
        },
        transitionToAnalysisView: () => { uploadView.classList.add('hidden'); analysisView.classList.remove('hidden'); },
        updateStatus: (statusMessage) => {
            const match = statusMessage.match(/(\d+)% \((.*)\)/);
            if (match) {
                const [_, percent, text] = match;
                progressBar.style.width = `${percent}%`;
                statusText.textContent = text;
            }
        },
        analysisComplete: () => {
            progressBar.style.width = '100%';
            progressBar.style.backgroundColor = 'var(--success-color)';
            statusText.textContent = 'Analysis Complete!';
            setTimeout(() => statusBar.classList.add('hidden'), 2000);
            chatInput.disabled = false; sendBtn.disabled = false; chatInput.focus();
            exportBtn.classList.remove('hidden');
            ui.populateTimeline(fullAnalysisData);
            ui.showSuggestedPrompts();
            ui.appendMessage("Your video has been fully analyzed. Explore the timeline or ask me anything!", 'bot');
        },
        populateTimeline: (data) => {
            eventsTimeline.innerHTML = '';
            let events = [];
            if(data.transcript) data.transcript.forEach(t => events.push({type: 'transcript', time: t.timestamp[0], text: t.text}));
            if(data.scenes) data.scenes.forEach(s => events.push({type: 'scene', time: s.timestamp, text: s.description}));
            
            events.sort((a, b) => a.time - b.time);
            
            events.forEach(event => {
                const item = document.createElement('div');
                item.className = 'event-item';
                item.dataset.timestamp = event.time;
                const icon = event.type === 'transcript' ? '🗣️' : '🖼️';
                item.innerHTML = `<p><span class="timestamp">${icon} ${event.time.toFixed(2)}s:</span> ${event.text}</p>`;
                eventsTimeline.appendChild(item);
            });
        },
        showSuggestedPrompts: () => {
            suggestedPromptsContainer.innerHTML = '';
            const prompts = ["Summarize the video", "List all people detected", "What is the main topic?"];
            prompts.forEach(p => {
                const btn = document.createElement('button');
                btn.className = 'prompt-btn';
                btn.dataset.prompt = p;
                btn.textContent = p;
                suggestedPromptsContainer.appendChild(btn);
            });
            suggestedPromptsContainer.classList.remove('hidden');
        },
        appendMessage: (text, type) => {
            const messageDiv = document.createElement('div');
            messageDiv.className = `chat-message ${type}-message`;
            const p = document.createElement('p');
            p.textContent = text;
            messageDiv.appendChild(p);
            chatWindow.appendChild(messageDiv);
            chatWindow.scrollTop = chatWindow.scrollHeight;
        },
        toggleChatInput: (enabled) => { chatInput.disabled = !enabled; sendBtn.disabled = !enabled; },
        drawBoundingBoxes: (objects = []) => {
            const ctx = overlayCanvas.getContext('2d');
            overlayCanvas.width = videoPlayer.clientWidth;
            overlayCanvas.height = videoPlayer.clientHeight;
            const scaleX = overlayCanvas.width / videoPlayer.videoWidth;
            const scaleY = overlayCanvas.height / videoPlayer.videoHeight;
            ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
            if (!objects || objects.length === 0) return;
            objects.forEach(obj => {
                const [x, y, w, h] = obj.box;
                ctx.strokeStyle = 'var(--highlight-color)'; ctx.lineWidth = 3;
                ctx.strokeRect(x * scaleX, y * scaleY, w * scaleX, h * scaleY);
                ctx.fillStyle = 'var(--highlight-color)'; ctx.font = 'bold 14px sans-serif';
                ctx.fillText(obj.label, x * scaleX, y * scaleY - 5);
            });
            setTimeout(() => ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height), 4000);
        },
        zoom: {
            apply: () => { videoPlayer.style.transformOrigin = `0 0`; videoPlayer.style.transform = `translate(${zoomState.x}px, ${zoomState.y}px) scale(${zoomState.scale})`; },
            zoomIn: () => { zoomState.scale *= 1.2; ui.zoom.apply(); },
            zoomOut: () => { zoomState.scale /= 1.2; if (zoomState.scale < 1) ui.zoom.reset(); else ui.zoom.apply(); },
            reset: () => { zoomState = { scale: 1, x: 0, y: 0 }; videoPlayer.style.transform = 'none'; }
        }
    };

    const api = {
        uploadFile: (file, analysisType) => {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('analysis_type', analysisType);
            fetch('/upload', { method: 'POST', body: formData })
                .then(res => res.json()).then(data => {
                    if (data.error) throw new Error(data.error);
                    currentVideoId = data.video_id; videoPlayer.src = data.video_url;
                    api.pollAnalysisStatus(data.video_id);
                }).catch(err => { alert(`Upload Error: ${err.message}`); window.location.reload(); });
        },
        pollAnalysisStatus: (videoId) => {
            const interval = setInterval(() => {
                fetch(`/status/${videoId}`).then(res => res.json()).then(data => {
                    const status = data.status || 'unknown';
                    if (status.startsWith('processing:')) { ui.updateStatus(status); }
                    else if (status === 'completed') {
                        clearInterval(interval);
                        fetch(`/analysis_data/${videoId}`).then(res => res.json()).then(analysisData => {
                            fullAnalysisData = analysisData;
                            ui.analysisComplete();
                        });
                    } else if (status.startsWith('failed:')) {
                        clearInterval(interval);
                        progressBar.style.backgroundColor = '#ef4444';
                        statusText.textContent = `Analysis Failed`;
                    }
                });
            }, 2000);
        },
        sendChatMessage: () => {
            const message = chatInput.value.trim();
            if (!message || !currentVideoId) return;
            ui.appendMessage(message, 'user'); chatInput.value = ''; ui.toggleChatInput(false);
            fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message, video_id: currentVideoId })
            }).then(res => res.json()).then(data => {
                const { text, timestamp, highlight_objects } = data.response;
                ui.appendMessage(text, 'bot');
                if (timestamp !== undefined && timestamp !== null) { videoPlayer.currentTime = timestamp; videoPlayer.play(); }
                if (highlight_objects && highlight_objects.length > 0) { setTimeout(() => ui.drawBoundingBoxes(highlight_objects), 500); }
            }).finally(() => { ui.toggleChatInput(true); chatInput.focus(); });
        }
    };
    
    const utils = {
        exportAnalysis: () => {
            if (!fullAnalysisData) return;
            let markdown = `# Video Analysis Report\n\n`;
            
            markdown += `## 📝 Full Transcript\n\n`;
            if (fullAnalysisData.transcript && fullAnalysisData.transcript.length > 0) {
                fullAnalysisData.transcript.forEach(t => {
                    markdown += `**${t.timestamp[0].toFixed(2)}s:** ${t.text.trim()}\n\n`;
                });
            } else {
                markdown += `No audio transcript available.\n\n`;
            }
            
            markdown += `## 🖼️ Key Visual Scenes\n\n`;
            if (fullAnalysisData.scenes && fullAnalysisData.scenes.length > 0) {
                fullAnalysisData.scenes.forEach(s => {
                    markdown += `### Scene at ${s.timestamp.toFixed(2)}s\n`;
                    markdown += `${s.description.trim()}\n\n`;
                    if (s.objects && s.objects.length > 0) {
                        markdown += `**Detected Objects:** ${s.objects.map(o => o.label).join(', ')}\n\n`;
                    }
                });
            } else {
                markdown += `No visual scenes analyzed.\n\n`;
            }

            const blob = new Blob([markdown], { type: 'text/markdown' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `video_analysis_${currentVideoId.substring(0,8)}.md`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }
    };
    
    eventListeners.attachAll();
});