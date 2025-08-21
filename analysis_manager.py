import threading

class AnalysisManager:
    def __init__(self):
        self._analyses = {}
        self._statuses = {}
        self._lock = threading.Lock()

    def store_results(self, video_id, results):
        with self._lock:
            self._analyses[video_id] = results

    def get_results(self, video_id):
        with self._lock:
            return self._analyses.get(video_id)

    def set_status(self, video_id, status):
        with self._lock:
            self._statuses[video_id] = status

    def get_status(self, video_id):
        with self._lock:
            return self._statuses.get(video_id)