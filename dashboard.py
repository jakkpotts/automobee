from flask import Flask, render_template, Response, jsonify
from pathlib import Path
import json
import asyncio
from datetime import datetime
import threading
from queue import Queue
import logging

app = Flask(__name__)
event_queue = Queue()

# Shared state between detector and dashboard
class DashboardState:
    def __init__(self):
        self.camera_stats = {}
        self.total_detections = 0
        self.total_errors = 0
        self.uptime_start = datetime.now()
        self.recent_detections = []  # Keep last 10 detections
        self.active_alerts = []

dashboard_state = DashboardState()

def update_dashboard_state(stats_file: Path):
    """Update dashboard state from stats file"""
    try:
        if stats_file.exists():
            with open(stats_file) as f:
                stats = json.load(f)
                dashboard_state.camera_stats = stats['camera_stats']
                dashboard_state.total_detections = stats['total_detections']
                dashboard_state.total_errors = stats['total_errors']
    except Exception as e:
        logging.error(f"Error updating dashboard state: {e}")

@app.route('/')
def index():
    """Render main dashboard"""
    return render_template('dashboard.html')

@app.route('/api/stats')
def get_stats():
    """Get current system stats"""
    update_dashboard_state(Path('monitoring/stats.json'))
    return jsonify({
        'camera_stats': dashboard_state.camera_stats,
        'total_detections': dashboard_state.total_detections,
        'total_errors': dashboard_state.total_errors,
        'uptime': (datetime.now() - dashboard_state.uptime_start).total_seconds(),
        'recent_detections': dashboard_state.recent_detections,
        'active_alerts': dashboard_state.active_alerts
    })

@app.route('/stream')
def stream():
    """SSE endpoint for real-time updates"""
    def event_stream():
        while True:
            event = event_queue.get()
            yield f"data: {json.dumps(event)}\n\n"
    
    return Response(
        event_stream(),
        mimetype='text/event-stream'
    )

def run_dashboard():
    """Run the dashboard server"""
    app.run(host='0.0.0.0', port=8675, debug=True)

if __name__ == '__main__':
    run_dashboard() 