from flask import Flask, render_template, Response, jsonify
from pathlib import Path
import json
import asyncio
from datetime import datetime, timedelta
import threading
from queue import Queue
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

app = Flask(__name__)
event_queue = Queue()

@dataclass
class CameraLocation:
    lat: float
    lng: float
    name: str

@dataclass
class Detection:
    timestamp: str
    camera: str
    image: str
    type: str
    confidence: float
    make: Optional[str] = None
    model: Optional[str] = None
    color: Optional[str] = None

@dataclass
class Alert:
    type: str
    message: str
    timestamp: str
    camera: Optional[str] = None

class DashboardState:
    MAX_DETECTIONS = 50  # Limit stored detections
    MAX_ALERTS = 20      # Limit stored alerts
    
    def __init__(self):
        self.camera_stats = {}
        self.total_detections = 0
        self.total_errors = 0
        self.uptime_start = datetime.now()
        self.recent_detections: List[Detection] = []
        self.active_alerts: List[Alert] = []
        self.camera_locations = self._load_camera_locations()
        
    def _load_camera_locations(self) -> Dict[str, CameraLocation]:
        """Load camera locations from config file"""
        try:
            with open('config/camera_locations.json') as f:
                locations = json.load(f)
                return {
                    name: CameraLocation(**loc)
                    for name, loc in locations.items()
                }
        except FileNotFoundError:
            return {}

    def add_detection(self, detection: Detection):
        """Add a new detection and maintain history"""
        self.recent_detections.insert(0, detection)
        if len(self.recent_detections) > self.MAX_DETECTIONS:
            self.recent_detections.pop()
        self.total_detections += 1

    def add_alert(self, alert: Alert):
        """Add a new alert"""
        self.active_alerts.append(alert)
        # Keep only last MAX_ALERTS alerts
        if len(self.active_alerts) > self.MAX_ALERTS:
            self.active_alerts.pop(0)

    def clear_alert(self, camera: str):
        """Clear alerts for a specific camera"""
        self.active_alerts = [
            alert for alert in self.active_alerts 
            if alert.camera != camera
        ]

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
                
                # Add camera locations to stats
                for name, stats in dashboard_state.camera_stats.items():
                    if name in dashboard_state.camera_locations:
                        stats['location'] = asdict(dashboard_state.camera_locations[name])
                        
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
    
    # Calculate time-based stats
    now = datetime.now()
    hour_ago = now - timedelta(hours=1)
    
    # Limit detections to last hour only
    recent_detections = [
        d for d in dashboard_state.recent_detections
        if datetime.fromisoformat(d.timestamp) > hour_ago
    ][:30]  # Only return last 30 detections max
    
    # Count active cameras (those without errors in last 5 minutes)
    active_cameras = sum(1 for stats in dashboard_state.camera_stats.values() 
                        if not stats.get('error') or 
                        (datetime.fromisoformat(stats['last_update']) > now - timedelta(minutes=5)))
    
    # Calculate error rate
    total_cameras = len(dashboard_state.camera_stats)
    error_rate = ((total_cameras - active_cameras) / total_cameras * 100) if total_cameras > 0 else 0
    
    # Calculate uptime
    uptime_seconds = (now - dashboard_state.uptime_start).total_seconds()
    uptime_hours = int(uptime_seconds // 3600)
    uptime_minutes = int((uptime_seconds % 3600) // 60)
    
    return jsonify({
        'camera_stats': dashboard_state.camera_stats,
        'total_detections': dashboard_state.total_detections,
        'total_errors': dashboard_state.total_errors,
        'uptime': {
            'hours': uptime_hours,
            'minutes': uptime_minutes,
            'total_seconds': uptime_seconds
        },
        'recent_detections': [asdict(d) for d in recent_detections],  # Limited to 30 most recent
        'active_alerts': [asdict(a) for a in dashboard_state.active_alerts[-10:]],  # Only last 10 alerts
        'detection_rate': len(recent_detections),
        'active_cameras': active_cameras,
        'error_rate': error_rate,
        'camera_locations': {
            name: asdict(loc) for name, loc in dashboard_state.camera_locations.items()
        }
    })

@app.route('/stream')
def stream():
    """SSE endpoint for real-time updates"""
    def event_stream():
        while True:
            event = event_queue.get()
            if isinstance(event, (dict, list)):
                yield f"data: {json.dumps(event)}\n\n"
            else:
                yield f"data: {json.dumps(asdict(event))}\n\n"
    
    return Response(
        event_stream(),
        mimetype='text/event-stream'
    )

def run_dashboard():
    """Run the dashboard server"""
    app.run(host='0.0.0.0', port=8675, debug=True)

if __name__ == '__main__':
    run_dashboard() 