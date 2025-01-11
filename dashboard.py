from typing import Dict, List, Optional
from dataclasses import dataclass
from functools import wraps
from datetime import datetime
import logging
import json
import queue
import asyncio
import time
from async_generator import async_generator, yield_
from flask import Flask, jsonify, Response, request, render_template, send_from_directory, current_app, Blueprint
from flask_cors import CORS
from asgiref.wsgi import WsgiToAsgi
from contextlib import asynccontextmanager
from app_factory import app

dashboard_bp = Blueprint("dashboard_bp", __name__)

# Initialize Flask app with ASGI wrapper and lifespan support
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
# Make sure run_dashboard is explicitly exported
__all__ = ['Alert', 'Detection', 'dashboard_state', 'event_queue', 'run_dashboard']

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

@dataclass
class Detection:
    timestamp: str
    camera: str
    image: str
    type: str
    confidence: float
    make: Optional[str] = None
    model: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert Detection object to dictionary for JSON serialization."""
        return {
            'timestamp': self.timestamp,
            'camera': self.camera,
            'image': self.image,
            'type': self.type,
            'confidence': self.confidence,
            'make': self.make,
            'model': self.model
        }

@dataclass
class Alert:
    type: str
    message: str
    timestamp: str
    camera: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert Alert object to dictionary for JSON serialization."""
        return {
            'type': self.type,
            'message': self.message,
            'timestamp': self.timestamp,
            'camera': self.camera
        }

@dataclass
class AppConfig:
    """Application configuration."""
    DEBUG: bool = False
    CAMERA_TIMEOUT: int = 30
    MAX_DETECTIONS: int = 100
    STREAM_RETRY_TIMEOUT: int = 5000

class DashboardState:
    """Manages dashboard state."""
    def __init__(self):
        # Initialize with proper camera data structure
        self.cameras = {
            'cam1': {
                'id': 'cam1', 
                'name': 'Camera 1', 
                'active': True,
                'location': {
                    'lat': 40.7128,
                    'lng': -74.0060
                },
                'stream_url': 'http://example.com/stream1'
            },
            'cam2': {
                'id': 'cam2', 
                'name': 'Camera 2', 
                'active': True,
                'location': {
                    'lat': 40.7580,
                    'lng': -73.9855
                },
                'stream_url': 'http://example.com/stream2'
            }
        }
        self.detections: List[Detection] = []
        self.alerts: List[Alert] = []
        self._max_detections = AppConfig.MAX_DETECTIONS
        self.camera_locations: Dict[str, dict] = {
            'cam1': {'lat': 40.7128, 'lng': -74.0060},
            'cam2': {'lat': 40.7580, 'lng': -73.9855}
        }
        self.stats = {
            'total_detections': 0,
            'active_cameras': 0,
            'recent_detections': [],
            'alerts': []
        }

    def add_detection(self, detection: Detection) -> None:
        self.detections.insert(0, detection)
        if len(self.detections) > self._max_detections:
            self.detections.pop()

    def add_alert(self, alert: Alert) -> None:
        self.alerts.insert(0, alert)

    def clear_alert(self, camera: str) -> None:
        self.alerts = [a for a in self.alerts if a.camera != camera]

    def get_stats(self) -> dict:
        return {
            'total_detections': len(self.detections),
            'active_cameras': len([c for c in self.cameras.values() if c.get('active')]),
            'recent_detections': self.detections[:10],
            'active_alerts': self.alerts[:5],
            'camera_stats': self.cameras,
            'camera_locations': self.camera_locations  # Include camera locations in stats
        }

    def update_camera(self, camera_id: str, data: dict) -> None:
        """Update camera status and information."""
        if camera_id not in self.cameras:
            self.cameras[camera_id] = {'id': camera_id}
        
        self.cameras[camera_id].update(data)
        logger.info(f"Updated camera {camera_id}: {data}")

# Initialize global state and event queue
dashboard_state = DashboardState()
# Add some test detections
test_detection = Detection(
    timestamp=datetime.utcnow().isoformat(),
    camera='cam1',
    image='test_image.jpg',
    type='ford_f150',
    confidence=0.92,
    make='ford',
    model='f150'
)
dashboard_state.add_detection(test_detection)

# Add test alert
test_alert = Alert(
    type='vehicle_detected',
    message='Ford F-150 detected at main entrance',
    timestamp=datetime.utcnow().isoformat(),
    camera='cam1'
)
dashboard_state.add_alert(test_alert)

event_queue = queue.Queue()

# Add lifespan handler
@asynccontextmanager
async def lifespan(app):
    # Startup
    logger.info("Starting dashboard application...")
    yield
    # Shutdown
    logger.info("Shutting down dashboard application...")


# Error handling middleware
def handle_errors(f):
    """Decorator for consistent error handling."""
    @wraps(f)
    async def decorated_function(*args, **kwargs):
        try:
            return await f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {str(e)}", exc_info=True)
            return jsonify({
                'error': str(e),
                'status': 'error'
            }), 500
    return decorated_function

# Input validation
def validate_detection(data: dict) -> bool:
    """Validate detection data."""
    required = {'type', 'confidence', 'camera', 'timestamp', 'image'}
    return all(k in data for k in required)

def validate_camera_update(data: dict) -> bool:
    """Validate camera update data."""
    required = {'status', 'timestamp'}
    return all(k in data for k in required)

# Route handlers
def format_camera_data(camera_id: str, camera_info: dict) -> dict:
    """Format camera data for API response."""
    try:
        # Handle location data safely
        location = camera_info.get('location', {})
        if isinstance(location, str):
            # If location is a string, create default location
            location = {'lat': 0, 'lng': 0}
        elif not isinstance(location, dict):
            location = {'lat': 0, 'lng': 0}

        return {
            "name": camera_info.get('name', f"Camera {camera_id}"),
            "status": "active" if camera_info.get('active', False) else "error",
            "location": {
                "latitude": float(location.get('lat', 0)),
                "longitude": float(location.get('lng', 0))
            },
            "stream_url": camera_info.get('stream_url', ''),
            "last_update": camera_info.get('last_update', datetime.utcnow().isoformat())
        }
    except Exception as e:
        logger.error(f"Error formatting camera data for {camera_id}: {e}")
        # Return safe default values if formatting fails
        return {
            "name": f"Camera {camera_id}",
            "status": "error",
            "location": {"latitude": 0, "longitude": 0},
            "stream_url": "",
            "last_update": datetime.utcnow().isoformat()
        }


# Remove this duplicate route from the Blueprint
# @dashboard_bp.route("/api/stats")
# def get_stats():
#     # ...remove this entire function...

@app.route('/api/stats')
def get_stats():
    """Get dashboard statistics with proper data structure."""
    try:
        # Format camera data
        formatted_cameras = {
            camera_id: format_camera_data(camera_id, camera_info)
            for camera_id, camera_info in dashboard_state.cameras.items()
        }

        # Calculate detection rate
        current_time = datetime.utcnow()
        hour_ago = current_time.timestamp() - 3600
        recent_detections = [d for d in dashboard_state.detections 
                           if datetime.fromisoformat(d.timestamp).timestamp() > hour_ago]
        
        # Format response using to_dict methods
        response_data = {
            "cameras": formatted_cameras,
            "recent_detections": [
                detection.to_dict() for detection in dashboard_state.detections[-50:]
            ],
            "alerts": [
                alert.to_dict() for alert in dashboard_state.alerts
            ],
            "stats": {
                "total_detections": len(dashboard_state.detections),
                "detection_rate": len(recent_detections)
            }
        }

        logger.debug(f"Returning stats: {response_data}")
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error getting stats: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/detections', methods=['POST'])
@handle_errors
def add_detection():  # removed async
    """Add new detection."""
    data = request.json
    if not validate_detection(data):
        return jsonify({'error': 'Invalid detection data'}), 400

    detection = Detection(**data)
    dashboard_state.add_detection(detection)
    return jsonify({'status': 'success'})

@app.route('/api/cameras/<camera_id>', methods=['PUT'])
@handle_errors
def update_camera(camera_id: str):  # removed async
    """Update camera status."""
    data = request.json
    if not validate_camera_update(data):
        return jsonify({'error': 'Invalid camera data'}), 400

    dashboard_state.update_camera(camera_id, data)
    return jsonify({'status': 'success'})

@app.after_request
def add_header(response):
    """Add headers to optimize performance."""
    response.headers['Cache-Control'] = 'public, max-age=300'  # 5 minutes cache
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response

# Update the stream function to use chunked transfer
@app.route('/stream')
def stream():
    def event_stream():
        try:
            last_time = datetime.utcnow()
            while True:
                # Get any new events from the queue
                try:
                    while True:  # Drain all available events
                        event = event_queue.get_nowait()
                        if event:
                            # Ensure event data is serializable
                            if isinstance(event.get('data'), (Detection, Alert)):
                                event['data'] = event['data'].to_dict()
                            yield f"data: {json.dumps(event)}\n\n"
                except queue.Empty:
                    pass  # No more events in queue

                # Periodically send stats update
                current_time = datetime.utcnow()
                if (current_time - last_time).total_seconds() >= 1:
                    stats = dashboard_state.get_stats()
                    # Convert Detection and Alert objects to dicts
                    serializable_stats = {
                        'total_detections': stats['total_detections'],
                        'active_cameras': stats['active_cameras'],
                        'recent_detections': [d.to_dict() for d in stats.get('recent_detections', [])],
                        'active_alerts': [a.to_dict() for a in stats.get('active_alerts', [])],
                        'camera_stats': stats['camera_stats'],
                        'camera_locations': stats['camera_locations']
                    }
                    yield f"data: {json.dumps({'type': 'stats', 'data': serializable_stats})}\n\n"
                    last_time = current_time

                time.sleep(0.1)
        except GeneratorExit:
            pass

    return Response(event_stream(), mimetype="text/event-stream")

# Health check endpoint
@app.route('/health')
async def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0'
    })

# Add static file handling
@app.route('/static/<path:path>')
async def send_static(path):
    """Serve static files."""
    return send_from_directory('static', path)

# Replace the index route with template rendering
@app.route('/')
async def index():
    """Serve the main dashboard page."""
    return render_template('dashboard.html')

# Add routes for serving all static files
@app.route('/static/js/<path:filename>')
def serve_js(filename):
    return send_from_directory('static/js', filename)

@app.route('/static/css/<path:filename>')
def serve_css(filename):
    return send_from_directory('static/css', filename)

@app.route('/static/img/<path:filename>')
def serve_img(filename):
    return send_from_directory('static/img', filename)

def run_dashboard():
    """Run the dashboard application."""
    try:
        # Simplified server configuration
        app.run(
            host="0.0.0.0",
            port=8675,
            debug=True,  # Enable debug mode for better error messages
            use_reloader=False  # Disable reloader to prevent conflicts
        )
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        raise

if __name__ == '__main__':
    run_dashboard()