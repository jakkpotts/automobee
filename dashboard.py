from typing import Dict, List, Optional
from dataclasses import dataclass
from functools import wraps
from datetime import datetime, timezone
import json
import queue
import asyncio
import time
from async_generator import async_generator, yield_
from flask import (
    Flask, jsonify, Response, request, render_template, 
    send_from_directory, current_app, Blueprint, stream_with_context
)
from flask_cors import CORS
from asgiref.wsgi import WsgiToAsgi
from contextlib import asynccontextmanager
import logging
from logger_config import logger  # Import the shared logger
from sympy import fu
from app_factory import app

dashboard_bp = Blueprint("dashboard_bp", __name__)

# Initialize Flask app with ASGI wrapper and lifespan support
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
# Make sure run_dashboard is explicitly exported
__all__ = ['Alert', 'Detection', 'dashboard_state', 'event_queue', 'run_dashboard']

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

def serialize_event(event_type: str, data: any) -> dict:
    """Serialize event data before putting in queue."""
    if hasattr(data, 'to_dict'):
        serialized_data = data.to_dict()
    elif isinstance(data, dict):
        serialized_data = data
    elif isinstance(data, (list, tuple)):
        serialized_data = [item.to_dict() if hasattr(item, 'to_dict') else item for item in data]
    else:
        serialized_data = str(data)
    
    return {
        'type': event_type,
        'data': serialized_data
    }

def format_camera_data(camera_id: str, camera_info: dict) -> dict:
    """Format camera data for frontend consumption."""
    try:
        return {
            "id": camera_id,
            "data": {  # Add this wrapper
                "id": camera_id,
                "name": camera_info.get('name', f'Camera {camera_id}'),
                "location": camera_info.get('location', {'lat': 0, 'lng': 0}),
                "active": camera_info.get('active', False),
                "stream_url": camera_info.get('stream_url', ''),
                "last_update": camera_info.get('last_update', 
                              datetime.now(timezone.utc).isoformat())
            }
        }
    except Exception as e:
        logger.error(f"Error formatting camera data: {str(e)}", exc_info=True)
        return None

class DashboardState:
    """Manages dashboard state."""
    def __init__(self):
        self.cameras = {}  # Empty initially
        self.detections: List[Detection] = []
        self.alerts: List[Alert] = []
        self._max_detections = AppConfig.MAX_DETECTIONS
        self.camera_locations = {}  # Add this line
        self.map_points = {}  # Add this line to track map points
        self.stats = {
            'total_detections': 0,
            'active_cameras': 0,
            'recent_detections': [],
            'alerts': []
        }
        self._last_update = time.time()

    def add_detection(self, detection: Detection) -> None:
        """Add a new detection and emit the event."""
        self.detections.insert(0, detection)
        if len(self.detections) > self._max_detections:
            self.detections.pop()
            
        detection_event = serialize_event('detection', detection)
        event_queue.put(detection_event)
        
        self.stats['total_detections'] = len(self.detections)
        self.stats['recent_detections'] = [d.to_dict() for d in self.detections[:10]]
        
        logger.debug(f"Added detection: {detection}")

    def add_alert(self, alert: Alert) -> None:
        """Add a new alert and emit the event."""
        self.alerts.insert(0, alert)
        
        alert_event = serialize_event('alert', alert)
        event_queue.put(alert_event)
        
        self.stats['alerts'] = [a.to_dict() for a in self.alerts[:5]]
        
        logger.debug(f"Added alert: {alert}")

    def clear_alert(self, camera: str) -> None:
        self.alerts = [a for a in self.alerts if a.camera != camera]

    def get_stats(self) -> dict:
        """Get dashboard statistics with properly serialized data."""
        current_time = datetime.now(timezone.utc)
        hour_ago = current_time.timestamp() - 3600
        
        stats = {
            'total_detections': len(self.detections),
            'active_cameras': len([c for c in self.cameras.values() if c.get('active')]),
            'recent_detections': [d.to_dict() for d in self.detections[:10]],  # Serialize detections
            'active_alerts': [a.to_dict() for a in self.alerts[:5]],  # Serialize alerts
            'camera_stats': self.cameras,
            'camera_locations': self.camera_locations,
            'map_points': list(self.map_points.values())  # Add this line
        }
        return stats

    def update_camera(self, camera_id: str, data: dict) -> None:
        """Update camera status and information."""
        if camera_id not in self.cameras:
            self.cameras[camera_id] = {'id': camera_id}
        
        self.cameras[camera_id].update(data)
        self._last_update = time.time()
        
        # Handle location data and map points
        if 'location' in data:
            location = data['location']
            if isinstance(location, dict) and 'lat' in location and 'lng' in location:
                self.camera_locations[camera_id] = location
                self.map_points[camera_id] = {
                    'id': camera_id,
                    'name': data.get('name', f'Camera {camera_id}'),
                    'latitude': float(location['lat']),
                    'longitude': float(location['lng']),
                    'type': 'camera'
                }
                # Emit map update event
                event_queue.put(serialize_event('map_update', {
                    'points': list(self.map_points.values())
                }))
        
        # Ensure we emit a camera update event
        event_queue.put(serialize_event('camera_update', {
            'camera_id': camera_id,
            'data': format_camera_data(camera_id, self.cameras[camera_id])
        }))
        logger.info(f"Updated camera {camera_id}: {data}")
        
def init_test_data():
    """Initialize test data with camera information."""
    try:
        if not dashboard_state.cameras:
            # Add a test camera
            dashboard_state.update_camera('cam1', {
                'name': 'Main Entrance',
                'location': {'lat': 36.1699, 'lng': -115.1398},
                'active': True,
                'stream_url': 'test-stream'
            })
            
        if not dashboard_state.detections:
            logger.info("Initializing test data...")
            
            test_detection = Detection(
                timestamp=datetime.now(tz=timezone.utc).isoformat(),
                camera='cam1',
                image='no-image.png',
                type='ford_f150',
                confidence=0.92,
                make='ford',
                model='f150'
            )
            
            test_alert = Alert(
                type='vehicle_detected',
                message='Ford F-150 detected at main entrance',
                timestamp=datetime.now(tz=timezone.utc).isoformat(),
                camera='cam1'
            )
            
            # Verify serialization
            success, issues = debug_serialization('detection', test_detection)
            if not success:
                logger.error(f"Detection serialization failed: {issues}")
                raise ValueError(f"Invalid detection object: {issues}")
            
            success, issues = debug_serialization('alert', test_alert)
            if not success:
                logger.error(f"Alert serialization failed: {issues}")
                raise ValueError(f"Invalid alert object: {issues}")
            
            dashboard_state.add_detection(test_detection)
            dashboard_state.add_alert(test_alert)
            
            logger.info("Test data initialized successfully")
            
    except Exception as e:
        logger.error(f"Error initializing test data: {e}")
        pass

def debug_serialization(obj_type: str, obj):
    """Debug serialization issues by checking each field"""
    try:
        # Test full serialization first
        serialized = serialize_event(obj_type, obj)
        json.dumps(serialized)
        return True, None
    except Exception as e:
        # Field-by-field analysis
        problems = {}
        for field, value in obj.__dict__.items():
            try:
                json.dumps({field: value})
            except Exception as e:
                problems[field] = {
                    'value': str(value),
                    'type': str(type(value)),
                    'error': str(e)
                }
        return False, problems

# Initialize global state and event queue
dashboard_state = DashboardState()
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
        current_time = datetime.now(timezone.utc)
        hour_ago = current_time.timestamp() - 3600
        recent_detections = [d for d in dashboard_state.detections 
                           if datetime.fromisoformat(d.timestamp).timestamp() > hour_ago]
        detection_rate = len(recent_detections) / 3600  # detections per second
        
        response_data = {
            'total_detections': len(dashboard_state.detections),
            'active_cameras': len([c for c in dashboard_state.cameras.values() if c.get('active')]),
            'detection_rate': detection_rate,
            'camera_stats': {
                camera_id: {
                    'name': info.get('location', f'Camera {camera_id}'),
                    'active': info.get('active', False),
                    'stream_url': info.get('stream_url', '')
                }
                for camera_id, info in dashboard_state.cameras.items()
            }
        }

        logger.debug(f"Returning stats: {response_data}")
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error getting stats: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/detections', methods=['POST'])
@handle_errors
def add_detection():
    """Add new detection."""
    data = request.json
    if not validate_detection(data):
        return jsonify({'error': 'Invalid detection data'}), 400

    detection = Detection(**data)
    dashboard_state.add_detection(detection)
    
    # Ensure detection is serialized
    event_queue.put({
        'type': 'detection',
        'data': detection.to_dict()
    })
    
    return jsonify({'status': 'success'})

@app.route('/api/cameras/<camera_id>', methods=['PUT'])
@handle_errors
def update_camera(camera_id: str):
    """Update camera status."""
    try:
        data = request.json
        if not validate_camera_update(data):
            return jsonify({'error': 'Invalid camera data'}), 400

        dashboard_state.update_camera(camera_id, data)
        
        # Emit camera update event
        event_queue.put({
            'type': 'camera_update',
            'data': {
                'camera_id': camera_id,
                'status': data
            }
        })
        
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Camera update error: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.after_request
def add_header(response):
    """Add headers to optimize performance."""
    response.headers['Cache-Control'] = 'public, max-age=300'  # 5 minutes cache
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response

def event_stream():
    try:
        logger.info("Starting event stream...")
        
        def generate():
            # Send initial state immediately
            initial_state = {
                'type': 'initial_state',
                'data': {
                    'cameras': {
                        cid: format_camera_data(cid, camera)
                        for cid, camera in dashboard_state.cameras.items()
                    }
                }
            }
            yield f"data: {json.dumps(initial_state)}\n\n"
            
            # Send initial map points
            map_points = []
            for camera_id, camera in dashboard_state.cameras.items():
                if 'location' in camera:
                    try:
                        point = {
                            'id': camera_id,
                            'name': camera.get('name', f'Camera {camera_id}'),
                            'latitude': float(camera['location']['lat']),
                            'longitude': float(camera['location']['lng']),
                            'type': 'camera',
                            'stream_url': camera.get('stream_url', '')
                        }
                        map_points.append(point)
                    except Exception as e:
                        logger.error(f"Error processing camera {camera_id}: {e}")
                        continue
            
            if map_points:
                map_update = serialize_event('map_update', {'points': map_points})
                yield f"data: {json.dumps(map_update)}\n\n"
            
            # Continue with regular event processing
            while True:
                try:
                    # Process queued events
                    while not event_queue.empty():
                        event = event_queue.get_nowait()
                        if event:
                            yield f"data: {json.dumps(event)}\n\n"
                    
                    # Send keepalive every second
                    yield ': keepalive\n\n'
                    time.sleep(1)
                    
                except queue.Empty:
                    yield ': keepalive\n\n'
                    time.sleep(0.1)
                
        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no',
                'Content-Type': 'text/event-stream'
            }
        )
        
    except Exception as e:
        logger.error(f"Error in event stream: {e}", exc_info=True)
        return Response("Error", status=500)

def _generate_events():
    """Generator function for SSE events"""
    try:
        # Send initial state
        initial_state = {
            'type': 'initial_state',
            'data': {
                'cameras': {
                    cid: format_camera_data(cid, cam) 
                    for cid, cam in dashboard_state.cameras.items()
                }
            }
        }
        yield f"data: {json.dumps(initial_state)}\n\n"
        
        while True:
            try:
                # Process queue events
                while not event_queue.empty():
                    event = event_queue.get_nowait()
                    if event:
                        yield f"data: {json.dumps(event)}\n\n"
                        
                # Keep connection alive
                yield ': keepalive\n\n'
                
            except queue.Empty:
                pass
                
            time.sleep(0.1)  # Prevent tight loop
            
    except GeneratorExit:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Error generating events: {e}")
        
# Update the route to use the new event stream
@app.route('/stream')
def stream():
    """Event stream endpoint."""
    try:
        return Response(
            stream_with_context(_generate_events()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive'
            }
        )
    except Exception as e:
        logger.error(f"Stream error: {e}")
        return Response("Error", status=500)

# Health check endpoint
@app.route('/health')
async def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now(timezone.utc).isoformat(),
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
    return render_template('dashboard.html', initalize_map=True)

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

def run_dashboard(shared_state=None):
    """Run the dashboard application.
    
    Args:
        shared_state: Optional shared state dictionary for multiprocessing
    """
    try:
        # Update dashboard state from shared state if provided
        if shared_state is not None:
            logger.info(f"Initializing dashboard with {len(shared_state['cameras'])} cameras")
            dashboard_state.cameras.update(shared_state.get('cameras', {}))
            dashboard_state.detections.extend(shared_state.get('detections', []))
            
            # Initialize map points from cameras
            for camera_id, camera in shared_state['cameras'].items():
                if 'location' in camera:
                    dashboard_state.map_points[camera_id] = {
                        'id': camera_id,
                        'name': camera.get('name', f'Camera {camera_id}'),
                        'latitude': float(camera['location']['lat']),
                        'longitude': float(camera['location']['lng']),
                        'type': 'camera',
                        'stream_url': camera.get('stream_url', '')
                    }
            
        # Simplified server configuration
        app.run(
            host="0.0.0.0",
            port=8675,
            debug=True,
            use_reloader=False
        )
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        raise

if __name__ == '__main__':
    run_dashboard()