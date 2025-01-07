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
from flask import Flask, jsonify, Response, request, render_template, send_from_directory, current_app
from flask_cors import CORS
from asgiref.wsgi import WsgiToAsgi
from contextlib import asynccontextmanager

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@dataclass
class Detection:
    timestamp: str
    camera: str
    image: str
    type: str
    confidence: float
    make: Optional[str] = None
    model: Optional[str] = None

@dataclass
class Alert:
    type: str
    message: str
    timestamp: str
    camera: Optional[str] = None

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
        self.cameras: Dict[str, dict] = {}
        self.detections: List[Detection] = []
        self.alerts: List[Alert] = []
        self._max_detections = AppConfig.MAX_DETECTIONS
        self.camera_locations: Dict[str, dict] = {}  # Add camera locations storage

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

# Initialize Flask app with ASGI wrapper and lifespan support
app = Flask(__name__)
CORS(app)
asgi_app = WsgiToAsgi(app)

# Add lifespan protocol support
async def app_with_lifespan(scope, receive, send):
    if scope["type"] == "lifespan":
        async with lifespan(app):
            await receive()
            await send({"type": "lifespan.startup.complete"})
            while True:
                message = await receive()
                if message["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    break
    else:
        await asgi_app(scope, receive, send)

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
@handle_errors
async def get_stats():
    """Get dashboard statistics."""
    logger.info("Fetching dashboard stats")
    return jsonify(dashboard_state.get_stats())  # camera_locations already included in get_stats()

@app.route('/api/detections', methods=['POST'])
@handle_errors
async def add_detection():
    """Add new detection."""
    data = request.json
    if not validate_detection(data):
        return jsonify({'error': 'Invalid detection data'}), 400

    detection = Detection(**data)
    dashboard_state.add_detection(detection)
    return jsonify({'status': 'success'})

@app.route('/api/cameras/<camera_id>', methods=['PUT'])
@handle_errors
async def update_camera(camera_id: str):
    """Update camera status."""
    data = request.json
    if not validate_camera_update(data):
        return jsonify({'error': 'Invalid camera data'}), 400

    dashboard_state.update_camera(camera_id, data)
    return jsonify({'status': 'success'})

@app.route('/stream')
def stream():
    """SSE stream for real-time updates with improved error handling."""
    def generate():
        try:
            while True:
                if dashboard_state.detections:
                    try:
                        data = {
                            'type': 'detection',
                            'data': dashboard_state.detections[0]
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                    except Exception as e:
                        logger.error(f"Error sending SSE data: {e}")
                # Use a small sleep to prevent CPU overload
                time.sleep(1)
        except GeneratorExit:
            logger.info("SSE connection closed by client")
        except Exception as e:
            logger.error(f"SSE connection error: {e}")

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )

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

# Add root route to serve dashboard
@app.route('/')
async def index():
    """Serve the main dashboard page."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AutomoBee Dashboard</title>
        <!-- Use local resources -->
  <script src="https://cdn.tailwindcss.com"></script>
        <!-- Replace OpenLayers with Leaflet -->
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <script src="/static/js/hls.min.js"></script>
        <style>
            .notification-show { opacity: 1; }
            .detection-card { @apply bg-slate-800 rounded-lg overflow-hidden mb-4; }
            .map-container { @apply w-full h-[400px] rounded-lg overflow-hidden relative; }
            .map-overlay { @apply absolute bottom-0 left-0 right-0 bg-slate-800 bg-opacity-75 p-2 text-xs text-white z-[1000]; }
            /* Override Leaflet styles to match theme */
            .leaflet-container { 
                @apply bg-slate-800;
                height: 400px;
                border-radius: 0.5rem;
            }
            .leaflet-popup-content-wrapper { 
                @apply bg-slate-700 text-white border-0;
            }
            .leaflet-popup-tip { 
                @apply bg-slate-700 border-0;
            }
            .camera-popup {
                @apply p-2 text-sm;
            }
            .camera-popup h3 {
                @apply font-bold mb-2;
            }
            .camera-popup .status {
                @apply flex items-center gap-2 mb-2;
            }
            .camera-popup .status-dot {
                @apply w-2 h-2 rounded-full;
            }
            .camera-popup .status-dot.active {
                @apply bg-green-500;
            }
            .camera-popup .status-dot.error {
                @apply bg-red-500;
            }
        </style>
    </head>
    <body class="bg-slate-900 text-white">
        <div id="notification" class="fixed top-4 right-4 bg-green-500 text-white px-4 py-2 rounded-lg opacity-0 transition-opacity duration-300"></div>
        
        <!-- Main Content -->
        <div class="container mx-auto px-4 py-8">
            <h1 class="text-3xl font-bold mb-8">AutomoBee Dashboard</h1>
            
            <!-- Stats Bar -->
            <div class="grid grid-cols-3 gap-4 mb-8">
                <div class="bg-slate-800 p-4 rounded-lg">
                    <h3 class="text-sm text-slate-400">Total Detections</h3>
                    <p id="total-detections" class="text-2xl font-bold">0</p>
                </div>
                <div class="bg-slate-800 p-4 rounded-lg">
                    <h3 class="text-sm text-slate-400">Active Cameras</h3>
                    <p id="active-cameras" class="text-2xl font-bold">0</p>
                </div>
                <div class="bg-slate-800 p-4 rounded-lg">
                    <h3 class="text-sm text-slate-400">Detection Rate</h3>
                    <p id="detection-rate" class="text-2xl font-bold">0/hr</p>
                </div>
            </div>
            
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <!-- Map Section -->
                <div class="bg-slate-800 p-4 rounded-lg">
                    <h2 class="text-xl font-semibold mb-4">Camera Locations</h2>
                    <div id="map" class="w-full h-[400px] rounded-lg"></div>
                    <div id="map-loading"></div>
                </div>
                
                <!-- Detections Section -->
                <div class="bg-slate-800 p-4 rounded-lg">
                    <div class="flex justify-between items-center mb-4">
                        <h2 class="text-xl font-semibold">Recent Detections</h2>
                        <div class="flex items-center space-x-2">
                            <select id="detection-filter" class="bg-slate-700 text-white rounded px-2 py-1">
                                <option value="all">All Types</option>
                                <option value="ford_f150">Ford F-150</option>
                                <option value="other">Other</option>
                            </select>
                            <span id="detection-count" class="bg-blue-500 px-3 py-1 rounded-full text-sm">0</span>
                        </div>
                    </div>
                    <div id="recent-detections" class="space-y-4 max-h-[400px] overflow-y-auto"></div>
                </div>
                
                <!-- Alerts Section -->
                <div class="bg-slate-800 p-4 rounded-lg">
                    <h2 class="text-xl font-semibold mb-4">Active Alerts</h2>
                    <div id="alerts-container"></div>
                </div>
                
                <!-- Camera Status Section -->
                <div class="bg-slate-800 p-4 rounded-lg">
                    <div class="flex justify-between items-center mb-4">
                        <h2 class="text-xl font-semibold">Camera Status</h2>
                        <input type="text" id="camera-search" placeholder="Search cameras..." 
                               class="bg-slate-700 text-white rounded px-2 py-1">
                    </div>
                    <div id="camera-status-list" class="space-y-4"></div>
                </div>
            </div>
        </div>

        <!-- Detection Modal -->
        <div id="detection-modal" class="fixed inset-0 bg-black bg-opacity-50 hidden items-center justify-center">
            <div class="bg-white rounded-lg max-w-2xl w-full mx-4">
                <div class="p-4">
                    <h3 id="detection-modal-title" class="text-xl font-bold text-gray-900"></h3>
                    <img id="detection-modal-image" class="w-full h-auto mt-4 rounded" alt="Detection">
                    <div id="detection-modal-details" class="mt-4 space-y-4"></div>
                    <div class="mt-4 flex justify-end space-x-2">
                        <button id="download-image" class="bg-blue-500 text-white px-4 py-2 rounded">Download</button>
                        <button onclick="document.getElementById('detection-modal').classList.add('hidden')" 
                                class="bg-gray-500 text-white px-4 py-2 rounded">Close</button>
                    </div>
                </div>
            </div>
        </div>

        <!-- Stream Modal -->
        <div id="stream-modal" class="fixed inset-0 bg-black bg-opacity-50 hidden items-center justify-center">
            <div class="bg-white rounded-lg max-w-4xl w-full mx-4">
                <div class="p-4">
                    <h3 id="stream-title" class="text-xl font-bold text-gray-900"></h3>
                    <video id="stream-player" class="w-full mt-4" controls></video>
                    <div class="mt-4 flex justify-end">
                        <button onclick="closeStreamModal()" class="bg-gray-500 text-white px-4 py-2 rounded">Close</button>
                    </div>
                </div>
            </div>
        </div>

        <script src="/static/js/dashboard.js"></script>
    </body>
    </html>
    """

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

# Update run_dashboard to use the new ASGI app
def run_dashboard():
    """Run the dashboard application."""
    import uvicorn
    
    config = uvicorn.Config(
        app=app_with_lifespan,
        host="0.0.0.0",
        port=8675,
        log_level="debug",
        reload=AppConfig.DEBUG,
        loop="asyncio"
    )
    
    server = uvicorn.Server(config)
    try:
        server.run()
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        raise

# Update ASGI handling for async routes
def make_async_handler(handler):
    """Convert async route handlers to ASGI-compatible handlers."""
    @wraps(handler)
    def wrapped(*args, **kwargs):
        return asyncio.run(handler(*args, **kwargs))
    return wrapped

# Decorate async routes with sync wrapper
app.view_functions['get_stats'] = make_async_handler(app.view_functions['get_stats'])
app.view_functions['add_detection'] = make_async_handler(app.view_functions['add_detection'])
app.view_functions['update_camera'] = make_async_handler(app.view_functions['update_camera'])
app.view_functions['health_check'] = make_async_handler(app.view_functions['health_check'])
app.view_functions['index'] = make_async_handler(app.view_functions['index'])

if __name__ == '__main__':
    run_dashboard()

# Make sure run_dashboard is explicitly exported
__all__ = ['Alert', 'Detection', 'dashboard_state', 'event_queue', 'run_dashboard']