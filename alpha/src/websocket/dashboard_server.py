import asyncio
import json
import logging
import websockets
from typing import Set, Dict, List
from dataclasses import dataclass, asdict
from ..zones.zone_manager import ZoneType
from ..visualization.zone_dashboard import ZoneDashboard, ZoneMetrics
from ..camera.stream_manager import StreamStatus
from ..auth.token_manager import TokenManager
from ..processing.types import ZoneFocusConfig, FocusConfig  # Import from types instead
from ..auth.security_middleware import SecurityMiddleware
from ..utils.error_manager import ErrorManager, ErrorCategory, ErrorSeverity
import zlib
from ..config.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class DashboardWebSocket:
    """WebSocket server for real-time dashboard updates."""
    
    def __init__(self, dashboard: ZoneDashboard, host: str = 'localhost', port: int = 8765):
        self.dashboard = dashboard
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.running = False
        config_manager = ConfigManager()
        self.token_manager = TokenManager(config_manager)
        self.security_middleware = SecurityMiddleware(self.token_manager)
        self.error_manager = ErrorManager()
        
    async def start(self):
        """Start the WebSocket server."""
        self.running = True
        
        # Connect to scheduler
        if hasattr(self.dashboard.zone_manager.stream_manager, 'scheduler'):
            scheduler = self.dashboard.zone_manager.stream_manager.scheduler
            scheduler.websocket_server = self
        
        server = await websockets.serve(self._handle_client, self.host, self.port)
        logger.info(f"Dashboard WebSocket server started on ws://{self.host}:{self.port}")
        
        # Start update broadcast loop
        asyncio.create_task(self._broadcast_updates())
        
        return server
        
    async def stop(self):
        """Stop the WebSocket server."""
        self.running = False
        for client in self.clients:
            await client.close()
            
    async def _authenticate(self, websocket: websockets.WebSocketServerProtocol) -> bool:
        """Authenticate WebSocket connection with enhanced security."""
        try:
            auth_header = websocket.request_headers.get('Authorization')
            if not auth_header or not auth_header.startswith('Bearer '):
                await self.error_manager.handle_error(
                    ValueError("Missing or invalid authorization"),
                    ErrorCategory.AUTHENTICATION,
                    ErrorSeverity.MEDIUM
                )
                await websocket.close(1008, "Missing or invalid authorization")
                return False
                
            token = auth_header.split(' ')[1]
            required_scopes = {'read'}  # Basic scope requirement
            
            if not await self.security_middleware.validate_request(token, required_scopes):
                await websocket.close(1008, "Request validation failed")
                return False
                
            return True
            
        except Exception as e:
            await self.error_manager.handle_error(
                e, ErrorCategory.AUTHENTICATION, ErrorSeverity.HIGH
            )
            return False
            
    async def _handle_client(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """Handle new WebSocket client connection."""
        try:
            if not await self._authenticate(websocket):
                return
            
            self.clients.add(websocket)
            logger.info(f"New client connected. User: {websocket.user_id}")
            
            # Send initial state
            await self._send_dashboard_state(websocket)
            
            # Keep connection alive and handle messages
            async for message in websocket:
                await self._handle_message(websocket, message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client connection closed. User: {getattr(websocket, 'user_id', 'unknown')}")
        finally:
            self.clients.remove(websocket)
            
    async def _broadcast_updates(self):
        """Broadcast updates to all connected clients."""
        while self.running:
            if self.clients:
                state = self._get_dashboard_state()
                await self._broadcast(json.dumps(state))
            await asyncio.sleep(1)  # Update frequency
            
    async def _broadcast(self, message: str):
        """Broadcast message with error handling and rate limiting."""
        if not self.clients:
            return
            
        try:
            # Decompress if needed
            if message.startswith('compressed:'):
                compressed_data = bytes.fromhex(message.split(':', 1)[1])
                message = zlib.decompress(compressed_data).decode()
                
            disconnected = set()
            
            for client in self.clients:
                try:
                    if client.open:
                        await asyncio.wait_for(client.send(message), timeout=2.0)
                    else:
                        disconnected.add(client)
                except Exception as e:
                    await self.error_manager.handle_error(
                        e, ErrorCategory.NETWORK, ErrorSeverity.MEDIUM,
                        {'client_id': getattr(client, 'user_id', 'unknown')}
                    )
                    disconnected.add(client)
                    
            # Remove disconnected clients
            self.clients -= disconnected
            
        except Exception as e:
            await self.error_manager.handle_error(
                e, ErrorCategory.SYSTEM, ErrorSeverity.HIGH
            )
        
    def _get_dashboard_state(self) -> dict:
        """Get current dashboard state with memory-efficient detection history."""
        try:
            scheduler = self.dashboard.zone_manager.stream_manager.scheduler
            state = {
                'zones': {},
                'alerts': self._get_active_alerts(),
                'focus_modes': {},
                'recent_detections': {}
            }
            
            # Get metrics for each zone
            for zone_def in self.dashboard.zone_manager.ZONE_DEFINITIONS:
                metrics = self.dashboard.get_zone_metrics(zone_def.name)
                state['zones'][zone_def.name.value] = asdict(metrics)
                
                if scheduler:
                    if zone_def.name in scheduler.focus_config:
                        state['focus_modes'][zone_def.name.value] = asdict(
                            scheduler.focus_config[zone_def.name]
                        )
                    # Add limited recent detections for the zone
                    if hasattr(scheduler, 'detection_logger'):
                        zone_detections = [
                            asdict(d) for d in scheduler.detection_logger.detections.get(
                                zone_def.name, []
                            )[-100:]  # Last 100 detections
                        ]
                        state['recent_detections'][zone_def.name.value] = zone_detections
            
            return state
            
        except Exception as e:
            logger.error(f"Error getting dashboard state: {str(e)}")
            return {'error': str(e)}
        
    def _get_active_alerts(self) -> List[dict]:
        """Get list of active alerts."""
        alerts = []
        scheduler = self.dashboard.zone_manager.stream_manager.scheduler
        
        if scheduler:
            for stream_id, task in scheduler.active_tasks.items():
                if task.error_count > 0:
                    alerts.append({
                        'stream_id': stream_id,
                        'type': 'error',
                        'message': f"Stream processing error (attempt {task.error_count}/{task.max_retries})",
                        'timestamp': task.last_processed.isoformat()
                    })
                    
        return alerts 
        
    async def _handle_message(self, websocket: websockets.WebSocketServerProtocol, message: str):
        """Handle incoming WebSocket messages."""
        try:
            data = json.loads(message)
            command = data.get('command')
            
            if command == 'set_zone_focus':
                if 'admin' not in websocket.scopes:
                    await websocket.send(json.dumps({
                        'error': 'Unauthorized - Admin access required'
                    }))
                    return
                    
                zone_name = data.get('zone')
                enabled = data.get('enabled', False)
                config_data = data.get('config', {})
                
                try:
                    zone_type = ZoneType(zone_name)
                except ValueError:
                    await websocket.send(json.dumps({
                        'error': f'Invalid zone type: {zone_name}'
                    }))
                    return

                # Create focus configuration
                focus_config = ZoneFocusConfig(
                    is_active=enabled,
                    reserve_slots=config_data.get('reserve_slots', 10),
                    focus_timeout=config_data.get('focus_timeout', 3600),
                    priority_config=FocusConfig(
                        priority_method=config_data.get('priority_method', 'vehicle_density'),
                        min_vehicle_threshold=config_data.get('min_vehicle_threshold', 5),
                        confidence_threshold=config_data.get('confidence_threshold', 0.7),
                        update_interval=config_data.get('update_interval', 60)
                    )
                )
                
                scheduler = self.dashboard.zone_manager.stream_manager.scheduler
                await scheduler.set_zone_focus(zone_type, enabled, focus_config)
                
                # Broadcast focus mode change
                await self._broadcast(json.dumps({
                    'event': 'zone_focus_changed',
                    'zone': zone_name,
                    'enabled': enabled,
                    'config': asdict(focus_config)
                }))
                
        except Exception as e:
            logger.error(f"Error handling message: {str(e)}")
            await websocket.send(json.dumps({'error': str(e)})) 