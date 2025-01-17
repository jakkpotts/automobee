import asyncio
from typing import Dict, Set, Optional, Any
import logging
import json
import zlib
from datetime import datetime
import aiohttp
from aiohttp import web
import jwt
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class WebSocketMessage:
    """Container for WebSocket messages"""
    type: str
    data: Dict
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_json(self) -> str:
        """Convert message to JSON string"""
        return json.dumps(asdict(self))

class DashboardWebSocket:
    """WebSocket server for real-time dashboard updates"""
    
    def __init__(
        self,
        state_manager: 'StateManager',
        host: str = '0.0.0.0',
        port: int = 8765,
        jwt_secret: Optional[str] = None,
        compression_level: int = 6
    ):
        """Initialize WebSocket server
        
        Args:
            state_manager: System state manager
            host: Server host
            port: Server port
            jwt_secret: Secret for JWT authentication
            compression_level: Compression level (0-9)
        """
        self.state_manager = state_manager
        self.host = host
        self.port = port
        self.jwt_secret = jwt_secret
        self.compression_level = compression_level
        
        # Client management
        self.clients: Set[web.WebSocketResponse] = set()
        self._client_info: Dict[web.WebSocketResponse, Dict] = {}
        
        # Server management
        self.app = web.Application()
        self.app.router.add_get('/ws', self.websocket_handler)
        self.runner: Optional[web.AppRunner] = None
        self._server_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Message compression
        self._compressor = zlib.compressobj(level=compression_level)
    
    async def start(self) -> None:
        """Start WebSocket server"""
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        
        site = web.TCPSite(self.runner, self.host, self.port)
        await site.start()
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_disconnected())
        
        logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")
    
    async def websocket_handler(self, request: web.Request) -> web.WebSocketResponse:
        """Handle WebSocket connections
        
        Args:
            request: HTTP request
        
        Returns:
            WebSocket response
        """
        # Create WebSocket response
        ws = web.WebSocketResponse(
            heartbeat=30,
            compress=self.compression_level > 0
        )
        await ws.prepare(request)
        
        # Authenticate client
        if not await self._authenticate_client(ws, request):
            await ws.close(code=4001, message=b'Authentication failed')
            return ws
        
        # Add client to active set
        self.clients.add(ws)
        client_info = {
            'connected_at': datetime.now(),
            'remote': request.remote,
            'user_agent': request.headers.get('User-Agent'),
            'last_ping': datetime.now()
        }
        self._client_info[ws] = client_info
        
        try:
            # Send initial state
            await self._send_initial_state(ws)
            
            # Handle messages
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._handle_message(ws, msg.data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f'WebSocket error: {ws.exception()}')
                    break
                
        finally:
            # Cleanup on disconnect
            self.clients.discard(ws)
            self._client_info.pop(ws, None)
        
        return ws
    
    async def _authenticate_client(self, ws: web.WebSocketResponse, 
                                 request: web.Request) -> bool:
        """Authenticate WebSocket client
        
        Args:
            ws: WebSocket response
            request: HTTP request
        
        Returns:
            True if authenticated, False otherwise
        """
        if not self.jwt_secret:
            return True  # No authentication required
        
        try:
            token = request.headers.get('Authorization', '').split(' ')[1]
            jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            return True
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return False
    
    async def _send_initial_state(self, ws: web.WebSocketResponse) -> None:
        """Send initial state to client
        
        Args:
            ws: WebSocket response
        """
        # Send active streams
        stream_states = self.state_manager.get_all_stream_states()
        await self._send_message(ws, WebSocketMessage(
            type='stream_states',
            data=stream_states
        ))
        
        # Send current health metrics
        health_metrics = self.state_manager.health_metrics
        await self._send_message(ws, WebSocketMessage(
            type='health_metrics',
            data=health_metrics
        ))
    
    async def _handle_message(self, ws: web.WebSocketResponse, data: str) -> None:
        """Handle incoming WebSocket message
        
        Args:
            ws: WebSocket response
            data: Message data
        """
        try:
            message = json.loads(data)
            message_type = message.get('type')
            
            if message_type == 'ping':
                # Update last ping time
                self._client_info[ws]['last_ping'] = datetime.now()
                await self._send_message(ws, WebSocketMessage(
                    type='pong',
                    data={}
                ))
                
            elif message_type == 'subscribe':
                # Handle subscription requests
                topics = message.get('data', {}).get('topics', [])
                self._client_info[ws]['subscribed_topics'] = set(topics)
                
            else:
                logger.warning(f"Unknown message type: {message_type}")
                
        except json.JSONDecodeError:
            logger.error("Invalid JSON message")
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def _send_message(self, ws: web.WebSocketResponse, 
                          message: WebSocketMessage) -> None:
        """Send message to WebSocket client
        
        Args:
            ws: WebSocket response
            message: Message to send
        """
        try:
            # Convert message to JSON
            data = message.to_json()
            
            # Compress if enabled
            if self.compression_level > 0:
                data = self._compressor.compress(data.encode()) + \
                      self._compressor.flush(zlib.Z_SYNC_FLUSH)
            
            # Send message
            await ws.send_bytes(data) if isinstance(data, bytes) else \
                  await ws.send_str(data)
                  
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            # Remove client if send fails
            self.clients.discard(ws)
            self._client_info.pop(ws, None)
    
    async def broadcast_health_update(self, metrics: Dict[str, Any]) -> None:
        """Broadcast health metrics update
        
        Args:
            metrics: Health metrics
        """
        message = WebSocketMessage(
            type='health_update',
            data=metrics
        )
        await self.broadcast_message(message)
    
    async def broadcast_detection(self, camera_id: str, detection: Dict) -> None:
        """Broadcast vehicle detection
        
        Args:
            camera_id: ID of the camera
            detection: Detection data
        """
        message = WebSocketMessage(
            type='detection',
            data={
                'camera_id': camera_id,
                'detection': detection
            }
        )
        await self.broadcast_message(message)
    
    async def broadcast_stream_update(self, camera_id: str, state: Dict) -> None:
        """Broadcast stream state update
        
        Args:
            camera_id: ID of the camera
            state: Stream state
        """
        message = WebSocketMessage(
            type='stream_update',
            data={
                'camera_id': camera_id,
                'state': state
            }
        )
        await self.broadcast_message(message)
    
    async def broadcast_message(self, message: WebSocketMessage) -> None:
        """Broadcast message to all connected clients
        
        Args:
            message: Message to broadcast
        """
        if not self.clients:
            return
        
        # Broadcast to all clients
        tasks = []
        for ws in self.clients:
            # Check if client is subscribed to this message type
            subscribed_topics = self._client_info[ws].get('subscribed_topics')
            if subscribed_topics is None or message.type in subscribed_topics:
                tasks.append(self._send_message(ws, message))
        
        if tasks:
            # Wait for all sends to complete
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _cleanup_disconnected(self) -> None:
        """Periodically cleanup disconnected clients"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Find disconnected clients
                now = datetime.now()
                to_remove = set()
                
                for ws, info in self._client_info.items():
                    if (now - info['last_ping']).total_seconds() > 90:  # No ping for 90s
                        to_remove.add(ws)
                
                # Remove disconnected clients
                for ws in to_remove:
                    self.clients.discard(ws)
                    self._client_info.pop(ws, None)
                    try:
                        await ws.close()
                    except Exception:
                        pass
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(5)
    
    async def shutdown(self) -> None:
        """Shutdown WebSocket server"""
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        close_tasks = []
        for ws in self.clients:
            close_tasks.append(ws.close())
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
        
        # Cleanup client tracking
        self.clients.clear()
        self._client_info.clear()
        
        # Shutdown server
        if self.runner:
            await self.runner.cleanup()
            
        logger.info("WebSocket server shut down") 