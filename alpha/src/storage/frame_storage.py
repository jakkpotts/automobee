import asyncio
import logging
from typing import Optional, Dict, Any
import numpy as np
from datetime import datetime, timedelta
import cv2
from ..utils.error_manager import ErrorManager, ErrorCategory, ErrorSeverity

logger = logging.getLogger(__name__)

class FrameStorage:
    """Manages temporary storage of frames for deferred processing."""
    
    def __init__(self, max_age_seconds: int = 300):
        self.frames: Dict[str, Dict[str, Any]] = {}
        self.max_age = timedelta(seconds=max_age_seconds)
        self.error_manager = ErrorManager()
        self._cleanup_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start the frame storage service."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
    async def stop(self):
        """Stop the frame storage service."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        # Clear frames
        self.frames.clear()
                
    async def store_frame(self, frame_id: str, frame: np.ndarray, metadata: Dict[str, Any] = None):
        """Store a frame for deferred processing."""
        try:
            self.frames[frame_id] = {
                'frame': frame,
                'timestamp': datetime.now(),
                'metadata': metadata or {}
            }
        except Exception as e:
            await self.error_manager.handle_error(
                e, ErrorCategory.STORAGE, ErrorSeverity.MEDIUM
            )
            
    async def get_frame(self, frame_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a stored frame."""
        return self.frames.get(frame_id)
        
    async def _cleanup_loop(self):
        """Periodically clean up old frames."""
        while True:
            try:
                current_time = datetime.now()
                expired_frames = [
                    frame_id for frame_id, data in self.frames.items()
                    if current_time - data['timestamp'] > self.max_age
                ]
                
                for frame_id in expired_frames:
                    del self.frames[frame_id]
                    
                await asyncio.sleep(60)  # Cleanup every minute
                
            except Exception as e:
                await self.error_manager.handle_error(
                    e, ErrorCategory.SYSTEM, ErrorSeverity.LOW
                )
                await asyncio.sleep(5)  # Back off on error 