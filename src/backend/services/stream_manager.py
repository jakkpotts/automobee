import asyncio
from typing import Dict, Optional, Set
import logging
import cv2
import numpy as np
from datetime import datetime
import aiohttp
import tempfile
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class AsyncStreamManager:
    """Manages multiple video streams asynchronously"""
    
    def __init__(
        self,
        state_manager: 'StateManager',
        detector: 'AsyncVehicleDetector',
        max_concurrent_streams: int = 8,
        frame_buffer_size: int = 30,
        processing_interval: float = 0.1  # seconds between frame processing
    ):
        """Initialize stream manager
        
        Args:
            state_manager: System state manager
            detector: Vehicle detector service
            max_concurrent_streams: Maximum number of concurrent streams
            frame_buffer_size: Size of frame buffer per stream
            processing_interval: Interval between frame processing
        """
        self.state_manager = state_manager
        self.detector = detector
        self.max_concurrent_streams = max_concurrent_streams
        self.frame_buffer_size = frame_buffer_size
        self.processing_interval = processing_interval
        
        # Stream management
        self.active_streams: Dict[str, Dict] = {}
        self.stream_tasks: Dict[str, asyncio.Task] = {}
        self._stream_locks: Dict[str, asyncio.Lock] = {}
        
        # HLS stream management
        self.hls_output_dir = Path("static/streams")
        self.hls_segment_time = 2  # seconds
        self.hls_playlist_size = 5  # number of segments in playlist
    
    async def initialize(self) -> None:
        """Initialize the stream manager"""
        # Ensure HLS output directory exists
        os.makedirs(self.hls_output_dir, exist_ok=True)
        logger.info("Stream manager initialized")
    
    async def add_stream(self, camera_id: str, url: str, fps: int = 30) -> None:
        """Add a new stream
        
        Args:
            camera_id: ID of the camera
            url: Stream URL (RTSP/HTTP)
            fps: Frames per second to process
        """
        if len(self.active_streams) >= self.max_concurrent_streams:
            raise RuntimeError(f"Maximum number of streams ({self.max_concurrent_streams}) reached")
        
        if camera_id in self.active_streams:
            logger.warning(f"Stream {camera_id} already exists, stopping existing stream")
            await self.remove_stream(camera_id)
        
        # Initialize stream state
        self.active_streams[camera_id] = {
            'url': url,
            'fps': fps,
            'frame_count': 0,
            'last_frame_time': None,
            'status': 'initializing',
            'error': None
        }
        
        self._stream_locks[camera_id] = asyncio.Lock()
        
        # Start stream processing
        self.stream_tasks[camera_id] = asyncio.create_task(
            self._process_stream(camera_id)
        )
        
        logger.info(f"Added stream: {camera_id}")
    
    async def remove_stream(self, camera_id: str) -> None:
        """Remove a stream
        
        Args:
            camera_id: ID of the camera to remove
        """
        if camera_id in self.stream_tasks:
            # Cancel stream task
            self.stream_tasks[camera_id].cancel()
            try:
                await self.stream_tasks[camera_id]
            except asyncio.CancelledError:
                pass
            
            # Cleanup stream resources
            del self.stream_tasks[camera_id]
            del self.active_streams[camera_id]
            del self._stream_locks[camera_id]
            
            # Cleanup HLS files
            await self._cleanup_hls_files(camera_id)
            
            logger.info(f"Removed stream: {camera_id}")
    
    async def _process_stream(self, camera_id: str) -> None:
        """Process frames from a stream
        
        Args:
            camera_id: ID of the camera to process
        """
        stream_info = self.active_streams[camera_id]
        cap = None
        frame_interval = 1.0 / stream_info['fps']
        last_frame_time = 0
        
        try:
            # Open video capture
            cap = cv2.VideoCapture(stream_info['url'])
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open stream: {stream_info['url']}")
            
            # Update stream status
            stream_info['status'] = 'running'
            self.state_manager.update_stream_state(camera_id, stream_info)
            
            while True:
                current_time = datetime.now().timestamp()
                if current_time - last_frame_time < frame_interval:
                    await asyncio.sleep(0.001)  # Small sleep to prevent CPU spinning
                    continue
                
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError("Failed to read frame")
                
                # Update stream info
                stream_info['frame_count'] += 1
                stream_info['last_frame_time'] = current_time
                last_frame_time = current_time
                
                # Process frame
                await self.detector.process_frame(
                    frame=frame,
                    camera_id=camera_id,
                    frame_id=stream_info['frame_count']
                )
                
                # Update HLS stream
                await self._update_hls_stream(camera_id, frame)
                
                # Update state
                self.state_manager.update_stream_state(camera_id, stream_info)
                
        except asyncio.CancelledError:
            logger.info(f"Stream processing cancelled: {camera_id}")
            raise
            
        except Exception as e:
            logger.error(f"Error processing stream {camera_id}: {e}")
            stream_info['status'] = 'error'
            stream_info['error'] = str(e)
            self.state_manager.update_stream_state(camera_id, stream_info)
            
        finally:
            if cap:
                cap.release()
    
    async def _update_hls_stream(self, camera_id: str, frame: np.ndarray) -> None:
        """Update HLS stream with new frame
        
        Args:
            camera_id: ID of the camera
            frame: Video frame
        """
        stream_dir = self.hls_output_dir / camera_id
        os.makedirs(stream_dir, exist_ok=True)
        
        # Get current segment number
        segment_number = self.active_streams[camera_id]['frame_count'] // \
            (self.hls_segment_time * self.active_streams[camera_id]['fps'])
        
        # Write frame to current segment
        segment_path = stream_dir / f"segment_{segment_number}.ts"
        
        async with self._stream_locks[camera_id]:
            # Write frame using ffmpeg-python (implementation details omitted)
            # This would involve using ffmpeg to write the frame to a transport stream
            pass
        
        # Update playlist
        await self._update_hls_playlist(camera_id, segment_number)
    
    async def _update_hls_playlist(self, camera_id: str, current_segment: int) -> None:
        """Update HLS playlist
        
        Args:
            camera_id: ID of the camera
            current_segment: Current segment number
        """
        stream_dir = self.hls_output_dir / camera_id
        playlist_path = stream_dir / "playlist.m3u8"
        
        # Calculate segment range
        start_segment = max(0, current_segment - self.hls_playlist_size + 1)
        segments = range(start_segment, current_segment + 1)
        
        # Generate playlist content
        content = "#EXTM3U\n"
        content += f"#EXT-X-VERSION:3\n"
        content += f"#EXT-X-TARGETDURATION:{self.hls_segment_time}\n"
        content += f"#EXT-X-MEDIA-SEQUENCE:{start_segment}\n"
        
        for segment in segments:
            content += f"#EXTINF:{self.hls_segment_time},\n"
            content += f"segment_{segment}.ts\n"
        
        # Write playlist
        async with aiohttp.ClientSession() as session:
            async with session.put(f"file://{playlist_path}", data=content):
                pass
    
    async def _cleanup_hls_files(self, camera_id: str) -> None:
        """Cleanup HLS files for a stream
        
        Args:
            camera_id: ID of the camera
        """
        stream_dir = self.hls_output_dir / camera_id
        if stream_dir.exists():
            for file in stream_dir.glob("*"):
                try:
                    os.remove(file)
                except Exception as e:
                    logger.error(f"Error removing file {file}: {e}")
            try:
                os.rmdir(stream_dir)
            except Exception as e:
                logger.error(f"Error removing directory {stream_dir}: {e}")
    
    @property
    def active_stream_count(self) -> int:
        """Get number of active streams
        
        Returns:
            Number of active streams
        """
        return len(self.active_streams)
    
    def get_stream_status(self, camera_id: str) -> Optional[Dict]:
        """Get status of a stream
        
        Args:
            camera_id: ID of the camera
        
        Returns:
            Stream status if exists, None otherwise
        """
        return self.active_streams.get(camera_id)
    
    def get_stream_url(self, camera_id: str) -> Optional[str]:
        """Get HLS URL for a stream
        
        Args:
            camera_id: ID of the camera
        
        Returns:
            HLS URL if stream exists, None otherwise
        """
        if camera_id in self.active_streams:
            return f"/static/streams/{camera_id}/playlist.m3u8"
        return None
    
    async def cleanup(self) -> None:
        """Cleanup stream manager resources"""
        # Stop all streams
        for camera_id in list(self.active_streams.keys()):
            await self.remove_stream(camera_id)
        
        logger.info("Stream manager cleaned up") 