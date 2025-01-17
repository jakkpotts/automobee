import asyncio
import logging
from typing import Dict, Optional
from datetime import datetime
from .detection_logger import DetectionLogger  # Add this import
from ..utils.error_manager import ErrorManager, ErrorCategory, ErrorSeverity

logger = logging.getLogger(__name__)

class ProcessingScheduler:
    """Manages processing tasks and scheduling."""
    
    def __init__(self):
        self.active_tasks = {}
        self.focus_config = {}
        self.detection_logger = None
        self.running = False
        self.error_manager = ErrorManager() 
        
    async def initialize(self):
        """Initialize the scheduler."""
        try:
            self.running = True
            self.detection_logger = DetectionLogger()
            logger.info("Processing scheduler initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize processing scheduler: {str(e)}")
            raise
            
    async def start(self):
        """Start the scheduler."""
        self.running = True
        logger.info("Processing scheduler started") 