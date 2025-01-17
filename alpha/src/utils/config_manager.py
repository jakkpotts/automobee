import yaml
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages application configuration loading and validation."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("config/default.yaml")
        self.config: Dict[str, Any] = {}
        self._load_config()
        
    def _load_config(self):
        """Load configuration from YAML file."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
            else:
                logger.warning(f"Config file not found at {self.config_path}, using defaults")
                self.config = self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            self.config = self._get_default_config()
            
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values."""
        return {
            'processing': {
                'batch_size': 32,
                'max_queue_size': 100,
                'processing_interval': 0.1
            },
            'storage': {
                'frame_retention_seconds': 300
            },
            'security': {
                'token_refresh_window': 300,
                'max_failed_attempts': 5
            }
        }
        
    def get_all(self) -> Dict[str, Any]:
        """Get complete configuration."""
        return self.config
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        return self.config.get(key, default) 