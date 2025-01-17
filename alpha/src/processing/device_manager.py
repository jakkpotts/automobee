from dataclasses import dataclass
from typing import List, Optional
import torch
import logging
import psutil
import GPUtil
from enum import Enum

logger = logging.getLogger(__name__)

class DeviceType(Enum):
    """Supported device types for processing."""
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"

@dataclass
class DeviceConfig:
    """Configuration for device management."""
    device_fallback_order: List[str]
    cuda_memory_threshold: float = 0.9  # 90% memory usage threshold
    cpu_memory_threshold: float = 0.85  # 85% memory usage threshold
    dynamic_batch_sizing: bool = True
    min_batch_size: int = 1
    max_batch_size: int = 32
    performance_window: int = 100  # Number of inferences to average

class DeviceManager:
    """Manages compute devices and their resources."""
    
    def __init__(self, config: Optional[DeviceConfig] = None):
        self.config = config or DeviceConfig(
            device_fallback_order=self._get_default_device_order()
        )
        self.current_device = self._initialize_device()
        self.performance_metrics = {
            "inference_times": [],
            "memory_usage": [],
            "batch_sizes": [],
            "errors": []
        }
        
    def _get_default_device_order(self) -> List[str]:
        """Determine default device order based on availability."""
        devices = []
        if torch.cuda.is_available():
            devices.append("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            devices.append("mps")
        devices.append("cpu")
        return devices
        
    def _initialize_device(self) -> str:
        """Initialize the primary compute device."""
        for device in self.config.device_fallback_order:
            if self._test_device(device):
                logger.info(f"Initialized primary device: {device}")
                return device
        return "cpu"
        
    def _test_device(self, device: str) -> bool:
        """Test if a device is available and working."""
        try:
            if device == "cuda":
                return torch.cuda.is_available()
            elif device == "mps":
                return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            return device == "cpu"
        except Exception as e:
            logger.error(f"Device test failed for {device}: {str(e)}")
            return False
            
    def get_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on device and memory."""
        if not self.config.dynamic_batch_sizing:
            return self.config.max_batch_size
            
        try:
            if self.current_device == "cuda":
                gpu = GPUtil.getGPUs()[0]
                memory_usage = gpu.memoryUtil
                if memory_usage > self.config.cuda_memory_threshold:
                    return max(self.config.min_batch_size,
                             self.config.max_batch_size // 2)
            else:
                memory_usage = psutil.virtual_memory().percent / 100
                if memory_usage > self.config.cpu_memory_threshold:
                    return max(self.config.min_batch_size,
                             self.config.max_batch_size // 2)
                    
            return self.config.max_batch_size
            
        except Exception as e:
            logger.error(f"Error calculating batch size: {str(e)}")
            return self.config.min_batch_size
            
    def update_performance_metrics(self, inference_time: float, batch_size: int,
                                 error: Optional[Exception] = None):
        """Update device performance metrics."""
        self.performance_metrics["inference_times"].append(inference_time)
        self.performance_metrics["batch_sizes"].append(batch_size)
        
        if self.current_device == "cuda":
            try:
                gpu = GPUtil.getGPUs()[0]
                self.performance_metrics["memory_usage"].append(gpu.memoryUtil)
            except:
                pass
        else:
            self.performance_metrics["memory_usage"].append(
                psutil.virtual_memory().percent / 100
            )
            
        if error:
            self.performance_metrics["errors"].append(str(error))
            
        # Keep only recent metrics
        window = self.config.performance_window
        for key in self.performance_metrics:
            self.performance_metrics[key] = self.performance_metrics[key][-window:]
            
    def get_device_stats(self) -> dict:
        """Get current device statistics."""
        stats = {
            "device": self.current_device,
            "avg_inference_time": 0,
            "avg_memory_usage": 0,
            "avg_batch_size": 0,
            "error_rate": 0
        }
        
        if self.performance_metrics["inference_times"]:
            stats["avg_inference_time"] = sum(self.performance_metrics["inference_times"]) / \
                                        len(self.performance_metrics["inference_times"])
                                        
        if self.performance_metrics["memory_usage"]:
            stats["avg_memory_usage"] = sum(self.performance_metrics["memory_usage"]) / \
                                      len(self.performance_metrics["memory_usage"])
                                      
        if self.performance_metrics["batch_sizes"]:
            stats["avg_batch_size"] = sum(self.performance_metrics["batch_sizes"]) / \
                                    len(self.performance_metrics["batch_sizes"])
                                    
        if self.performance_metrics["errors"]:
            stats["error_rate"] = len(self.performance_metrics["errors"]) / \
                                 self.config.performance_window
                                 
        return stats 