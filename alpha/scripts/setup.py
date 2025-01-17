import platform
import subprocess
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class SetupManager:
    """Manages platform-specific setup and dependency installation."""
    
    def __init__(self):
        self.system = platform.system()
        self.root_dir = Path(__file__).parent.parent
        self.requirements_file = self.root_dir / "requirements.txt"
        
    def check_environment(self) -> bool:
        """Check if the environment is properly configured."""
        try:
            if not self.requirements_file.exists():
                logger.error("Requirements file not found")
                return False
                
            if self.system == "Linux":
                return self._check_linux_environment()
            elif self.system == "Darwin":
                return self._check_macos_environment()
            else:
                logger.error("Unsupported operating system")
                return False
                
        except Exception as e:
            logger.error(f"Environment check failed: {str(e)}")
            return False
            
    def _check_linux_environment(self) -> bool:
        """Check Linux-specific requirements."""
        try:
            import torch
            if not torch.cuda.is_available():
                logger.error("CUDA not detected. Please install CUDA toolkit and drivers.")
                logger.error("Visit: https://developer.nvidia.com/cuda-downloads")
                return False
                
            cuda_version = torch.version.cuda
            logger.info(f"CUDA version {cuda_version} detected")
            return True
            
        except ImportError:
            logger.error("PyTorch not found or CUDA support not properly configured")
            return False
            
    def _check_macos_environment(self) -> bool:
        """Check macOS-specific requirements."""
        try:
            import torch
            if not torch.backends.mps.is_available():
                logger.error("Metal Performance Shaders (MPS) not available")
                logger.error("Please ensure you're using macOS 12.3+ and have compatible hardware")
                return False
                
            logger.info("MPS support detected")
            return True
            
        except ImportError:
            logger.error("PyTorch not found or MPS support not properly configured")
            return False
            
    def install_dependencies(self) -> bool:
        """Install required dependencies."""
        try:
            logger.info("Installing dependencies...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "-r", str(self.requirements_file)
            ])
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {str(e)}")
            return False
            
    def verify_installation(self) -> bool:
        """Verify that all required packages are installed and working."""
        required_packages = [
            'torch', 'torchvision', 'tensorflow', 'opencv-python',
            'ultralytics', 'aiohttp', 'websockets', 'folium'
        ]
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                logger.error(f"Required package {package} not found")
                return False
                
        return True 