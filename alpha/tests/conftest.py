import pytest
import asyncio
from pathlib import Path
from src.config.config_manager import ConfigManager

@pytest.fixture
def config_manager():
    """Fixture for configuration manager."""
    return ConfigManager(Path("tests/test_config.yaml"))

@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close() 