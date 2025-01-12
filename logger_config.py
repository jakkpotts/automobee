import logging
import os

# Configure logging
# Define the log file path
log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'main.log')

# Ensure the log directory exists
os.makedirs(os.path.dirname(log_path), exist_ok=True)

# Set up the logger
logger = logging.getLogger('shared_logger')
logger.setLevel(logging.DEBUG)

# File handler
file_handler = logging.FileHandler(log_path, mode='a')  # 'a' for appending logs to the file
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Console handler (optional)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# The logger is now set up and can be used in other modules


# Test logging
logger.debug('This is a debug message')
logger.info('This is an info message')