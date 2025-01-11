# AutomoBee Vehicle Detection System

AutomoBee is a real-time traffic monitoring system that analyzes the city's public cctv camera network a modern web interface for locating, tracking, and monitoring a target vehicle's location using computer vision and AI.

## Features

- Real-time vehicle detection and classification
- Real-time vehicle tracking
- Live map visualization
- Real-time statistics and monitoring
- Supports multiple camera feed analysis
- Responsive layout supporting both desktop and mobile

## Architecture

### Frontend

- **Dashboard UI**
  - Built with vanilla JavaScript, HTML5 and Tailwind CSS
  - Live map powered by Leaflet.js for geographic visualization
  - Real-time data updates using WebSocket
  - HLS.js integration for video streaming
  - Nord color theme for consistent visual design

- **Key Components**
  - Live camera feed manager
  - Interactive map with camera locations target vehicle detected zones
  - Real-time detection statistics
  - Camera feed status monitoring
  - Alert system for detections

### Data Handling

- Video stream processing using HLS (HTTP Live Streaming)
- Efficient buffering and frame processing
- Camera feed management
- Detection metadata handling

## Project Structure

```text
automobee/
├── main.py             # Application entry point and process management
├── dashboard.py        # Dashboard web server and state management
├── vehicle_detect.py   # Vehicle detection and classification logic
├── stream_handler.py   # Video stream processing and management
├── config.py          # Configuration and environment settings
├── templates/
│   └── dashboard.html # Main dashboard interface template
├── static/
│   ├── css/
│   │   └── leaflet.css # Map styling and controls
│   └── js/
│       ├── dashboard.js # Dashboard UI logic and controls
│       └── hls.min.js  # HTTP Live Streaming client
└── .gitignore         # Version control exclusions
```

Each file serves a specific purpose:

- `dashboard.py`: Dashboard web server and state management
- `dashboard.py`: Implements web server with Flask, handles real-time updates and state
- `vehicle_detect.py`: Core detection and classification algorithms
- `stream_handler.py`: Manages video stream processing and buffering
- `config.py`: Environment and application configuration
- `dashboard.html`: Main dashboard interface template built with Tailwind CSS
- `dashboard.js`: Implements map functionality and UI interactions
- `leaflet.css`: Provides map styling and controls
- `hls.min.js`: Enables video stream playback

## Security Notes

- Environment variables and secrets should be properly configured
- Model files are excluded from version control
- Sensitive configuration files are gitignored

## License

MIT standard license. Open source is the way.
