# AutomoBee Vehicle Detection System

Real-time traffic monitoring system using computer vision and AI, with WebSocket-based communication for efficient real-time updates.

## Architecture

### Backend

- **Core Controller (`AutomoBee`)**
  - Centralized system initialization and coordination
  - Resource management and cleanup
  - Component lifecycle management

- **Services**
  - WebSocket server for real-time communication
  - Async stream manager for video processing
  - Zone-based detection service
  - Performance monitoring and optimization

### Frontend

- **Components**
  - Real-time dashboard with WebSocket updates
  - Interactive map with zone visualization
  - HLS video stream viewer
  - Performance metrics display

### Features

- Asynchronous video stream processing
- Batch-based vehicle detection
- Zone-based monitoring
- Real-time WebSocket updates
- Automatic performance optimization
- Resource usage monitoring
- Comprehensive error handling

## Setup

1. Create Python virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment:
```bash
cp .env.example .env
# Edit .env with your settings
```

4. Initialize configuration:
```bash
mkdir -p config logs
cp config/cameras.json.example config/cameras.json
# Edit cameras.json with your camera configurations
```

5. Start the system:
```bash
python main.py
```

## Configuration

### Camera Configuration (config/cameras.json)
```json
{
  "camera_id": {
    "name": "Camera Name",
    "lat": 36.1699,
    "lng": -115.1398,
    "stream_url": "rtsp://camera-url",
    "fps": 30,
    "zones": [
      {
        "name": "Zone 1",
        "coordinates": [
          {"lat": 36.1699, "lng": -115.1398},
          {"lat": 36.1700, "lng": -115.1399},
          {"lat": 36.1701, "lng": -115.1397}
        ]
      }
    ]
  }
}
```

### Environment Variables

See `.env.example` for all available configuration options.

## Development

### Project Structure
```
automobee/
├── src/
│   ├── backend/
│   │   ├── core/
│   │   │   └── automobee.py
│   │   └── services/
│   │       ├── detection_service.py
│   │       ├── stream_manager.py
│   │       ├── websocket_server.py
│   │       └── metrics_collector.py
│   └── frontend/
│       ├── components/
│       │   ├── map.js
│       │   └── stream-viewer.js
│       └── services/
│           └── websocket.js
├── config/
│   ├── cameras.json
│   └── logging.yaml
├── models/
│   ├── yolov8m.pt
│   └── vehicle_classifier.pt
└── logs/
    ├── automobee.log
    ├── error.log
    └── metrics.log
```

### Testing

Run tests with:
```bash
pytest tests/
```

### Monitoring

Access metrics and health information:
- System metrics: `logs/metrics.log`
- Application logs: `logs/automobee.log`
- Error tracking: `logs/error.log`

## Performance Optimization

The system automatically optimizes:
- Batch processing size
- Frame skip rate
- Detection confidence threshold
- Resource utilization

Monitor performance through:
- Detection rate per camera
- Classification confidence
- Processing latency
- Memory usage
- GPU utilization

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - see LICENSE file for details
