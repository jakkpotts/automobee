# Las Vegas Traffic Camera Vehicle Detector

A Python-based system that monitors Las Vegas traffic camera feeds and uses AI to detect and track specific vehicles across multiple intersections. Features a real-time web dashboard for monitoring system status and detections.

## Features

- Real-time monitoring of Las Vegas traffic cameras
- Vehicle detection using YOLOv8
- Color and vehicle type classification
- Make/model detection (currently optimized for Ford F-150)
- Automated screenshot capture of matches
- Detailed match logging with timestamp and location data
- Multi-camera support with intersection mapping
- Real-time web dashboard
- Rate limiting and error handling
- System health monitoring

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/las-vegas-vehicle-detector.git
cd las-vegas-vehicle-detector
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the monitoring dashboard:
```bash
python dashboard.py
```

2. In a new terminal, run the main detection system:
```bash
python main.py
```

3. Access the dashboard at: http://localhost:8675

### Dashboard Features

- Real-time statistics overview
- Camera status monitoring
- Recent detection display with images
- Detection history graph
- System health alerts
- Camera uptime tracking

### Configuration

Target vehicle details can be modified in `main.py`:
```python
target_vehicle = {
    "type": "truck",        # Options: car, motorcycle, bus, truck
    "color": "black",       # Options: red, blue, white, black
    "make": "Ford",         # Currently supports Ford vehicles
    "model": "F-150"        # Currently supports F-150 model
}
```

System parameters can be adjusted when initializing the detector:
```python
detector = VehicleDetector(
    sample_interval=30,     # Seconds between camera checks
    max_retries=3,         # Number of retries for failed camera access
    retry_delay=5,         # Seconds between retries
    alert_threshold=5      # Consecutive failures before alert
)
```

## Project Structure

```
.
├── dashboard.py          # Web dashboard server
├── feed_selector.py      # Camera feed management
├── vehicle_detector.py   # Vehicle detection and tracking
├── rate_limiter.py      # Request rate limiting
├── train_classifier.py  # Make/model classifier training
├── collect_data.py     # Dataset collection for make/model training
├── main.py              # Main application entry
├── requirements.txt      # Project dependencies
└── templates/
    └── dashboard.html    # Dashboard template
```

## Output

The system generates three types of output:

1. Match Files (in `matches/` directory):
   - Full frame images
   - Cropped vehicle images
   - JSON metadata files

2. Monitoring Data (in `monitoring/` directory):
   - System statistics
   - Camera health data
   - Detection history

3. Web Dashboard:
   - Real-time statistics
   - Camera status cards
   - Recent detections feed
   - Performance graphs

## Performance Considerations

- Processing speed depends on:
  - Number of camera feeds being monitored
  - Hardware capabilities (CPU/GPU)
  - Network bandwidth
- GPU acceleration is automatically used if available
- M2 Mac support via MPS backend
- Rate limiting prevents camera feed overload
- Automatic retry on camera feed failures
- Concurrent processing of multiple cameras

## Error Handling

The system includes robust error handling for:
- Network connectivity issues
- Camera feed failures
- Image processing errors
- Resource management
- System resource constraints

## Monitoring

The dashboard provides real-time monitoring of:
- System uptime
- Camera health
- Detection rates
- Error rates
- Resource usage
- Recent detections

## Limitations

- Color detection accuracy may vary based on lighting conditions
- Make/model detection currently optimized for Ford F-150
- Requires stable internet connection
- Camera feed availability depends on NV Roads system

## System Requirements

- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended)
- NVIDIA GPU with CUDA support (optional)
- Apple M1/M2 chip (supported via MPS)
- Stable internet connection (minimum 10Mbps)
- 500MB free disk space for base installation

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This tool is for educational purposes only. Ensure compliance with all local laws and regulations regarding traffic camera usage and vehicle tracking. This is not an official tool of the Nevada Department of Transportation.

## Acknowledgments

- Nevada DOT for providing traffic camera feeds
- YOLOv8 team for the object detection model
- OpenCV community for computer vision tools
- Flask team for the web framework
- TailwindCSS for dashboard styling

## Model Training

The system includes a make/model classifier trained on the VMMRdb dataset:

1. Data Collection:
```bash
python collect_data.py
```
This will download and organize vehicle images, focusing on Ford F-150 samples.

2. Model Training:
```bash
python train_classifier.py
```
This trains an EfficientNet-based classifier optimized for F-150 detection.