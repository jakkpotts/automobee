# Las Vegas Traffic Camera Vehicle Detector

A Python-based system that monitors Las Vegas traffic camera feeds and uses AI to detect and track specific vehicles across multiple intersections.

## Features

- Real-time monitoring of Las Vegas traffic cameras
- Vehicle detection using YOLOv8
- Color and vehicle type classification
- Automated screenshot capture of matches
- Detailed match logging with timestamp and location data
- Multi-camera support with intersection mapping

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

1. Run the main script:
```bash
python main.py
```

2. Select cameras to monitor from the displayed list of available intersections

3. The system will begin monitoring the selected feeds for vehicles matching the target description

### Configuration

Target vehicle details can be modified in `main.py`:
```python
target_vehicle = {
    "type": "sedan",  # Options: car, truck, bus, motorcycle
    "color": "red",   # Options: red, blue, white, black
    "make": "Toyota",
    "model": "Camry"
}
```

## Supported Vehicle Types

- Car
- Motorcycle
- Bus
- Truck

## Supported Colors

- Red
- Blue
- White
- Black

Additional colors can be added by modifying the `color_ranges` dictionary in `vehicle_detector.py`.

## Project Structure

```
.
├── feed_selector.py    # Camera feed management
├── vehicle_detector.py # Vehicle detection and tracking
├── main.py            # Main application entry
└── requirements.txt   # Project dependencies
```

## Output

When a matching vehicle is detected, the system:

1. Saves the full frame as an image
2. Saves a cropped image of the detected vehicle
3. Creates a JSON metadata file containing:
   - Timestamp with timezone
   - Intersection location
   - Camera details
   - Detection confidence
   - Vehicle characteristics

Files are saved in the `matches/` directory with the following structure:
```
matches/
├── {camera_id}_{timestamp}.jpg           # Full frame
├── {camera_id}_{timestamp}_vehicle_0.jpg # Vehicle crop
└── {camera_id}_{timestamp}_metadata.json # Detection metadata
```

## Performance Considerations

- Processing speed depends on:
  - Number of camera feeds being monitored
  - Hardware capabilities (CPU/GPU)
  - Network bandwidth
- GPU acceleration is automatically used if available
- Rate limiting is implemented to prevent overwhelming camera feeds

## Limitations

- Color detection accuracy may vary based on lighting conditions
- Vehicle make/model detection not currently implemented
- Requires stable internet connection
- Camera feed availability depends on NV Roads system

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

## System Requirements

- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended)
- NVIDIA GPU with CUDA support (optional, for GPU acceleration)
- Stable internet connection (minimum 10Mbps)
- 500MB free disk space for base installation

## Additional Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Nevada DOT Camera Feeds](https://www.nvroads.com/)