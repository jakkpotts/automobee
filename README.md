# AutomoBee - Intelligent Vehicle Detection System

A sophisticated Python-based system that leverages AI to monitor traffic camera feeds for vehicle detection, classification, and tracking. The system features a real-time web dashboard for monitoring multiple camera feeds, with advanced vehicle detection capabilities and comprehensive system health monitoring.

## 🚀 Features

- **Advanced Vehicle Detection**
  - Real-time processing using YOLOv8
  - Multi-class vehicle type classification (cars, trucks, motorcycles, buses)
  - Color detection and analysis
  - Make/model classification
  - High-accuracy vehicle tracking

- **Interactive Dashboard**
  - Real-time monitoring web interface
  - Live detection feed with SSE updates
  - Interactive map with camera locations
  - System health monitoring and alerts
  - Comprehensive performance metrics
  - Mobile-responsive design

- **Smart Camera Management**
  - Multi-camera feed support
  - Automatic feed selection and prioritization
  - Built-in rate limiting and error handling
  - Robust recovery mechanisms
  - Geographic location tracking

- **Real-time Analytics**
  - Detection rate monitoring
  - Camera uptime tracking
  - Error rate analysis
  - System performance metrics
  - Historical data tracking

## 🛠 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/automobee.git
cd automobee
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📋 Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, for acceleration)
- Apple Silicon Mac (optional, for MPS acceleration)
- Stable internet connection
- Sufficient storage for detection archives

## 🚦 Usage

1. Start the monitoring dashboard:
```bash
python dashboard.py
```

2. Launch the main detection system:
```bash
python main.py
```

3. Access the dashboard at `http://localhost:8675`

## ⚙️ Configuration

### Vehicle Detection Parameters

Modify target vehicle specifications in `main.py`:
```python
target_vehicle = {
    "type": "truck",     # Options: car, motorcycle, bus, truck
    "color": "black",    # Options: red, blue, white, black
    "make": "ford",      # Currently optimized for Ford
    "model": "f-150"     # Currently optimized for F-150
}
```

### System Settings

Adjust detector parameters in `main.py`:
```python
detector = VehicleDetector(
    sample_interval=30,    # Seconds between camera checks
    max_retries=3,        # Number of retries on failure
    retry_delay=5,        # Seconds between retries
    alert_email="your.email@domain.com",
    alert_threshold=5     # Failures before alerting
)
```

## 📁 Project Structure

```
automobee/
├── vehicle_detector.py   # Core detection engine
├── vehicle_classifier.py # Make/model classification
├── feed_selector.py     # Camera feed management
├── rate_limiter.py     # Request rate limiting
├── dashboard.py        # Web interface server
├── train_classifier.py # Model training utilities
├── collect_data.py    # Training data collection
├── main.py           # Application entry point
├── requirements.txt  # Project dependencies
└── monitoring/      # System statistics and logs
```

## 📊 Output

The system generates three types of output:

1. **Detection Records** (`matches/`)
   - Full frame captures
   - Cropped vehicle images
   - Detection metadata (JSON)

2. **System Monitoring** (`monitoring/`)
   - Performance statistics
   - Camera health data
   - Detection history

3. **Live Dashboard**
   - Real-time detection feed
   - System status overview
   - Performance metrics
   - Alert notifications

## 🔧 Advanced Usage

### Training Custom Models

1. Collect training data:
```bash
python collect_data.py
```

2. Train the make/model classifier:
```bash
python train_classifier.py
```

### Performance Optimization

- Adjust `sample_interval` based on hardware capabilities
- Use GPU acceleration when available
- Configure rate limiting based on network capacity
- Optimize camera feed selection for coverage

## 🛡 Error Handling

The system includes comprehensive error handling for:
- Network connectivity issues
- Camera feed failures
- Processing errors
- Resource constraints
- System overload conditions

## 📈 Monitoring

The dashboard provides real-time monitoring of:
- System uptime
- Camera health
- Detection rates
- Error statistics
- Resource utilization
- Recent detections

## ⚠️ Limitations

- Color detection accuracy varies with lighting conditions
- Make/model detection optimized for specific vehicles
- Requires stable network connection
- Performance depends on hardware capabilities

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- YOLOv8 for object detection
- EfficientNet for vehicle classification
- OpenCV for image processing
- PyTorch for deep learning capabilities