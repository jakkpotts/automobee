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
