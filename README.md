# Multi-Camera Detection System

A real-time computer vision system that performs object detection across multiple camera feeds using YOLOv8 and provides a web interface for monitoring.

## Features

- **Multi-Camera Support**: Simultaneously processes up to 4 camera feeds
- **Real-time Object Detection**: Uses YOLOv8 for accurate object detection
- **Web Interface**: Clean, responsive web interface for monitoring
- **Docker Support**: Easy deployment with Docker containers
- **Live Streaming**: Real-time video streaming with detection overlays
- **Detection Analytics**: Live statistics and detection summaries

## Camera Support

This system is configured to work with your specific cameras:
- MacBook Pro Camera (Built-in)
- HERO11 Black (USB)
- Insta360 X3 (USB)
- OBS Virtual Camera (Virtual)

## Prerequisites

- Docker and Docker Compose
- Python 3.9+ (for local development)
- OpenCV compatible cameras
- At least 4GB RAM recommended
- GPU support recommended (optional)

## Quick Start with Docker

1. **Clone and navigate to the project directory**:
   ```bash
   cd multiview_detection_system
   ```

2. **Build and run with Docker Compose**:
   ```bash
   docker-compose up --build
   ```

3. **Access the web interface**:
   Open your browser and go to `http://localhost:5000`

## Local Development Setup

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application**:
   ```bash
   python app.py
   ```

3. **Access the web interface**:
   Open your browser and go to `http://localhost:5000`

## Configuration

### Camera Configuration
Edit `config.py` to customize camera settings:
- Resolution and FPS settings
- Enable/disable specific cameras
- Camera names and types

### YOLO Model Configuration
Choose different YOLO models for speed vs accuracy:
- `yolov8n.pt`: Fastest, lowest accuracy
- `yolov8s.pt`: Balanced
- `yolov8m.pt`: Medium accuracy
- `yolov8l.pt`: High accuracy
- `yolov8x.pt`: Highest accuracy, slowest

### Detection Settings
Adjust detection parameters:
- Confidence threshold (0.0-1.0)
- IOU threshold for non-max suppression
- Maximum detections per frame
- Specific classes to detect

## Usage

1. **Start the System**:
   - Click "Start Detection" in the web interface
   - The system will initialize all available cameras
   - Live feeds will appear in the grid layout

2. **Monitor Detection**:
   - View real-time detection results
   - See bounding boxes around detected objects
   - Monitor detection statistics

3. **Stop the System**:
   - Click "Stop Detection" to stop processing
   - All camera feeds will be released

## Web Interface Features

- **Live Camera Feeds**: 2x2 grid showing all camera feeds
- **Detection Overlays**: Bounding boxes and labels on detected objects
- **Real-time Statistics**: Object counts, FPS, and camera status
- **Responsive Design**: Works on desktop and mobile devices

## API Endpoints

- `GET /`: Main web interface
- `GET /api/cameras`: List of available cameras
- `GET /api/detection-summary`: Current detection statistics
- WebSocket events for real-time updates

## Docker Configuration

The system includes Docker support with:
- Multi-stage builds for optimization
- Camera device mapping
- Volume mounting for configuration
- Health checks
- Automatic restart policies

## Troubleshooting

### Camera Issues
- Ensure cameras are properly connected
- Check camera permissions (macOS may require camera access)
- Verify camera IDs in the system
- Try different camera indices if detection fails

### Performance Issues
- Reduce camera resolution in `config.py`
- Use a lighter YOLO model (yolov8n.pt)
- Decrease FPS settings
- Enable GPU acceleration if available

### Docker Issues
- Ensure Docker has camera access permissions
- Check device mapping in `docker-compose.yml`
- Verify privileged mode is enabled for camera access

## File Structure

```
multiview_detection_system/
├── app.py                 # Main application
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
├── Dockerfile           # Docker configuration
├── docker-compose.yml   # Docker Compose setup
├── README.md           # This file
└── templates/
    └── index.html      # Web interface template
```

## Performance Optimization

1. **Hardware Acceleration**: Enable GPU support for faster inference
2. **Model Selection**: Choose appropriate YOLO model based on your hardware
3. **Resolution Settings**: Lower resolution for better performance
4. **Batch Processing**: Process multiple frames together (if supported)

## Security Considerations

- Change the default secret key in production
- Consider authentication for web interface
- Restrict network access as needed
- Monitor resource usage

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the logs for error messages
3. Ensure all dependencies are properly installed
4. Verify camera compatibility and connections
