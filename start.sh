#!/bin/bash

# Multi-Camera Detection System Startup Script
# This script handles the deployment and management of the detection system

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    print_status "Checking Docker installation..."
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_success "Docker and Docker Compose are installed"
}

# Check camera permissions on macOS
check_camera_permissions() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        print_warning "On macOS, you may need to grant camera permissions to Docker"
        print_warning "Go to System Preferences > Security & Privacy > Camera"
        print_warning "Make sure Docker Desktop has camera access"
    fi
}

# List available cameras
list_cameras() {
    print_status "Scanning for available cameras..."
    
    # Try to list video devices
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        system_profiler SPCameraDataType 2>/dev/null || print_warning "Could not list cameras automatically on macOS"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        ls /dev/video* 2>/dev/null || print_warning "No video devices found in /dev/"
    fi
}

# Build the Docker image
build_image() {
    print_status "Building Docker image..."
    docker-compose build
    print_success "Docker image built successfully"
}

# Start the services
start_services() {
    print_status "Starting multi-camera detection system..."
    docker-compose up -d
    
    # Wait for services to be ready
    print_status "Waiting for services to start..."
    sleep 10
    
    # Check if container is running
    if docker-compose ps | grep -q "Up"; then
        print_success "Services started successfully"
        print_success "Web interface available at: http://localhost:5001"
    else
        print_error "Failed to start services"
        docker-compose logs
        exit 1
    fi
}

# Stop the services
stop_services() {
    print_status "Stopping multi-camera detection system..."
    docker-compose down
    print_success "Services stopped successfully"
}

# Show logs
show_logs() {
    print_status "Showing application logs..."
    docker-compose logs -f
}

# Clean up (remove containers and images)
cleanup() {
    print_status "Cleaning up Docker resources..."
    docker-compose down --rmi all --volumes --remove-orphans
    print_success "Cleanup completed"
}

# Development mode (run without Docker)
dev_mode() {
    print_status "Starting development mode..."
    
    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        exit 1
    fi
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        print_status "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install dependencies
    print_status "Installing dependencies..."
    pip install -r requirements.txt
    
    # Run the application
    print_status "Starting application in development mode..."
    python app.py
}

# Show help
show_help() {
    echo "Multi-Camera Detection System Management Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start     - Start the detection system with Docker"
    echo "  stop      - Stop the detection system"
    echo "  restart   - Restart the detection system"
    echo "  logs      - Show application logs"
    echo "  dev       - Run in development mode (without Docker)"
    echo "  build     - Build Docker image"
    echo "  cleanup   - Remove all Docker resources"
    echo "  cameras   - List available cameras"
    echo "  check     - Check system requirements"
    echo "  help      - Show this help message"
}

# Main script logic
case "${1:-help}" in
    "start")
        check_docker
        check_camera_permissions
        build_image
        start_services
        ;;
    "stop")
        stop_services
        ;;
    "restart")
        stop_services
        sleep 2
        start_services
        ;;
    "logs")
        show_logs
        ;;
    "dev")
        dev_mode
        ;;
    "build")
        check_docker
        build_image
        ;;
    "cleanup")
        cleanup
        ;;
    "cameras")
        list_cameras
        ;;
    "check")
        check_docker
        check_camera_permissions
        list_cameras
        ;;
    "help"|*)
        show_help
        ;;
esac
