from vehicle_detector import TrafficCameraMonitor
from feed_selector import CameraFeedSelector

def main():
    # Initialize the camera feed selector
    feed_selector = CameraFeedSelector()
    available_feeds = feed_selector.fetch_available_feeds()
    
    # Initialize monitor with model
    monitor = TrafficCameraMonitor(model_path="path/to/model.weights")
    
    # Print available intersections
    print("\nAvailable camera feeds:")
    for idx, intersection in enumerate(feed_selector.get_all_intersections(), 1):
        print(f"{idx}. {intersection}")
    
    # Get user input for feed selection
    selected_indices = input("\nEnter the numbers of feeds to monitor (comma-separated): ")
    
    # Add selected feeds to monitor
    intersections = feed_selector.get_all_intersections()
    for idx in selected_indices.split(','):
        try:
            idx = int(idx.strip()) - 1
            if 0 <= idx < len(intersections):
                intersection = intersections[idx]
                camera_info = feed_selector.get_camera_info(intersection)
                
                # Add camera with full information
                monitor.add_camera_feed(
                    camera_id=camera_info['id'],
                    url=camera_info['url'],
                    camera_info={
                        'name': camera_info['name'],
                        'coordinates': camera_info.get('coordinates'),
                        'intersection': intersection
                    }
                )
                print(f"Added camera: {intersection}")
        except ValueError:
            print(f"Invalid input: {idx}")
    
    # Define target vehicle
    target_vehicle = {
        "type": "sedan",
        "color": "red",
        "make": "Toyota",
        "model": "Camry",
    }
    
    # Start monitoring
    monitor.monitor_feeds(target_vehicle)

if __name__ == "__main__":
    main() 