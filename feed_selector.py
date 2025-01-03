from bs4 import BeautifulSoup
import requests
from typing import Dict, List
import re
import json
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

class CameraFeedSelector:
    def __init__(self):
        self.base_url = "https://www.nvroads.com/region/Las%20Vegas"
        self.feeds: Dict[str, Dict] = {}

    def fetch_available_feeds(self) -> Dict[str, Dict]:
        """
        Scrapes the NV Roads website to get available camera feeds
        Returns a dictionary of camera information including intersection names and feed URLs
        """
        try:
            response = requests.get(self.base_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find camera information in the MyCameras section
            my_cameras = soup.find('div', class_='myCamerasContainer')
            if my_cameras:
                # Extract camera title and ID
                camera_title = my_cameras.find('p', id='myCameraTitle')
                camera_location = my_cameras.find('a', id='myCameraLocation')
                
                if camera_title and camera_location:
                    camera_id = camera_location.get('href', '').replace('#camera-', '')
                    self.feeds[camera_title.text] = {
                        'id': camera_id,
                        'name': camera_title.text,
                        'url': f"https://www.nvroads.com/cameras/{camera_id}/snapshot"
                    }

            # Find all script tags that load camera data
            scripts = soup.find_all('script', src=re.compile(r'/scripts/jsresources/map/map\?'))
            
            # Extract camera data from each script
            for script in scripts:
                script_url = script.get('src')
                if script_url:
                    try:
                        script_response = requests.get(f"https://www.nvroads.com{script_url}")
                        # Parse camera data from JavaScript
                        camera_data = self._extract_camera_data(script_response.text)
                        self.feeds.update(camera_data)
                    except Exception as e:
                        print(f"Error parsing camera script: {e}")
            
            return self.feeds
            
        except Exception as e:
            print(f"Error fetching camera feeds: {e}")
            return {}

    def _extract_camera_data(self, script_text: str) -> Dict[str, Dict]:
        """
        Extracts camera information from JavaScript code
        """
        cameras = {}
        try:
            # Look for camera data in the script
            camera_matches = re.findall(r'camera\s*=\s*{([^}]+)}', script_text)
            for match in camera_matches:
                # Parse camera properties
                props = dict(re.findall(r'(\w+)\s*:\s*[\'"]([^\'"]+)[\'"]', match))
                if 'id' in props and 'name' in props:
                    cameras[props['name']] = {
                        'id': props['id'],
                        'name': props['name'],
                        'url': f"https://www.nvroads.com/cameras/{props['id']}/snapshot"
                    }
        except Exception as e:
            print(f"Error extracting camera data: {e}")
        return cameras

    def get_feed_url(self, intersection_name: str) -> str:
        """
        Returns the feed URL for a given intersection name
        """
        camera = self.feeds.get(intersection_name, {})
        return camera.get('url', '')

    def get_all_intersections(self) -> List[str]:
        """
        Returns a list of all available intersection names
        """
        return list(self.feeds.keys())

    def get_camera_info(self, intersection_name: str) -> Dict:
        """
        Returns all available information for a given intersection
        """
        return self.feeds.get(intersection_name, {}) 