from bs4 import BeautifulSoup
import requests
from typing import Dict, List
import re
import json

class CameraFeedSelector:
    def __init__(self):
        self.base_url = "https://www.nvroads.com"
        self.feeds: Dict[str, Dict] = {}
        self.session = requests.Session()

    def fetch_available_feeds(self) -> Dict[str, Dict]:
        """
        Fetches camera feeds using the Nevada 511 API
        Returns a dictionary of camera information including names and feed URLs
        """
        try:
            # Get the main page first to get cookies and token
            main_response = self.session.get(f"{self.base_url}/cctv")
            soup = BeautifulSoup(main_response.text, 'html.parser')
            
            # Extract verification token
            token_tag = soup.find('input', {'name': '__RequestVerificationToken'})
            token = token_tag.get('value') if token_tag else None
            
            if not token:
                print("Warning: No verification token found")
            
            cameras = {}
            start = 0
            page_size = 100
            
            while True:
                # Prepare the query for Las Vegas cameras
                query = {
                    "columns": [
                        {"data": None, "name": ""},
                        {"name": "sortOrder", "s": True},
                        {
                            "name": "region",
                            "search": {"value": "Las Vegas Area"},
                            "s": True
                        },
                        {"name": "roadway", "s": True},
                        {"data": 4, "name": ""}
                    ],
                    "order": [
                        {"column": 1, "dir": "asc"},
                        {"column": 2, "dir": "asc"},
                        {"column": 3, "dir": "asc"}
                    ],
                    "start": start,
                    "length": page_size,
                    "search": {"value": ""}
                }
                
                # Get camera list from the data endpoint
                cameras_response = self.session.get(
                    f"{self.base_url}/List/GetData/Cameras",
                    params={
                        'query': json.dumps(query),
                        'lang': 'en'
                    },
                    headers={
                        'accept': 'application/json',
                        'content-type': 'application/json',
                        'x-requested-with': 'XMLHttpRequest',
                        '__requestverificationtoken': token
                    }
                )
                
                camera_data = cameras_response.json()
                
                # Process camera data from this page
                for camera in camera_data.get('data', []):
                    camera_id = camera.get('DT_RowId')
                    if camera_id:
                        roadway = camera.get('roadway', '')
                        region = camera.get('region', '')
                        name = roadway if roadway else f"{region} Camera {camera_id}"
                        
                        cameras[name] = {
                            'id': camera_id,
                            'name': name,
                            'region': region,
                            'roadway': roadway,
                            'url': f"{self.base_url}/cameras/{camera_id}/snapshot",
                            'stream_url': f"{self.base_url}/cameras/{camera_id}/stream"
                        }
                
                # Check if we've got all cameras
                records_total = camera_data.get('recordsTotal', 0)
                if start + page_size >= records_total:
                    break
                    
                start += page_size
                print(f"Fetched {len(cameras)} cameras so far...")
            
            self.feeds = cameras
            print(f"\nFound {len(cameras)} Las Vegas area cameras")
            
            return self.feeds
            
        except Exception as e:
            print(f"Error fetching camera feeds: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def get_feed_url(self, intersection_name: str) -> str:
        """Returns the feed URL for a given intersection name"""
        camera = self.feeds.get(intersection_name, {})
        return camera.get('url', '')

    def get_all_intersections(self) -> List[str]:
        """Returns a list of all available intersection names"""
        return list(self.feeds.keys())

    def get_camera_info(self, intersection_name: str) -> Dict:
        """Returns all available information for a given intersection"""
        return self.feeds.get(intersection_name, {}) 

    def select_strategic_cameras(self) -> Dict[str, Dict]:
        """
        Selects strategic cameras based on:
        - Major intersections/highways
        - Entry/exit points to the city
        - Even distribution across the network
        """
        strategic_locations = [
            "I-15",      # Major N-S highway
            "US 95",     # Major E-W highway
            "I-515",     # Downtown connector
            "I-215",     # Beltway
            "Las Vegas Blvd",  # The Strip
            "Sahara Ave",      # Major E-W arterial
            "Charleston Blvd",  # Major E-W arterial
            "Tropicana Ave",   # Major E-W arterial
            "Flamingo Rd",     # Major E-W arterial
            "Blue Diamond Rd", # Southern access
            "Lake Mead Blvd",  # Northern access
            "Boulder Hwy"      # Eastern access
        ]
        
        strategic_cameras = {}
        for name, camera in self.feeds.items():
            # Check if camera is at a strategic location
            if any(loc in name for loc in strategic_locations):
                strategic_cameras[name] = camera
                
        print(f"\nSelected {len(strategic_cameras)} strategic cameras at major intersections:")
        for name in strategic_cameras.keys():
            print(f"- {name}")
            
        return strategic_cameras 