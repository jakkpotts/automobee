import aiohttp
import asyncio
import json
import logging
from logger_config import logger
import cv2
import requests
from typing import Dict, List, Optional

class CameraFeedSelector:
    def __init__(self):
        self.feeds = {}
        self.session = None
        self.logger = self._setup_logging()
        self.validated_feeds = self._load_and_validate_camera_data()
        self.base_url = "https://www.nvroads.com/List/GetData/Cameras"
        self.camera_locations = self._load_camera_locations()
        
    def _load_and_validate_camera_data(self) -> Dict:
        """Load and validate camera data from config file"""
        try:
            with open('config/camera_locations.json') as f:
                data = json.load(f)
                
            # Validate data structure
            if isinstance(data, dict):
                # Check if each camera has required fields
                valid_data = {}
                for name, camera in data.items():
                    if isinstance(camera, dict) and all(
                        key in camera for key in ['lat', 'lng', 'name']
                    ):
                        valid_data[name] = camera
                
                if valid_data:
                    self.logger.info(f"Loaded {len(valid_data)} valid camera locations")
                    return valid_data
                    
            self.logger.warning("No valid camera data found in config file")
            return {}
            
        except FileNotFoundError:
            self.logger.warning("Camera locations file not found")
            return {}
        except json.JSONDecodeError:
            self.logger.error("Invalid JSON in camera locations file")
            return {}
        except Exception as e:
            self.logger.error(f"Error loading camera data: {e}")
            return {}

    def _load_camera_locations(self) -> Dict[str, Dict]:
        """Load camera locations from config file or transform from API data"""
        try:
            with open('config/camera_locations.json') as f:
                data = json.load(f)
                
                # Validate the format
                if 'data' in data:
                    # Handle raw API data format
                    return self._transform_camera_data(data)
                elif all('lat' in cam and 'lng' in cam and 'name' in cam 
                        for cam in data.values()):
                    # Current format is valid
                    self.logger.info("Loaded pre-configured camera locations")
                    return data
                else:
                    self.logger.error("Invalid camera locations format")
                    return {}
                
        except FileNotFoundError:
            self.logger.warning("Camera locations file not found")
            return {}
        except Exception as e:
            self.logger.error(f"Error loading camera locations: {e}")
            return {}

    def _setup_logging(self) -> logging.Logger:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('camera_feeds.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)

    async def init_session(self):
        """Initialize aiohttp session"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)

    async def close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None

    async def fetch_camera_data(self, start: int = 0, length: int = 100) -> Optional[Dict]:
        """
        Fetch camera data from NVROADS API with proper query parameters
        Args:
            start: Starting index for pagination
            length: Number of records to fetch
        """
        query = {
            "columns": [
                {"data": None, "name": ""},
                {"name": "sortOrder", "s": True},
                {"name": "region", "search": {"value": "Las Vegas Area"}, "s": True},
                {"name": "roadway", "s": True},
                {"data": 4, "name": ""}
            ],
            "order": [
                {"column": 1, "dir": "asc"},
                {"column": 2, "dir": "asc"},
                {"column": 3, "dir": "asc"}
            ],
            "start": start,
            "length": length,
            "search": {"value": ""}
        }

        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'X-Requested-With': 'XMLHttpRequest'
        }
        
        params = {
            "query": json.dumps(query),
            "lang": "en"
        }

        try:
            async with self.session.get(
                self.base_url, 
                params=params,
                headers=headers,
                ssl=False  # Add this if having SSL verification issues
            ) as response:
                if response.status == 200:  # Changed from status_code to status
                    return await response.json()
                else:
                    self.logger.error(f"API request failed with status {response.status}")
                    return None
        except Exception as e:
            self.logger.error(f"Error fetching camera data: {e}")
            return None

    async def update_feeds(self):
        """Update camera feeds from API using pagination"""
        await self.init_session()
        
        try:
            # First request to get total count
            initial_data = await self.fetch_camera_data(0, 700)
            if not initial_data:
                return
                
            total_records = initial_data.get('recordsTotal', 0)
            self.logger.info(f"Found {total_records} total cameras")
            
            # Fetch all cameras in batches
            all_cameras = []
            batch_size = 100
            for start in range(0, total_records, batch_size):
                batch_data = await self.fetch_camera_data(start, batch_size)
                if batch_data and 'data' in batch_data:
                    all_cameras.extend(batch_data['data'])
                    self.logger.info(f"Fetched cameras {start} to {start + len(batch_data['data'])}")
                else:
                    self.logger.error(f"Failed to fetch batch starting at {start}")
                    
            # Process camera data and prepare for JSON
            camera_locations = {}
            for camera in all_cameras:
                try:
                    location = camera.get('location', '')
                    if not location:
                        continue

                    # Extract coordinates and video URL
                    lat, lng = 0.0, 0.0
                    video_url = None
                    
                    if 'latLng' in camera and 'geography' in camera['latLng']:
                        wkt = camera['latLng']['geography']['wellKnownText']
                        coords = wkt.replace('POINT (', '').replace(')', '').split()
                        if len(coords) == 2:
                            lng, lat = map(float, coords)
                    
                    # Extract video URL from images array
                    for image in camera.get('images', []):
                        if image.get('videoUrl'):
                            video_url = image['videoUrl']
                            break

                    if lat != 0.0 and lng != 0.0:
                        camera_locations[location] = {
                            "lat": lat,
                            "lng": lng,
                            "name": location,
                            "roadway": camera.get('roadway', ''),
                            "direction": camera.get('direction', ''),
                            "id": camera.get('id', ''),
                            "url": video_url
                        }

                except Exception as e:
                    self.logger.error(f"Error processing camera for JSON: {e}")
                    continue

            # Save to JSON file if we have data
            if camera_locations:
                try:
                    with open('config/camera_locations.json', 'w') as f:
                        json.dump(camera_locations, f, indent=4, sort_keys=True)
                    self.logger.info(f"Saved {len(camera_locations)} camera locations to JSON")
                except Exception as e:
                    self.logger.error(f"Error saving camera locations to JSON: {e}")

            # Process camera data
            for camera in all_cameras:
                try:
                    camera_id = camera.get('id')
                    if not camera_id:
                        continue
                        
                    # Extract video URL from images array
                    video_url = None
                    for image in camera.get('images', []):
                        if image.get('videoUrl'):
                            video_url = image['videoUrl']
                            break
                            
                    if not video_url:
                        continue
                        
                    location = camera.get('location', '')
                    self.feeds[location] = {
                        'url': video_url,
                        'id': camera_id,
                        'roadway': camera.get('roadway', ''),
                        'direction': camera.get('direction', 0)
                    }
                    
                except Exception as e:
                    self.logger.error(f"Error processing camera data: {e}")
                    continue
                    
            self.logger.info(f"Successfully loaded {len(self.feeds)} camera feeds")
            self.logger.info(self.feeds)
            

            
        except Exception as e:
            self.logger.error(f"Error updating feeds: {e}")
        finally:
            await self.close_session()

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
        """Select strategic camera locations."""
        try:
            # First try to use validated feeds
            if self.validated_feeds:
                self.logger.info("Using pre-validated camera feeds")
                strategic_cameras = {}
                for name, camera in self.validated_feeds.items():
                    strategic_cameras[name] = {
                        'location': {
                            'lat': camera['lat'],
                            'lng': camera['lng']
                        },
                        'name': camera['name'],
                        'url': camera.get('url', '')  # Add URL from new JSON format
                    }
                    # Add optional fields if they exist
                    if 'roadway' in camera:
                        strategic_cameras[name]['roadway'] = camera['roadway']
                    if 'direction' in camera:
                        strategic_cameras[name]['direction'] = camera['direction']
                    if 'id' in camera:
                        strategic_cameras[name]['id'] = camera['id']
                return strategic_cameras

            # If no validated feeds, try real camera feeds
            feeds = self.feeds.copy()
            if feeds:
                # Transform feed data to include required location info
                strategic_cameras = {}
                for name, feed in feeds.items():
                    try:
                        location = feed.get('location', '')
                        lat, lng = self._extract_coordinates(location)
                        
                        if lat and lng:
                            strategic_cameras[name] = {
                                'url': feed['url'],
                                'location': {
                                    'lat': lat,
                                    'lng': lng
                                },
                                'id': feed.get('id', ''),
                                'roadway': feed.get('roadway', ''),
                                'name': name
                            }
                    except Exception as e:
                        self.logger.error(f"Error processing camera {name}: {e}")
                        continue

                if strategic_cameras:
                    return strategic_cameras

            # Fall back to test data if no other sources available
            self.logger.warning("No real cameras found, using test data")
            return {
                "Camera-NW-01": {
                    "location": {"lat": 36.2159, "lng": -115.2351},
                    "url": "https://camera1.stream"
                },
                "Camera-NE-02": {
                    "location": {"lat": 36.2159, "lng": -115.0351},
                    "url": "https://camera2.stream"
                },
                "Camera-SW-03": {
                    "location": {"lat": 36.1159, "lng": -115.2351},
                    "url": "https://camera3.stream"
                },
                "Camera-SE-04": {
                    "location": {"lat": 36.1159, "lng": -115.0351},
                    "url": "https://camera4.stream"
                }
            }
                
        except Exception as e:
            self.logger.error(f"Error selecting strategic cameras: {e}")
            return {}

    def _extract_coordinates(self, location_str: str) -> tuple[float, float]:
        """Extract coordinates from location string."""
        try:
            # Handle WKT format: 'POINT (-115.1398 36.1699)'
            if location_str.startswith('POINT'):
                coords = location_str.replace('POINT (', '').replace(')', '').split()
                if len(coords) == 2:
                    return float(coords[1]), float(coords[0])  # lat, lng
                    
            # Handle direct lat/lng object
            elif isinstance(location_str, dict):
                lat = location_str.get('lat')
                lng = location_str.get('lng')
                if lat is not None and lng is not None:
                    return float(lat), float(lng)
                    
        except Exception as e:
            self.logger.debug(f"Could not extract coordinates from {location_str}: {e}")
            
        return 0.0, 0.0  # Default to origin if parsing fails

    async def fetch_available_feeds(self) -> Dict[str, Dict]:
        """Return already validated feeds instead of re-fetching"""
        return self.validated_feeds

    def fetch_available_feeds_sync(self) -> Dict[str, Dict]:
        """
        Synchronous version of fetch_available_feeds using requests library instead of asyncio
        Returns:
            Dict[str, Dict]: Dictionary of camera feeds with location as key
        """
        try:
            # First request to get total count
            initial_data = self._fetch_camera_data_sync(0, 10)
            if not initial_data:
                return {}
                
            total_records = initial_data.get('recordsTotal', 0)
            self.logger.info(f"Found {total_records} total cameras")
            
            # Fetch all cameras in batches
            all_cameras = []
            batch_size = 100
            for start in range(0, total_records, batch_size):
                batch_data = self._fetch_camera_data_sync(start, batch_size)
                if batch_data and 'data' in batch_data:
                    all_cameras.extend(batch_data['data'])
                    self.logger.info(f"Fetched cameras {start} to {start + len(batch_data['data'])}")
                else:
                    self.logger.error(f"Failed to fetch batch starting at {start}")
            
            # Process camera data
            feeds = {}
            for camera in all_cameras:
                try:
                    camera_id = camera.get('id')
                    if not camera_id:
                        continue
                        
                    # Extract video URL from images array
                    video_url = None
                    for image in camera.get('images', []):
                        if image.get('videoUrl'):
                            video_url = image['videoUrl']
                            break
                            
                    if not video_url:
                        continue
                        
                    location = camera.get('location', '')
                    feeds[location] = {
                        'url': video_url,
                        'id': camera_id,
                        'roadway': camera.get('roadway', ''),
                        'direction': camera.get('direction', 0)
                    }
                    
                except Exception as e:
                    self.logger.error(f"Error processing camera data: {e}")
                    continue
                    
            self.logger.info(f"Successfully loaded {len(feeds)} camera feeds")
            return feeds
            
        except Exception as e:
            self.logger.error(f"Error in fetch_available_feeds_sync: {e}")
            return {}

    def _fetch_camera_data_sync(self, start: int = 0, length: int = 10) -> Optional[Dict]:
        """
        Synchronous version of fetch_camera_data using requests
        Args:
            start: Starting index for pagination
            length: Number of records to fetch
        """
        query = {
            "columns": [
                {"data": None, "name": ""},
                {"name": "sortOrder", "s": True},
                {"name": "region", "search": {"value": "Las Vegas Area"}, "s": True},
                {"name": "roadway", "s": True},
                {"data": 4, "name": ""}
            ],
            "order": [
                {"column": 1, "dir": "asc"},
                {"column": 2, "dir": "asc"},
                {"column": 3, "dir": "asc"}
            ],
            "start": start,
            "length": length,
            "search": {"value": ""}
        }

        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'X-Requested-With': 'XMLHttpRequest'
        }
        
        params = {
            "query": json.dumps(query),
            "lang": "en"
        }

        try:
            response = requests.get(
                self.base_url,
                params=params,
                headers=headers,
                timeout=30
            )
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"API request failed with status {response.status_code}")
                return None
        except Exception as e:
            self.logger.error(f"Error fetching camera data: {e}")
            return None

    def _transform_camera_data(self, api_data):
        """Transform raw API camera data into simplified location format"""
        locations = {}
        
        if not isinstance(api_data, dict) or 'data' not in api_data:
            self.logger.error(f"Invalid API data format: {type(api_data)}")
            return {}
        
        for camera in api_data.get('data', []):
            name = camera.get('location')
            if not name:
                continue
            
            if not isinstance(camera, dict):
                self.logger.error(f"Invalid camera data type: {type(camera)}")
                continue
            
            if 'latLng' not in camera:
                self.logger.debug(f"No location data for camera: {name}")
                continue
            
            try:
                # Extract lat/lng from wellKnownText format
                wkt = camera['latLng']['geography']['wellKnownText']
                coords = wkt.replace('POINT (', '').replace(')', '').split()
                if len(coords) == 2:
                    lng, lat = map(float, coords)
                    
                    # Extract video URL
                    video_url = None
                    for image in camera.get('images', []):
                        if image.get('videoUrl'):
                            video_url = image['videoUrl']
                            break
                    
                    # Format to match CameraLocation dataclass structure
                    locations[name] = {
                        "lat": float(lat),
                        "lng": float(lng),
                        "name": str(name),
                        "url": video_url
                    }
                    self.logger.debug(f"Processed camera location: {name} -> {locations[name]}")
            except Exception as e:
                self.logger.error(f"Error parsing camera location for {name}: {e}")
                continue
        
        # Verify data before saving
        if not locations:
            self.logger.warning("No camera locations were transformed")
            return {}
        
        # Save transformed data
        try:
            self.logger.info(f"Saving {len(locations)} camera locations")
            with open('config/camera_locations.json', 'w') as f:
                json.dump(locations, f, indent=4, sort_keys=True)
            
            # Verify saved data
            with open('config/camera_locations.json', 'r') as f:
                saved_data = json.load(f)
                if saved_data != locations:
                    self.logger.error("Saved data does not match transformed data")
                else:
                    self.logger.info("Camera locations saved successfully")
                
        except Exception as e:
            self.logger.error(f"Error saving camera locations: {e}")
        
        return locations 

    def verify_camera_locations(self):
        """Debug method to verify camera locations"""
        self.logger.info("\nVerifying camera locations:")
        self.logger.info(f"Total cameras with location data: {len(self.camera_locations)}")
        
        # Print first few cameras as sample
        for i, (name, location) in enumerate(self.camera_locations.items()):
            if i >= 5:  # Only show first 5
                break
            self.logger.info(
                f"Camera: {name}\n"
                f"  Lat: {location['lat']}\n"
                f"  Lng: {location['lng']}"
            )

    def verify_feeds(self):
        """No need to re-verify since feeds are validated during initialization"""
        pass

def open_video_stream(url):
    """
    Open a video stream with proper URL handling and fallback mechanisms
    Args:
        url: The original camera URL
    Returns:
        cv2.VideoCapture object or None if failed
    """
    logger = logging.getLogger(__name__)
    
    # Validate and clean URL
    if not url:
        logger.warning("Empty URL provided")
        return None
        
    # Convert nvroads.com URLs to its.nv.gov format
    if 'nvroads.com/cameras' in url:
        # Extract camera ID and server info from URL
        try:
            camera_id = url.split('/cameras/')[1].split('/')[0]
        except IndexError:
            logger.error(f"Failed to extract camera ID from URL: {url}")
            return None
            
        # Try to extract server number if present in the URL
        server_num = None
        if 'xcd' in url:
            try:
                server_num = url.split('xcd')[1][:2]  # Extract "01", "05" etc
            except:
                logger.debug(f"Could not extract server number from URL: {url}")
                pass
        
        # Construct proper its.nv.gov URL with fallbacks
        if server_num:
            # Try the specific server first
            stream_url = f"https://d1wse{server_num}.its.nv.gov/vegasxcd{server_num}/{camera_id}_lvflirxcd{server_num}_public.stream/playlist.m3u8"
        else:
            # Default to server 5 if no specific server found
            stream_url = f"https://d1wse5.its.nv.gov/vegasxcd05/{camera_id}_lvflirxcd05_public.stream/playlist.m3u8"
    
    # For HLS streams (m3u8)
    elif 'playlist.m3u8' in url:
        # Remove explicit port and clean up URL
        stream_url = url.replace(':443', '')
        
        # Fix inconsistent server numbers in URL
        try:
            # Extract server numbers from URL
            wse_num = stream_url.split('wse')[1][0]
            xcd_nums = [x[3:5] for x in stream_url.split('xcd')[1:]]
            
            # If there's inconsistency in server numbers, use the first one
            if len(set(xcd_nums)) > 1 or wse_num != xcd_nums[0]:
                logger.warning(f"Inconsistent server numbers in URL: wse{wse_num}, xcd{xcd_nums}")
                # Use the wse number for consistency
                fixed_url = stream_url
                for xcd_num in xcd_nums:
                    fixed_url = fixed_url.replace(f'xcd{xcd_num}', f'xcd{wse_num}')
                stream_url = fixed_url
        except Exception as e:
            logger.debug(f"Error while trying to fix server numbers: {e}")
            
    elif url.startswith('rtsp://'):
        stream_url = url
    else:
        stream_url = url
    
    logger.debug(f"Attempting to open stream: {stream_url}")
    
    # Try different methods to open the stream
    cap = None
    
    # Try with ffmpeg backend first (more reliable for streaming)
    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
    if cap is not None and cap.isOpened():
        logger.debug("Successfully opened stream with FFMPEG backend")
        return cap
        
    # Fallback to default backend
    cap = cv2.VideoCapture(stream_url)
    if cap is not None and cap.isOpened():
        logger.debug("Successfully opened stream with default backend")
        return cap
        
    # If original stream failed and it's from its.nv.gov, try alternate servers
    if 'its.nv.gov' in stream_url:
        for server_num in ['05', '01', '02', '03', '04']:
            try:
                alt_url = stream_url.replace(stream_url.split('wse')[1][:1], server_num)
                alt_url = alt_url.replace(f'xcd{stream_url.split("xcd")[1][:2]}', f'xcd{server_num}')
                logger.debug(f"Trying alternate server {server_num}: {alt_url}")
                
                cap = cv2.VideoCapture(alt_url, cv2.CAP_FFMPEG)
                if cap is not None and cap.isOpened():
                    logger.info(f"Successfully opened stream using alternate server {server_num}")
                    return cap
            except Exception as e:
                logger.debug(f"Failed to open stream with server {server_num}: {e}")
                continue
    
    # If all attempts failed, try streamlink as last resort
    if cap is None or not cap.isOpened():
        logger.debug("Attempting to open stream with streamlink")
        cap = open_stream_alternative(stream_url)
        if cap is not None and cap.isOpened():
            logger.debug("Successfully opened stream with streamlink")
            return cap
    
    logger.error(f"Failed to open stream: {url}")
    return None

def check_ffmpeg_support():
    try:
        # Try to open a test stream with FFMPEG backend using Las Vegas camera feed
        test_url = "https://d1wse5.its.nv.gov/vegasxcd05/b9f374d4-513b-463e-9114-14da5c7e13f4_lvflirxcd05_public.stream/playlist.m3u8"
        test_cap = cv2.VideoCapture(test_url, cv2.CAP_FFMPEG)
        has_ffmpeg = test_cap is not None and test_cap.isOpened()
        
        if test_cap is not None:
            test_cap.release()
            
        if not has_ffmpeg:
            print("Warning: FFmpeg backend not available or cannot open Las Vegas test stream")
            
        return has_ffmpeg
    except Exception as e:
        print(f"Warning: Error checking FFmpeg support: {e}")
        return False

# Make the check less strict - just warn instead of raising an error
if not check_ffmpeg_support():
    print("Warning: FFmpeg support may not be available - some streams might not work")

def open_stream_alternative(url):
    try:
        import streamlink
        streams = streamlink.streams(url)
        if streams:
            # Get the best quality stream URL
            stream_url = streams['best'].url
            return cv2.VideoCapture(stream_url)
    except ImportError:
        print("streamlink not installed. Try: pip install streamlink")
    return None