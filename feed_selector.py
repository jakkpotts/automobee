import aiohttp
import asyncio
import json
import logging
import cv2
import requests
from typing import Dict, List, Optional

class CameraFeedSelector:
    def __init__(self):
        self.feeds = {}
        self.session = None
        self.logger = self._setup_logging()
        self.base_url = "https://www.nvroads.com/List/GetData/Cameras"
        self.camera_locations = self._load_camera_locations()
        
    def _load_camera_locations(self) -> Dict[str, Dict]:
        """Load camera locations from config file"""
        try:
            with open('config/camera_locations.json') as f:
                return json.load(f)
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

    async def fetch_camera_data(self, start: int = 0, length: int = 10) -> Optional[Dict]:
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
                headers=headers
            ) as response:
                if response.status == 200:
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
            initial_data = await self.fetch_camera_data(0, 10)
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
        """
        Selects strategic cameras based on:
        1. Coverage - Ensure even distribution across the network
        2. Traffic Volume - Prioritize high-traffic areas
        3. Entry/Exit Points - Cover major routes in/out of the city
        4. Intersection Types - Focus on major intersections
        5. Historical Data - Consider past detection success rates
        """
        strategic_cameras = {}
        
        # Key areas to monitor (with weights)
        priority_areas = {
            'highways': {
                'patterns': ['I-15', 'US 95', 'I-515', 'I-215'],
                'weight': 1.0  # Highest priority
            },
            'major_roads': {
                'patterns': ['Las Vegas Blvd', 'Sahara', 'Charleston', 'Tropicana', 'Flamingo', 'Eastern'],
                'weight': 0.8
            },
            'entry_points': {
                'patterns': ['Blue Diamond', 'Lake Mead', 'Summerlin', 'Boulder Hwy'],
                'weight': 0.9
            }
        }
        
        # Track coverage to ensure even distribution
        coverage_grid = {}  # Grid sectors to track camera distribution
        
        for name, camera in self.feeds.items():
            score = 0.0
            matches = []
            
            # Skip cameras without location data
            if not camera.get('location'):
                continue
                
            location = camera['location']
            lat, lng = location.get('lat', 0), location.get('lng', 0)
            
            # Create grid sector key (rough 1km squares)
            grid_key = f"{int(lat * 100)},{int(lng * 100)}"
            if grid_key not in coverage_grid:
                coverage_grid[grid_key] = 0
                score += 0.2  # Bonus for covering new areas
            
            # Calculate priority score based on location matches
            for area_type, data in priority_areas.items():
                for pattern in data['patterns']:
                    if pattern.lower() in name.lower():
                        score += data['weight']
                        matches.append(area_type)
            
            # Adjust score based on coverage distribution
            coverage_grid[grid_key] += 1
            if coverage_grid[grid_key] > 1:
                score *= 0.8  # Reduce score if area already covered
            
            # Add camera if it meets minimum score threshold
            if score >= 0.5:  # Threshold for selection
                strategic_cameras[name] = camera
                strategic_cameras[name]['strategic_score'] = score
                strategic_cameras[name]['priority_matches'] = matches
        
        # Ensure we have a reasonable number of cameras
        max_cameras = 20  # Adjust based on system capacity
        if len(strategic_cameras) > max_cameras:
            # Sort by score and take top cameras
            sorted_cameras = dict(sorted(
                strategic_cameras.items(),
                key=lambda x: x[1]['strategic_score'],
                reverse=True
            )[:max_cameras])
            strategic_cameras = sorted_cameras
        
        # Log selected cameras and their strategic importance
        self.logger.info(f"\nSelected {len(strategic_cameras)} strategic cameras:")
        for name, data in strategic_cameras.items():
            self.logger.info(
                f"- {name}: Score={data['strategic_score']:.2f}, "
                f"Priorities={', '.join(data['priority_matches'])}"
            )
        
        return strategic_cameras

    async def fetch_available_feeds(self) -> Dict[str, Dict]:
        """
        Fetch all available camera feeds from NVROADS API
        Returns:
            Dict[str, Dict]: Dictionary of camera feeds with location as key
        """
        await self.update_feeds()
        return self.feeds

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