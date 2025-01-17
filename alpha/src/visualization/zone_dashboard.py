import folium
from dataclasses import dataclass
from typing import Dict, List
from ..zones.zone_manager import ZoneManager, ZoneType, ZoneDefinition
from ..camera.types import StreamStatus

@dataclass
class ZoneMetrics:
    """Metrics for zone health monitoring."""
    total_streams: int
    active_streams: int
    error_streams: int
    offline_streams: int
    processing_load: float  # Percentage of max capacity
    average_processing_time: float

class ZoneDashboard:
    """Generates real-time visualization of zone status and metrics."""
    
    # Color schemes for different states
    STATUS_COLORS = {
        StreamStatus.ACTIVE: '#28a745',     # Green
        StreamStatus.ERROR: '#dc3545',      # Red
        StreamStatus.OFFLINE: '#6c757d',    # Gray
        StreamStatus.INITIALIZING: '#ffc107' # Yellow
    }
    
    def __init__(self, zone_manager: ZoneManager):
        self.zone_manager = zone_manager
        
    def generate_map(self, output_path: str = "zone_status.html"):
        """Generate an interactive map showing zones and camera status."""
        # Center the map on Las Vegas
        m = folium.Map(
            location=[36.1699, -115.1398],
            zoom_start=11,
            tiles='CartoDB positron'
        )
        
        # Add zones as circles
        for zone_def in self.zone_manager.ZONE_DEFINITIONS:
            self._add_zone_circle(m, zone_def)
            
        # Add camera markers
        for stream_id, stream in self.zone_manager.streams.items():
            self._add_camera_marker(m, stream)
            
        m.save(output_path)
        
    def _add_zone_circle(self, m: folium.Map, zone_def: ZoneDefinition):
        """Add a zone circle to the map."""
        metrics = self._get_zone_metrics(zone_def.name)
        
        # Create circle with tooltip showing metrics
        folium.Circle(
            location=[zone_def.center_lat, zone_def.center_lng],
            radius=zone_def.radius_km * 1000,  # Convert km to meters
            color='#3388ff',
            fill=True,
            fillOpacity=0.2,
            popup=self._create_zone_popup(zone_def, metrics),
            tooltip=f"{zone_def.name.value}: {metrics.active_streams}/{metrics.total_streams} active"
        ).add_to(m)
        
    def _create_zone_popup(self, zone_def: ZoneDefinition, metrics: ZoneMetrics) -> str:
        """Create popup content with safe division."""
        error_rate = (metrics.error_streams / metrics.total_streams * 100) if metrics.total_streams > 0 else 0
        return f"""
            <h3>{zone_def.name}</h3>
            <p>Total Streams: {metrics.total_streams}</p>
            <p>Active Streams: {metrics.active_streams}</p>
            <p>Error Rate: {error_rate:.1f}%</p>
        """
        
    def _add_camera_marker(self, m: folium.Map, stream):
        """Add a camera marker to the map."""
        color = self.STATUS_COLORS.get(stream.status, '#6c757d')
        
        folium.CircleMarker(
            location=[stream.lat, stream.lng],
            radius=6,
            color=color,
            fill=True,
            fillOpacity=0.7,
            popup=self._create_camera_popup(stream),
            tooltip=f"{stream.name}: {stream.status.value}"
        ).add_to(m)
        
    def _create_camera_popup(self, stream) -> str:
        """Create HTML popup content for camera."""
        return f"""
            <div style="width:200px">
                <h4>{stream.name}</h4>
                <p>Location: {stream.location}</p>
                <p>Status: {stream.status.value}</p>
                <p>Stream ID: {stream.id}</p>
                <img src="{stream.image_url}" style="width:100%;max-width:200px">
                <button onclick="window.open('{stream.video_url}', '_blank')">
                    View Live Stream
                </button>
            </div>
        """
        
    def _get_zone_metrics(self, zone_type: ZoneType) -> ZoneMetrics:
        """Calculate metrics for a zone."""
        streams = self.zone_manager.get_zone_streams(zone_type)
        total = len(streams)
        
        if total == 0:
            return ZoneMetrics(0, 0, 0, 0, 0.0, 0.0)
            
        active = sum(1 for sid in streams 
                    if self.zone_manager.stream_manager.streams[sid].status == StreamStatus.ACTIVE)
        error = sum(1 for sid in streams 
                   if self.zone_manager.stream_manager.streams[sid].status == StreamStatus.ERROR)
        offline = sum(1 for sid in streams 
                     if self.zone_manager.stream_manager.streams[sid].status == StreamStatus.OFFLINE)
        
        # Get processing metrics from scheduler
        scheduler = self.zone_manager.stream_manager.scheduler
        zone_load = scheduler.get_zone_load(zone_type) if scheduler else 0.0
        avg_time = scheduler.get_zone_average_processing_time(zone_type) if scheduler else 0.0
        
        return ZoneMetrics(
            total_streams=total,
            active_streams=active,
            error_streams=error,
            offline_streams=offline,
            processing_load=zone_load,
            average_processing_time=avg_time
        ) 