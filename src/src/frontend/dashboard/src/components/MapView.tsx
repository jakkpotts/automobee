import { useEffect, useRef } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import type { Camera, Detection } from '../types';

// Fix for default markers in React-Leaflet
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: '/leaflet/marker-icon-2x.png',
  iconUrl: '/leaflet/marker-icon.png',
  shadowUrl: '/leaflet/marker-shadow.png',
});

interface MapViewProps {
  cameras: Camera[];
  detections: Detection[];
}

// Custom hook to fit bounds when markers change
function FitBounds({ cameras }: { cameras: Camera[] }) {
  const map = useMap();

  useEffect(() => {
    if (cameras.length > 0) {
      const bounds = L.latLngBounds(cameras.map(cam => [cam.location.lat, cam.location.lng]));
      map.fitBounds(bounds, { padding: [50, 50] });
    }
  }, [cameras, map]);

  return null;
}

// Function to create icon URL with fallback
function getIconUrl(iconName: string): string {
  // Try PNG first, fallback to SVG
  const pngPath = `/icons/${iconName}.png`;
  const svgPath = `/icons/${iconName}.svg`;
  
  // Check if PNG exists
  try {
    // This is a simple check - in production you might want a more robust solution
    const img = new Image();
    img.src = pngPath;
    return pngPath;
  } catch {
    return svgPath;
  }
}

export function MapView({ cameras, detections }: MapViewProps) {
  const mapRef = useRef<L.Map>(null);

  // Custom marker icons
  const cameraIcon = new L.Icon({
    iconUrl: getIconUrl('camera-icon'),
    iconSize: [32, 32],
    iconAnchor: [16, 32],
    popupAnchor: [0, -32],
  });

  const detectionIcon = new L.Icon({
    iconUrl: getIconUrl('detection-icon'),
    iconSize: [24, 24],
    iconAnchor: [12, 24],
    popupAnchor: [0, -24],
  });

  return (
    <div className="h-full w-full">
      <MapContainer
        ref={mapRef}
        center={[0, 0]}
        zoom={2}
        className="h-full w-full"
        style={{ background: '#f8fafc' }}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        <FitBounds cameras={cameras} />

        {/* Render Cameras */}
        {cameras.map((camera) => (
          <Marker
            key={camera.id}
            position={[camera.location.lat, camera.location.lng]}
            icon={cameraIcon}
          >
            <Popup>
              <div className="p-2">
                <h3 className="font-semibold">{camera.name}</h3>
                <p className="text-sm text-gray-600">Status: {camera.status}</p>
                <button
                  className="mt-2 px-3 py-1 bg-blue-500 text-white rounded-md text-sm hover:bg-blue-600"
                  onClick={() => {
                    // Handle view stream
                  }}
                >
                  View Stream
                </button>
              </div>
            </Popup>
          </Marker>
        ))}

        {/* Render Recent Detections */}
        {detections.map((detection) => (
          <Marker
            key={detection.id}
            position={[detection.location.lat, detection.location.lng]}
            icon={detectionIcon}
          >
            <Popup>
              <div className="p-2">
                <h3 className="font-semibold">{detection.vehicle_type}</h3>
                <p className="text-sm text-gray-600">
                  Make: {detection.make}<br />
                  Model: {detection.model}<br />
                  Confidence: {(detection.confidence * 100).toFixed(1)}%
                </p>
              </div>
            </Popup>
          </Marker>
        ))}
      </MapContainer>
    </div>
  );
} 