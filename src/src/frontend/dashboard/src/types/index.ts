export interface Camera {
  id: string;
  name: string;
  location: {
    lat: number;
    lng: number;
  };
  status: 'active' | 'connecting' | 'error';
  stream_url: string;
}

export interface Detection {
  id: string;
  camera_id: string;
  timestamp: string;
  vehicle_type: string;
  make: string;
  model: string;
  confidence: number;
  location: {
    lat: number;
    lng: number;
  };
}

export interface SystemMetrics {
  totalDetections: number;
  activeCameras: number;
  detectionRate: number;
  cpuUsage: number;
  memoryUsage: number;
  gpuUtilization: number;
}

export interface Alert {
  id: string;
  title: string;
  message: string;
  type: 'info' | 'warning' | 'error';
  timestamp: string;
  isRead: boolean;
} 