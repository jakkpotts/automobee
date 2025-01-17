import { toast } from 'react-hot-toast';
import type { Camera, Detection, SystemMetrics } from '../types';

type WebSocketMessage = {
  type: 'camera_update' | 'detection' | 'metrics' | 'alert';
  payload: any;
};

type WebSocketCallbacks = {
  onCameraUpdate?: (cameras: Camera[]) => void;
  onDetection?: (detection: Detection) => void;
  onMetricsUpdate?: (metrics: SystemMetrics) => void;
};

class WebSocketService {
  private ws: WebSocket | null = null;
  private callbacks: WebSocketCallbacks = {};
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectTimeout = 1000; // Start with 1 second

  constructor(private url: string) {}

  connect(callbacks: WebSocketCallbacks) {
    this.callbacks = callbacks;
    this.establishConnection();
  }

  private establishConnection() {
    try {
      this.ws = new WebSocket(this.url);
      this.setupEventListeners();
    } catch (error) {
      console.error('WebSocket connection error:', error);
      this.handleReconnect();
    }
  }

  private setupEventListeners() {
    if (!this.ws) return;

    this.ws.onopen = () => {
      console.log('WebSocket connected');
      this.reconnectAttempts = 0;
      this.reconnectTimeout = 1000;
      toast.success('Connected to server');
    };

    this.ws.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data);
        this.handleMessage(message);
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };

    this.ws.onclose = () => {
      console.log('WebSocket disconnected');
      toast.error('Disconnected from server');
      this.handleReconnect();
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      this.ws?.close();
    };
  }

  private handleMessage(message: WebSocketMessage) {
    switch (message.type) {
      case 'camera_update':
        this.callbacks.onCameraUpdate?.(message.payload);
        break;
      case 'detection':
        this.callbacks.onDetection?.(message.payload);
        break;
      case 'metrics':
        this.callbacks.onMetricsUpdate?.(message.payload);
        break;
      case 'alert':
        toast(message.payload.message, {
          icon: message.payload.type === 'error' ? '🔴' :
                message.payload.type === 'warning' ? '⚠️' : 'ℹ️',
        });
        break;
      default:
        console.warn('Unknown message type:', message.type);
    }
  }

  private handleReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      toast.error('Could not reconnect to server. Please refresh the page.');
      return;
    }

    setTimeout(() => {
      this.reconnectAttempts++;
      this.reconnectTimeout *= 2; // Exponential backoff
      this.establishConnection();
    }, this.reconnectTimeout);
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  // Method to send messages to the server if needed
  send(type: string, payload: any) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type, payload }));
    } else {
      console.warn('WebSocket is not connected');
    }
  }
}

// Create a singleton instance
const wsService = new WebSocketService(
  `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws`
);

export default wsService; 