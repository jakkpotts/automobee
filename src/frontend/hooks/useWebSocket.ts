import { useEffect, useRef, useState, useCallback } from 'react';
import { WebSocketClient, WebSocketMessage, HealthMetrics, Detection, StreamState } from '../services/websocket';

interface WebSocketHookOptions {
    url: string;
    authToken?: string;
    autoReconnect?: boolean;
    subscriptions?: string[];
    onMessage?: (message: WebSocketMessage) => void;
    onHealthUpdate?: (metrics: HealthMetrics) => void;
    onDetection?: (data: { camera_id: string; detection: Detection }) => void;
    onStreamUpdate?: (data: { camera_id: string; state: StreamState }) => void;
    onError?: (error: Event) => void;
}

interface WebSocketHookResult {
    isConnected: boolean;
    lastMessage: WebSocketMessage | null;
    send: (message: Omit<WebSocketMessage, 'timestamp'>) => boolean;
    subscribe: (topics: string[]) => void;
    unsubscribe: (topics: string[]) => void;
    reconnect: () => void;
}

export function useWebSocket({
    url,
    authToken,
    autoReconnect = true,
    subscriptions = [],
    onMessage,
    onHealthUpdate,
    onDetection,
    onStreamUpdate,
    onError
}: WebSocketHookOptions): WebSocketHookResult {
    const wsRef = useRef<WebSocketClient | null>(null);
    const [isConnected, setIsConnected] = useState(false);
    const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);

    // Memoize callbacks to prevent unnecessary re-renders
    const memoizedCallbacks = {
        onMessage: useCallback(onMessage ?? (() => {}), [onMessage]),
        onHealthUpdate: useCallback(onHealthUpdate ?? (() => {}), [onHealthUpdate]),
        onDetection: useCallback(onDetection ?? (() => {}), [onDetection]),
        onStreamUpdate: useCallback(onStreamUpdate ?? (() => {}), [onStreamUpdate]),
        onError: useCallback(onError ?? (() => {}), [onError])
    };

    // Initialize WebSocket client
    useEffect(() => {
        wsRef.current = new WebSocketClient(url, authToken);
        const ws = wsRef.current;

        // Setup event handlers
        ws.on('connected', () => {
            setIsConnected(true);
            if (subscriptions.length > 0) {
                ws.subscribe(subscriptions);
            }
        });

        ws.on('disconnected', () => {
            setIsConnected(false);
        });

        ws.on('message', (message: WebSocketMessage) => {
            setLastMessage(message);
            memoizedCallbacks.onMessage(message);
        });

        ws.on('healthUpdate', (metrics: HealthMetrics) => {
            memoizedCallbacks.onHealthUpdate(metrics);
        });

        ws.on('detection', (data: { camera_id: string; detection: Detection }) => {
            memoizedCallbacks.onDetection(data);
        });

        ws.on('streamUpdate', (data: { camera_id: string; state: StreamState }) => {
            memoizedCallbacks.onStreamUpdate(data);
        });

        ws.on('error', (error: Event) => {
            memoizedCallbacks.onError(error);
        });

        // Cleanup on unmount
        return () => {
            ws.close();
            wsRef.current = null;
        };
    }, [
        url,
        authToken,
        subscriptions,
        memoizedCallbacks.onMessage,
        memoizedCallbacks.onHealthUpdate,
        memoizedCallbacks.onDetection,
        memoizedCallbacks.onStreamUpdate,
        memoizedCallbacks.onError
    ]);

    // Subscribe to topics when subscriptions prop changes
    useEffect(() => {
        if (wsRef.current && isConnected && subscriptions.length > 0) {
            wsRef.current.subscribe(subscriptions);
        }
    }, [subscriptions, isConnected]);

    // Send message wrapper
    const send = useCallback((message: Omit<WebSocketMessage, 'timestamp'>): boolean => {
        return wsRef.current?.send(message) ?? false;
    }, []);

    // Subscribe wrapper
    const subscribe = useCallback((topics: string[]): void => {
        wsRef.current?.subscribe(topics);
    }, []);

    // Unsubscribe wrapper
    const unsubscribe = useCallback((topics: string[]): void => {
        wsRef.current?.unsubscribe(topics);
    }, []);

    // Reconnect wrapper
    const reconnect = useCallback((): void => {
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = new WebSocketClient(url, authToken);
        }
    }, [url, authToken]);

    return {
        isConnected,
        lastMessage,
        send,
        subscribe,
        unsubscribe,
        reconnect
    };
}

// Example usage:
/*
function MyComponent() {
    const {
        isConnected,
        send,
        subscribe
    } = useWebSocket({
        url: 'ws://localhost:8765/ws',
        authToken: 'my-token',
        subscriptions: ['health_update', 'detection'],
        onHealthUpdate: (metrics) => {
            console.log('Health metrics:', metrics);
        },
        onDetection: ({ camera_id, detection }) => {
            console.log(`Detection on camera ${camera_id}:`, detection);
        }
    });

    useEffect(() => {
        if (isConnected) {
            console.log('Connected to WebSocket server');
        }
    }, [isConnected]);

    return (
        <div>
            <p>Connection status: {isConnected ? 'Connected' : 'Disconnected'}</p>
            <button onClick={() => subscribe(['new_topic'])}>
                Subscribe to new topic
            </button>
        </div>
    );
}
*/ 