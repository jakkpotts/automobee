import { EventEmitter } from 'events';
import * as pako from 'pako';

// Message types
export interface WebSocketMessage<T = any> {
    type: string;
    data: T;
    timestamp: string;
}

export interface HealthMetrics {
    cpu_usage: number;
    memory_usage: {
        rss: number;
        vms: number;
        percent: number;
    };
    gpu_metrics: {
        memory_used_mb: number;
        memory_cached_mb: number;
        device_name: string;
    };
    detection_latency: number;
    active_streams: number;
    timestamp: string;
}

export interface Detection {
    type: string;
    confidence: number;
    bbox: number[];
    frame_id: number;
    metadata: {
        processing_device: string;
        model: string;
    };
}

export interface StreamState {
    url: string;
    fps: number;
    frame_count: number;
    last_frame_time: string | null;
    status: 'initializing' | 'running' | 'error';
    error: string | null;
}

export enum MessagePriority {
    HIGH = 0,
    MEDIUM = 1,
    LOW = 2
}

interface QueuedMessage {
    message: Omit<WebSocketMessage, 'timestamp'>;
    priority: MessagePriority;
    timestamp: number;
}

const PERSISTENT_QUEUE_KEY = 'websocket_persistent_queue';
const MAX_PERSISTENT_MESSAGES = 50;

interface PersistentQueueMessage extends QueuedMessage {
    persistUntil: number;  // Timestamp when message should expire
}

export class WebSocketClient extends EventEmitter {
    private ws: WebSocket | null = null;
    private reconnectAttempts = 0;
    private maxReconnectAttempts = 5;
    private reconnectDelay = 1000; // Start with 1 second
    private maxReconnectDelay = 30000; // Max 30 seconds
    private pingInterval = 30000; // 30 seconds
    private pingTimer: number | null = null;
    private subscriptions = new Set<string>();
    private isConnecting = false;
    private shouldReconnect = true;
    private messageQueue: QueuedMessage[] = [];
    private maxQueueSizeByPriority = {
        [MessagePriority.HIGH]: 100,   // Keep more high priority messages
        [MessagePriority.MEDIUM]: 50,  // Medium capacity for regular updates
        [MessagePriority.LOW]: 20      // Limited space for low priority
    };
    private persistentQueue: PersistentQueueMessage[] = [];
    private persistenceEnabled = true;
    private readonly persistentMessageTypes = new Set([
        // System Critical
        'authenticate',
        'error',
        'critical_alert',
        'system_config',
        
        // Security Related
        'auth_refresh',
        'permission_change',
        'access_revoked',
        'security_alert',
        
        // Configuration Changes
        'camera_config_update',
        'detection_config_update',
        'zone_config_update',
        'system_settings_update',
        
        // Critical Business Events
        'license_expiry',
        'storage_warning',
        'hardware_failure',
        'model_update_required',
        
        // Important User Actions
        'user_preferences',
        'saved_views',
        'custom_alerts',
        'notification_settings'
    ]);

    constructor(
        private url: string,
        private authToken?: string,
        private persistenceOptions = {
            enabled: true,
            maxAge: 24 * 60 * 60 * 1000, // 24 hours
            maxSize: MAX_PERSISTENT_MESSAGES
        }
    ) {
        super();
        this.persistenceEnabled = persistenceOptions.enabled;
        if (this.persistenceEnabled) {
            this.loadPersistedMessages();
        }
        this.connect();
    }

    /**
     * Connect to WebSocket server
     */
    private connect(): void {
        if (this.ws?.readyState === WebSocket.CONNECTING || this.isConnecting) {
            return;
        }

        this.isConnecting = true;

        try {
            this.ws = new WebSocket(this.url);
            this.setupEventHandlers();
        } catch (error) {
            console.error('WebSocket connection error:', error);
            this.handleConnectionError();
        }
    }

    /**
     * Setup WebSocket event handlers
     */
    private setupEventHandlers(): void {
        if (!this.ws) return;

        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.isConnecting = false;
            this.reconnectAttempts = 0;
            this.reconnectDelay = 1000;

            // Authenticate if token provided
            if (this.authToken) {
                this.send({
                    type: 'authenticate',
                    data: { token: this.authToken }
                });
            }

            // Resubscribe to topics
            if (this.subscriptions.size > 0) {
                this.send({
                    type: 'subscribe',
                    data: { topics: Array.from(this.subscriptions) }
                });
            }

            // Flush queued messages
            this.flushMessageQueue();

            // Start ping interval
            this.startPingInterval();

            this.emit('connected');
        };

        this.ws.onclose = (event) => {
            console.log('WebSocket closed:', event.code, event.reason);
            this.isConnecting = false;
            this.cleanup();

            if (this.shouldReconnect) {
                this.handleConnectionError();
            }

            this.emit('disconnected', event);
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.emit('error', error);
        };

        this.ws.onmessage = (event) => {
            try {
                let data = event.data;
                
                // Handle compressed data
                if (data instanceof Blob) {
                    this.handleBlobMessage(data);
                    return;
                }

                // Parse JSON message
                const message = JSON.parse(data) as WebSocketMessage;
                this.handleMessage(message);
            } catch (error) {
                console.error('Error handling message:', error);
            }
        };
    }

    /**
     * Decompress data using zlib
     */
    private decompress(data: Uint8Array): string {
        try {
            const decompressed = pako.inflate(data);
            return new TextDecoder().decode(decompressed);
        } catch (error) {
            console.error('Error decompressing data:', error);
            // Fallback to raw data if decompression fails
            return new TextDecoder().decode(data);
        }
    }

    /**
     * Compress data using zlib
     */
    private compress(data: string): Uint8Array {
        try {
            const compressed = pako.deflate(new TextEncoder().encode(data));
            return compressed;
        } catch (error) {
            console.error('Error compressing data:', error);
            // Fallback to raw data if compression fails
            return new TextEncoder().encode(data);
        }
    }

    /**
     * Handle blob messages (compressed data)
     */
    private async handleBlobMessage(blob: Blob): Promise<void> {
        try {
            // Convert blob to array buffer
            const arrayBuffer = await blob.arrayBuffer();
            
            // Decompress data
            const decompressed = this.decompress(new Uint8Array(arrayBuffer));
            
            // Parse JSON message
            const message = JSON.parse(decompressed) as WebSocketMessage;
            this.handleMessage(message);
        } catch (error) {
            console.error('Error handling blob message:', error);
        }
    }

    /**
     * Handle incoming messages
     */
    private handleMessage(message: WebSocketMessage): void {
        switch (message.type) {
            case 'pong':
                // Handle pong response
                break;

            case 'health_update':
                this.emit('healthUpdate', message.data as HealthMetrics);
                break;

            case 'detection':
                this.emit('detection', message.data as {
                    camera_id: string;
                    detection: Detection;
                });
                break;

            case 'stream_update':
                this.emit('streamUpdate', message.data as {
                    camera_id: string;
                    state: StreamState;
                });
                break;

            case 'stream_states':
                this.emit('streamStates', message.data as Record<string, StreamState>);
                break;

            case 'health_metrics':
                this.emit('healthMetrics', message.data as HealthMetrics);
                break;

            default:
                console.warn('Unknown message type:', message.type);
        }

        // Emit raw message for custom handling
        this.emit('message', message);
    }

    /**
     * Start ping interval
     */
    private startPingInterval(): void {
        this.stopPingInterval();
        this.pingTimer = window.setInterval(() => {
            this.send({ type: 'ping', data: {} });
        }, this.pingInterval);
    }

    /**
     * Stop ping interval
     */
    private stopPingInterval(): void {
        if (this.pingTimer !== null) {
            clearInterval(this.pingTimer);
            this.pingTimer = null;
        }
    }

    /**
     * Handle connection errors and reconnection
     */
    private handleConnectionError(): void {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.error('Max reconnection attempts reached');
            this.emit('maxReconnectAttemptsReached');
            return;
        }

        this.reconnectAttempts++;
        const delay = Math.min(
            this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1),
            this.maxReconnectDelay
        );

        console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
        setTimeout(() => this.connect(), delay);
    }

    /**
     * Get message priority based on type
     */
    private getMessagePriority(type: string): MessagePriority {
        switch (type) {
            // High priority messages
            case 'authenticate':
            case 'error':
            case 'subscribe':
                return MessagePriority.HIGH;

            // Medium priority messages
            case 'detection':
            case 'stream_update':
                return MessagePriority.MEDIUM;

            // Low priority messages
            case 'health_update':
            case 'ping':
            default:
                return MessagePriority.LOW;
        }
    }

    /**
     * Load persisted messages from localStorage
     */
    private loadPersistedMessages(): void {
        try {
            const persistedData = localStorage.getItem(PERSISTENT_QUEUE_KEY);
            if (persistedData) {
                const messages = JSON.parse(persistedData) as PersistentQueueMessage[];
                const now = Date.now();
                
                // Filter out expired messages
                this.persistentQueue = messages.filter(msg => msg.persistUntil > now);
                
                // Add valid messages to the regular queue
                this.persistentQueue.forEach(msg => {
                    this.queueMessage(msg.message, true);
                });
                
                // Update storage with cleaned up queue
                this.persistMessages();
            }
        } catch (error) {
            console.error('Error loading persisted messages:', error);
        }
    }

    /**
     * Persist messages to localStorage
     */
    private persistMessages(): void {
        if (!this.persistenceEnabled) return;

        try {
            const now = Date.now();
            // Clean up expired messages
            this.persistentQueue = this.persistentQueue.filter(msg => msg.persistUntil > now);
            
            // Ensure we don't exceed max size
            if (this.persistentQueue.length > this.persistenceOptions.maxSize) {
                this.persistentQueue = this.persistentQueue
                    .sort((a, b) => b.priority - a.priority || a.timestamp - b.timestamp)
                    .slice(0, this.persistenceOptions.maxSize);
            }

            localStorage.setItem(PERSISTENT_QUEUE_KEY, JSON.stringify(this.persistentQueue));
        } catch (error) {
            console.error('Error persisting messages:', error);
        }
    }

    /**
     * Check if message should be persisted
     */
    private shouldPersistMessage(message: Omit<WebSocketMessage, 'timestamp'>): boolean {
        return (
            this.persistenceEnabled &&
            this.persistentMessageTypes.has(message.type) &&
            message.data?.persist !== false // Allow opt-out via message data
        );
    }

    /**
     * Add message to queue with priority
     */
    private queueMessage(
        message: Omit<WebSocketMessage, 'timestamp'>,
        isPersisted = false
    ): boolean {
        const priority = this.getMessagePriority(message.type);
        
        // Handle message persistence
        if (!isPersisted && this.shouldPersistMessage(message)) {
            const persistentMessage: PersistentQueueMessage = {
                message,
                priority,
                timestamp: Date.now(),
                persistUntil: Date.now() + this.persistenceOptions.maxAge
            };
            this.persistentQueue.push(persistentMessage);
            this.persistMessages();
        }

        // Count messages of this priority
        const priorityCount = this.messageQueue.filter(m => m.priority === priority).length;
        
        // Check if we've reached the limit for this priority
        if (priorityCount >= this.maxQueueSizeByPriority[priority]) {
            // If high priority, try to remove a lower priority message
            if (priority === MessagePriority.HIGH) {
                // Find and remove the oldest lowest priority message
                const lowestPriorityIndex = this.messageQueue.findIndex(m => m.priority === MessagePriority.LOW);
                if (lowestPriorityIndex >= 0) {
                    this.messageQueue.splice(lowestPriorityIndex, 1);
                } else {
                    // Try medium priority if no low priority messages
                    const mediumPriorityIndex = this.messageQueue.findIndex(m => m.priority === MessagePriority.MEDIUM);
                    if (mediumPriorityIndex >= 0) {
                        this.messageQueue.splice(mediumPriorityIndex, 1);
                    } else {
                        return false; // Queue is full of high priority messages
                    }
                }
            } else {
                return false; // Queue is full for this priority
            }
        }

        // Add message to queue
        this.messageQueue.push({
            message,
            priority,
            timestamp: Date.now()
        });

        // Sort queue by priority and timestamp
        this.messageQueue.sort((a, b) => {
            if (a.priority !== b.priority) {
                return a.priority - b.priority; // Lower number = higher priority
            }
            return a.timestamp - b.timestamp; // Older messages first within same priority
        });

        return true;
    }

    /**
     * Send message to server
     */
    public send(message: Omit<WebSocketMessage, 'timestamp'>): boolean {
        if (this.ws?.readyState !== WebSocket.OPEN) {
            // Queue message if disconnected
            return this.queueMessage(message);
        }

        try {
            const fullMessage: WebSocketMessage = {
                ...message,
                timestamp: new Date().toISOString()
            };
            
            // Compress message before sending
            const compressed = this.compress(JSON.stringify(fullMessage));
            this.ws.send(compressed);
            return true;
        } catch (error) {
            console.error('Error sending message:', error);
            return this.queueMessage(message);
        }
    }

    /**
     * Flush queued messages
     */
    private flushMessageQueue(): void {
        // Messages are already sorted by priority and timestamp
        while (this.messageQueue.length > 0 && this.ws?.readyState === WebSocket.OPEN) {
            const queuedMessage = this.messageQueue.shift();
            if (queuedMessage) {
                this.send(queuedMessage.message);
            }
        }
    }

    /**
     * Get queue statistics including persistence info
     */
    public getQueueStats(): Record<string, number> {
        const stats = {
            [MessagePriority.HIGH]: this.messageQueue.filter(m => m.priority === MessagePriority.HIGH).length,
            [MessagePriority.MEDIUM]: this.messageQueue.filter(m => m.priority === MessagePriority.MEDIUM).length,
            [MessagePriority.LOW]: this.messageQueue.filter(m => m.priority === MessagePriority.LOW).length,
            persistentMessages: this.persistentQueue.length
        };
        return stats;
    }

    /**
     * Subscribe to topics
     */
    public subscribe(topics: string[]): void {
        topics.forEach(topic => this.subscriptions.add(topic));
        
        if (this.ws?.readyState === WebSocket.OPEN) {
            this.send({
                type: 'subscribe',
                data: { topics: Array.from(this.subscriptions) }
            });
        }
    }

    /**
     * Unsubscribe from topics
     */
    public unsubscribe(topics: string[]): void {
        topics.forEach(topic => this.subscriptions.delete(topic));
        
        if (this.ws?.readyState === WebSocket.OPEN) {
            this.send({
                type: 'subscribe',
                data: { topics: Array.from(this.subscriptions) }
            });
        }
    }

    /**
     * Check if connected to server
     */
    public isConnected(): boolean {
        return this.ws?.readyState === WebSocket.OPEN;
    }

    /**
     * Cleanup resources
     */
    private cleanup(): void {
        this.stopPingInterval();
        if (this.ws) {
            this.ws.onclose = null;
            this.ws.onerror = null;
            this.ws.onmessage = null;
            this.ws.onopen = null;
        }
    }

    /**
     * Clear message queue
     */
    public clearMessageQueue(clearPersisted = false): void {
        this.messageQueue = [];
        if (clearPersisted) {
            this.persistentQueue = [];
            if (this.persistenceEnabled) {
                localStorage.removeItem(PERSISTENT_QUEUE_KEY);
            }
        }
    }

    /**
     * Get current queue size
     */
    public getQueueSize(): number {
        return this.messageQueue.length;
    }

    /**
     * Close connection
     */
    public close(): void {
        this.shouldReconnect = false;
        this.clearMessageQueue();
        this.cleanup();
        this.ws?.close();
    }
}

// Example usage:
/*
const ws = new WebSocketClient('ws://localhost:8765/ws', 'optional-auth-token');

ws.on('connected', () => {
    console.log('Connected to server');
    ws.subscribe(['health_update', 'detection']);
});

ws.on('healthUpdate', (metrics: HealthMetrics) => {
    console.log('Health metrics:', metrics);
});

ws.on('detection', ({ camera_id, detection }) => {
    console.log(`Detection on camera ${camera_id}:`, detection);
});

ws.on('disconnected', () => {
    console.log('Disconnected from server');
});
*/ 