// State management
const dashboardState = {
    map: null,
    markers: {},
    cameras: {},
    eventSource: null,
    stats: {
        totalDetections: 0,
        activeCameras: 0,
        detectionRate: 0
    }
};

// Map configuration
const mapConfig = {
    center: [36.1699, -115.1398], // Las Vegas center
    zoom: 11,
    minZoom: 3,
    maxZoom: 18
};

const cameraMarkerSettings = {
    connecting: {
        className: 'custom-marker connecting',
        riseOnHover: true,
        color: '#FFA500'  // Orange for connecting
    },
    active: {
        className: 'custom-marker active',
        riseOnHover: true,
        color: '#4CAF50'  // Green for active
    },
    error: {
        className: 'custom-marker error',
        riseOnHover: true,
        color: '#F44336'  // Red for error
    }
};

// Wait for Leaflet to be ready
const waitForLeaflet = () => {
    return new Promise((resolve, reject) => {
        if (window.L) {
            resolve();
        } else {
            const checkInterval = setInterval(() => {
                if (window.L) {
                    clearInterval(checkInterval);
                    resolve();
                }
            }, 100);

            setTimeout(() => {
                clearInterval(checkInterval);
                reject(new Error('Leaflet failed to load'));
            }, 10000);
        }
    });
};

// Create custom marker
const createCustomMarker = (type, title, position, status = 'initializing') => {
    const markerElement = document.createElement('div');
    markerElement.className = `${type === 'camera' ? 'camera-marker' : 'detection-marker'} ${status}`;
    
    if (type === 'camera') {
        markerElement.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 9a3 3 0 100 6 3 3 0 000-6zM2 9a3 3 0 100 6 3 3 0 000-6zM22 9a3 3 0 100 6 3 3 0 000-6z"/>
            </svg>
        `;
    } else {
        markerElement.textContent = title;
    }

    return L.marker(position, {
        icon: L.divIcon({
            className: 'custom-marker-container',
            html: markerElement,
            iconSize: [24, 24],
            iconAnchor: [12, 12],
            popupAnchor: [0, -12]
        }),
        title: title // This enables the hover tooltip
    });
};

// Initialize Leaflet map
const initializeMap = async () => {
    console.log('Initializing map...');
    
    try {
        await waitForLeaflet();
        
        dashboardState.map = L.map('main-map', mapConfig);
        console.log('Map created:', dashboardState.map);

        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(dashboardState.map);
        
        console.log('Map initialization complete');
    } catch (error) {
        console.error('Error initializing map:', error);
    }
};

// Initialize UI event handlers
const initializeUIHandlers = () => {
    console.log('Initializing UI handlers...');
    
    try {
        // Map mode selector
        document.getElementById('map-mode')?.addEventListener('change', (e) => {
            updateMapMode(e.target.value);
        });

        // Recenter map button
        document.getElementById('recenter-map')?.addEventListener('click', () => {
            if (dashboardState.map) {
                dashboardState.map.setView(mapConfig.center, mapConfig.zoom);
            }
        });

        // Camera search
        document.getElementById('camera-search')?.addEventListener('input', (e) => {
            filterCameras(e.target.value);
        });

        console.log('UI handlers initialized successfully');
    } catch (error) {
        console.error('Error initializing UI handlers:', error);
    }
};

// Initialize event stream
const initializeEventStream = () => {
    console.log('Initializing event stream...');
    
    try {
        if (dashboardState.eventSource) {
            dashboardState.eventSource.close();
        }

        dashboardState.eventSource = new EventSource('/stream');
        
        dashboardState.eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                handleStreamData(data);
            } catch (error) {
                console.error('Error parsing stream data:', error);
            }
        };

        dashboardState.eventSource.onerror = (error) => {
            console.error('Stream error:', error);
            setTimeout(initializeEventStream, 5000);
        };

        console.log('Event stream initialized successfully');
    } catch (error) {
        console.error('Error initializing event stream:', error);
    }
};

// Update camera feed
const updateCameraFeed = (feed) => {
    console.log('Processing camera feed update:', feed);

    try {
        // Check if we're dealing with multiple cameras
        if (typeof feed === 'object' && !feed.hasOwnProperty('camera_id')) {
            // Multiple camera update
            Object.values(feed).forEach(cameraData => {
                updateSingleCamera(cameraData);
            });
        } else {
            // Single camera update
            updateSingleCamera(feed.data || feed);
        }
    } catch (error) {
        console.error('Error in updateCameraFeed:', error);
    }
};

const updateSingleCamera = (cameraData) => {
    if (!dashboardState.map) {
        console.error('Map not initialized');
        return;
    }

    console.log('Updating single camera:', cameraData);

    const position = [cameraData.location.lat, cameraData.location.lng];
    const cameraId = cameraData.id;

    if (dashboardState.markers[cameraId]) {
        console.log('Updating existing marker:', cameraId);
        dashboardState.markers[cameraId].setLatLng(position);
    } else {
        console.log('Creating new marker:', cameraId);
        const marker = createCustomMarker(
            'camera',
            cameraData.name || cameraId,
            position
        );
        dashboardState.markers[cameraId] = marker;
        marker.addTo(dashboardState.map);
    }
};

// Update dashboard statistics
const updateDashboardStats = (stats) => {
    console.log('Updating dashboard stats:', stats);
    
    if (!stats) return;
    
    dashboardState.stats = {
        ...dashboardState.stats,
        ...stats
    };
    
    // Update UI elements with new stats
    if (stats.totalDetections !== undefined) {
        const totalDetections = document.getElementById('total-detections');
        if (totalDetections) {
            totalDetections.textContent = stats.totalDetections;
        }
    }
    
    if (stats.activeCameras !== undefined) {
        const activeCameras = document.getElementById('active-cameras');
        if (activeCameras) {
            activeCameras.textContent = stats.activeCameras;
            console.log('Updated active cameras count:', stats.activeCameras);
        }
    }
    
    if (stats.detectionRate !== undefined) {
        const detectionRate = document.getElementById('detection-rate');
        if (detectionRate) {
            detectionRate.textContent = stats.detectionRate.toFixed(2);
        }
    }
};

// Handle stream data
const handleStreamData = (data) => {
    try {
        console.log('Received stream data:', data.type);
        
        switch (data.type) {
            case 'initial_state':
                // Handle initial state - set up cameras and map
                if (data.data && data.data.cameras) {
                    Object.entries(data.data.cameras).forEach(([id, camera]) => {
                        updateSingleCamera(camera);
                    });
                    // Update stats
                    updateDashboardStats({
                        activeCameras: Object.keys(data.data.cameras).length
                    });
                }
                break;
                
            case 'camera_feed':
            case 'camera_update':    
                updateCameraFeed(data.data);
                break;
            case 'stats':
                updateDashboardStats(data.data);
                break;
            case 'detection':
                handleDetection(data.data);
                break;
            case 'alert':
                handleAlert(data.data);
                break;
            case 'map_update':
                handleMapUpdate(data.data);
                break;
            case 'camera_status':
                handleCameraStatus(data.data);
                break;
            default:
                console.log('Unhandled event type:', data.type);
        }
    } catch (error) {
        console.error('Error handling stream data:', error);
    }
};

// Handle map updates
const handleMapUpdate = (data) => {
    try {
        if (!data.points || !Array.isArray(data.points)) {
            return;
        }

        data.points.forEach(point => {
            if (point.latitude && point.longitude) {
                const markerId = point.id;
                const markerPosition = [point.latitude, point.longitude];
                
                // Get camera status
                const status = point.status || 'initializing';
                
                // Create or update marker
                const marker = createCustomMarker(
                    point.type || 'camera',
                    point.name || point.id,
                    markerPosition,
                    status
                );
                
                // Add click handler for camera markers
                if (point.type === 'camera') {
                    marker.on('click', () => {
                        console.log('Opening stream:', point);
                        openCameraStream(point.id, formatStreamUrl(point.stream_url), point.name);
                    });
                }
                
                // Remove existing marker if it exists
                if (dashboardState.markers[markerId]) {
                    dashboardState.map.removeLayer(dashboardState.markers[markerId]);
                }
                
                marker.addTo(dashboardState.map);
                dashboardState.markers[markerId] = marker;
                dashboardState.cameras[markerId] = point;
            }
        });
    } catch (error) {
        console.error('Error handling map update:', error);
    }
};

// Add function to format stream URL
const formatStreamUrl = (url) => {
    if (!url) return '';
    
    // Remove any explicit port numbers
    url = url.replace(':443', '');
    
    // Ensure we're using https
    if (!url.startsWith('https://')) {
        url = url.replace('http://', 'https://');
    }
    
    return url;
};

// Add handler for camera status updates
const handleCameraStatus = (data) => {
    const { id, status } = data;
    console.log(`Camera ${id} status update:`, status);
    
    const marker = dashboardState.markers[id];
    if (marker) {
        // Update marker appearance
        const settings = cameraMarkerSettings[status] || cameraMarkerSettings.connecting;
        marker.getElement().className = settings.className;
        
        // Update tooltip
        const tooltip = `${dashboardState.cameras[id]?.name || id} (${status})`;
        marker.setTooltipContent(tooltip);
    }
    
    // Update camera info in state
    if (dashboardState.cameras[id]) {
        dashboardState.cameras[id].status = status;
    }
};

// Add function to handle camera stream modal
const openCameraStream = (cameraId, streamUrl, cameraName) => {
    console.log(`Opening stream for camera ${cameraId}: ${streamUrl}`);
    
    // Create or get modal
    let modal = document.getElementById('camera-modal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'camera-modal';
        modal.className = 'modal';
        document.body.appendChild(modal);
    }
    
    // Update modal content with HLS.js player
    modal.innerHTML = `
        <div class="modal-content">
            <div class="modal-header">
                <h2>${cameraName || `Camera ${cameraId}`}</h2>
                <span class="close">&times;</span>
            </div>
            <div class="modal-body">
                <video id="video-player" controls></video>
            </div>
        </div>
    `;
    
    // Show modal
    modal.style.display = 'block';
    
    // Initialize HLS.js
    const video = document.getElementById('video-player');
    if (Hls.isSupported()) {
        const hls = new Hls({
            debug: false,
            enableWorker: true,
            lowLatencyMode: true,
            backBufferLength: 90
        });
        hls.loadSource(streamUrl);
        hls.attachMedia(video);
        hls.on(Hls.Events.MANIFEST_PARSED, () => {
            video.play().catch(e => console.error('Playback failed:', e));
        });
    } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
        // Fallback for Safari
        video.src = streamUrl;
        video.addEventListener('loadedmetadata', () => {
            video.play().catch(e => console.error('Playback failed:', e));
        });
    }
    
    // Add close functionality
    const closeBtn = modal.querySelector('.close');
    closeBtn.onclick = () => {
        modal.style.display = 'none';
        if (hls) {
            hls.destroy();
        }
        video.src = '';
    };
    
    // Close on outside click
    window.onclick = (event) => {
        if (event.target === modal) {
            modal.style.display = 'none';
            if (hls) {
                hls.destroy();
            }
            video.src = '';
        }
    };
};

// Handle new detection
const handleDetection = (detection) => {
    try {
        // Update detection count
        const detectionCount = document.getElementById('detection-count');
        if (detectionCount) {
            const currentCount = parseInt(detectionCount.textContent);
            detectionCount.textContent = currentCount + 1;
        }

        // Add detection to recent detections grid
        const recentDetections = document.getElementById('recent-detections');
        if (recentDetections) {
            const detectionCard = createDetectionCard(detection);
            recentDetections.insertBefore(detectionCard, recentDetections.firstChild);

            // Limit the number of visible detections
            while (recentDetections.children.length > 9) {
                recentDetections.removeChild(recentDetections.lastChild);
            }
        }

        // Add detection marker to map if coordinates are available
        if (detection.location) {
            const marker = createCustomMarker(
                'detection',
                `${detection.type} (${Math.round(detection.confidence * 100)}%)`,
                [detection.location.lat, detection.location.lng]
            );
            marker.addTo(dashboardState.map);
            
            // Remove marker after 5 seconds
            setTimeout(() => {
                dashboardState.map.removeLayer(marker);
            }, 5000);
        }
    } catch (error) {
        console.error('Error handling detection:', error);
    }
};

// Create detection card element
const createDetectionCard = (detection) => {
    const card = document.createElement('div');
    card.className = 'detection-card';
    
    const timestamp = new Date(detection.timestamp).toLocaleTimeString();
    const confidencePercent = Math.round(detection.confidence * 100);
    
    card.innerHTML = `
        <div class="flex justify-between items-start mb-2">
            <div>
                <h4 class="font-semibold text-nord-0">${detection.type}</h4>
                <p class="text-sm text-nord-3">${detection.camera}</p>
            </div>
            <span class="bg-nord-10 text-white text-sm px-2 py-1 rounded-full">${confidencePercent}%</span>
        </div>
        <div class="text-sm text-nord-3">${timestamp}</div>
    `;
    
    return card;
};

// Handle new alert
const handleAlert = (alert) => {
    try {
        const alertsContainer = document.getElementById('alerts-container');
        if (alertsContainer) {
            const alertElement = document.createElement('div');
            alertElement.className = 'alert-item';
            
            const timestamp = new Date(alert.timestamp).toLocaleTimeString();
            
            alertElement.innerHTML = `
                <div class="flex-1">
                    <p class="font-medium">${alert.message}</p>
                    <p class="text-sm">${timestamp}</p>
                </div>
                ${alert.camera ? `<span class="text-sm bg-nord-11/20 px-2 py-1 rounded">${alert.camera}</span>` : ''}
            `;
            
            alertsContainer.insertBefore(alertElement, alertsContainer.firstChild);
            
            // Limit the number of visible alerts
            while (alertsContainer.children.length > 5) {
                alertsContainer.removeChild(alertsContainer.lastChild);
            }
        }
    } catch (error) {
        console.error('Error handling alert:', error);
    }
};

// Initialize the dashboard
const initializeDashboard = async () => {
    console.log('Starting dashboard initialization...');
    
    try {
        await waitForLeaflet();
        console.log('Leaflet loaded successfully');

        const mapContainer = document.getElementById('main-map');
        if (!mapContainer) {
            throw new Error('Map container not found');
        }

        await initializeMap();
        initializeEventStream();
        initializeUIHandlers();
        
        console.log('Dashboard initialized successfully');
    } catch (error) {
        console.error('Dashboard initialization failed:', error);
        const mapContainer = document.getElementById('main-map');
        if (mapContainer) {
            mapContainer.innerHTML = `<div class="p-4 text-red-500">Error loading map: ${error.message}</div>`;
        }
    }
};

// Initialize when the page is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeDashboard);
} else {
    initializeDashboard();
}

// Export the module
export {
    initializeDashboard,
    updateCameraFeed,
    handleStreamData,
    handleDetection,
    handleAlert,
    handleMapUpdate,
    dashboardState
};