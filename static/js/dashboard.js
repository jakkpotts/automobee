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
    maxZoom: 18,
    // Dark theme settings
    preferCanvas: true,
    renderer: L.canvas({ tolerance: 5 }),
    zoomControl: false // We'll add custom styled controls
};

const cameraMarkerSettings = {
    connecting: {
        className: 'custom-marker connecting',
        riseOnHover: true,
        color: '#EBCB8B'  // Nord13 for connecting
    },
    active: {
        className: 'custom-marker active',
        riseOnHover: true,
        color: '#A3BE8C'  // Nord14 for active
    },
    error: {
        className: 'custom-marker error',
        riseOnHover: true,
        color: '#BF616A'  // Nord11 for error
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

        // Add dark theme tile layer
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors',
            className: 'dark-tiles'
        }).addTo(dashboardState.map);

        // Add custom zoom controls
        L.control.zoom({
            position: 'bottomright',
            zoomInText: '<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6"/></svg>',
            zoomOutText: '<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20 12H4"/></svg>'
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

    // Validate camera data
    if (!cameraData || !cameraData.location || 
        typeof cameraData.location.lat === 'undefined' || 
        typeof cameraData.location.lng === 'undefined') {
        console.error('Invalid camera data:', cameraData);
        return;
    }

    const position = [cameraData.location.lat, cameraData.location.lng];
    const cameraId = cameraData.id;

    try {
        if (dashboardState.markers[cameraId]) {
            console.log('Updating existing marker:', cameraId);
            dashboardState.markers[cameraId].setLatLng(position);
        } else {
            console.log('Creating new marker:', cameraId);
            const marker = createCustomMarker(
                'camera',
                cameraData.name || cameraId,
                position,
                cameraData.status || 'initializing'
            );
            dashboardState.markers[cameraId] = marker;
            marker.addTo(dashboardState.map);
        }

        // Update cameras state
        dashboardState.cameras[cameraId] = cameraData;
    } catch (error) {
        console.error('Error in updateSingleCamera:', error);
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
        console.log('Received stream data:', data);
        
        if (!data || !data.type) {
            console.warn('Invalid stream data received');
            return;
        }

        switch (data.type) {
            case 'initial_state':
                if (data.data?.cameras) {
                    Object.entries(data.data.cameras).forEach(([id, camera]) => {
                        if (camera) {
                            // Ensure camera has proper location structure
                            const cameraData = {
                                id: id,
                                location: {
                                    lat: camera.lat || camera.latitude || (camera.location && camera.location.lat),
                                    lng: camera.lng || camera.longitude || (camera.location && camera.location.lng)
                                },
                                name: camera.name || `Camera ${id}`,
                                status: camera.status || 'initializing'
                            };
                            updateSingleCamera(cameraData);
                        }
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
    console.log('Camera status update:', data);
    const { id, status, name } = data;
    
    // Update marker status
    const marker = dashboardState.markers[id];
    if (marker) {
        // Get marker element
        const markerElement = marker.getElement();
        if (markerElement) {
            // Remove all status classes first
            markerElement.classList.remove('connecting', 'active', 'error', 'processing');
            // Add new status class
            markerElement.classList.add(status);
            
            // Update marker color based on status
            const settings = cameraMarkerSettings[status] || cameraMarkerSettings.connecting;
            markerElement.style.backgroundColor = settings.color;
            
            // Update tooltip
            const tooltip = `${name || id} (${status})`;
            if (marker.getTooltip()) {
                marker.setTooltipContent(tooltip);
            } else {
                marker.bindTooltip(tooltip);
            }
        }
    }
    
    // Update camera card
    updateCameraCard(id, data);
};

const updateCameraCard = (cameraId, data) => {
    const camerasContainer = document.getElementById('camera-list');
    if (!camerasContainer) return;

    let card = document.getElementById(`camera-card-${cameraId}`);
    const { name, status, timestamp, stream_url } = data;

    if (!card) {
        // Create new card if it doesn't exist
        card = document.createElement('div');
        card.id = `camera-card-${cameraId}`;
        card.className = 'camera-card';
        camerasContainer.appendChild(card);
    }

    // Update card content
    card.innerHTML = `
        <div class="camera-card ${status}">
            <div class="camera-header">
                <h3>${name || `Camera ${cameraId}`}</h3>
                <span class="status-badge ${status}">${status}</span>
            </div>
            <div class="camera-details">
                <p>Last update: ${new Date(timestamp).toLocaleTimeString()}</p>
                ${stream_url ? `
                    <button class="stream-button" onclick="openCameraStream('${cameraId}', '${stream_url}', '${name}')">
                        View Stream
                    </button>
                ` : ''}
            </div>
        </div>
    `;
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
                <h4 class="font-semibold text-nord-6">${detection.type}</h4>
                <p class="text-sm text-nord-4">${detection.camera}</p>
            </div>
            <span class="bg-nord-8 text-nord-0 text-sm px-2 py-1 rounded-full">${confidencePercent}%</span>
        </div>
        <div class="text-sm text-nord-4">${timestamp}</div>
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
                    <p class="text-sm text-nord-4">${timestamp}</p>
                </div>
                ${alert.camera ? `<span class="text-sm bg-nord-11/20 text-nord-11 px-2 py-1 rounded">${alert.camera}</span>` : ''}
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

// Add this CSS to your stylesheet
const addStyles = () => {
    const style = document.createElement('style');
    style.textContent = `
        .camera-card {
            border: 1px solid #ddd;
            padding: 1rem;
            margin: 0.5rem;
            border-radius: 0.5rem;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .camera-card.active { border-left: 4px solid #4CAF50; }
        .camera-card.error { border-left: 4px solid #F44336; }
        .camera-card.connecting { border-left: 4px solid #FFA500; }
        .camera-card.processing { border-left: 4px solid #2196F3; }

        .status-badge {
            padding: 0.25rem 0.5rem;
            border-radius: 1rem;
            font-size: 0.875rem;
            font-weight: 500;
        }

        .status-badge.active { background: #E8F5E9; color: #2E7D32; }
        .status-badge.error { background: #FFEBEE; color: #C62828; }
        .status-badge.connecting { background: #FFF3E0; color: #EF6C00; }
        .status-badge.processing { background: #E3F2FD; color: #1565C0; }

        .camera-marker {
            width: 24px;
            height: 24px;
            border-radius: 50%;
            border: 2px solid white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
            transition: all 0.3s ease;
        }

        .camera-marker.active { background-color: #4CAF50 !important; }
        .camera-marker.error { background-color: #F44336 !important; }
        .camera-marker.connecting { background-color: #FFA500 !important; }
        .camera-marker.processing { background-color: #2196F3 !important; }
    `;
    document.head.appendChild(style);
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
        
        // Add debug call
        setInterval(debugDashboardState, 5000);
        
        console.log('Dashboard initialized successfully');
    } catch (error) {
        console.error('Dashboard initialization failed:', error);
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

const transitionToZone = (zoneId) => {
    const zone = dashboardState.zones.get(zoneId);
    if (!zone) return;

    // Store previous zone for transition
    const prevZoneId = dashboardState.zoneFilter.activeZone;
    
    // Smooth map transition
    dashboardState.map.flyTo(
        [zone.center.lat, zone.center.lng],
        Math.min(14, dashboardState.map.getZoom()),
        {
            duration: 1.5,
            easeLinearity: 0.25
        }
    );

    // Fade out previous zone
    if (prevZoneId !== 'all' && prevZoneId !== zoneId) {
        const prevZone = dashboardState.zones.get(prevZoneId);
        if (prevZone && prevZone.layer) {
            fadeOutZone(prevZone.layer);
        }
    }

    // Fade in new zone with delay
    setTimeout(() => {
        highlightActiveZone(zoneId);
        updateZoneMarkers(zoneId);
    }, 750);
};

const fadeOutZone = (layer) => {
    let opacity = 1;
    const fadeInterval = setInterval(() => {
        opacity -= 0.1;
        if (opacity <= 0) {
            clearInterval(fadeInterval);
            dashboardState.map.removeLayer(layer);
        } else {
            layer.setStyle({ 
                fillOpacity: opacity * 0.15,
                opacity: opacity 
            });
        }
    }, 50);
};

const updateZoneMarkers = (zoneId) => {
    const zone = dashboardState.zones.get(zoneId);
    if (!zone) return;

    Object.entries(dashboardState.markers).forEach(([markerId, marker]) => {
        const element = marker.getElement();
        const isInZone = zone.cameras.has(markerId);
        
        element.style.transition = 'all 0.5s ease-in-out';
        if (isInZone) {
            element.classList.add('in-active-zone');
        } else {
            element.classList.add('inactive-zone');
        }
    });
};