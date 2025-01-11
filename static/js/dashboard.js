// DashboardState class to manage application state
class DashboardState {
    constructor() {
        this.cameras = new Map();
        this.detections = [];
        this.alerts = [];
        this.map = null;
        this.markers = new Map();
        this.selectedCamera = null;
        this.markerSource = null;
        this.popup = null;
        this.subscribers = new Map();
        this.camerasMap = null;
        this.detectionsMap = null;
        this.selectedCamera = null;
        this.detectionMarkers = new Map();
        this.cameraMarkers = new Map();
    }

    subscribe(event, callback) {
        if (!this.subscribers.has(event)) {
            this.subscribers.set(event, new Set());
        }
        this.subscribers.get(event).add(callback);
    }

    notify(event, data) {
        if (this.subscribers.has(event)) {
            this.subscribers.get(event).forEach(callback => callback(data));
        }
    }

    updateDetections(detections) {
        this.detections = detections;
        this.notify('detectionsUpdated', detections);
    }

    // Add more state management methods...
}

// Create global state instance
const dashboardState = new DashboardState();

// Define status colors
const statusColors = {
    'active': 'bg-green-100 text-green-600',
    'processing': 'bg-blue-100 text-blue-600',
    'analyzing': 'bg-yellow-100 text-yellow-600',
    'error': 'bg-red-100 text-red-600',
    'inactive': 'bg-gray-100 text-gray-600'
};

// Debounced update function
const debouncedUpdate = debounce(updateStats, 500);

// Error boundary for async operations
async function withErrorBoundary(operation, fallback) {
    try {
        return await operation();
    } catch (error) {
        console.error(`Operation failed: ${error.message}`);
        return fallback;
    }
}


// Handle map click events
function handleMapClick(evt) {
    const feature = dashboardState.map.forEachFeatureAtPixel(evt.pixel, feature => feature);
    
    if (feature) {
        const coords = feature.getGeometry().getCoordinates();
        const properties = feature.getProperties();
        
        const popupContent = `
            <div class="ol-popup">
                <h3 class="font-semibold mb-2">${properties.name}</h3>
                <button onclick="openStreamModal('${properties.name}', '${properties.url}')" 
                        class="bg-blue-500 text-white px-4 py-2 rounded-lg text-sm hover:bg-blue-600 transition-colors w-full">
                    View Stream
                </button>
            </div>
        `;

        dashboardState.popup.getElement().innerHTML = popupContent;
        dashboardState.popup.setPosition(coords);
    } else {
        dashboardState.popup.setPosition(undefined);
    }
}

// Handle map hover events
function handleMapHover(evt) {
    const pixel = dashboardState.map.getEventPixel(evt.originalEvent);
    const hit = dashboardState.map.hasFeatureAtPixel(pixel);
    dashboardState.map.getTargetElement().style.cursor = hit ? 'pointer' : '';
}

// Create legend element
function createLegendElement() {
    const div = document.createElement('div');
    div.className = 'bg-white p-4 rounded-lg shadow-lg';
    div.innerHTML = `
        <div class="text-sm space-y-2">
            <div class="flex items-center">
                <span class="w-3 h-3 rounded-full bg-green-500 mr-2"></span>
                <span>Active Camera</span>
            </div>
            <div class="flex items-center">
                <span class="w-3 h-3 rounded-full bg-red-500 mr-2"></span>
                <span>Inactive Camera</span>
            </div>
        </div>
    `;
    return div;
}

// Add retry functionality
window.retryMapLoad = async function() {
    const loadingElement = document.getElementById('map-loading');
    loadingElement.innerHTML = '<div class="text-gray-500">Loading map...</div>';
    await initMap();
};

// Initialize SSE connection with retry
let eventSource = null;

function initEventSource() {
    try {
        if (eventSource) {
            eventSource.close();
            eventSource = null;
        }

        eventSource = new EventSource('/stream');
        
        eventSource.onmessage = function(event) {
            try {
                const data = JSON.parse(event.data);
                handleRealtimeUpdate(data);
            } catch (e) {
                console.error("Failed to parse SSE message:", e);
            }
        };

        eventSource.onerror = function(error) {
            console.error("SSE Error:", error);
            if (eventSource) {
                eventSource.close();
                eventSource = null;
            }
            setTimeout(() => retryConnection(), 1000);
        };

        console.log("SSE Connection established at /stream");
    } catch (error) {
        console.error("Error initializing EventSource:", error);
        setTimeout(() => retryConnection(), 1000);
    }
}

// Retry connection with backoff
function retryConnection(attempt = 0) {
    const maxAttempts = 5;
    const backoffMs = Math.min(1000 * Math.pow(2, attempt), 10000);
    
    if (attempt < maxAttempts) {
        console.log(`Retrying connection in ${backoffMs/1000}s... (Attempt ${attempt + 1}/${maxAttempts})`);
        setTimeout(() => {
            initEventSource();
        }, backoffMs);
    } else {
        console.error("Max retry attempts reached. Please refresh the page.");
    }
}

// Enhanced realtime update handler with validation
function handleRealtimeUpdate(data) {
    if (!data || typeof data !== 'object') {
        console.error('Invalid update data received');
        return;
    }

    const handlers = {
        detection: addDetection,
        alert: addAlert,
        camera_status: updateCameraStatus
    };

    if (data.type in handlers) {
        withErrorBoundary(() => handlers[data.type](data.data), null);
    }
}

// Add updateStatsCounters function
function updateStatsCounters(data) {
    // Update total detections
    const totalDetectionsElement = document.getElementById('total-detections');
    if (totalDetectionsElement) {
        totalDetectionsElement.textContent = data.total_detections || 0;
    }

    // Update active cameras
    const activeCamerasElement = document.getElementById('active-cameras');
    if (activeCamerasElement) {
        activeCamerasElement.textContent = data.active_cameras || 0;
    }

    // Update detection rate
    const detectionRateElement = document.getElementById('detection-rate');
    if (detectionRateElement) {
        detectionRateElement.textContent = `${data.detection_rate || 0}/hr`;
    }
}

// Update dashboard stats with error handling
function updateStats() {
    console.log('Fetching updated stats...');
    
    fetch('/api/stats')
        .then(response => {
            if (!response.ok) throw new Error('Stats fetch failed');
            return response.json();
        })
        .then(data => {
            console.log('Received stats:', data);
            
            // Update state with real data
            dashboardState.detections = data.recent_detections || [];
            dashboardState.alerts = data.alerts || [];
            
            // Process camera data
            const cameraData = {};
            if (data.cameras) {
                Object.entries(data.cameras).forEach(([id, camera]) => {
                    cameraData[id] = {
                        name: camera.name,
                        status: camera.status,
                        lat: camera.location?.latitude,
                        lng: camera.location?.longitude,
                        url: camera.stream_url,
                        last_update: camera.last_update
                    };
                });
            }
            
            // Update UI elements
            updateDetectionsList();
            updateAlertsList();
            updateCameraList(cameraData);
            
            // Update map markers with real camera locations
            updateMapMarkers(cameraData);
            
            // Update stats counters with real data
            updateStatsCounters({
                total_detections: data.stats?.total_detections || 0,
                active_cameras: Object.values(data.cameras || {})
                    .filter(c => c.status === 'active').length,
                detection_rate: data.stats?.detection_rate || 0
            });
        })
        .catch(error => {
            console.error('Error updating stats:', error);
        });
}

// Ensure regular updates
function startUpdateCycle() {
    // Initial update
    updateStats();
    
    // Regular updates every 5 seconds
    setInterval(updateStats, 5000);
}

// Initialize everything when the page loads
document.addEventListener('DOMContentLoaded', async () => {
    console.log('DOM Content Loaded, starting initialization...');
    
    try {
        // Initialize map
        await initMap();
        
        // Initialize UI components
        initializeUIComponents();
        
        // Close existing SSE connection if any
        if (eventSource) {
            eventSource.close();
        }
        
        // Initialize single SSE connection
        initEventSource();
        
        // Start update cycle
        startUpdateCycle();
    } catch (error) {
        console.error('Failed to initialize dashboard:', error);
    }
});

// Update markers for Leaflet
function updateMapMarkers(cameras) {
    if (!dashboardState.camerasMap || !dashboardState.cameraMarkers) {
        console.error('Camera map or markers not initialized');
        return;
    }

    try {
        // Clear existing markers
        dashboardState.cameraMarkers.clearLayers();

        // Add new markers for each camera
        Object.entries(cameras).forEach(([id, camera]) => {
            if (!camera.lat || !camera.lng) {
                console.warn(`Invalid coordinates for camera ${camera.name}`);
                return;
            }

            const marker = L.marker(
                [camera.lat, camera.lng],
                { 
                    icon: camera.status === 'error' ? 
                        dashboardState.icons.error : 
                        dashboardState.icons.active
                }
            );

            // Add popup with camera info and stream button
            const popup = L.popup().setContent(`
                <div class="camera-popup p-3">
                    <h3 class="font-semibold mb-2">${camera.name}</h3>
                    <div class="status mb-2">
                        <span class="inline-block w-2 h-2 rounded-full ${
                            camera.status === 'active' ? 'bg-green-500' : 'bg-red-500'
                        } mr-2"></span>
                        <span class="text-sm">${camera.status}</span>
                    </div>
                    ${camera.last_update ? 
                        `<p class="text-xs text-gray-500 mb-2">Last update: ${formatTime(camera.last_update)}</p>` 
                        : ''}
                    <button onclick="openStreamModal('${camera.name}', '${camera.url}')"
                            class="bg-blue-500 text-white px-3 py-1 rounded text-sm hover:bg-blue-600 transition-colors w-full">
                        View Stream
                    </button>
                </div>
            `);

            marker.bindPopup(popup);

            // Add to feature group
            dashboardState.cameraMarkers.addLayer(marker);
        });

        // Fit map to markers if we have any
        if (dashboardState.cameraMarkers.getLayers().length > 0) {
            dashboardState.camerasMap.fitBounds(dashboardState.cameraMarkers.getBounds(), {
                padding: [50, 50],
                maxZoom: 13
            });
        }

    } catch (e) {
        console.error('Error updating map markers:', e);
    }
}

// Update camera list
function updateCameraList(stats) {
    const container = document.getElementById('camera-list');
    if (!container) {
        console.warn('Camera list container not found');
        return;
    }
    
    container.innerHTML = '';
    
    for (const [name, stat] of Object.entries(stats)) {
        const div = document.createElement('div');
        div.className = `p-3 rounded-lg ${stat.error ? 'bg-red-50' : 'bg-green-50'} cursor-pointer hover:scale-105 transition-transform`;
        div.innerHTML = `
            <div class="flex items-center justify-between">
                <span class="font-medium">${name}</span>
                <span class="w-2 h-2 rounded-full ${stat.error ? 'bg-red-500' : 'bg-green-500'}"></span>
            </div>
            <p class="text-sm text-gray-500 mt-1">Last update: ${formatTime(stat.last_update)}</p>
        `;
        container.appendChild(div);
    }
}

// Enhanced detection list update with validation
function updateDetectionsList() {
    const container = document.getElementById('recent-detections');
    const countElement = document.getElementById('detection-count');
    if (!container || !countElement) {
        console.warn('Detections container not found');
        return;
    }

    const filter = document.getElementById('detection-filter')?.value || 'all';
    const sort = document.getElementById('detection-sort')?.value || 'newest';
    
    // Validate detections array
    if (!Array.isArray(dashboardState.detections)) {
        console.error('Invalid detections data');
        return;
    }

    let filteredDetections = filter === 'all' 
        ? dashboardState.detections 
        : dashboardState.detections.filter(d => d.type === filter);

    // Sort detections
    filteredDetections.sort((a, b) => {
        if (sort === 'newest') {
            return new Date(b.timestamp) - new Date(a.timestamp);
        } else if (sort === 'confidence') {
            return b.confidence - a.confidence;
        }
        return 0;
    });

    // Update count
    countElement.textContent = filteredDetections.length;
    
    container.innerHTML = '';
    
    if (filteredDetections.length === 0) {
        container.innerHTML = `
            <div class="col-span-full flex flex-col items-center justify-center py-12 text-slate-400">
                <svg class="w-16 h-16 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                          d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/>
                </svg>
                <p class="text-lg">No detections found</p>
            </div>
        `;
        return;
    }

    for (const detection of filteredDetections) {
        const div = document.createElement('div');
        div.className = 'detection-card';
        
        div.innerHTML = `
            <div class="relative">
                <img 
                    src="/matches/${detection.image}" 
                    alt="Detection ${detection.type}"
                    class="w-full h-64 object-cover"
                    onerror="this.src='/static/img/no-image.png'"
                />
                <div class="absolute top-2 right-2 flex space-x-2">
                    <span class="bg-blue-500 text-white px-3 py-1 rounded-full text-sm">
                        ${Math.round(detection.confidence * 100)}%
                    </span>
                    ${detection.type === 'ford_f150' ? 
                        '<span class="bg-green-500 text-white px-3 py-1 rounded-full text-sm">Target</span>' 
                        : ''}
                </div>
            </div>
            <div class="p-4">
                <div class="flex justify-between items-start mb-4">
                    <div>
                        <h3 class="text-lg font-semibold text-slate-200">${detection.type}</h3>
                        <p class="text-sm text-slate-400">${formatTime(detection.timestamp)}</p>
                    </div>
                    <span class="bg-slate-700 text-slate-300 px-2 py-1 rounded text-sm">
                        ${detection.camera}
                    </span>
                </div>
                ${detection.make ? `
                    <div class="bg-slate-700 rounded-lg p-3 mt-3">
                        <p class="text-sm text-slate-300">
                            <span class="font-medium">Make/Model:</span> 
                            ${detection.make} ${detection.model || ''}
                        </p>
                    </div>
                ` : ''}
                <div class="flex justify-end mt-4 space-x-2">
                    <button onclick="showDetectionModal(${JSON.stringify(detection)})" 
                            class="bg-blue-500 text-white px-4 py-2 rounded-lg text-sm hover:bg-blue-600 transition-colors">
                        View Details
                    </button>
                    <button onclick="downloadDetectionImage(${JSON.stringify(detection)})"
                            class="bg-slate-700 text-slate-200 px-4 py-2 rounded-lg text-sm hover:bg-slate-600 transition-colors">
                        Download
                    </button>
                </div>
            </div>
        `;
        
        container.appendChild(div);
    }
}

// Show notification for new detections
function showNotification() {
    const notification = document.getElementById('notification');
    if (!notification) return;

    notification.classList.add('notification-show');
    setTimeout(() => {
        notification.classList.remove('notification-show');
    }, 3000);
}

// Update alerts list - Fix syntax error and complete the function
function updateAlertsList() {
    const container = document.getElementById('alerts-container');
    if (!container) {
        console.warn('Alerts container not found');
        return;
    }
    
    container.innerHTML = '';
    
    if (!Array.isArray(dashboardState.alerts) || dashboardState.alerts.length === 0) {
        container.innerHTML = '<p class="text-gray-500 text-center py-4">No active alerts</p>';
        return;
    }
    
    for (const alert of dashboardState.alerts) {
        const div = document.createElement('div');
        div.className = 'bg-red-50 border border-red-100 rounded-lg p-4 mb-3';
        div.innerHTML = `
            <div class="flex items-center justify-between">
                <div class="flex items-center">
                    <svg class="w-5 h-5 text-red-600 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/>
                    </svg>
                    <span class="font-medium text-red-800">${alert.type}</span>
                </div>
                <span class="text-sm text-gray-600">${formatTime(alert.timestamp)}</span>
            </div>
            <p class="mt-2 text-red-600">${alert.message}</p>
        `;
        container.appendChild(div);
    }
}

// Add time formatting function
function formatTime(timestamp) {
    if (!timestamp) return 'N/A';
    try {
        const date = new Date(timestamp);
        return date.toLocaleString();
    } catch (e) {
        console.error('Error formatting time:', e);
        return 'Invalid time';
    }
}

// Add error handling for image loading
function showDetectionModal(detection) {
    const modal = document.getElementById('detection-modal');
    const modalTitle = document.getElementById('detection-modal-title');
    const modalImage = document.getElementById('detection-modal-image');
    const modalDetails = document.getElementById('detection-modal-details');
    const downloadBtn = document.getElementById('download-image');
    
    // Set modal content
    modalTitle.textContent = `${detection.type} Detection`;
    
    // Add error handling for image loading
    modalImage.onerror = function() {
        this.src = '/static/img/no-image.png';
        console.error(`Failed to load image: ${detection.image}`);
    };
    modalImage.src = `/matches/${detection.image}`;
    
    // Setup download button
    downloadBtn.onclick = () => downloadDetectionImage(detection);
    
    // Format details
    modalDetails.innerHTML = `
        <div class="bg-gray-50 p-3 rounded-lg">
            <p class="text-sm font-medium text-gray-500">Time</p>
            <p class="text-base">${formatTime(detection.timestamp)}</p>
        </div>
        <div class="bg-gray-50 p-3 rounded-lg">
            <p class="text-sm font-medium text-gray-500">Camera</p>
            <p class="text-base">${detection.camera}</p>
        </div>
        <div class="bg-gray-50 p-3 rounded-lg">
            <p class="text-sm font-medium text-gray-500">Confidence</p>
            <p class="text-base">${Math.round(detection.confidence * 100)}%</p>
        </div>
        ${detection.make ? `
        <div class="bg-gray-50 p-3 rounded-lg">
            <p class="text-sm font-medium text-gray-500">Make/Model</p>
            <p class="text-base">${detection.make} ${detection.model || ''}</p>
        </div>
        ` : ''}
    `;
    
    // Show modal
    modal.classList.remove('hidden');
    modal.classList.add('flex');
}

// Add download functionality
function downloadDetectionImage(detection) {
    const link = document.createElement('a');
    link.href = `/matches/${detection.image}`;
    link.download = detection.image;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// Add clear detections functionality
document.getElementById('clear-detections')?.addEventListener('click', () => {
    dashboardState.detections = [];
    updateDetectionsList();
});

// Update camera status with enhanced error handling and logging
function updateCameraStatus(cameraData) {
    console.log('Updating camera status:', cameraData);
    
    const cameraStatusList = document.getElementById('camera-status-list');
    if (!cameraStatusList) {
        console.error('Camera status list container not found');
        return;
    }

    const statusColors = {
        'active': 'bg-green-500',
        'error': 'bg-red-500',
        'processing': 'bg-blue-500',
        'analyzing': 'bg-yellow-500',
        'inactive': 'bg-gray-500'
    };

    const statusDiv = document.createElement('div');
    statusDiv.className = 'bg-slate-800 rounded-lg p-4 border border-slate-700';
    
    const statusColor = statusColors[cameraData.status] || statusColors.inactive;
    
    statusDiv.innerHTML = `
        <div class="flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <div class="w-3 h-3 rounded-full ${statusColor}"></div>
                <span class="text-slate-200">${cameraData.name}</span>
            </div>
            <span class="text-sm text-slate-400">${formatTime(cameraData.timestamp)}</span>
        </div>
        ${cameraData.error ? `
            <p class="mt-2 text-sm text-red-400">${cameraData.error}</p>
        ` : ''}
    `;

    // Find existing status card and replace or append
    const existingCard = Array.from(cameraStatusList.children)
        .find(child => child.querySelector('span')?.textContent === cameraData.name);
    
    if (existingCard) {
        cameraStatusList.replaceChild(statusDiv, existingCard);
    } else {
        cameraStatusList.appendChild(statusDiv);
    }
}

// Stream modal functionality
function openStreamModal(cameraName, streamUrl) {
    const modal = document.getElementById('stream-modal');
    const title = document.getElementById('stream-title');
    const player = document.getElementById('stream-player');

    if (!modal || !title || !player || !streamUrl) {
        console.error('Missing required elements for stream modal');
        return;
    }

    title.textContent = `Camera Stream: ${cameraName}`;
    modal.classList.remove('hidden');
    modal.classList.add('flex');

    try {
        if (Hls.isSupported()) {
            const hls = new Hls({
                debug: false,
                enableWorker: true,
                lowLatencyMode: true
            });
            
            hls.on(Hls.Events.ERROR, function(event, data) {
                console.error('HLS Error:', data);
                if (data.fatal) {
                    hls.destroy();
                }
            });

            hls.loadSource(streamUrl);
            hls.attachMedia(player);
            hls.on(Hls.Events.MANIFEST_PARSED, function() {
                player.play().catch(e => console.error('Playback failed:', e));
            });
        } else if (player.canPlayType('application/vnd.apple.mpegurl')) {
            player.src = streamUrl;
            player.addEventListener('loadedmetadata', function() {
                player.play().catch(e => console.error('Playback failed:', e));
            });
        } else {
            console.error('HLS not supported');
        }
    } catch (e) {
        console.error('Error setting up stream:', e);
    }
}

function closeStreamModal() {
    const modal = document.getElementById('stream-modal');
    const player = document.getElementById('stream-player');
    
    player.pause();
    player.src = '';
    modal.classList.add('hidden');
    modal.classList.remove('flex');
}

// Add camera search handler
function handleCameraSearch(event) {
    const searchTerm = event.target.value.toLowerCase();
    const cameraList = document.getElementById('camera-list');
    
    if (!cameraList) return;
    
    Array.from(cameraList.children).forEach(camera => {
        const name = camera.querySelector('span')?.textContent.toLowerCase();
        if (name) {
            camera.style.display = name.includes(searchTerm) ? 'block' : 'none';
        }
    });
}

// Utility function for debouncing
function debounce(func, wait) {
    let timeout;
    return function(...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => {
            func(...args);
        }, wait);
    };
}

// Add missing handler functions for realtime updates
function addDetection(detection) {
    if (!detection) return;
    
    // Add to state
    dashboardState.detections.unshift(detection);
    if (dashboardState.detections.length > 50) {
        dashboardState.detections.pop();
    }
    
    // Update UI
    updateDetectionsList();
    showNotification();
}

function addAlert(alert) {
    if (!alert) return;
    
    // Add to state
    dashboardState.alerts.unshift(alert);
    if (dashboardState.alerts.length > 10) {
        dashboardState.alerts.pop();
    }
    
    // Update UI
    updateAlertsList();
}

// Single map instance
const mapState = {
    map: null,
    markers: null,
    currentMode: 'cameras'  // or 'detections'
};

// Initialize single map
async function initMap() {
    try {
        // Cleanup existing map
        if (mapState.map) {
            mapState.map.remove();
            mapState.map = null;
        }

        // Initialize main map
        mapState.map = L.map('main-map', {
            center: [36.1699, -115.1398],
            zoom: 11,
            zoomControl: true
        });

        L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
            attribution: '©CartoDB',
            maxZoom: 19
        }).addTo(mapState.map);

        // Initialize marker group
        mapState.markers = L.featureGroup().addTo(mapState.map);

        // Force a resize
        setTimeout(() => mapState.map.invalidateSize(), 100);

        return true;
    } catch (error) {
        console.error('Map initialization failed:', error);
        return false;
    }
}

// Handle map mode switching
function updateMapMode(mode) {
    mapState.currentMode = mode;
    mapState.markers.clearLayers();
    
    if (mode === 'cameras') {
        updateCameraMarkers();
    } else {
        updateDetectionMarkers();
    }
}

// Update camera markers
function updateCameraMarkers() {
    if (!mapState.map || !mapState.markers) return;

    mapState.markers.clearLayers();
    Object.entries(dashboardState.cameras).forEach(([id, camera]) => {
        if (!camera.lat || !camera.lng) return;

        const marker = L.marker([camera.lat, camera.lng], {
            icon: L.divIcon({
                className: `camera-marker ${camera.status}`,
                html: `<div class="camera-marker ${camera.status}"></div>`,
                iconSize: [24, 24]
            })
        });

        marker.bindPopup(createCameraPopup(camera));
        marker.on('click', () => showCameraAnalysis(camera));
        mapState.markers.addLayer(marker);
    });
}

// Update detection markers
function updateDetectionMarkers() {
    if (!mapState.map || !mapState.markers) return;

    mapState.markers.clearLayers();
    dashboardState.detections.forEach(detection => {
        const camera = dashboardState.cameras.get(detection.camera);
        if (!camera || !camera.lat || !camera.lng) return;

        const marker = L.marker([camera.lat, camera.lng], {
            icon: L.divIcon({
                className: 'target-marker',
                html: `<div class="target-marker"></div>`,
                iconSize: [32, 32]
            })
        });

        marker.bindPopup(createDetectionPopup(detection, camera));
        mapState.markers.addLayer(marker);
    });
}

// Helper functions for creating popups
function createCameraPopup(camera) {
    return `
        <div class="map-popup">
            <h3 class="font-semibold mb-2">${camera.name}</h3>
            <p class="text-sm text-slate-400 mb-2">${camera.location || 'Location not specified'}</p>
            <p class="text-xs text-slate-500">Last update: ${formatTime(camera.last_update)}</p>
            <button onclick="openStreamModal('${camera.name}', '${camera.url}')"
                    class="mt-2 w-full bg-blue-500 text-white px-3 py-1 rounded text-sm">
                View Stream
            </button>
        </div>
    `;
}

function createDetectionPopup(detection, camera) {
    return `
        <div class="map-popup">
            <h3 class="font-semibold mb-2">Detection at ${camera.name}</h3>
            <p class="text-sm text-slate-400">${detection.type}</p>
            <p class="text-xs text-slate-500">Detected: ${formatTime(detection.timestamp)}</p>
            <button onclick="showDetectionModal(${JSON.stringify(detection)})"
                    class="mt-2 w-full bg-blue-500 text-white px-3 py-1 rounded text-sm">
                View Details
            </button>
        </div>
    `;
}

// Remove all the old map initialization code and update existing functions to use mapState
// ...existing code...

document.addEventListener('DOMContentLoaded', function() {
    // Initialize the map and markers
    if (!window.mapInitialized) {
        var map = L.map('main-map').setView([51.505, -0.09], 13);

        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            maxZoom: 19,
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        }).addTo(map);

        window.mapInitialized = true;
    }

    // Fetch updated stats
    fetchStats();

    // Initialize SSE connection
    const eventSource = new EventSource('/stream');
    eventSource.onmessage = function(event) {
        const message = JSON.parse(event.data);
        if (message.type === 'detection') {
            updateDetectionsList(message.data);
        } else if (message.type === 'stats') {
            updateStats(message.data);
        }
    };

    eventSource.onerror = function() {
        console.error('SSE connection error');
    };
});

function updateDetectionsList(data) {
    const detectionsContainer = document.getElementById('recent-detections');
    if (!detectionsContainer) {
        console.warn('Detections container not found');
        return;
    }

    // Update the detections list with new data
    // ...existing code...
}

function updateStats(data) {
    // Update the stats with new data
    // ...existing code...
}

function fetchStats() {
    // Fetch updated stats from the server
    // ...existing code...
}

// Initialize dashboard
document.addEventListener('DOMContentLoaded', async () => {
    try {
        // Initialize map first
        await initMap();
        
        // Initialize UI components
        initializeUIComponents();
        
        // Start data updates
        initEventSource();
        startUpdateCycle();
    } catch (error) {
        console.error('Failed to initialize dashboard:', error);
    }
});

function initializeUIComponents() {
    // Initialize detection filters
    const filterSelect = document.getElementById('detection-filter');
    if (filterSelect) {
        filterSelect.addEventListener('change', () => updateDetectionsList());
    }

    // Initialize clear detections button
    const clearButton = document.getElementById('clear-detections');
    if (clearButton) {
        clearButton.addEventListener('click', () => {
            dashboardState.detections = [];
            updateDetectionsList();
        });
    }
}

function updateDetectionsList(newDetection = null) {
    const container = document.getElementById('recent-detections');
    const countElement = document.getElementById('detection-count');
    
    if (!container) {
        console.error('Detections container not found - check HTML structure');
        return;
    }

    // Add new detection if provided
    if (newDetection) {
        dashboardState.detections.unshift(newDetection);
    }

    // Update counter
    if (countElement) {
        countElement.textContent = dashboardState.detections.length;
    }

    // Clear container
    container.innerHTML = '';

    // Show message if no detections
    if (dashboardState.detections.length === 0) {
        container.innerHTML = `
            <div class="col-span-full text-center py-8 text-slate-400">
                <p>No detections found</p>
            </div>
        `;
        return;
    }

    // Add detection cards
    dashboardState.detections.forEach(detection => {
        const card = createDetectionCard(detection);
        container.appendChild(card);
    });
}

// Helper function to create detection cards
function createDetectionCard(detection) {
    const div = document.createElement('div');
    div.className = 'detection-card bg-slate-700 rounded-lg overflow-hidden';
    
    div.innerHTML = `
        <div class="relative">
            <img src="/matches/${detection.image}" 
                 alt="Detection ${detection.type}"
                 class="w-full h-48 object-cover"
                 onerror="this.src='/static/img/no-image.png'"
            />
            <div class="absolute top-2 right-2">
                <span class="bg-blue-500 text-white px-2 py-1 rounded text-sm">
                    ${Math.round(detection.confidence * 100)}%
                </span>
            </div>
        </div>
        <div class="p-4">
            <h3 class="text-lg font-semibold text-slate-200">${detection.type}</h3>
            <p class="text-sm text-slate-400">${formatTime(detection.timestamp)}</p>
        </div>
    `;
    
    return div;
}

// ...existing code...