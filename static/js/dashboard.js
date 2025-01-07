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

// Enhanced map initialization with Leaflet
async function initMap() {
    return withErrorBoundary(async () => {
        if (dashboardState.map) return;

        try {
            // Initialize map centered on Las Vegas
            dashboardState.map = L.map('map').setView([36.1699, -115.1398], 11);
            
            // Use Stadia Maps light theme
            L.tileLayer('https://tiles.stadiamaps.com/tiles/alidade_smooth/{z}/{x}/{y}{r}.png', {
                attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
                maxZoom: 19
            }).addTo(dashboardState.map);

            // Initialize marker layer group
            dashboardState.markers = new L.featureGroup().addTo(dashboardState.map);

            // Custom marker icons
            dashboardState.icons = {
                active: L.divIcon({
                    className: 'custom-marker',
                    html: `<div class="w-4 h-4 bg-green-500 rounded-full border-2 border-white shadow-lg"></div>`,
                    iconSize: [16, 16]
                }),
                error: L.divIcon({
                    className: 'custom-marker',
                    html: `<div class="w-4 h-4 bg-red-500 rounded-full border-2 border-white shadow-lg"></div>`,
                    iconSize: [16, 16]
                })
            };

            // Initial stats fetch will include camera locations
            updateStats();

        } catch (e) {
            console.error('Error initializing map:', e);
        }

    }, null);
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
function initEventSource() {
    let retryCount = 0;
    const maxRetries = 5;
    const baseDelay = 1000;

    function connect() {
        const eventSource = new EventSource('/stream');
        
        eventSource.onmessage = function(event) {
            console.log("🔄 Received SSE message:", event.data);
            try {
                const data = JSON.parse(event.data);
                handleRealtimeUpdate(data);
            } catch (e) {
                console.error("Failed to parse SSE message:", e);
            }
        };

        eventSource.onerror = function(error) {
            console.error("❌ SSE Error:", error);
            eventSource.close();
            
            if (retryCount < maxRetries) {
                const delay = Math.min(1000 * Math.pow(2, retryCount), 10000);
                retryCount++;
                console.log(`Reconnecting in ${delay/1000} seconds... (Attempt ${retryCount}/${maxRetries})`);
                setTimeout(connect, delay);
            } else {
                console.error("Max retry attempts reached. Please refresh the page.");
            }
        };

        eventSource.onopen = function() {
            console.log("✅ SSE Connection established");
            retryCount = 0; // Reset retry count on successful connection
        };
    }

    connect();
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
            
            // Update state
            dashboardState.updateDetections(data.recent_detections || []);
            dashboardState.alerts = data.active_alerts || [];
            
            // Update UI elements
            updateDetectionsList();
            updateAlertsList();
            updateCameraList(data.camera_stats || {});
            
            // Update map markers with camera locations and status
            if (data.camera_locations) {
                updateMapMarkers(data.camera_locations);
            }
            
            // Update stats counters
            updateStatsCounters({
                total_detections: data.total_detections || 0,
                active_cameras: data.active_cameras || 0,
                detection_rate: Math.round((data.total_detections || 0) / 
                    (((Date.now() - new Date(data.uptime_start || Date.now()).getTime()) / 3600000) || 1))
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
document.addEventListener('DOMContentLoaded', () => {
    console.log("Initializing dashboard...");
    
    const cleanup = new Set();

    // Initialize map
    const mapElement = document.getElementById('map');
    if (mapElement && !dashboardState.map) {
        initMap().catch(error => {
            console.error('Map initialization failed:', error);
            if (document.getElementById('map-loading')) {
                document.getElementById('map-loading').innerHTML = 
                    '<div class="text-red-500">Failed to load map. <button onclick="retryMapLoad()">Retry</button></div>';
            }
        });
    }
    
    // Initialize event handlers
    const searchInput = document.getElementById('camera-search');
    if (searchInput) {
        searchInput.addEventListener('input', handleCameraSearch);
        cleanup.add(() => searchInput.removeEventListener('input', handleCameraSearch));
    }

    const filterElement = document.getElementById('detection-filter');
    if (filterElement) {
        filterElement.addEventListener('change', () => debouncedUpdate());
        cleanup.add(() => filterElement.removeEventListener('change', debouncedUpdate));
    }

    // Initialize camera list if it doesn't exist
    if (!document.getElementById('camera-list')) {
        const cameraListContainer = document.getElementById('camera-status-list');
        if (cameraListContainer) {
            const cameraList = document.createElement('div');
            cameraList.id = 'camera-list';
            cameraListContainer.appendChild(cameraList);
        }
    }

    // Start data updates
    updateStats();
    initEventSource();
    startUpdateCycle();

    // Add cleanup on page unload
    window.addEventListener('unload', () => {
        cleanup.forEach(cleanupFn => cleanupFn());
    });
});

// Update markers for OpenLayers
function updateMapMarkers(locations) {
    if (!dashboardState.map || !dashboardState.markers) {
        console.error('Map or markers not initialized');
        return;
    }

    try {
        // Clear existing markers
        dashboardState.markers.clearLayers();

        // Add new markers
        Object.entries(locations).forEach(([name, location]) => {
            try {
                if (!location.lat || !location.lng) {
                    console.warn(`Invalid coordinates for camera ${name}`);
                    return;
                }

                const marker = L.marker(
                    [location.lat, location.lng],
                    { 
                        icon: location.status === 'error' ? 
                            dashboardState.icons.error : 
                            dashboardState.icons.active
                    }
                );

                // Add popup with camera info
                const popup = L.popup().setContent(`
                    <div class="camera-popup">
                        <h3>${name}</h3>
                        <div class="status">
                            <span class="status-dot ${location.status || 'active'}"></span>
                            <span>${location.status || 'Active'}</span>
                        </div>
                        <button onclick="openStreamModal('${name}', '${location.url}')"
                                class="bg-blue-500 text-white px-3 py-1 rounded text-sm hover:bg-blue-600 transition-colors w-full">
                            View Stream
                        </button>
                    </div>
                `);

                marker.bindPopup(popup);

                // Add hover effect
                marker.on('mouseover', function() {
                    this.openPopup();
                });

                // Add to feature group
                dashboardState.markers.addLayer(marker);

            } catch (e) {
                console.error(`Error adding marker for ${name}:`, e);
            }
        });

        // Fit map to markers if we have any
        if (dashboardState.markers.getLayers().length > 0) {
            const bounds = dashboardState.markers.getBounds();
            dashboardState.map.fitBounds(bounds, {
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
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}