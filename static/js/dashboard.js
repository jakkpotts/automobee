// Dashboard state management
let state = {
    cameras: new Map(),
    detections: [],
    alerts: [],
    map: null,
    markers: new Map(),
    selectedCamera: null
};

// Initialize map
function initMap() {
    state.map = L.map('map').setView([36.1699, -115.1398], 12);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
    }).addTo(state.map);
}

// Initialize SSE connection
const eventSource = new EventSource('/stream');
eventSource.onmessage = function(event) {
    const data = JSON.parse(event.data);
    handleRealtimeUpdate(data);
};

// Handle realtime updates
function handleRealtimeUpdate(data) {
    if (data.type === 'detection') {
        addDetection(data);
    } else if (data.type === 'alert') {
        addAlert(data);
    } else if (data.type === 'camera_status') {
        updateCameraStatus(data);
    }
}

// Update dashboard stats
function updateStats() {
    fetch('/api/stats')
        .then(response => response.json())
        .then(data => {
            // Update counters
            document.getElementById('total-detections').textContent = data.total_detections;
            document.getElementById('detection-rate').textContent = `${data.detection_rate}/hour`;
            document.getElementById('active-cameras').textContent = data.active_cameras;
            document.getElementById('error-rate').textContent = `${data.error_rate}%`;
            document.getElementById('uptime').textContent = `${data.uptime.hours}h ${data.uptime.minutes}m`;
            
            // Update camera markers
            updateMapMarkers(data.camera_locations);
            
            // Update camera list
            updateCameraList(data.camera_stats);
            
            // Limit stored detections
            state.detections = data.recent_detections.slice(0, 30);
            updateDetectionsList();
            
            // Update alerts
            state.alerts = data.active_alerts.slice(0, 10);
            updateAlertsList();
        });
}

// Update map markers
function updateMapMarkers(locations) {
    for (const [name, location] of Object.entries(locations)) {
        if (!state.markers.has(name)) {
            const marker = L.marker([location.lat, location.lng])
                .bindPopup(location.name)
                .addTo(state.map);
            state.markers.set(name, marker);
        }
    }
}

// Update camera list
function updateCameraList(stats) {
    const container = document.getElementById('camera-list');
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

// Update detections list
function updateDetectionsList() {
    const container = document.getElementById('recent-detections');
    const filter = document.getElementById('detection-filter').value;
    
    const filteredDetections = filter === 'all' 
        ? state.detections 
        : state.detections.filter(d => d.type === filter);
    
    container.innerHTML = '';
    
    for (const detection of filteredDetections) {
        const div = document.createElement('div');
        div.className = 'detection-card bg-gray-50 p-3 rounded-lg cursor-pointer';
        div.innerHTML = `
            <div class="flex items-center justify-between">
                <div>
                    <span class="font-medium">${detection.type}</span>
                    <span class="text-sm text-gray-500 ml-2">${formatTime(detection.timestamp)}</span>
                </div>
                <span class="text-sm bg-blue-100 text-blue-800 px-2 py-1 rounded">
                    ${Math.round(detection.confidence * 100)}%
                </span>
            </div>
            ${detection.make ? `<p class="text-sm text-gray-600 mt-1">${detection.make} ${detection.model || ''}</p>` : ''}
        `;
        div.onclick = () => showDetectionModal(detection);
        container.appendChild(div);
    }
}

// Update alerts list
function updateAlertsList() {
    const container = document.getElementById('alerts-container');
    container.innerHTML = '';
    
    for (const alert of state.alerts) {
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
                <span class="text-sm text-red-600">${formatTime(alert.timestamp)}</span>
            </div>
            <p class="text-sm text-red-700 mt-1">${alert.message}</p>
        `;
        container.appendChild(div);
    }
}

// Show detection modal
function showDetectionModal(detection) {
    const modal = document.getElementById('detection-modal');
    const title = document.getElementById('modal-title');
    const image = document.getElementById('modal-image');
    const details = document.getElementById('modal-details');
    
    title.textContent = `${detection.type} Detection`;
    image.src = detection.image;
    
    details.innerHTML = `
        <div>
            <p class="text-sm text-gray-500">Time</p>
            <p class="font-medium">${formatTime(detection.timestamp)}</p>
        </div>
        <div>
            <p class="text-sm text-gray-500">Camera</p>
            <p class="font-medium">${detection.camera}</p>
        </div>
        <div>
            <p class="text-sm text-gray-500">Confidence</p>
            <p class="font-medium">${Math.round(detection.confidence * 100)}%</p>
        </div>
        ${detection.make ? `
        <div>
            <p class="text-sm text-gray-500">Vehicle</p>
            <p class="font-medium">${detection.make} ${detection.model || ''}</p>
        </div>
        ` : ''}
    `;
    
    modal.classList.remove('hidden');
    modal.classList.add('flex');
}

// Close detection modal
function closeDetectionModal() {
    const modal = document.getElementById('detection-modal');
    modal.classList.add('hidden');
    modal.classList.remove('flex');
}

// Utility function to format timestamps
function formatTime(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleTimeString();
}

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    initMap();
    updateStats();
    
    // Set up periodic updates
    setInterval(updateStats, 30000);
    
    // Set up detection filter
    document.getElementById('detection-filter').addEventListener('change', updateDetectionsList);
}); 