import { useState, useEffect } from 'react';
import { CommandPalette } from './components/CommandPalette';
import { MapView } from './components/MapView';
import { MetricsDashboard } from './components/MetricsDashboard';
import { DetectionFeed } from './components/DetectionFeed';
import { AlertSystem } from './components/AlertSystem';
import { Toaster } from 'react-hot-toast';
import { FloatingActionMenu } from './components/FloatingActionMenu';
import type { Camera, Detection, SystemMetrics } from './types';
import wsService from './services/websocket';

function App() {
  const [isCommandPaletteOpen, setIsCommandPaletteOpen] = useState(false);
  const [layout, setLayout] = useState('default'); // 'default' | 'split' | 'fullscreen'
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [detections, setDetections] = useState<Detection[]>([]);

  useEffect(() => {
    // Connect to WebSocket and set up callbacks
    wsService.connect({
      onCameraUpdate: (updatedCameras) => {
        setCameras(updatedCameras);
      },
      onDetection: (newDetection) => {
        setDetections((prev) => {
          // Keep only the last 100 detections
          const updated = [newDetection, ...prev].slice(0, 100);
          return updated;
        });
      },
      onMetricsUpdate: (updatedMetrics) => {
        setMetrics(updatedMetrics);
      },
    });

    // Cleanup on unmount
    return () => {
      wsService.disconnect();
    };
  }, []);

  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setIsCommandPaletteOpen(true);
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, []);

  return (
    <div className="h-screen bg-gray-50 dark:bg-gray-900">
      {/* Command Palette */}
      <CommandPalette 
        isOpen={isCommandPaletteOpen} 
        onClose={() => setIsCommandPaletteOpen(false)} 
      />

      {/* Main Layout */}
      <div className="flex h-full">
        {/* Main Content Area */}
        <main className="flex-1 overflow-hidden">
          <div className="h-full flex flex-col">
            {/* Top Bar */}
            <div className="bg-white dark:bg-gray-800 shadow-sm z-10 p-4">
              <h1 className="text-2xl font-semibold text-gray-900 dark:text-white">
                AutomoBee Dashboard
              </h1>
            </div>

            {/* Dynamic Content Area */}
            <div className="flex-1 overflow-hidden">
              {layout === 'default' && (
                <div className="grid grid-cols-12 gap-4 h-full p-4">
                  {/* Map View - 8 columns */}
                  <div className="col-span-8 bg-white dark:bg-gray-800 rounded-lg shadow-sm">
                    <MapView cameras={cameras} detections={detections} />
                  </div>

                  {/* Right Side Panel - 4 columns */}
                  <div className="col-span-4 space-y-4">
                    {/* Metrics Dashboard */}
                    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm p-4">
                      <MetricsDashboard metrics={metrics} />
                    </div>

                    {/* Detection Feed */}
                    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm p-4 flex-1">
                      <DetectionFeed detections={detections} />
                    </div>
                  </div>
                </div>
              )}

              {layout === 'split' && (
                <div className="grid grid-cols-2 gap-4 h-full p-4">
                  <MapView cameras={cameras} detections={detections} />
                  <DetectionFeed detections={detections} />
                </div>
              )}

              {layout === 'fullscreen' && (
                <div className="h-full">
                  <MapView cameras={cameras} detections={detections} />
                </div>
              )}
            </div>
          </div>
        </main>

        {/* Alert System - Slide-over panel */}
        <AlertSystem />
      </div>

      {/* Floating Action Menu */}
      <FloatingActionMenu onLayoutChange={setLayout} />

      {/* Toast Notifications */}
      <Toaster position="top-right" />
    </div>
  );
}

export default App;
