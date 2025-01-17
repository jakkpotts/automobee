import { useState } from 'react';
import type { Detection } from '../types';

interface DetectionFeedProps {
  detections: Detection[];
}

export function DetectionFeed({ detections }: DetectionFeedProps) {
  const [filter, setFilter] = useState('all'); // 'all' | 'car' | 'truck' | 'motorcycle'

  const filteredDetections = detections.filter(detection => {
    if (filter === 'all') return true;
    return detection.vehicle_type.toLowerCase() === filter;
  });

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between pb-4 border-b border-gray-200">
        <h2 className="text-lg font-semibold text-gray-900">Live Detections</h2>
        <div className="flex space-x-2">
          {['all', 'car', 'truck', 'motorcycle'].map((type) => (
            <button
              key={type}
              onClick={() => setFilter(type)}
              className={`px-3 py-1 text-sm rounded-full ${
                filter === type
                  ? 'bg-blue-100 text-blue-700'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              {type.charAt(0).toUpperCase() + type.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Detection List */}
      <div className="flex-1 overflow-y-auto">
        {filteredDetections.length === 0 ? (
          <div className="flex items-center justify-center h-full text-gray-500">
            No detections found
          </div>
        ) : (
          <div className="space-y-4 py-4">
            {filteredDetections.map((detection) => (
              <div
                key={detection.id}
                className="bg-white rounded-lg shadow-sm border border-gray-200 p-4 hover:shadow-md transition-shadow"
              >
                <div className="flex items-start justify-between">
                  <div>
                    <h3 className="font-medium text-gray-900">
                      {detection.make} {detection.model}
                    </h3>
                    <p className="text-sm text-gray-500">
                      {detection.vehicle_type}
                    </p>
                  </div>
                  <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                    {(detection.confidence * 100).toFixed(1)}%
                  </span>
                </div>
                
                <div className="mt-2 text-sm text-gray-600">
                  <div className="flex items-center space-x-2">
                    <svg
                      className="h-4 w-4 text-gray-400"
                      fill="none"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <span>
                      {new Date(detection.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                </div>

                <div className="mt-3 flex items-center space-x-2">
                  <button className="text-sm text-blue-600 hover:text-blue-700">
                    View Details
                  </button>
                  <span className="text-gray-300">|</span>
                  <button className="text-sm text-blue-600 hover:text-blue-700">
                    Show on Map
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
} 