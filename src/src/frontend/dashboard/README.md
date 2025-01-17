# AutomoBee Dashboard 🚗

A modern, real-time vehicle detection and monitoring system built with React, TypeScript, and WebSocket.

![AutomoBee Dashboard](./docs/dashboard-preview.png)

## ✨ Features

### 🎯 Core Functionality
- **Real-time Vehicle Detection** - Live monitoring of vehicle detections across multiple cameras
- **Interactive Map View** - Geospatial visualization of cameras and detections
- **Metrics Dashboard** - Live system performance metrics and statistics
- **Detection Feed** - Real-time feed of vehicle detections with filtering capabilities
- **Alert System** - Priority-based notification system for important events

### 💅 UI/UX Features
- **Command Palette (⌘K)** - Quick access to all dashboard features
- **Multiple Layouts**
  - Default (Map + Sidebar)
  - Split View
  - Full Screen Map
  - Camera Grid
- **Dark Mode Support** - Full dark mode implementation with system preference detection
- **Responsive Design** - Optimized for different screen sizes
- **Modern UI Components**
  - Interactive maps with clustering
  - Real-time charts and graphs
  - Toast notifications
  - Slide-over panels
  - Floating action menu

### 🔧 Technical Features
- **WebSocket Integration** - Real-time data updates with automatic reconnection
- **Type Safety** - Full TypeScript implementation
- **Modern Styling** - Tailwind CSS with custom configuration
- **Component Architecture** - Modular and reusable components
- **Performance Optimized** - Efficient rendering and data management

## 🚀 Getting Started

### Prerequisites
- Node.js (v18 or higher)
- npm or yarn
- Modern web browser

### Installation

1. Clone the repository:
\`\`\`bash
git clone https://github.com/your-username/automobee.git
cd automobee/src/frontend/dashboard
\`\`\`

2. Install dependencies:
\`\`\`bash
npm install
# or
yarn install
\`\`\`

3. Set up static assets:
\`\`\`bash
# Make the script executable
chmod +x scripts/download-assets.sh

# Run the asset download script
./scripts/download-assets.sh
\`\`\`

The script will:
- Create necessary directories in `/public`
- Download Leaflet map marker icons
- Create placeholder icons for cameras and detections
- Convert SVG icons to PNG if ImageMagick is available

4. Start the development server:
\`\`\`bash
npm run dev
# or
yarn dev
\`\`\`

## 🏗️ Project Structure

\`\`\`
src/
├── components/          # React components
│   ├── AlertSystem/    # Alert management
│   ├── CommandPalette/ # Quick command access
│   ├── DetectionFeed/  # Real-time detections
│   ├── MapView/        # Interactive map
│   └── MetricsDashboard/ # System metrics
├── services/           # Core services
│   └── websocket.ts    # WebSocket management
├── types/              # TypeScript definitions
├── styles/            # Global styles
└── App.tsx            # Main application
\`\`\`

## 🛠️ Technology Stack

- **Frontend Framework**: React 18
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **State Management**: React Hooks
- **Real-time Communication**: WebSocket
- **UI Components**:
  - Headless UI
  - Tremor
  - React Leaflet
  - CMDK
  - React Hot Toast

## 🔌 WebSocket Events

The dashboard listens to the following WebSocket events:
- \`camera_update\`: Camera status and location updates
- \`detection\`: New vehicle detections
- \`metrics\`: System performance metrics
- \`alert\`: System alerts and notifications

## 🎨 Customization

### Theme Configuration
The dashboard uses Tailwind CSS for styling. Customize the theme in \`tailwind.config.js\`:
- Colors
- Typography
- Spacing
- Breakpoints
- Dark mode preferences

### Layout Options
Modify available layouts in the \`FloatingActionMenu\` component:
- Default view
- Split view
- Fullscreen map
- Custom layouts

### Static Assets
The dashboard uses several static assets located in the `/public` directory:

#### Map Icons
- `/public/leaflet/` - Contains Leaflet map marker icons
  - `marker-icon.png`
  - `marker-icon-2x.png`
  - `marker-shadow.png`

#### Custom Icons
- `/public/icons/` - Contains application-specific icons
  - `camera-icon.png` (or .svg) - Camera location marker
  - `detection-icon.png` (or .svg) - Vehicle detection marker

To customize these icons:
1. Replace the existing files in the respective directories
2. Update the icon configurations in `MapView.tsx` if necessary
3. Run the development server to see changes

## 📝 Contributing

1. Fork the repository
2. Create your feature branch (\`git checkout -b feature/AmazingFeature\`)
3. Commit your changes (\`git commit -m 'Add some AmazingFeature'\`)
4. Push to the branch (\`git push origin feature/AmazingFeature\`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Support

For support, email support@automobee.com or join our Slack channel.
