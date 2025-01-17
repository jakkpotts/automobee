#!/bin/bash

# Set the base directory to the project root
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PUBLIC_DIR="$BASE_DIR/public"

echo "📁 Creating asset directories..."
mkdir -p "$PUBLIC_DIR/icons"
mkdir -p "$PUBLIC_DIR/leaflet"

# Function to download with error checking
download_file() {
    local url=$1
    local output=$2
    echo "⬇️  Downloading $(basename "$output")..."
    if curl -f -L -o "$output" "$url"; then
        echo "✅ Downloaded $(basename "$output")"
    else
        echo "❌ Failed to download $(basename "$output")"
        return 1
    fi
}

echo "🗺️  Downloading Leaflet assets..."
# Download Leaflet marker icons
download_file "https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-blue.png" "$PUBLIC_DIR/leaflet/marker-icon-2x.png"
download_file "https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-blue.png" "$PUBLIC_DIR/leaflet/marker-icon.png"
download_file "https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-shadow.png" "$PUBLIC_DIR/leaflet/marker-shadow.png"

echo "🎨 Creating placeholder icons..."
# Create placeholder icons using base64 encoded SVGs
# Camera icon placeholder
echo '<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/></svg>' > "$PUBLIC_DIR/icons/camera-icon.svg"

# Detection icon placeholder
echo '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 8v8m-4-4h8"/></svg>' > "$PUBLIC_DIR/icons/detection-icon.svg"

# Convert SVGs to PNGs using ImageMagick if available
if command -v convert >/dev/null 2>&1; then
    echo "🎨 Converting SVGs to PNGs..."
    convert "$PUBLIC_DIR/icons/camera-icon.svg" "$PUBLIC_DIR/icons/camera-icon.png"
    convert "$PUBLIC_DIR/icons/detection-icon.svg" "$PUBLIC_DIR/icons/detection-icon.png"
    rm "$PUBLIC_DIR/icons/camera-icon.svg" "$PUBLIC_DIR/icons/detection-icon.svg"
    echo "✅ Icons converted to PNG format"
else
    echo "⚠️  ImageMagick not found. SVG icons will be used instead of PNGs."
    echo "   Install ImageMagick to convert icons to PNG format:"
    echo "   - macOS: brew install imagemagick"
    echo "   - Ubuntu: sudo apt-get install imagemagick"
    echo "   - Windows: https://imagemagick.org/script/download.php"
fi

echo "✨ Asset setup complete!"
echo "📍 Assets location: $PUBLIC_DIR"
ls -R "$PUBLIC_DIR" 