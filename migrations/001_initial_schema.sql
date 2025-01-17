-- Initial schema for AutomoBee

-- Zones table
CREATE TABLE zones (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description VARCHAR(255),
    coordinates JSONB NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Zone cameras table
CREATE TABLE zone_cameras (
    id SERIAL PRIMARY KEY,
    zone_id INTEGER REFERENCES zones(id),
    camera_id VARCHAR(100) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Detections table
CREATE TABLE detections (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    camera_id VARCHAR(100) NOT NULL,
    zone_id INTEGER REFERENCES zones(id),
    vehicle_type VARCHAR(50),
    make VARCHAR(50),
    model VARCHAR(50),
    confidence FLOAT,
    image_path VARCHAR(255),
    extra_data JSONB
);

-- Target matches table
CREATE TABLE target_matches (
    id SERIAL PRIMARY KEY,
    detection_id INTEGER REFERENCES detections(id),
    target_make VARCHAR(50),
    target_model VARCHAR(50),
    confidence FLOAT,
    matched_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_verified BOOLEAN DEFAULT FALSE,
    extra_data JSONB
);

-- System metrics table
CREATE TABLE system_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metric_type VARCHAR(50) NOT NULL,
    value FLOAT NOT NULL,
    unit VARCHAR(20),
    component VARCHAR(50),
    extra_data JSONB
);

-- Performance logs table
CREATE TABLE performance_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    event_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    message VARCHAR(255),
    component VARCHAR(50),
    extra_data JSONB
);

-- Create indexes
CREATE INDEX idx_detections_timestamp ON detections(timestamp);
CREATE INDEX idx_detections_camera_id ON detections(camera_id);
CREATE INDEX idx_detections_zone_id ON detections(zone_id);
CREATE INDEX idx_target_matches_detection_id ON target_matches(detection_id);
CREATE INDEX idx_system_metrics_timestamp ON system_metrics(timestamp);
CREATE INDEX idx_system_metrics_type ON system_metrics(metric_type);
CREATE INDEX idx_performance_logs_timestamp ON performance_logs(timestamp);
CREATE INDEX idx_performance_logs_severity ON performance_logs(severity); 