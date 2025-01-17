from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class VehicleClassification(Base):
    __tablename__ = 'vehicle_classifications'
    
    id = Column(Integer, primary_key=True)
    detection_id = Column(Integer, ForeignKey('vehicle_detections.id'), nullable=False)
    make = Column(String)
    model = Column(String)
    year = Column(Integer, nullable=True)  # Optional
    color = Column(String, nullable=True)  # Optional
    vehicle_type = Column(String, nullable=False)  # car, truck, bus, motorcycle, etc.
    sub_type = Column(String)  # sedan, suv, pickup, etc.
    confidence_scores = Column(JSONB)  # Confidence scores for each classification
    features = Column(JSONB)  # Extracted visual features
    timestamp = Column(DateTime, nullable=False, index=True)

class VehicleDetectionRecord(Base):
    __tablename__ = 'vehicle_detections'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    stream_id = Column(String, nullable=False, index=True)
    zone = Column(String, nullable=False, index=True)
    location = Column(JSON)  # (lat, lng)
    direction = Column(Float)  # heading in degrees
    speed = Column(Float)  # speed in km/h
    frame_id = Column(String)  # Reference to captured frame
    bbox = Column(JSON)  # Bounding box coordinates
    classification = relationship("VehicleClassification", uselist=False, backref="detection")
    target_match_id = Column(String, ForeignKey('target_vehicles.id'))
    target_match = relationship("TargetVehicleRecord", back_populates="detections")
    priority_level = Column(Integer, default=1)
    processing_metadata = Column(JSONB)

class StreamMetricsRecord(Base):
    __tablename__ = 'stream_metrics'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    stream_id = Column(String, nullable=False, index=True)
    vehicle_count = Column(Integer)
    processing_time = Column(Float)
    frame_rate = Column(Float)
    detection_confidence = Column(Float)

class ZoneMetricsRecord(Base):
    __tablename__ = 'zone_metrics'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    zone = Column(String, nullable=False, index=True)
    total_vehicles = Column(Integer)
    vehicle_types = Column(JSONB)  # Dict of vehicle types and counts
    peak_hour_count = Column(Integer)
    peak_hour_start = Column(DateTime)
    active_streams = Column(Integer)
    vehicle_classifications = Column(JSONB)  # Extended stats including make/model
    popular_vehicles = Column(JSONB)  # Most common make/models in zone

class SystemAlert(Base):
    __tablename__ = 'system_alerts'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    alert_type = Column(String, nullable=False)
    stream_id = Column(String, index=True)
    message = Column(String)
    metrics = Column(JSONB)  # Additional metrics/context 

class TargetVehicleRecord(Base):
    """Database model for target vehicle configurations."""
    __tablename__ = 'target_vehicles'
    
    id = Column(String, primary_key=True)
    vehicle_type = Column(String, nullable=False, index=True)
    make = Column(String, index=True)
    model = Column(String, index=True)
    priority_level = Column(Integer, default=1)
    pre_screening_threshold = Column(Float, default=0.6)
    classification_threshold = Column(Float, default=0.85)
    active = Column(Boolean, default=True)
    metadata = Column(JSONB)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    detections = relationship("VehicleDetectionRecord", back_populates="target_match") 