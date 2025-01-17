from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import json

Base = declarative_base()

class Zone(Base):
    __tablename__ = 'zones'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(String(255))
    coordinates = Column(JSON, nullable=False)  # GeoJSON format
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    cameras = relationship("ZoneCamera", back_populates="zone")
    detections = relationship("Detection", back_populates="zone")

class ZoneCamera(Base):
    __tablename__ = 'zone_cameras'
    
    id = Column(Integer, primary_key=True)
    zone_id = Column(Integer, ForeignKey('zones.id'))
    camera_id = Column(String(100), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    zone = relationship("Zone", back_populates="cameras")

class Detection(Base):
    __tablename__ = 'detections'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    camera_id = Column(String(100), nullable=False)
    zone_id = Column(Integer, ForeignKey('zones.id'))
    vehicle_type = Column(String(50))
    make = Column(String(50))
    model = Column(String(50))
    confidence = Column(Float)
    image_path = Column(String(255))
    extra_data = Column(JSON)  # Additional detection data
    
    # Relationships
    zone = relationship("Zone", back_populates="detections")
    matches = relationship("TargetMatch", back_populates="detection")

class TargetMatch(Base):
    __tablename__ = 'target_matches'
    
    id = Column(Integer, primary_key=True)
    detection_id = Column(Integer, ForeignKey('detections.id'))
    target_make = Column(String(50))
    target_model = Column(String(50))
    confidence = Column(Float)
    matched_at = Column(DateTime, default=datetime.utcnow)
    is_verified = Column(Boolean, default=False)
    extra_data = Column(JSON)  # Additional match data
    
    # Relationships
    detection = relationship("Detection", back_populates="matches")

class SystemMetric(Base):
    __tablename__ = 'system_metrics'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    metric_type = Column(String(50), nullable=False)  # e.g., 'latency', 'memory', 'cpu'
    value = Column(Float, nullable=False)
    unit = Column(String(20))  # e.g., 'ms', 'MB', '%'
    component = Column(String(50))  # e.g., 'detector', 'classifier', 'dashboard'
    extra_data = Column(JSON)  # Additional metric data

class PerformanceLog(Base):
    __tablename__ = 'performance_logs'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    event_type = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False)  # 'info', 'warning', 'error', 'critical'
    message = Column(String(255))
    component = Column(String(50))
    extra_data = Column(JSON)  # Additional log data

def init_db(db_url):
    """Initialize the database and create all tables"""
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    return engine 