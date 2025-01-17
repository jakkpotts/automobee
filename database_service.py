from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.sql import func
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import logging
from models import Base, Zone, ZoneCamera, Detection, TargetMatch, SystemMetric, PerformanceLog

logger = logging.getLogger(__name__)

class DatabaseService:
    def __init__(self, engine):
        self.engine = engine
        self.Session = scoped_session(sessionmaker(bind=engine))
    
    def get_session(self):
        return self.Session()
    
    # Zone Operations
    def create_zone(self, name: str, coordinates: Dict, description: str = None) -> Zone:
        """Create a new zone"""
        session = self.get_session()
        try:
            zone = Zone(
                name=name,
                coordinates=coordinates,
                description=description
            )
            session.add(zone)
            session.commit()
            return zone
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating zone: {e}")
            raise
        finally:
            session.close()
    
    def get_active_zones(self) -> List[Zone]:
        """Get all active zones"""
        session = self.get_session()
        try:
            return session.query(Zone).filter(Zone.is_active == True).all()
        finally:
            session.close()
    
    def update_zone(self, zone_id: int, data: Dict[str, Any]) -> Optional[Zone]:
        """Update zone information"""
        session = self.get_session()
        try:
            zone = session.query(Zone).filter(Zone.id == zone_id).first()
            if zone:
                for key, value in data.items():
                    if hasattr(zone, key):
                        setattr(zone, key, value)
                session.commit()
            return zone
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating zone: {e}")
            raise
        finally:
            session.close()
    
    # Detection Operations
    def add_detection(self, detection_data: Dict[str, Any]) -> Detection:
        """Add a new detection record"""
        session = self.get_session()
        try:
            detection = Detection(**detection_data)
            session.add(detection)
            session.commit()
            return detection
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding detection: {e}")
            raise
        finally:
            session.close()
    
    def get_recent_detections(self, limit: int = 100) -> List[Detection]:
        """Get recent detections"""
        session = self.get_session()
        try:
            return session.query(Detection)\
                .order_by(Detection.timestamp.desc())\
                .limit(limit)\
                .all()
        finally:
            session.close()
    
    def get_detections_by_timeframe(self, start_time: datetime, end_time: datetime) -> List[Detection]:
        """Get detections within a specific timeframe"""
        session = self.get_session()
        try:
            return session.query(Detection)\
                .filter(Detection.timestamp.between(start_time, end_time))\
                .order_by(Detection.timestamp.desc())\
                .all()
        finally:
            session.close()
    
    # Target Match Operations
    def add_target_match(self, match_data: Dict[str, Any]) -> TargetMatch:
        """Add a new target match record"""
        session = self.get_session()
        try:
            match = TargetMatch(**match_data)
            session.add(match)
            session.commit()
            return match
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding target match: {e}")
            raise
        finally:
            session.close()
    
    def get_recent_matches(self, limit: int = 50) -> List[TargetMatch]:
        """Get recent target matches"""
        session = self.get_session()
        try:
            return session.query(TargetMatch)\
                .order_by(TargetMatch.matched_at.desc())\
                .limit(limit)\
                .all()
        finally:
            session.close()
    
    # System Metrics Operations
    def add_system_metric(self, metric_type: str, value: float, component: str,
                         unit: str = None, extra_data: Dict = None) -> SystemMetric:
        """Add a system metric"""
        try:
            metric = SystemMetric(
                metric_type=metric_type,
                value=value,
                unit=unit,
                component=component,
                extra_data=extra_data or {}
            )
            self.session.add(metric)
            self.session.commit()
            return metric
        except Exception as e:
            self.session.rollback()
            raise DatabaseError(f"Error adding system metric: {e}")
    
    def get_system_metrics(self, metric_type: str, hours: int = 24) -> List[SystemMetric]:
        """Get system metrics for a specific type within the last N hours"""
        session = self.get_session()
        try:
            start_time = datetime.utcnow() - timedelta(hours=hours)
            return session.query(SystemMetric)\
                .filter(SystemMetric.metric_type == metric_type,
                       SystemMetric.timestamp >= start_time)\
                .order_by(SystemMetric.timestamp.desc())\
                .all()
        finally:
            session.close()
    
    def get_performance_stats(self, component: str = None, hours: int = 24) -> Dict[str, Any]:
        """Get aggregated performance statistics"""
        session = self.get_session()
        try:
            start_time = datetime.utcnow() - timedelta(hours=hours)
            query = session.query(
                SystemMetric.metric_type,
                func.avg(SystemMetric.value).label('avg_value'),
                func.max(SystemMetric.value).label('max_value'),
                func.min(SystemMetric.value).label('min_value')
            ).filter(SystemMetric.timestamp >= start_time)
            
            if component:
                query = query.filter(SystemMetric.component == component)
            
            query = query.group_by(SystemMetric.metric_type)
            
            results = {}
            for row in query.all():
                results[row.metric_type] = {
                    'avg': float(row.avg_value),
                    'max': float(row.max_value),
                    'min': float(row.min_value)
                }
            return results
        finally:
            session.close()
    
    # Performance Logging Operations
    def add_performance_log(self, event_type: str, severity: str, message: str,
                           component: str = None, extra_data: Dict = None) -> PerformanceLog:
        """Add a performance log entry"""
        try:
            log = PerformanceLog(
                event_type=event_type,
                severity=severity,
                message=message,
                component=component,
                extra_data=extra_data or {}
            )
            self.session.add(log)
            self.session.commit()
            return log
        except Exception as e:
            self.session.rollback()
            raise DatabaseError(f"Error adding performance log: {e}")
    
    def get_recent_performance_logs(self, severity: str = None, 
                                  component: str = None, limit: int = 100) -> List[PerformanceLog]:
        """Get recent performance logs with optional filtering"""
        session = self.get_session()
        try:
            query = session.query(PerformanceLog)
            
            if severity:
                query = query.filter(PerformanceLog.severity == severity)
            if component:
                query = query.filter(PerformanceLog.component == component)
            
            return query.order_by(PerformanceLog.timestamp.desc()).limit(limit).all()
        finally:
            session.close()
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary statistics"""
        session = self.get_session()
        try:
            start_time = datetime.utcnow() - timedelta(hours=hours)
            
            # Get severity counts
            severity_counts = dict(
                session.query(
                    PerformanceLog.severity,
                    func.count(PerformanceLog.id)
                ).filter(PerformanceLog.timestamp >= start_time)
                .group_by(PerformanceLog.severity)
                .all()
            )
            
            # Get component event counts
            component_counts = dict(
                session.query(
                    PerformanceLog.component,
                    func.count(PerformanceLog.id)
                ).filter(PerformanceLog.timestamp >= start_time)
                .group_by(PerformanceLog.component)
                .all()
            )
            
            return {
                'severity_distribution': severity_counts,
                'component_distribution': component_counts,
                'total_events': sum(severity_counts.values())
            }
        finally:
            session.close() 