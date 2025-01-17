from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sqlalchemy import func, and_
import logging
from models import Detection, TargetMatch, SystemMetric, PerformanceLog, Zone
from database_service import DatabaseService

logger = logging.getLogger(__name__)

class AnalyticsService:
    def __init__(self, db_service: DatabaseService):
        self.db = db_service
    
    def get_detection_trends(self, days: int = 7) -> Dict[str, Any]:
        """Get detection trends over time"""
        session = self.db.get_session()
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days)
            
            # Get hourly detection counts
            hourly_counts = session.query(
                func.date_trunc('hour', Detection.timestamp).label('hour'),
                func.count(Detection.id).label('count')
            ).filter(
                Detection.timestamp.between(start_time, end_time)
            ).group_by('hour').order_by('hour').all()
            
            # Convert to pandas for analysis
            df = pd.DataFrame(hourly_counts, columns=['hour', 'count'])
            
            return {
                'hourly_counts': df.to_dict('records'),
                'total_detections': int(df['count'].sum()),
                'avg_hourly_rate': float(df['count'].mean()),
                'peak_hour': df.loc[df['count'].idxmax(), 'hour'].isoformat(),
                'peak_count': int(df['count'].max())
            }
        finally:
            session.close()
    
    def get_vehicle_distribution(self, days: int = 7) -> Dict[str, Any]:
        """Get distribution of vehicle types and makes"""
        session = self.db.get_session()
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days)
            
            # Get vehicle type distribution
            type_dist = dict(
                session.query(
                    Detection.vehicle_type,
                    func.count(Detection.id)
                ).filter(
                    Detection.timestamp.between(start_time, end_time)
                ).group_by(Detection.vehicle_type).all()
            )
            
            # Get make/model distribution
            make_dist = dict(
                session.query(
                    Detection.make,
                    func.count(Detection.id)
                ).filter(
                    Detection.timestamp.between(start_time, end_time)
                ).group_by(Detection.make).all()
            )
            
            return {
                'vehicle_types': type_dist,
                'vehicle_makes': make_dist,
                'total_vehicles': sum(type_dist.values())
            }
        finally:
            session.close()
    
    def get_target_match_analytics(self, days: int = 7) -> Dict[str, Any]:
        """Get analytics for target vehicle matches"""
        session = self.db.get_session()
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days)
            
            # Get match statistics
            matches = session.query(TargetMatch).filter(
                TargetMatch.matched_at.between(start_time, end_time)
            ).all()
            
            if not matches:
                return {
                    'total_matches': 0,
                    'avg_confidence': 0,
                    'verified_ratio': 0
                }
            
            df = pd.DataFrame([{
                'confidence': m.confidence,
                'is_verified': m.is_verified,
                'matched_at': m.matched_at
            } for m in matches])
            
            return {
                'total_matches': len(matches),
                'avg_confidence': float(df['confidence'].mean()),
                'verified_ratio': float(df['is_verified'].mean()),
                'hourly_distribution': df.groupby(
                    df['matched_at'].dt.hour
                )['confidence'].count().to_dict()
            }
        finally:
            session.close()
    
    def get_zone_analytics(self, days: int = 7) -> Dict[str, Any]:
        """Get analytics for detection zones"""
        session = self.db.get_session()
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days)
            
            # Get zone activity
            zone_activity = {}
            zones = session.query(Zone).filter(Zone.is_active == True).all()
            
            for zone in zones:
                detections = session.query(Detection).filter(
                    and_(
                        Detection.zone_id == zone.id,
                        Detection.timestamp.between(start_time, end_time)
                    )
                ).all()
                
                if detections:
                    df = pd.DataFrame([{
                        'timestamp': d.timestamp,
                        'vehicle_type': d.vehicle_type
                    } for d in detections])
                    
                    zone_activity[zone.name] = {
                        'total_detections': len(detections),
                        'vehicle_distribution': df['vehicle_type'].value_counts().to_dict(),
                        'hourly_average': float(len(detections) / (days * 24))
                    }
            
            return {
                'zone_activity': zone_activity,
                'total_zones': len(zones),
                'active_zones': len(zone_activity)
            }
        finally:
            session.close()
    
    def get_system_health_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate system health report"""
        session = self.db.get_session()
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours)
            
            # Get performance metrics
            metrics = session.query(SystemMetric).filter(
                SystemMetric.timestamp.between(start_time, end_time)
            ).all()
            
            if not metrics:
                return {'status': 'No metrics available'}
            
            df = pd.DataFrame([{
                'timestamp': m.timestamp,
                'metric_type': m.metric_type,
                'value': m.value,
                'component': m.component
            } for m in metrics])
            
            # Calculate component health scores
            component_health = {}
            for component in df['component'].unique():
                component_df = df[df['component'] == component]
                
                health_score = self._calculate_health_score(component_df)
                component_health[component] = {
                    'health_score': health_score,
                    'metrics': component_df.groupby('metric_type')['value'].agg([
                        'mean', 'min', 'max'
                    ]).to_dict('index')
                }
            
            # Get error rates
            error_logs = session.query(PerformanceLog).filter(
                and_(
                    PerformanceLog.timestamp.between(start_time, end_time),
                    PerformanceLog.severity.in_(['error', 'critical'])
                )
            ).all()
            
            error_rate = len(error_logs) / hours if hours > 0 else 0
            
            return {
                'component_health': component_health,
                'system_status': self._determine_system_status(component_health),
                'error_rate': error_rate,
                'total_errors': len(error_logs)
            }
        finally:
            session.close()
    
    def _calculate_health_score(self, df: pd.DataFrame) -> float:
        """Calculate health score for a component based on its metrics"""
        try:
            # Normalize values
            normalized = df.groupby('metric_type').transform(
                lambda x: (x - x.mean()) / x.std()
            )
            
            # Calculate score (0-100)
            score = 100 * (1 - np.abs(normalized['value']).mean())
            return float(np.clip(score, 0, 100))
        except Exception:
            return 0.0
    
    def _determine_system_status(self, component_health: Dict) -> str:
        """Determine overall system status based on component health"""
        scores = [c['health_score'] for c in component_health.values()]
        avg_score = np.mean(scores) if scores else 0
        
        if avg_score >= 90:
            return 'Healthy'
        elif avg_score >= 70:
            return 'Warning'
        else:
            return 'Critical'
    
    def generate_daily_report(self) -> Dict[str, Any]:
        """Generate comprehensive daily report"""
        try:
            detection_trends = self.get_detection_trends(days=1)
            vehicle_dist = self.get_vehicle_distribution(days=1)
            target_analytics = self.get_target_match_analytics(days=1)
            zone_analytics = self.get_zone_analytics(days=1)
            health_report = self.get_system_health_report(hours=24)
            
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'detection_metrics': detection_trends,
                'vehicle_distribution': vehicle_dist,
                'target_matches': target_analytics,
                'zone_activity': zone_analytics,
                'system_health': health_report
            }
        except Exception as e:
            logger.error(f"Error generating daily report: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time system metrics for dashboard"""
        try:
            # Get last hour's metrics
            metrics = self.get_system_health_report(hours=1)
            
            # Get recent detections
            recent_detections = self.db.get_recent_detections(limit=10)
            detection_rate = len(self.db.get_detections_by_timeframe(
                datetime.utcnow() - timedelta(hours=1),
                datetime.utcnow()
            ))
            
            # Get active zones
            active_zones = self.db.get_active_zones()
            zone_stats = {
                zone.name: len(zone.detections) 
                for zone in active_zones
            }
            
            return {
                'current_status': metrics['system_status'],
                'detection_rate': detection_rate,
                'active_zones': len(active_zones),
                'zone_activity': zone_stats,
                'recent_detections': [d.to_dict() for d in recent_detections],
                'component_health': metrics['component_health'],
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting real-time metrics: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            } 