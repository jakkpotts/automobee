from functools import wraps
from flask import jsonify, current_app
import logging
import traceback
from typing import Dict, Any, Optional, Type
import sys
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ErrorContext:
    timestamp: datetime
    error_type: str
    message: str
    traceback: str
    severity: ErrorSeverity
    component: str
    metadata: Dict[str, Any]

class ErrorHandler:
    def __init__(self, app=None):
        self.app = app
        self.error_handlers = {}
        self.recovery_handlers = {}
        self.error_counts = {}
        self.alert_thresholds = {
            ErrorSeverity.LOW: 100,      # Alert after 100 low severity errors
            ErrorSeverity.MEDIUM: 50,     # Alert after 50 medium severity errors
            ErrorSeverity.HIGH: 10,       # Alert after 10 high severity errors
            ErrorSeverity.CRITICAL: 1     # Alert immediately for critical errors
        }
        
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize error handler with Flask app"""
        self.app = app
        
        # Register default error handlers
        @app.errorhandler(Exception)
        def handle_exception(e):
            return self.handle_error(e)
    
    def register_error_handler(self, error_type: Type[Exception], handler: callable):
        """Register a custom error handler for a specific exception type"""
        self.error_handlers[error_type] = handler
    
    def register_recovery_handler(self, error_type: Type[Exception], handler: callable):
        """Register a recovery handler for a specific exception type"""
        self.recovery_handlers[error_type] = handler
    
    def handle_error(self, error: Exception, component: str = None) -> tuple:
        """Handle an error and return appropriate response"""
        error_context = self._create_error_context(error, component)
        
        # Log error
        self._log_error(error_context)
        
        # Update error counts
        self._update_error_counts(error_context)
        
        # Check if we need to alert
        self._check_alert_threshold(error_context)
        
        # Try to recover
        self._attempt_recovery(error_context)
        
        # Get custom handler if exists
        handler = self.error_handlers.get(type(error))
        if handler:
            return handler(error)
        
        # Default error response
        return self._create_error_response(error_context)
    
    def _create_error_context(self, error: Exception, component: str = None) -> ErrorContext:
        """Create error context with all relevant information"""
        return ErrorContext(
            timestamp=datetime.utcnow(),
            error_type=type(error).__name__,
            message=str(error),
            traceback=traceback.format_exc(),
            severity=self._determine_severity(error),
            component=component or 'unknown',
            metadata=self._gather_error_metadata(error)
        )
    
    def _determine_severity(self, error: Exception) -> ErrorSeverity:
        """Determine error severity based on error type and context"""
        if isinstance(error, (SystemError, MemoryError)):
            return ErrorSeverity.CRITICAL
        elif isinstance(error, (ValueError, TypeError)):
            return ErrorSeverity.MEDIUM
        elif isinstance(error, (KeyError, AttributeError)):
            return ErrorSeverity.HIGH
        return ErrorSeverity.LOW
    
    def _gather_error_metadata(self, error: Exception) -> Dict[str, Any]:
        """Gather additional metadata about the error"""
        try:
            return {
                'python_version': sys.version,
                'error_module': error.__module__,
                'error_line': traceback.extract_tb(error.__traceback__)[-1].lineno,
                'error_file': traceback.extract_tb(error.__traceback__)[-1].filename,
                'memory_usage': self._get_memory_usage()
            }
        except Exception as e:
            logger.error(f"Error gathering metadata: {e}")
            return {}
    
    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return None
    
    def _log_error(self, context: ErrorContext):
        """Log error with appropriate severity"""
        log_message = (
            f"Error in {context.component}: {context.error_type}\n"
            f"Message: {context.message}\n"
            f"Severity: {context.severity.value}\n"
            f"Traceback:\n{context.traceback}"
        )
        
        if context.severity in (ErrorSeverity.CRITICAL, ErrorSeverity.HIGH):
            logger.error(log_message, extra={'metadata': context.metadata})
        elif context.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message, extra={'metadata': context.metadata})
        else:
            logger.info(log_message, extra={'metadata': context.metadata})
    
    def _update_error_counts(self, context: ErrorContext):
        """Update error count statistics"""
        key = f"{context.component}:{context.error_type}"
        if key not in self.error_counts:
            self.error_counts[key] = {
                'count': 0,
                'first_seen': context.timestamp,
                'last_seen': context.timestamp,
                'severity': context.severity
            }
        
        self.error_counts[key]['count'] += 1
        self.error_counts[key]['last_seen'] = context.timestamp
    
    def _check_alert_threshold(self, context: ErrorContext):
        """Check if error count exceeds alert threshold"""
        key = f"{context.component}:{context.error_type}"
        count = self.error_counts[key]['count']
        threshold = self.alert_thresholds[context.severity]
        
        if count >= threshold:
            self._send_alert(context)
    
    def _send_alert(self, context: ErrorContext):
        """Send alert for error threshold exceeded"""
        try:
            alert_message = {
                'type': 'error_alert',
                'severity': context.severity.value,
                'component': context.component,
                'error_type': context.error_type,
                'count': self.error_counts[f"{context.component}:{context.error_type}"]['count'],
                'message': context.message,
                'timestamp': context.timestamp.isoformat()
            }
            
            # Send to alert system (implement your alert mechanism)
            logger.critical(f"ALERT: {alert_message}")
            
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
    
    def _attempt_recovery(self, context: ErrorContext):
        """Attempt to recover from error if recovery handler exists"""
        handler = self.recovery_handlers.get(context.error_type)
        if handler:
            try:
                handler(context)
            except Exception as e:
                logger.error(f"Error in recovery handler: {e}")
    
    def _create_error_response(self, context: ErrorContext) -> tuple:
        """Create standardized error response"""
        response = {
            'error': {
                'type': context.error_type,
                'message': context.message,
                'severity': context.severity.value,
                'timestamp': context.timestamp.isoformat()
            }
        }
        
        # Include debug information in development
        if current_app.debug:
            response['error']['debug'] = {
                'traceback': context.traceback,
                'metadata': context.metadata
            }
        
        status_code = self._get_status_code(context.error_type)
        return jsonify(response), status_code
    
    def _get_status_code(self, error_type: str) -> int:
        """Get appropriate HTTP status code for error type"""
        status_codes = {
            'ValueError': 400,
            'TypeError': 400,
            'KeyError': 404,
            'AuthenticationError': 401,
            'PermissionError': 403,
            'NotFoundError': 404,
            'DatabaseError': 503,
            'TimeoutError': 504
        }
        return status_codes.get(error_type, 500)
    
    def error_handler(self, component: str = None):
        """Decorator for error handling"""
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                try:
                    return f(*args, **kwargs)
                except Exception as e:
                    return self.handle_error(e, component)
            return decorated_function
        return decorator 