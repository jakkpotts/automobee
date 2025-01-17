class AutomoBeeException(Exception):
    """Base exception class for AutomoBee system"""
    def __init__(self, message: str = None, code: str = None):
        self.message = message or "An unexpected error occurred"
        self.code = code or "INTERNAL_ERROR"
        super().__init__(self.message)

class AuthenticationError(AutomoBeeException):
    """Raised when authentication fails"""
    def __init__(self, message: str = None):
        super().__init__(
            message or "Authentication failed",
            code="AUTH_ERROR"
        )

class AuthorizationError(AutomoBeeException):
    """Raised when user lacks required permissions"""
    def __init__(self, message: str = None):
        super().__init__(
            message or "Insufficient permissions",
            code="AUTH_PERMISSION_ERROR"
        )

class ValidationError(AutomoBeeException):
    """Raised when input validation fails"""
    def __init__(self, message: str = None, field: str = None):
        self.field = field
        super().__init__(
            message or f"Validation failed for field: {field}",
            code="VALIDATION_ERROR"
        )

class RateLimitError(AutomoBeeException):
    """Raised when rate limit is exceeded"""
    def __init__(self, message: str = None):
        super().__init__(
            message or "Rate limit exceeded",
            code="RATE_LIMIT_ERROR"
        )

class DatabaseError(AutomoBeeException):
    """Raised when database operations fail"""
    def __init__(self, message: str = None, operation: str = None):
        self.operation = operation
        super().__init__(
            message or f"Database operation failed: {operation}",
            code="DB_ERROR"
        )

class DetectionError(AutomoBeeException):
    """Raised when vehicle detection fails"""
    def __init__(self, message: str = None, camera_id: str = None):
        self.camera_id = camera_id
        super().__init__(
            message or f"Detection failed for camera: {camera_id}",
            code="DETECTION_ERROR"
        )

class ClassificationError(AutomoBeeException):
    """Raised when vehicle classification fails"""
    def __init__(self, message: str = None, vehicle_id: str = None):
        self.vehicle_id = vehicle_id
        super().__init__(
            message or f"Classification failed for vehicle: {vehicle_id}",
            code="CLASSIFICATION_ERROR"
        )

class ZoneConfigError(AutomoBeeException):
    """Raised when zone configuration is invalid"""
    def __init__(self, message: str = None, zone_id: str = None):
        self.zone_id = zone_id
        super().__init__(
            message or f"Invalid zone configuration: {zone_id}",
            code="ZONE_CONFIG_ERROR"
        )

class CameraError(AutomoBeeException):
    """Raised when camera operations fail"""
    def __init__(self, message: str = None, camera_id: str = None):
        self.camera_id = camera_id
        super().__init__(
            message or f"Camera error: {camera_id}",
            code="CAMERA_ERROR"
        )

class SystemResourceError(AutomoBeeException):
    """Raised when system resources are constrained"""
    def __init__(self, message: str = None, resource: str = None):
        self.resource = resource
        super().__init__(
            message or f"System resource error: {resource}",
            code="RESOURCE_ERROR"
        )

class NetworkError(AutomoBeeException):
    """Raised when network operations fail"""
    def __init__(self, message: str = None, service: str = None):
        self.service = service
        super().__init__(
            message or f"Network error for service: {service}",
            code="NETWORK_ERROR"
        )

class ConfigurationError(AutomoBeeException):
    """Raised when system configuration is invalid"""
    def __init__(self, message: str = None, config_key: str = None):
        self.config_key = config_key
        super().__init__(
            message or f"Configuration error for: {config_key}",
            code="CONFIG_ERROR"
        )

class PerformanceError(AutomoBeeException):
    """Raised when performance thresholds are exceeded"""
    def __init__(self, message: str = None, metric: str = None, threshold: float = None):
        self.metric = metric
        self.threshold = threshold
        super().__init__(
            message or f"Performance threshold exceeded for {metric}: {threshold}",
            code="PERFORMANCE_ERROR"
        ) 