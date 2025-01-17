from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any
import logging
import traceback
from datetime import datetime

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for better organization."""
    AUTHENTICATION = "auth"
    AUTHORIZATION = "authz"
    VALIDATION = "validation"
    PROCESSING = "processing"
    DATABASE = "database"
    NETWORK = "network"
    SYSTEM = "system"

@dataclass
class ErrorContext:
    """Context information for errors."""
    timestamp: datetime
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    details: Dict[str, Any]
    stack_trace: Optional[str] = None
    user_id: Optional[str] = None
    recovery_attempted: bool = False

class ErrorManager:
    """Manages error handling, reporting, and recovery strategies."""
    
    def __init__(self):
        self.error_counts: Dict[ErrorCategory, int] = {
            category: 0 for category in ErrorCategory
        }
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
    async def handle_error(self, error: Exception, category: ErrorCategory, 
                          severity: ErrorSeverity, context: Dict[str, Any] = None) -> ErrorContext:
        """Handle an error with proper logging and recovery attempts."""
        try:
            error_id = f"{category.value}_{datetime.now().timestamp()}"
            error_context = ErrorContext(
                timestamp=datetime.now(),
                error_id=error_id,
                category=category,
                severity=severity,
                message=str(error),
                details=context or {},
                stack_trace=traceback.format_exc()
            )
            
            # Update error counts
            self.error_counts[category] += 1
            
            # Log error
            self._log_error(error_context)
            
            # Attempt recovery if possible
            if severity != ErrorSeverity.CRITICAL:
                await self._attempt_recovery(error_context)
                
            # Report to monitoring system
            await self._report_to_monitoring(error_context)
            
            return error_context
            
        except Exception as e:
            logger.error(f"Error in error handler: {str(e)}")
            raise
            
    def _log_error(self, error_context: ErrorContext):
        """Log error with appropriate severity."""
        log_message = (
            f"Error {error_context.error_id} [{error_context.category.value}] "
            f"[{error_context.severity.value}]: {error_context.message}"
        )
        
        if error_context.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message, extra={'error_context': error_context})
        elif error_context.severity == ErrorSeverity.HIGH:
            logger.error(log_message, extra={'error_context': error_context})
        elif error_context.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message, extra={'error_context': error_context})
        else:
            logger.info(log_message, extra={'error_context': error_context})
            
    async def _attempt_recovery(self, error_context: ErrorContext):
        """Attempt to recover from error based on category."""
        try:
            if error_context.category == ErrorCategory.DATABASE:
                await self._handle_database_error(error_context)
            elif error_context.category == ErrorCategory.NETWORK:
                await self._handle_network_error(error_context)
            # Add more recovery strategies as needed
            
            error_context.recovery_attempted = True
            
        except Exception as e:
            logger.error(f"Recovery attempt failed for {error_context.error_id}: {str(e)}")
            
    async def _report_to_monitoring(self, error_context: ErrorContext):
        """Report error to monitoring system."""
        # Implement your monitoring system integration here
        pass
        
    async def _handle_database_error(self, error_context: ErrorContext):
        """Handle database-specific errors."""
        try:
            if "connection" in str(error_context.message).lower():
                # Attempt to reconnect
                await self._reconnect_database()
            elif "deadlock" in str(error_context.message).lower():
                # Handle deadlock
                await self._resolve_deadlock(error_context)
            else:
                # Log unhandled database error
                logger.error(f"Unhandled database error: {error_context.message}")
        except Exception as e:
            logger.error(f"Error recovery failed: {str(e)}")
        
    async def _handle_network_error(self, error_context: ErrorContext):
        """Handle network-specific errors."""
        # Implement network error recovery strategy
        pass

class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(self, name: str, failure_threshold: int = 5, 
                 reset_timeout: int = 60):
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure_time = None
        self.state = "closed"
        
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker pattern."""
        if self.state == "open":
            if self._should_reset():
                self.state = "half-open"
            else:
                raise Exception(f"Circuit breaker {self.name} is open")
                
        try:
            result = await func(*args, **kwargs)
            if self.state == "half-open":
                self.state = "closed"
                self.failures = 0
            return result
            
        except Exception as e:
            self._handle_failure()
            raise
            
    def _handle_failure(self):
        """Handle a failure and potentially open the circuit."""
        self.failures += 1
        self.last_failure_time = datetime.now()
        
        if self.failures >= self.failure_threshold:
            self.state = "open"
            
    def _should_reset(self) -> bool:
        """Check if circuit breaker should be reset."""
        if not self.last_failure_time:
            return True
            
        elapsed = (datetime.now() - self.last_failure_time).seconds
        return elapsed >= self.reset_timeout 