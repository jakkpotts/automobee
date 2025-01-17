from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
from typing import Dict, Optional, Set, List
import logging
from .token_manager import TokenManager

logger = logging.getLogger(__name__)

@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    requests_per_minute: int = 60
    burst_size: int = 10
    cleanup_interval: int = 300  # 5 minutes

@dataclass
class SecurityConfig:
    """Security configuration."""
    token_refresh_window: int = 300  # 5 minutes before expiry
    max_failed_attempts: int = 5
    lockout_duration: int = 900  # 15 minutes
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)

class SecurityMiddleware:
    """Handles security concerns including rate limiting and token refresh."""
    
    def __init__(self, token_manager: TokenManager):
        self.token_manager = token_manager
        self.config = SecurityConfig()
        self.rate_limits: Dict[str, List[datetime]] = {}
        self.failed_attempts: Dict[str, int] = {}
        self.locked_out: Set[str] = set()
        self.last_cleanup = datetime.now()
        
    async def validate_request(self, token: str, required_scopes: Set[str]) -> bool:
        """Validate request including rate limiting and scope checking."""
        try:
            # Validate token and extract user_id
            payload = self.token_manager.validate_token(token)
            user_id = payload['sub']
            
            # Check if user is locked out
            if user_id in self.locked_out:
                logger.warning(f"Blocked request from locked out user: {user_id}")
                return False
                
            # Check rate limits
            if not self._check_rate_limit(user_id):
                logger.warning(f"Rate limit exceeded for user: {user_id}")
                return False
                
            # Validate scopes
            if not self._validate_scopes(payload.get('scopes', []), required_scopes):
                logger.warning(f"Insufficient scopes for user: {user_id}")
                return False
                
            # Check if token needs refresh
            if self._needs_refresh(payload):
                await self._handle_token_refresh(user_id, payload)
                
            return True
            
        except Exception as e:
            logger.error(f"Request validation failed: {str(e)}")
            return False
            
    def _check_rate_limit(self, user_id: str) -> bool:
        """Check if request is within rate limits."""
        current_time = datetime.now()
        
        # Cleanup old entries
        if (current_time - self.last_cleanup).seconds > self.config.rate_limit.cleanup_interval:
            self._cleanup_rate_limits()
            
        # Initialize user's rate limit tracking
        if user_id not in self.rate_limits:
            self.rate_limits[user_id] = []
            
        # Remove old requests
        minute_ago = current_time - timedelta(minutes=1)
        self.rate_limits[user_id] = [
            ts for ts in self.rate_limits[user_id]
            if ts > minute_ago
        ]
        
        # Check limits
        if len(self.rate_limits[user_id]) >= self.config.rate_limit.requests_per_minute:
            return False
            
        # Add new request
        self.rate_limits[user_id].append(current_time)
        return True
        
    def _validate_scopes(self, user_scopes: List[str], required_scopes: Set[str]) -> bool:
        """Validate if user has all required scopes."""
        user_scope_set = set(user_scopes)
        return required_scopes.issubset(user_scope_set)
        
    def _needs_refresh(self, payload: dict) -> bool:
        """Check if token needs refresh."""
        exp_time = datetime.fromtimestamp(payload['exp'])
        refresh_threshold = exp_time - timedelta(seconds=self.config.token_refresh_window)
        return datetime.now() > refresh_threshold
        
    async def _handle_token_refresh(self, user_id: str, old_payload: dict):
        """Handle token refresh."""
        try:
            new_token = self.token_manager.generate_token(
                user_id=user_id,
                scopes=old_payload.get('scopes', [])
            )
            # Broadcast token refresh event
            # This should be implemented based on your notification system
            logger.info(f"Token refreshed for user: {user_id}")
            
        except Exception as e:
            logger.error(f"Token refresh failed for user {user_id}: {str(e)}")
            
    def _cleanup_rate_limits(self):
        """Clean up old rate limit entries."""
        current_time = datetime.now()
        self.last_cleanup = current_time
        
        # Clean up rate limits
        for user_id in list(self.rate_limits.keys()):
            minute_ago = current_time - timedelta(minutes=1)
            self.rate_limits[user_id] = [
                ts for ts in self.rate_limits[user_id]
                if ts > minute_ago
            ]
            if not self.rate_limits[user_id]:
                del self.rate_limits[user_id] 