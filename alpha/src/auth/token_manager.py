import jwt
from datetime import datetime, timedelta
from typing import Dict, Optional
from ..config.config_manager import ConfigManager

class TokenManager:
    """Manages JWT token generation and validation."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.secret_key = config.get("websocket.auth.secret_key")
        self.algorithm = config.get("websocket.auth.algorithm")
        self.token_expiry = config.get("websocket.auth.token_expiry")
        
    def generate_token(self, user_id: str, scopes: Optional[list] = None) -> str:
        """Generate a new JWT token."""
        now = datetime.utcnow()
        payload = {
            'sub': user_id,
            'scopes': scopes or [],
            'iat': now,
            'exp': now + timedelta(seconds=self.token_expiry)
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
    def validate_token(self, token: str) -> Dict:
        """Validate JWT token and return payload."""
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError:
            raise ValueError("Invalid token") 