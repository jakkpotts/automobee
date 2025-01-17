from functools import wraps
from flask import request, jsonify, current_app
from jose import jwt, JWTError
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict, Any
import redis
import hashlib
import json

logger = logging.getLogger(__name__)

class SecurityMiddleware:
    def __init__(self, app=None, redis_url: str = None):
        self.app = app
        self.redis_client = redis.from_url(redis_url) if redis_url else None
        self.rate_limits = {
            'default': {'calls': 100, 'period': 60},  # 100 calls per minute
            'analytics': {'calls': 300, 'period': 60},  # 300 calls per minute
            'stream': {'calls': 1000, 'period': 60}    # 1000 calls per minute
        }
        
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize the security middleware with a Flask app"""
        self.app = app
        
        # Set default security headers
        @app.after_request
        def add_security_headers(response):
            response.headers['X-Content-Type-Options'] = 'nosniff'
            response.headers['X-Frame-Options'] = 'DENY'
            response.headers['X-XSS-Protection'] = '1; mode=block'
            response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
            return response
    
    def generate_token(self, user_data: Dict[str, Any], expires_delta: timedelta = None) -> str:
        """Generate a JWT token"""
        if expires_delta is None:
            expires_delta = timedelta(minutes=15)
            
        expire = datetime.utcnow() + expires_delta
        to_encode = {
            **user_data,
            'exp': expire,
            'iat': datetime.utcnow()
        }
        
        return jwt.encode(
            to_encode,
            current_app.config['SECRET_KEY'],
            algorithm='HS256'
        )
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify a JWT token"""
        try:
            payload = jwt.decode(
                token,
                current_app.config['SECRET_KEY'],
                algorithms=['HS256']
            )
            return payload
        except JWTError as e:
            logger.error(f"JWT verification failed: {e}")
            return None
    
    def require_auth(self, roles=None):
        """Decorator to require authentication and optional role checking"""
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                auth_header = request.headers.get('Authorization')
                if not auth_header or not auth_header.startswith('Bearer '):
                    return jsonify({'error': 'Missing or invalid authorization header'}), 401
                
                token = auth_header.split(' ')[1]
                payload = self.verify_token(token)
                
                if not payload:
                    return jsonify({'error': 'Invalid or expired token'}), 401
                
                if roles and not any(role in payload.get('roles', []) for role in roles):
                    return jsonify({'error': 'Insufficient permissions'}), 403
                
                return f(*args, **kwargs)
            return decorated_function
        return decorator
    
    def rate_limit(self, limit_type='default'):
        """Decorator to apply rate limiting"""
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                if not self.redis_client:
                    logger.warning("Redis not configured, rate limiting disabled")
                    return f(*args, **kwargs)
                
                # Get client identifier (IP or API key)
                client_id = request.headers.get('X-API-Key') or request.remote_addr
                
                # Get rate limit settings
                limit = self.rate_limits.get(limit_type, self.rate_limits['default'])
                key = f"rate_limit:{limit_type}:{client_id}"
                
                try:
                    # Check current request count
                    current = self.redis_client.get(key)
                    if current is None:
                        # First request, set initial count
                        self.redis_client.setex(key, limit['period'], 1)
                    elif int(current) >= limit['calls']:
                        return jsonify({
                            'error': 'Rate limit exceeded',
                            'retry_after': self.redis_client.ttl(key)
                        }), 429
                    else:
                        # Increment request count
                        self.redis_client.incr(key)
                    
                    return f(*args, **kwargs)
                except redis.RedisError as e:
                    logger.error(f"Redis error in rate limiting: {e}")
                    return f(*args, **kwargs)
                    
            return decorated_function
        return decorator
    
    def validate_request(self, schema: Dict):
        """Decorator to validate request data against a schema"""
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                if request.is_json:
                    data = request.get_json()
                else:
                    data = request.form.to_dict()
                
                # Validate required fields
                missing_fields = [
                    field for field, rules in schema.items()
                    if rules.get('required', False) and field not in data
                ]
                
                if missing_fields:
                    return jsonify({
                        'error': 'Missing required fields',
                        'fields': missing_fields
                    }), 400
                
                # Validate field types and constraints
                invalid_fields = []
                for field, value in data.items():
                    if field in schema:
                        rules = schema[field]
                        
                        # Type validation
                        if 'type' in rules:
                            try:
                                if rules['type'] == 'int':
                                    int(value)
                                elif rules['type'] == 'float':
                                    float(value)
                                elif rules['type'] == 'bool':
                                    if not isinstance(value, bool):
                                        invalid_fields.append(field)
                                elif rules['type'] == 'list':
                                    if not isinstance(value, list):
                                        invalid_fields.append(field)
                            except (ValueError, TypeError):
                                invalid_fields.append(field)
                        
                        # Length validation
                        if 'max_length' in rules and len(str(value)) > rules['max_length']:
                            invalid_fields.append(field)
                        
                        # Range validation
                        if 'min' in rules and float(value) < rules['min']:
                            invalid_fields.append(field)
                        if 'max' in rules and float(value) > rules['max']:
                            invalid_fields.append(field)
                
                if invalid_fields:
                    return jsonify({
                        'error': 'Invalid field values',
                        'fields': invalid_fields
                    }), 400
                
                return f(*args, **kwargs)
            return decorated_function
        return decorator
    
    def api_key_auth(self):
        """Decorator to require API key authentication"""
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                api_key = request.headers.get('X-API-Key')
                if not api_key:
                    return jsonify({'error': 'Missing API key'}), 401
                
                # Verify API key (implement your own verification logic)
                if not self._verify_api_key(api_key):
                    return jsonify({'error': 'Invalid API key'}), 401
                
                return f(*args, **kwargs)
            return decorated_function
        return decorator
    
    def _verify_api_key(self, api_key: str) -> bool:
        """Verify API key against stored keys"""
        if not self.redis_client:
            logger.warning("Redis not configured, API key verification disabled")
            return True
        
        try:
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            stored_key = self.redis_client.hget('api_keys', key_hash)
            
            if stored_key:
                key_data = json.loads(stored_key)
                return (
                    key_data.get('active', False) and
                    datetime.fromisoformat(key_data['expires']) > datetime.utcnow()
                )
            return False
            
        except (redis.RedisError, json.JSONDecodeError) as e:
            logger.error(f"Error verifying API key: {e}")
            return False 