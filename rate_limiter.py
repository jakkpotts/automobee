from datetime import datetime, timedelta
from collections import defaultdict
import asyncio
from typing import Dict, DefaultDict

class RateLimiter:
    def __init__(self, calls: int, period: int):
        """
        Initialize rate limiter
        Args:
            calls: Number of calls allowed per period
            period: Time period in seconds
        """
        self.calls = calls
        self.period = period
        self.timestamps: DefaultDict[str, list] = defaultdict(list)
    
    async def acquire(self, key: str) -> bool:
        """
        Check if we can make a request for the given key
        Returns True if request is allowed, False otherwise
        """
        now = datetime.now()
        cutoff = now - timedelta(seconds=self.period)
        
        # Remove old timestamps
        self.timestamps[key] = [ts for ts in self.timestamps[key] if ts > cutoff]
        
        if len(self.timestamps[key]) < self.calls:
            self.timestamps[key].append(now)
            return True
            
        return False 