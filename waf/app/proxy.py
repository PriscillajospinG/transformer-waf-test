"""
HTTP Reverse Proxy
Forwards requests to backend while WAF analyzes them
"""

import httpx
import logging
from typing import Optional, Dict, Tuple
from app.config import WAFConfig

logger = logging.getLogger("proxy")

class ReverseProxy:
    """HTTP reverse proxy for forwarding requests to backend"""
    
    def __init__(self, target_url: str = None, timeout: int = None):
        self.target_url = target_url or WAFConfig.TARGET_WEBSITE_URL
        self.timeout = timeout or WAFConfig.REQUEST_TIMEOUT_SEC
        
        # Client with connection pooling
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            follow_redirects=True,
            limits=httpx.Limits(max_keepalive_connections=100)
        )
        
        logger.info(f"Proxy initialized. Target: {self.target_url}")
    
    async def forward_request(self, method: str, path: str, query: str = "",
                             headers: Dict = None, body: bytes = None) -> Tuple[int, Dict, bytes]:
        """
        Forward request to backend and return response
        
        Returns: (status_code, response_headers, body)
        """
        try:
            # Build target URL
            target_url = f"{self.target_url}{path}"
            if query:
                target_url += f"?{query}"
            
            # Clean headers - remove hop-by-hop headers
            clean_headers = self._clean_headers(headers or {})
            
            # Forward request
            response = await self.client.request(
                method=method,
                url=target_url,
                headers=clean_headers,
                content=body if body else None
            )
            
            # Get response body
            response_body = await response.aread()
            
            # Extract response headers
            response_headers = dict(response.headers)
            
            logger.info(f"Proxy: {method} {path} -> {response.status_code}")
            
            return response.status_code, response_headers, response_body
        
        except httpx.TimeoutException:
            logger.error(f"Request timeout to {self.target_url}{path}")
            return 504, {}, b"Gateway Timeout"
        except httpx.ConnectError:
            logger.error(f"Connection error to {self.target_url}")
            return 502, {}, b"Bad Gateway"
        except Exception as e:
            logger.error(f"Proxy error: {e}")
            return 500, {}, b"Internal Server Error"
    
    def _clean_headers(self, headers: Dict) -> Dict:
        """Remove hop-by-hop headers that shouldn't be forwarded"""
        hop_by_hop = {
            'connection', 'keep-alive', 'proxy-authenticate',
            'proxy-authorization', 'te', 'trailers', 'transfer-encoding',
            'upgrade', 'content-length'  # We let httpx handle content-length
        }
        
        clean = {}
        for key, value in headers.items():
            if key.lower() not in hop_by_hop:
                clean[key] = value
        
        return clean
    
    async def close(self):
        """Close proxy client"""
        await self.client.aclose()

# Global proxy instance
reverse_proxy = ReverseProxy()
