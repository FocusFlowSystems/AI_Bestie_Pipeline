import json
import logging
import time
import socket
from typing import Any, Optional
from pydantic import BaseModel, Field
import httpx

logger = logging.getLogger("HeadlessOAuth")

class OAuthState(BaseModel):
    """Pydantic model representing the cryptographic state of a platform."""
    platform_id: str
    access_token: str
    refresh_token: Optional[str] = None
    expires_at: Optional[int] = None
    absolute_expiry: Optional[int] = None
    client_id: str
    client_secret: str
    token_url: str
    status: str = Field(default="active")

    def is_expired(self, buffer_seconds: int = 300) -> bool:
        """Determines if the token is expired or within the safety buffer."""
        if self.expires_at is None:
            # Represents a Facebook System User Token (Never expires)
            return False
        return time.time() >= (self.expires_at - buffer_seconds)

    def is_nearing_cliff(self, days_buffer: int = 14) -> bool:
        """Checks if a LinkedIn refresh token is approaching its 365-day death."""
        if self.absolute_expiry is None:
            return False
        buffer_seconds = days_buffer * 86400
        return time.time() >= (self.absolute_expiry - buffer_seconds)


class TokenStorageError(Exception):
    pass


class TokenManager:
    """Manages the retrieval, storage, and locking of OAuth artifacts."""
    def __init__(self, platform: str, redis_client: Any):
        self.platform = platform
        self.store = redis_client
        self.lock_timeout = 10  # Seconds to hold the lock

    def get_state(self) -> OAuthState:
        """Retrieves decrypted credentials from secure storage."""
        raw_data = self.store.get(f"oauth_state:{self.platform}")
        if not raw_data:
            raise TokenStorageError(f"No credentials found for {self.platform}")
        # In production, decryption happens here before parsing
        return OAuthState.model_validate_json(raw_data)

    def save_state(self, state: OAuthState) -> None:
        """Encrypts and persists the updated credential object."""
        # In production, encryption happens here before saving
        self.store.set(f"oauth_state:{self.platform}", state.model_dump_json())

    def acquire_lock(self) -> bool:
        """Attempts to acquire a distributed lock to prevent concurrent refresh race conditions."""
        lock_key = f"lock:oauth_refresh:{self.platform}"
        # Redis SETNX with expiration (NX=True, EX=lock_timeout)
        return bool(self.store.set(lock_key, "locked", nx=True, ex=self.lock_timeout))

    def release_lock(self) -> None:
        """Releases the distributed lock."""
        lock_key = f"lock:oauth_refresh:{self.platform}"
        self.store.delete(lock_key)


class AlertDispatcher:
    """Dispatches terminal authentication failures to an external webhook."""
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self.hostname = socket.gethostname()

    def trigger_incident(self, platform: str, event_type: str, details: str) -> None:
        """Constructs and fires the high-cardinality telemetry payload."""
        if not self.webhook_url or "YOUR_DISCORD_WEBHOOK_HERE" in self.webhook_url:
            logger.warning("Discord webhook URL is not configured. Skipping alert.")
            return

        payload = {
            "content": f"🚨 **AI Bestie OAuth Alert** 🚨\n**Platform:** {platform}\n**Event:** {event_type}",
            "embeds": [{
                "title": "Headless Publishing Daemon Incident",
                "color": 16711680, # Red
                "fields": [
                    {"name": "Severity", "value": "CRITICAL", "inline": True},
                    {"name": "Host", "value": self.hostname, "inline": True},
                    {"name": "Platform", "value": platform, "inline": True},
                    {"name": "Event", "value": event_type, "inline": False},
                    {"name": "Details", "value": f"```json\n{details}\n```", "inline": False},
                    {"name": "Action Required", "value": f"Manual human-in-the-loop intervention required for {platform}.", "inline": False}
                ],
                "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime())
            }]
        }
        try:
            # Short timeouts prevent the script from hanging during emergency shutdown
            with httpx.Client(timeout=3.0) as client:
                response = client.post(self.webhook_url, json=payload)
                response.raise_for_status()
            logger.info(f"Successfully dispatched telemetry webhook for {platform}.")
        except httpx.RequestError as e:
            # If the alert fails, dump to local standard error as a last resort
            logger.error(f"FATAL: Failed to dispatch webhook alert: {str(e)} | Payload: {payload}")


class TerminalAuthError(Exception):
    pass


class OAuthValidator:
    def __init__(self, manager: TokenManager, alert: AlertDispatcher):
        self.manager = manager
        self.alert = alert

    def get_valid_token(self) -> str:
        """Returns a valid access token, refreshing it if necessary."""
        state = self.manager.get_state()
        
        if state.status == "quarantine":
            raise TerminalAuthError(f"{self.manager.platform} is in quarantine state. Publishing blocked.")
            
        if state.is_nearing_cliff():
            # Non-blocking alert for LinkedIn 365-day boundary
            self.alert.trigger_incident(
                self.manager.platform,
                "APPROACHING_ABSOLUTE_EXPIRY",
                "Refresh token will die in < 14 days."
            )
            
        if not state.is_expired():
            return state.access_token
            
        logger.info(f"Token for {self.manager.platform} expired. Initiating refresh sequence.")
        return self._execute_refresh_with_lock(state)

    def _execute_refresh_with_lock(self, state: OAuthState) -> str:
        """Manages the race condition via distributed Redis locks."""
        max_retries = 5
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            if self.manager.acquire_lock():
                try:
                    # Double-check state after acquiring the lock.
                    # Another concurrent cron process may have just refreshed it.
                    fresh_state = self.manager.get_state()
                    if not fresh_state.is_expired():
                        return fresh_state.access_token
                    
                    # Proceed with actual HTTP refresh to the provider
                    return self._perform_http_refresh(fresh_state)
                finally:
                    # Always release the lock, even if HTTP request fails
                    self.manager.release_lock()
            else:
                logger.debug(f"Refresh lock held by another process. Backing off {retry_delay}s.")
                time.sleep(retry_delay)
                # Exponential backoff for subsequent polling
                retry_delay *= 2
                
        # If we exit the loop, the lock was never acquired (potential deadlock or massive latency)
        self.alert.trigger_incident(self.manager.platform, "LOCK_ACQUISITION_FAILED", "Could not acquire Redis lock for refresh.")
        raise TimeoutError("Exceeded maximum retries waiting for OAuth refresh lock.")

    def _perform_http_refresh(self, state: OAuthState) -> str:
        """Executes the exact HTTP POST request to the provider's token endpoint."""
        payload = {
            "grant_type": "refresh_token",
            "refresh_token": state.refresh_token,
            "client_id": state.client_id,
            "client_secret": state.client_secret
        }
        
        try:
            # 5-second timeout to prevent stalling the cron process
            with httpx.Client(timeout=5.0) as client:
                response = client.post(state.token_url, data=payload)
                
                if response.status_code in (400, 401):
                    error_data = response.text
                    
                    # Quarantine the state to prevent infinite loops
                    state.status = "quarantine"
                    self.manager.save_state(state)
                    
                    # Trigger the terminal telemetry webhook
                    self.alert.trigger_incident(
                        self.manager.platform,
                        "INVALID_GRANT_REVOCATION",
                        f"Refresh rejected. API Response: {error_data}"
                    )
                    raise TerminalAuthError("Refresh token rejected. State quarantined.")
                    
                # If network succeeds but response is malformed, raise error
                response.raise_for_status()
                data = response.json()
                
                # Update the state object with new credentials
                state.access_token = data['access_token']
                state.expires_at = int(time.time()) + data['expires_in']
                
                # Implement refresh token rotation if the provider supplies a new one
                if 'refresh_token' in data:
                    state.refresh_token = data['refresh_token']
                    
                self.manager.save_state(state)
                logger.info(f"Successfully refreshed and persisted tokens for {self.manager.platform}.")
                return state.access_token
                
        except httpx.RequestError as e:
            logger.error(f"Network error during refresh for {self.manager.platform}: {str(e)}")
            raise
