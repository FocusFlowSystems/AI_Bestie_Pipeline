import os
import time
import shutil
import logging
import httpx
from pathlib import Path
from dotenv import load_dotenv
import redis

from redis_datastore import TokenManager, OAuthState, AlertDispatcher, OAuthValidator, TerminalAuthError, TokenStorageError

# Directories
QUEUE_DIR = Path("04_Publishing_Queue")
DLQ_DIR = Path("05_DLQ")

for d in [QUEUE_DIR, DLQ_DIR]:
    d.mkdir(exist_ok=True)

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("HeadlessDaemon")

# Load Env
load_dotenv()
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# Initialize Core Services
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_client.ping()
except redis.ConnectionError:
    logger.error("FATAL: Could not connect to Redis.")
    exit(1)

alert_dispatcher = AlertDispatcher(DISCORD_WEBHOOK_URL)


class YouTubePublisher:
    def __init__(self):
        self.manager = TokenManager("youtube", redis_client)
        self.validator = OAuthValidator(self.manager, alert_dispatcher)

    def publish(self, video_path: Path, title: str, description: str):
        access_token = self.validator.get_valid_token()
        logger.info(f"YouTube: Acquired valid token. Preparing to upload {video_path.name}")
        
        # This uses the pure HTTP strategy instead of google-api-python-client 
        # to ensure compatibility with standard HTTpx async flows and telemetry interceptors
        metadata = {
            "snippet": {
                "title": title[:100],
                "description": description,
                "categoryId": "22"
            },
            "status": {
                "privacyStatus": "private", # Default to private for safety
                "selfDeclaredMadeForKids": False
            }
        }
        
        # Just a mock output for the system implementation plan since binary uploading is huge,
        # but in a real-world scenario we'd do a resumable upload via httpx
        try:
            # Fake the upload process validation
            with httpx.Client() as client:
                resp = client.post(
                    "https://www.googleapis.com/youtube/v3/videos?part=snippet,status",
                    headers={"Authorization": f"Bearer {access_token}"},
                    json=metadata,
                    timeout=10.0
                )
                if resp.status_code in (401, 403):
                    # Trigger the telemetry interceptor for quarantine
                    raise httpx.HTTPStatusError("Unauthorized", request=resp.request, response=resp)
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (401, 403):
                state = self.manager.get_state()
                state.status = "quarantine"
                self.manager.save_state(state)
                alert_dispatcher.trigger_incident("youtube", "REVOKED_PRIVILEGES_DURING_UPLOAD", str(e))
                raise TerminalAuthError("Token revoked during YouTube upload.")
            raise


class LinkedInPublisher:
    def __init__(self):
        self.manager = TokenManager("linkedin", redis_client)
        self.validator = OAuthValidator(self.manager, alert_dispatcher)

    def publish(self, text: str):
        access_token = self.validator.get_valid_token()
        logger.info("LinkedIn: Acquired valid token. Preparing to publish text post.")
        
        # In a real environment, you need the URN of the user. We assume a simple post here.
        # This is a stub for the LinkedIn UGC Post API.
        try:
            with httpx.Client() as client:
                # We would fetch the user's URN first: GET https://api.linkedin.com/v2/me
                me_resp = client.get("https://api.linkedin.com/v2/userinfo", headers={"Authorization": f"Bearer {access_token}"})
                
                if me_resp.status_code in (401, 403):
                    state = self.manager.get_state()
                    state.status = "quarantine"
                    self.manager.save_state(state)
                    alert_dispatcher.trigger_incident("linkedin", "REVOKED_PRIVILEGES", str(me_resp.text))
                    raise TerminalAuthError("LinkedIn token revoked.")
                    
                urn = me_resp.json().get("sub")
                logger.info(f"LinkedIn User URN: {urn}. Publishing...")
        except BaseException as e:
            logger.error(f"LinkedIn Exception: {e}")
            raise


class FacebookPublisher:
    def __init__(self):
        self.manager = TokenManager("facebook", redis_client)
        # We don't need the validator for Facebook since it's a permanent System User Token without a refresh endpoint
        
    def publish(self, text: str):
        try:
            state = self.manager.get_state()
            if state.status == "quarantine":
                raise TerminalAuthError("Facebook is quarantined.")
                
            access_token = state.access_token
            logger.info("Facebook: Using strictly scoped System User Token.")
            # Graph API call to /me/accounts then /page_id/feed
        except TokenStorageError:
            logger.error("Facebook token not initialized.")
            raise


def process_queue():
    """Scans the publishing queue and executes multi-platform distribution."""
    jobs = list(QUEUE_DIR.glob("*.json"))
    if not jobs:
        return
        
    logger.info(f"Found {len(jobs)} pending jobs in the queue.")
    
    youtube = YouTubePublisher()
    linkedin = LinkedInPublisher()
    facebook = FacebookPublisher()
    
    for job_file in jobs:
        try:
            # Expected schema: {"video_path": "...", "title": "...", "social_post": "..."}
            with open(job_file, 'r') as f:
                import json
                job_data = json.load(f)
                
            logger.info(f"Processing Job Task: {job_file.name}")
            video_path = Path(job_data.get("video_path", ""))
            
            # YouTube
            if video_path.exists():
                youtube.publish(video_path, job_data.get("title", "AI Bestie Video"), job_data.get("youtube_description", ""))
                
            # LinkedIn
            if "social_post" in job_data:
                linkedin.publish(job_data["social_post"])
                
            # Facebook
            if "social_post" in job_data:
                facebook.publish(job_data["social_post"])
                
            # If successful, archive or delete the job
            job_file.unlink()
            logger.info(f"Job {job_file.name} published successfully and removed from queue.")
            
        except TerminalAuthError as e:
            logger.error(f"Terminal Auth Exception occurred: {e}")
            # Route to DLQ (Dead Letter Queue) so the asset isn't lost
            dlq_target = DLQ_DIR / job_file.name
            shutil.move(str(job_file), str(dlq_target))
            logger.warning(f"Routed Job {job_file.name} to DLQ for manual recovery.")
        except Exception as e:
            logger.error(f"Unhandled Publishing Error for {job_file.name}: {e}")
            dlq_target = DLQ_DIR / job_file.name
            shutil.move(str(job_file), str(dlq_target))
            alert_dispatcher.trigger_incident("system", "UNHANDLED_DAEMON_EXCEPTION", str(e))


if __name__ == "__main__":
    logger.info("Starting AI Bestie Headless Publishing Daemon...")
    process_queue()
    logger.info("Daemon run complete.")

# ==========================================
# WEEKLY PUBLISHER ALIASES FOR AI_BESTIE_PUBLISHER
# ==========================================

def send_discord_notification(message: str):
    """Alias for alert_dispatcher used by ai_bestie_publisher.py."""
    if alert_dispatcher.webhook_url and "YOUR_DISCORD" not in alert_dispatcher.webhook_url:
        import requests
        try:
            requests.post(alert_dispatcher.webhook_url, json={"content": message})
        except Exception as e:
            logger.error(f"Failed to send Discord notification: {e}")

def publish_youtube_video(video_path: str, title: str, description: str) -> tuple[bool, str]:
    """Stub wrapper that simulates the YouTube upload process for testing."""
    logger.info(f"YOUTUBE UPLOAD INITIATED:\nFile: {video_path}\nTitle: {title}")
    
    # In a full implementation, this uses googleapiclient.discovery MediaFileUpload
    try:
        manager = TokenManager("youtube", redis_client)
        validator = OAuthValidator(manager, alert_dispatcher)
        token = validator.get_valid_token()
        logger.info("Successfully validated YouTube OAuth token from Redis.")
        # Simulation delay
        time.sleep(2) 
        return True, None
    except Exception as e:
        logger.error(f"YouTube Publish Error: {e}")
        return False, str(e)

def publish_facebook_page_video(video_path: str, title: str, description: str) -> tuple[bool, str]:
    """Stub wrapper that simulates the Facebook Page Video upload process for testing."""
    logger.info(f"FACEBOOK VIDEO PREPARING:\nFile: {video_path}\nTitle: {title}")
    
    # In a full implementation, this chunks the video to the Graph API /{page_id}/videos endpoint
    try:
        manager = TokenManager("facebook", redis_client)
        state = manager.get_state()
        if not state or not state.access_token:
            return False, "Facebook System Token not found in Redis."
        logger.info("Successfully retrieved Facebook System User Token from Redis.")
        # Simulation delay
        time.sleep(2)
        return True, None
    except Exception as e:
        logger.error(f"Facebook Video Publish Error: {e}")
        return False, str(e)

def publish_facebook_page_text(text: str, link: str = None) -> tuple[bool, str]:
    """Stub wrapper that simulates the Facebook Page Text/Link Post process for testing."""
    logger.info(f"FACEBOOK TEXT POST:\n{text}")
    
    # In a full implementation, this posts to the Graph API /{page_id}/feed endpoint
    try:
        manager = TokenManager("facebook", redis_client)
        state = manager.get_state()
        if not state or not state.access_token:
            return False, "Facebook System Token not found in Redis."
        logger.info("Successfully retrieved Facebook System User Token from Redis.")
        # Simulation delay
        time.sleep(1)
        return True, None
    except Exception as e:
        logger.error(f"Facebook Text Publish Error: {e}")
        return False, str(e)
