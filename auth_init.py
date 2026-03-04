import os
import json
import time
import logging
from pathlib import Path
from dotenv import load_dotenv
import redis
import httpx
from redis_datastore import TokenManager, OAuthState

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AuthInit")

# Load Env
load_dotenv()

# Initialize Redis
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_client.ping()
except redis.ConnectionError:
    logger.error("Could not connect to Redis. Please ensure Redis server is running.")
    exit(1)


def init_youtube():
    """Interactive OAuth flow for YouTube."""
    from google_auth_oauthlib.flow import InstalledAppFlow
    
    # Needs a client_secrets.json file from Google Cloud Console
    client_secrets_file = "client_secrets.json"
    if not Path(client_secrets_file).exists():
        logger.error(f"Missing {client_secrets_file}. Please download from Google Cloud Console.")
        return

    scopes = ["https://www.googleapis.com/auth/youtube.upload", "https://www.googleapis.com/auth/youtube"]
    flow = InstalledAppFlow.from_client_secrets_file(client_secrets_file, scopes)
    
    # Request offline access to get the permanent refresh token
    creds = flow.run_local_server(port=8080, access_type='offline', prompt='consent')
    
    # Read the client_secrets to store in Redis
    with open(client_secrets_file, 'r') as f:
        secrets = json.load(f)
        web_info = secrets.get('installed', secrets.get('web', {}))
    
    state = OAuthState(
        platform_id="youtube",
        access_token=creds.token,
        refresh_token=creds.refresh_token,
        expires_at=int(time.time()) + 3599 if creds.expiry else None,
        client_id=web_info.get("client_id", ""),
        client_secret=web_info.get("client_secret", ""),
        token_url=web_info.get("token_uri", "https://oauth2.googleapis.com/token")
    )
    
    manager = TokenManager("youtube", redis_client)
    manager.save_state(state)
    logger.info("YouTube OAuth state successfully persisted to Redis.")


def init_linkedin():
    """Interactive OAuth flow for LinkedIn."""
    client_id = input("Enter LinkedIn Client ID: ").strip()
    client_secret = input("Enter LinkedIn Client Secret: ").strip()
    
    if not client_id or not client_secret:
        logger.error("Client ID and Secret are required.")
        return
        
    redirect_uri = "http://localhost:8080/callback"
    auth_url = (f"https://www.linkedin.com/oauth/v2/authorization"
                f"?response_type=code&client_id={client_id}"
                f"&redirect_uri={redirect_uri}"
                f"&scope=w_member_social%20r_liteprofile")
                
    print(f"\n1. Please visit this URL to authorize LinkedIn:\n{auth_url}\n")
    auth_code = input("2. Enter the 'code' parameter from the callback URL: ").strip()
    
    if not auth_code:
        logger.error("Auth code is required.")
        return
        
    token_url = "https://www.linkedin.com/oauth/v2/accessToken"
    payload = {
        "grant_type": "authorization_code",
        "code": auth_code,
        "redirect_uri": redirect_uri,
        "client_id": client_id,
        "client_secret": client_secret
    }
    
    response = httpx.post(token_url, data=payload)
    if response.status_code != 200:
        logger.error(f"Failed to retrieve token: {response.text}")
        return
        
    data = response.json()
    
    # LinkedIn refresh tokens typically die in 365 days
    state = OAuthState(
        platform_id="linkedin",
        access_token=data['access_token'],
        refresh_token=data.get('refresh_token'),
        expires_at=int(time.time()) + data.get('expires_in', 5184000), # ~60 days default
        absolute_expiry=int(time.time()) + data.get('refresh_token_expires_in', 31536000), # ~365 days default
        client_id=client_id,
        client_secret=client_secret,
        token_url=token_url
    )
    
    manager = TokenManager("linkedin", redis_client)
    manager.save_state(state)
    logger.info("LinkedIn OAuth state successfully persisted to Redis. A 14-day early warning cliff is enabled.")


def init_facebook():
    """Interactive setup for Facebook Pages."""
    print("\nMeta Business Manager -> System Users -> Generate Token (pages_manage_posts scope)")
    access_token = input("Enter Meta System User Token: ").strip()
    
    if not access_token:
        logger.error("Access token is required.")
        return
        
    state = OAuthState(
        platform_id="facebook",
        access_token=access_token,
        refresh_token=None,
        expires_at=None, # Never expires
        absolute_expiry=None,
        client_id="",
        client_secret="",
        token_url=""
    )
    
    manager = TokenManager("facebook", redis_client)
    manager.save_state(state)
    logger.info("Facebook Pages System User Token successfully persisted to Redis.")


if __name__ == "__main__":
    print("\n--- AI Bestie Headless OAuth Initialization ---")
    print("1: Initialize YouTube")
    print("2: Initialize LinkedIn")
    print("3: Initialize Facebook")
    
    choice = input("\nSelect platform to initialize (1/2/3): ").strip()
    
    if choice == "1":
        init_youtube()
    elif choice == "2":
        init_linkedin()
    elif choice == "3":
        init_facebook()
    else:
        print("Invalid choice.")
