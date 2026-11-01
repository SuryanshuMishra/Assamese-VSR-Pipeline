"""
Quick Google Drive OAuth Login Verification Test
"""
import sys
from pathlib import Path
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/drive"]

def test_login():
    creds_path = Path("credentials.json")
    token_path = Path("token.json")

    if not creds_path.exists():
        print("[ERROR] credentials.json not found in current folder!")
        return

    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("[INFO] Refreshing expired token...")
            creds.refresh(Request())
        else:
            print("\n[INFO] Opening your default browser for Google sign-in...")
            flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), SCOPES)
            creds = flow.run_local_server(port=0)

        with open(token_path, "w") as token:
            token.write(creds.to_json())
        print("[SUCCESS] Login successful! Saved token to token.json.")

    # Test Drive API connection
    service = build("drive", "v3", credentials=creds)
    about = service.about().get(fields="user").execute()
    user_email = about.get("user", {}).get("emailAddress", "unknown")
    print(f"\n[OK] Connected to Google Drive as: {user_email}")
    print("[OK] You are ready to run the batch pipeline!")

if __name__ == "__main__":
    test_login()

