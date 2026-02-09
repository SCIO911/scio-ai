#!/usr/bin/env python3
"""
SCIO - Enterprise Integrations

Integration mit:
- Google Workspace (Drive, Docs, Sheets, Calendar)
- Microsoft 365 (OneDrive, Word, Excel, Outlook)
- Slack
- GitHub
- Jira
- Notion
"""

import os
import json
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class IntegrationType(str, Enum):
    """Integrationstypen"""
    GOOGLE = "google"
    MICROSOFT = "microsoft"
    SLACK = "slack"
    GITHUB = "github"
    JIRA = "jira"
    NOTION = "notion"


@dataclass
class CalendarEvent:
    """Kalenderevent"""
    event_id: str
    title: str
    start: datetime
    end: datetime
    description: str = ""
    location: str = ""
    attendees: List[str] = field(default_factory=list)
    is_recurring: bool = False
    meeting_url: str = ""


@dataclass
class Document:
    """Dokument"""
    doc_id: str
    title: str
    content: str = ""
    mime_type: str = "text/plain"
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    url: str = ""
    owner: str = ""


@dataclass
class SlackMessage:
    """Slack Nachricht"""
    channel: str
    text: str
    timestamp: str = ""
    user: str = ""
    thread_ts: str = ""
    attachments: List[Dict] = field(default_factory=list)


@dataclass
class GitHubIssue:
    """GitHub Issue"""
    number: int
    title: str
    body: str = ""
    state: str = "open"
    labels: List[str] = field(default_factory=list)
    assignees: List[str] = field(default_factory=list)
    created_at: Optional[datetime] = None
    url: str = ""


class GoogleWorkspaceClient:
    """Google Workspace API Client"""

    def __init__(self):
        self.credentials_file = os.getenv("GOOGLE_CREDENTIALS_FILE", "credentials.json")
        self.token_file = os.getenv("GOOGLE_TOKEN_FILE", "token.json")
        self._service = None
        self._enabled = os.path.exists(self.credentials_file)

    def _init_service(self, api: str, version: str):
        """Initialisiert Google API Service"""
        try:
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.auth.transport.requests import Request
            from googleapiclient.discovery import build

            SCOPES = [
                'https://www.googleapis.com/auth/drive',
                'https://www.googleapis.com/auth/documents',
                'https://www.googleapis.com/auth/spreadsheets',
                'https://www.googleapis.com/auth/calendar'
            ]

            creds = None
            if os.path.exists(self.token_file):
                creds = Credentials.from_authorized_user_file(self.token_file, SCOPES)

            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(self.credentials_file, SCOPES)
                    creds = flow.run_local_server(port=0)

                with open(self.token_file, 'w') as token:
                    token.write(creds.to_json())

            return build(api, version, credentials=creds)

        except ImportError:
            print("[ERROR] Google API libraries nicht installiert")
            print("pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client")
            return None
        except Exception as e:
            print(f"[ERROR] Google Auth fehlgeschlagen: {e}")
            return None

    def list_drive_files(self, query: str = None, limit: int = 10) -> List[Document]:
        """Listet Google Drive Dateien"""
        service = self._init_service("drive", "v3")
        if not service:
            return []

        try:
            q = query or "mimeType != 'application/vnd.google-apps.folder'"
            results = service.files().list(
                q=q,
                pageSize=limit,
                fields="files(id, name, mimeType, createdTime, modifiedTime, webViewLink, owners)"
            ).execute()

            files = results.get("files", [])

            return [
                Document(
                    doc_id=f["id"],
                    title=f["name"],
                    mime_type=f.get("mimeType", ""),
                    created_at=datetime.fromisoformat(f["createdTime"].replace("Z", "+00:00")) if f.get("createdTime") else None,
                    modified_at=datetime.fromisoformat(f["modifiedTime"].replace("Z", "+00:00")) if f.get("modifiedTime") else None,
                    url=f.get("webViewLink", ""),
                    owner=f.get("owners", [{}])[0].get("displayName", "")
                )
                for f in files
            ]
        except Exception as e:
            print(f"[ERROR] Drive Fehler: {e}")
            return []

    def create_doc(self, title: str, content: str = "") -> Optional[str]:
        """Erstellt Google Doc"""
        service = self._init_service("docs", "v1")
        if not service:
            return None

        try:
            doc = service.documents().create(body={"title": title}).execute()
            doc_id = doc.get("documentId")

            if content:
                requests = [{
                    "insertText": {
                        "location": {"index": 1},
                        "text": content
                    }
                }]
                service.documents().batchUpdate(documentId=doc_id, body={"requests": requests}).execute()

            return doc_id
        except Exception as e:
            print(f"[ERROR] Doc erstellen fehlgeschlagen: {e}")
            return None

    def read_sheet(self, spreadsheet_id: str, range_name: str = "A1:Z100") -> List[List[Any]]:
        """Liest Google Sheet"""
        service = self._init_service("sheets", "v4")
        if not service:
            return []

        try:
            result = service.spreadsheets().values().get(
                spreadsheetId=spreadsheet_id,
                range=range_name
            ).execute()

            return result.get("values", [])
        except Exception as e:
            print(f"[ERROR] Sheet lesen fehlgeschlagen: {e}")
            return []

    def write_sheet(self, spreadsheet_id: str, range_name: str, values: List[List[Any]]) -> bool:
        """Schreibt in Google Sheet"""
        service = self._init_service("sheets", "v4")
        if not service:
            return False

        try:
            body = {"values": values}
            service.spreadsheets().values().update(
                spreadsheetId=spreadsheet_id,
                range=range_name,
                valueInputOption="RAW",
                body=body
            ).execute()
            return True
        except Exception as e:
            print(f"[ERROR] Sheet schreiben fehlgeschlagen: {e}")
            return False

    def list_calendar_events(self, days: int = 7) -> List[CalendarEvent]:
        """Listet Kalenderevents"""
        service = self._init_service("calendar", "v3")
        if not service:
            return []

        try:
            now = datetime.utcnow()
            time_min = now.isoformat() + "Z"
            time_max = (now + timedelta(days=days)).isoformat() + "Z"

            events_result = service.events().list(
                calendarId="primary",
                timeMin=time_min,
                timeMax=time_max,
                maxResults=50,
                singleEvents=True,
                orderBy="startTime"
            ).execute()

            events = events_result.get("items", [])

            return [
                CalendarEvent(
                    event_id=e["id"],
                    title=e.get("summary", ""),
                    start=datetime.fromisoformat(e["start"].get("dateTime", e["start"].get("date")).replace("Z", "+00:00")),
                    end=datetime.fromisoformat(e["end"].get("dateTime", e["end"].get("date")).replace("Z", "+00:00")),
                    description=e.get("description", ""),
                    location=e.get("location", ""),
                    attendees=[a.get("email", "") for a in e.get("attendees", [])],
                    meeting_url=e.get("hangoutLink", "")
                )
                for e in events
            ]
        except Exception as e:
            print(f"[ERROR] Kalender Fehler: {e}")
            return []


class SlackClient:
    """Slack API Client"""

    def __init__(self):
        self.token = os.getenv("SLACK_BOT_TOKEN", "")
        self.webhook_url = os.getenv("SLACK_WEBHOOK_URL", "")
        self._client = None
        self._enabled = bool(self.token or self.webhook_url)

    def _init_client(self):
        """Initialisiert Slack Client"""
        if self._client:
            return True

        if not self.token:
            return False

        try:
            from slack_sdk import WebClient
            self._client = WebClient(token=self.token)
            return True
        except ImportError:
            print("[ERROR] slack_sdk nicht installiert: pip install slack_sdk")
            return False

    def send_message(self, channel: str, text: str, blocks: List[Dict] = None) -> bool:
        """Sendet Slack Nachricht"""
        if not self._enabled:
            return False

        # Webhook Fallback
        if self.webhook_url and not self.token:
            return self._send_webhook(text)

        if not self._init_client():
            return False

        try:
            response = self._client.chat_postMessage(
                channel=channel,
                text=text,
                blocks=blocks
            )
            return response["ok"]
        except Exception as e:
            print(f"[ERROR] Slack Nachricht fehlgeschlagen: {e}")
            return False

    def _send_webhook(self, text: str) -> bool:
        """Sendet ueber Webhook"""
        try:
            import requests
            response = requests.post(self.webhook_url, json={"text": text})
            return response.status_code == 200
        except Exception as e:
            print(f"[ERROR] Slack Webhook fehlgeschlagen: {e}")
            return False

    def list_channels(self) -> List[Dict]:
        """Listet Kanaele"""
        if not self._init_client():
            return []

        try:
            response = self._client.conversations_list()
            return [
                {"id": c["id"], "name": c["name"]}
                for c in response.get("channels", [])
            ]
        except Exception as e:
            print(f"[ERROR] Channels listen fehlgeschlagen: {e}")
            return []

    def get_messages(self, channel: str, limit: int = 10) -> List[SlackMessage]:
        """Holt Nachrichten aus Kanal"""
        if not self._init_client():
            return []

        try:
            response = self._client.conversations_history(channel=channel, limit=limit)
            return [
                SlackMessage(
                    channel=channel,
                    text=m.get("text", ""),
                    timestamp=m.get("ts", ""),
                    user=m.get("user", ""),
                    thread_ts=m.get("thread_ts", "")
                )
                for m in response.get("messages", [])
            ]
        except Exception as e:
            print(f"[ERROR] Messages holen fehlgeschlagen: {e}")
            return []

    def upload_file(self, channels: List[str], file_path: str, title: str = "") -> bool:
        """Laedt Datei hoch"""
        if not self._init_client():
            return False

        try:
            self._client.files_upload(
                channels=",".join(channels),
                file=file_path,
                title=title or os.path.basename(file_path)
            )
            return True
        except Exception as e:
            print(f"[ERROR] File Upload fehlgeschlagen: {e}")
            return False


class GitHubClient:
    """GitHub API Client"""

    def __init__(self):
        self.token = os.getenv("GITHUB_TOKEN", "")
        self.owner = os.getenv("GITHUB_OWNER", "")
        self.repo = os.getenv("GITHUB_REPO", "")
        self._enabled = bool(self.token)
        self._base_url = "https://api.github.com"

    def _headers(self) -> Dict:
        return {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json"
        }

    def list_issues(self, state: str = "open", labels: str = None) -> List[GitHubIssue]:
        """Listet Issues"""
        if not self._enabled:
            return []

        try:
            import requests

            params = {"state": state}
            if labels:
                params["labels"] = labels

            response = requests.get(
                f"{self._base_url}/repos/{self.owner}/{self.repo}/issues",
                headers=self._headers(),
                params=params
            )

            if response.status_code != 200:
                return []

            return [
                GitHubIssue(
                    number=i["number"],
                    title=i["title"],
                    body=i.get("body", ""),
                    state=i["state"],
                    labels=[l["name"] for l in i.get("labels", [])],
                    assignees=[a["login"] for a in i.get("assignees", [])],
                    created_at=datetime.fromisoformat(i["created_at"].replace("Z", "+00:00")),
                    url=i["html_url"]
                )
                for i in response.json()
            ]
        except Exception as e:
            print(f"[ERROR] GitHub Issues fehlgeschlagen: {e}")
            return []

    def create_issue(self, title: str, body: str = "", labels: List[str] = None) -> Optional[int]:
        """Erstellt Issue"""
        if not self._enabled:
            return None

        try:
            import requests

            payload = {"title": title, "body": body}
            if labels:
                payload["labels"] = labels

            response = requests.post(
                f"{self._base_url}/repos/{self.owner}/{self.repo}/issues",
                headers=self._headers(),
                json=payload
            )

            if response.status_code == 201:
                return response.json()["number"]
            return None
        except Exception as e:
            print(f"[ERROR] Issue erstellen fehlgeschlagen: {e}")
            return None

    def create_pr(self, title: str, body: str, head: str, base: str = "main") -> Optional[int]:
        """Erstellt Pull Request"""
        if not self._enabled:
            return None

        try:
            import requests

            payload = {
                "title": title,
                "body": body,
                "head": head,
                "base": base
            }

            response = requests.post(
                f"{self._base_url}/repos/{self.owner}/{self.repo}/pulls",
                headers=self._headers(),
                json=payload
            )

            if response.status_code == 201:
                return response.json()["number"]
            return None
        except Exception as e:
            print(f"[ERROR] PR erstellen fehlgeschlagen: {e}")
            return None

    def get_repo_info(self) -> Dict:
        """Holt Repository Info"""
        if not self._enabled:
            return {}

        try:
            import requests

            response = requests.get(
                f"{self._base_url}/repos/{self.owner}/{self.repo}",
                headers=self._headers()
            )

            if response.status_code == 200:
                data = response.json()
                return {
                    "name": data["name"],
                    "full_name": data["full_name"],
                    "description": data.get("description", ""),
                    "stars": data["stargazers_count"],
                    "forks": data["forks_count"],
                    "open_issues": data["open_issues_count"],
                    "url": data["html_url"]
                }
            return {}
        except Exception as e:
            print(f"[ERROR] Repo Info fehlgeschlagen: {e}")
            return {}


class NotionClient:
    """Notion API Client"""

    def __init__(self):
        self.token = os.getenv("NOTION_TOKEN", "")
        self._enabled = bool(self.token)
        self._base_url = "https://api.notion.com/v1"

    def _headers(self) -> Dict:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28"
        }

    def search(self, query: str) -> List[Dict]:
        """Sucht in Notion"""
        if not self._enabled:
            return []

        try:
            import requests

            response = requests.post(
                f"{self._base_url}/search",
                headers=self._headers(),
                json={"query": query}
            )

            if response.status_code == 200:
                return response.json().get("results", [])
            return []
        except Exception as e:
            print(f"[ERROR] Notion Suche fehlgeschlagen: {e}")
            return []

    def get_page(self, page_id: str) -> Dict:
        """Holt Notion Page"""
        if not self._enabled:
            return {}

        try:
            import requests

            response = requests.get(
                f"{self._base_url}/pages/{page_id}",
                headers=self._headers()
            )

            if response.status_code == 200:
                return response.json()
            return {}
        except Exception as e:
            print(f"[ERROR] Notion Page fehlgeschlagen: {e}")
            return {}

    def create_page(self, parent_id: str, title: str, content: str = "") -> Optional[str]:
        """Erstellt Notion Page"""
        if not self._enabled:
            return None

        try:
            import requests

            payload = {
                "parent": {"database_id": parent_id},
                "properties": {
                    "title": {
                        "title": [{"text": {"content": title}}]
                    }
                },
                "children": [
                    {
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [{"type": "text", "text": {"content": content}}]
                        }
                    }
                ] if content else []
            }

            response = requests.post(
                f"{self._base_url}/pages",
                headers=self._headers(),
                json=payload
            )

            if response.status_code == 200:
                return response.json()["id"]
            return None
        except Exception as e:
            print(f"[ERROR] Notion Page erstellen fehlgeschlagen: {e}")
            return None


class EnterpriseManager:
    """Zentrale Verwaltung aller Enterprise-Integrationen"""

    def __init__(self):
        self.google = GoogleWorkspaceClient()
        self.slack = SlackClient()
        self.github = GitHubClient()
        self.notion = NotionClient()

    def get_available_integrations(self) -> List[IntegrationType]:
        """Gibt verfuegbare Integrationen zurueck"""
        available = []

        if self.google._enabled:
            available.append(IntegrationType.GOOGLE)
        if self.slack._enabled:
            available.append(IntegrationType.SLACK)
        if self.github._enabled:
            available.append(IntegrationType.GITHUB)
        if self.notion._enabled:
            available.append(IntegrationType.NOTION)

        return available

    def notify_team(self, message: str, channel: str = None) -> bool:
        """Benachrichtigt Team ueber verfuegbare Kanaele"""
        success = False

        if self.slack._enabled:
            success = self.slack.send_message(channel or "#general", message)

        return success

    def create_task(self, title: str, description: str = "", integration: IntegrationType = None) -> Optional[str]:
        """Erstellt Task/Issue im entsprechenden System"""
        if integration == IntegrationType.GITHUB or (not integration and self.github._enabled):
            issue_num = self.github.create_issue(title, description)
            if issue_num:
                return f"github:issue:{issue_num}"

        if integration == IntegrationType.NOTION or (not integration and self.notion._enabled):
            page_id = self.notion.create_page("", title, description)
            if page_id:
                return f"notion:page:{page_id}"

        return None

    def get_calendar_summary(self, days: int = 7) -> str:
        """Gibt Kalender-Zusammenfassung"""
        if not self.google._enabled:
            return "Google Kalender nicht konfiguriert"

        events = self.google.list_calendar_events(days)

        if not events:
            return f"Keine Events in den naechsten {days} Tagen"

        lines = [f"Kalender fuer die naechsten {days} Tage:", ""]
        for event in events[:10]:
            lines.append(f"- {event.start.strftime('%d.%m %H:%M')} - {event.title}")

        return "\n".join(lines)


# Singleton
_manager: Optional[EnterpriseManager] = None


def get_enterprise_manager() -> EnterpriseManager:
    """Gibt Enterprise Manager zurueck"""
    global _manager
    if _manager is None:
        _manager = EnterpriseManager()
    return _manager
