#!/usr/bin/env python3
"""
SCIO - Social Media Integration

Ermoeglicht Interaktion mit:
- Twitter/X API
- LinkedIn API
- Reddit API
- Discord Webhooks
- Telegram Bot
"""

import os
import time
import json
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class Platform(str, Enum):
    """Social Media Plattformen"""
    TWITTER = "twitter"
    LINKEDIN = "linkedin"
    REDDIT = "reddit"
    DISCORD = "discord"
    TELEGRAM = "telegram"
    MASTODON = "mastodon"


class PostType(str, Enum):
    """Post-Typen"""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    LINK = "link"
    POLL = "poll"
    THREAD = "thread"


@dataclass
class SocialPost:
    """Social Media Post"""
    platform: Platform
    content: str
    post_type: PostType = PostType.TEXT
    media_urls: List[str] = field(default_factory=list)
    hashtags: List[str] = field(default_factory=list)
    mentions: List[str] = field(default_factory=list)
    link: str = ""
    scheduled_time: Optional[datetime] = None
    post_id: str = ""
    created_at: Optional[datetime] = None


@dataclass
class SocialProfile:
    """Social Media Profil"""
    platform: Platform
    username: str
    display_name: str = ""
    followers: int = 0
    following: int = 0
    posts_count: int = 0
    verified: bool = False
    bio: str = ""
    profile_url: str = ""


@dataclass
class SocialMetrics:
    """Engagement Metriken"""
    likes: int = 0
    comments: int = 0
    shares: int = 0
    views: int = 0
    engagement_rate: float = 0.0
    reach: int = 0


class TwitterClient:
    """Twitter/X API Client"""

    def __init__(self):
        self.api_key = os.getenv("TWITTER_API_KEY", "")
        self.api_secret = os.getenv("TWITTER_API_SECRET", "")
        self.access_token = os.getenv("TWITTER_ACCESS_TOKEN", "")
        self.access_secret = os.getenv("TWITTER_ACCESS_SECRET", "")
        self.bearer_token = os.getenv("TWITTER_BEARER_TOKEN", "")
        self._client = None
        self._enabled = bool(self.bearer_token or (self.api_key and self.api_secret))

    def _init_client(self):
        """Initialisiert Tweepy Client"""
        if self._client:
            return True

        try:
            import tweepy

            if self.bearer_token:
                self._client = tweepy.Client(bearer_token=self.bearer_token)
            else:
                auth = tweepy.OAuthHandler(self.api_key, self.api_secret)
                auth.set_access_token(self.access_token, self.access_secret)
                self._client = tweepy.Client(
                    consumer_key=self.api_key,
                    consumer_secret=self.api_secret,
                    access_token=self.access_token,
                    access_token_secret=self.access_secret
                )
            return True

        except ImportError:
            print("[ERROR] tweepy nicht installiert: pip install tweepy")
            return False
        except Exception as e:
            print(f"[ERROR] Twitter Client Fehler: {e}")
            return False

    def post_tweet(self, text: str, media_ids: List[str] = None) -> Optional[str]:
        """Postet einen Tweet"""
        if not self._enabled or not self._init_client():
            return None

        try:
            response = self._client.create_tweet(text=text, media_ids=media_ids)
            return str(response.data["id"])
        except Exception as e:
            print(f"[ERROR] Tweet fehlgeschlagen: {e}")
            return None

    def search_tweets(self, query: str, max_results: int = 10) -> List[Dict]:
        """Sucht Tweets"""
        if not self._enabled or not self._init_client():
            return []

        try:
            tweets = self._client.search_recent_tweets(
                query=query,
                max_results=max_results,
                tweet_fields=["created_at", "public_metrics", "author_id"]
            )

            if not tweets.data:
                return []

            return [
                {
                    "id": str(t.id),
                    "text": t.text,
                    "created_at": str(t.created_at),
                    "metrics": t.public_metrics
                }
                for t in tweets.data
            ]
        except Exception as e:
            print(f"[ERROR] Twitter Suche fehlgeschlagen: {e}")
            return []

    def get_user(self, username: str) -> Optional[SocialProfile]:
        """Holt User-Profil"""
        if not self._enabled or not self._init_client():
            return None

        try:
            user = self._client.get_user(
                username=username,
                user_fields=["description", "public_metrics", "verified"]
            )

            if not user.data:
                return None

            return SocialProfile(
                platform=Platform.TWITTER,
                username=username,
                display_name=user.data.name,
                followers=user.data.public_metrics.get("followers_count", 0),
                following=user.data.public_metrics.get("following_count", 0),
                posts_count=user.data.public_metrics.get("tweet_count", 0),
                verified=user.data.verified or False,
                bio=user.data.description or "",
                profile_url=f"https://twitter.com/{username}"
            )
        except Exception as e:
            print(f"[ERROR] Twitter User Fehler: {e}")
            return None


class RedditClient:
    """Reddit API Client"""

    def __init__(self):
        self.client_id = os.getenv("REDDIT_CLIENT_ID", "")
        self.client_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
        self.username = os.getenv("REDDIT_USERNAME", "")
        self.password = os.getenv("REDDIT_PASSWORD", "")
        self.user_agent = os.getenv("REDDIT_USER_AGENT", "SCIO/1.0")
        self._reddit = None
        self._enabled = bool(self.client_id and self.client_secret)

    def _init_client(self):
        """Initialisiert PRAW Client"""
        if self._reddit:
            return True

        try:
            import praw

            self._reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                username=self.username,
                password=self.password,
                user_agent=self.user_agent
            )
            return True

        except ImportError:
            print("[ERROR] praw nicht installiert: pip install praw")
            return False
        except Exception as e:
            print(f"[ERROR] Reddit Client Fehler: {e}")
            return False

    def get_subreddit_posts(self, subreddit: str, limit: int = 10, sort: str = "hot") -> List[Dict]:
        """Holt Subreddit Posts"""
        if not self._enabled or not self._init_client():
            return []

        try:
            sub = self._reddit.subreddit(subreddit)

            if sort == "hot":
                posts = sub.hot(limit=limit)
            elif sort == "new":
                posts = sub.new(limit=limit)
            elif sort == "top":
                posts = sub.top(limit=limit)
            else:
                posts = sub.hot(limit=limit)

            return [
                {
                    "id": p.id,
                    "title": p.title,
                    "selftext": p.selftext[:500] if p.selftext else "",
                    "url": p.url,
                    "score": p.score,
                    "num_comments": p.num_comments,
                    "author": str(p.author),
                    "created_utc": p.created_utc
                }
                for p in posts
            ]
        except Exception as e:
            print(f"[ERROR] Reddit Posts fehlgeschlagen: {e}")
            return []

    def search(self, query: str, subreddit: str = "all", limit: int = 10) -> List[Dict]:
        """Sucht auf Reddit"""
        if not self._enabled or not self._init_client():
            return []

        try:
            sub = self._reddit.subreddit(subreddit)
            results = sub.search(query, limit=limit)

            return [
                {
                    "id": p.id,
                    "title": p.title,
                    "subreddit": str(p.subreddit),
                    "score": p.score,
                    "url": f"https://reddit.com{p.permalink}"
                }
                for p in results
            ]
        except Exception as e:
            print(f"[ERROR] Reddit Suche fehlgeschlagen: {e}")
            return []

    def post(self, subreddit: str, title: str, text: str = None, url: str = None) -> Optional[str]:
        """Erstellt Reddit Post"""
        if not self._enabled or not self._init_client():
            return None

        try:
            sub = self._reddit.subreddit(subreddit)

            if text:
                submission = sub.submit(title=title, selftext=text)
            elif url:
                submission = sub.submit(title=title, url=url)
            else:
                return None

            return submission.id
        except Exception as e:
            print(f"[ERROR] Reddit Post fehlgeschlagen: {e}")
            return None


class DiscordWebhook:
    """Discord Webhook Client"""

    def __init__(self, webhook_url: str = None):
        self.webhook_url = webhook_url or os.getenv("DISCORD_WEBHOOK", "")
        self._enabled = bool(self.webhook_url)

    def send(self, content: str, username: str = "SCIO", avatar_url: str = None, embeds: List[Dict] = None) -> bool:
        """Sendet Discord Nachricht"""
        if not self._enabled:
            return False

        try:
            import requests

            payload = {
                "content": content,
                "username": username
            }

            if avatar_url:
                payload["avatar_url"] = avatar_url
            if embeds:
                payload["embeds"] = embeds

            response = requests.post(self.webhook_url, json=payload)
            return response.status_code == 204

        except Exception as e:
            print(f"[ERROR] Discord Webhook fehlgeschlagen: {e}")
            return False

    def send_embed(self, title: str, description: str, color: int = 0x00ff00,
                   fields: List[Dict] = None, thumbnail: str = None) -> bool:
        """Sendet Discord Embed"""
        embed = {
            "title": title,
            "description": description,
            "color": color,
            "timestamp": datetime.utcnow().isoformat()
        }

        if fields:
            embed["fields"] = fields
        if thumbnail:
            embed["thumbnail"] = {"url": thumbnail}

        return self.send("", embeds=[embed])


class TelegramBot:
    """Telegram Bot Client"""

    def __init__(self):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self._enabled = bool(self.bot_token)
        self._base_url = f"https://api.telegram.org/bot{self.bot_token}"

    def send_message(self, text: str, chat_id: str = None, parse_mode: str = "HTML") -> bool:
        """Sendet Telegram Nachricht"""
        if not self._enabled:
            return False

        try:
            import requests

            payload = {
                "chat_id": chat_id or self.chat_id,
                "text": text,
                "parse_mode": parse_mode
            }

            response = requests.post(f"{self._base_url}/sendMessage", json=payload)
            return response.status_code == 200

        except Exception as e:
            print(f"[ERROR] Telegram Nachricht fehlgeschlagen: {e}")
            return False

    def send_photo(self, photo_url: str, caption: str = "", chat_id: str = None) -> bool:
        """Sendet Foto"""
        if not self._enabled:
            return False

        try:
            import requests

            payload = {
                "chat_id": chat_id or self.chat_id,
                "photo": photo_url,
                "caption": caption
            }

            response = requests.post(f"{self._base_url}/sendPhoto", json=payload)
            return response.status_code == 200

        except Exception as e:
            print(f"[ERROR] Telegram Foto fehlgeschlagen: {e}")
            return False

    def get_updates(self, offset: int = 0) -> List[Dict]:
        """Holt Bot Updates"""
        if not self._enabled:
            return []

        try:
            import requests

            response = requests.get(f"{self._base_url}/getUpdates", params={"offset": offset})

            if response.status_code == 200:
                data = response.json()
                return data.get("result", [])
            return []

        except Exception as e:
            print(f"[ERROR] Telegram Updates fehlgeschlagen: {e}")
            return []


class LinkedInClient:
    """LinkedIn API Client (OAuth 2.0)"""

    def __init__(self):
        self.client_id = os.getenv("LINKEDIN_CLIENT_ID", "")
        self.client_secret = os.getenv("LINKEDIN_CLIENT_SECRET", "")
        self.access_token = os.getenv("LINKEDIN_ACCESS_TOKEN", "")
        self._enabled = bool(self.access_token)
        self._base_url = "https://api.linkedin.com/v2"

    def _headers(self) -> Dict:
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
            "X-Restli-Protocol-Version": "2.0.0"
        }

    def get_profile(self) -> Optional[SocialProfile]:
        """Holt eigenes Profil"""
        if not self._enabled:
            return None

        try:
            import requests

            response = requests.get(f"{self._base_url}/me", headers=self._headers())

            if response.status_code == 200:
                data = response.json()
                return SocialProfile(
                    platform=Platform.LINKEDIN,
                    username=data.get("id", ""),
                    display_name=f"{data.get('firstName', {}).get('localized', {}).get('en_US', '')} "
                                 f"{data.get('lastName', {}).get('localized', {}).get('en_US', '')}".strip()
                )
            return None

        except Exception as e:
            print(f"[ERROR] LinkedIn Profil fehlgeschlagen: {e}")
            return None

    def share_post(self, text: str, visibility: str = "PUBLIC") -> bool:
        """Teilt einen Post"""
        if not self._enabled:
            return False

        try:
            import requests

            # Erst User-ID holen
            me_response = requests.get(f"{self._base_url}/me", headers=self._headers())
            if me_response.status_code != 200:
                return False

            user_id = me_response.json().get("id")

            payload = {
                "author": f"urn:li:person:{user_id}",
                "lifecycleState": "PUBLISHED",
                "specificContent": {
                    "com.linkedin.ugc.ShareContent": {
                        "shareCommentary": {
                            "text": text
                        },
                        "shareMediaCategory": "NONE"
                    }
                },
                "visibility": {
                    "com.linkedin.ugc.MemberNetworkVisibility": visibility
                }
            }

            response = requests.post(f"{self._base_url}/ugcPosts", headers=self._headers(), json=payload)
            return response.status_code == 201

        except Exception as e:
            print(f"[ERROR] LinkedIn Post fehlgeschlagen: {e}")
            return False


class SocialMediaManager:
    """Zentrale Verwaltung aller Social Media Plattformen"""

    def __init__(self):
        self.twitter = TwitterClient()
        self.reddit = RedditClient()
        self.discord = DiscordWebhook()
        self.telegram = TelegramBot()
        self.linkedin = LinkedInClient()

        self._post_history: List[SocialPost] = []
        self._scheduled_posts: List[SocialPost] = []

    def get_available_platforms(self) -> List[Platform]:
        """Gibt verfuegbare Plattformen zurueck"""
        available = []

        if self.twitter._enabled:
            available.append(Platform.TWITTER)
        if self.reddit._enabled:
            available.append(Platform.REDDIT)
        if self.discord._enabled:
            available.append(Platform.DISCORD)
        if self.telegram._enabled:
            available.append(Platform.TELEGRAM)
        if self.linkedin._enabled:
            available.append(Platform.LINKEDIN)

        return available

    def post(self, platform: Platform, content: str, **kwargs) -> bool:
        """Postet auf einer Plattform"""
        success = False

        if platform == Platform.TWITTER:
            post_id = self.twitter.post_tweet(content)
            success = post_id is not None
        elif platform == Platform.REDDIT:
            subreddit = kwargs.get("subreddit", "test")
            title = kwargs.get("title", content[:100])
            post_id = self.reddit.post(subreddit, title, text=content)
            success = post_id is not None
        elif platform == Platform.DISCORD:
            success = self.discord.send(content)
        elif platform == Platform.TELEGRAM:
            success = self.telegram.send_message(content)
        elif platform == Platform.LINKEDIN:
            success = self.linkedin.share_post(content)

        if success:
            self._post_history.append(SocialPost(
                platform=platform,
                content=content,
                created_at=datetime.now()
            ))

        return success

    def cross_post(self, content: str, platforms: List[Platform] = None) -> Dict[Platform, bool]:
        """Postet auf mehreren Plattformen"""
        if platforms is None:
            platforms = self.get_available_platforms()

        results = {}
        for platform in platforms:
            results[platform] = self.post(platform, content)

        return results

    def search_all(self, query: str, limit: int = 5) -> Dict[Platform, List[Dict]]:
        """Sucht auf allen Plattformen"""
        results = {}

        if self.twitter._enabled:
            results[Platform.TWITTER] = self.twitter.search_tweets(query, limit)

        if self.reddit._enabled:
            results[Platform.REDDIT] = self.reddit.search(query, limit=limit)

        return results

    def schedule_post(self, post: SocialPost):
        """Plant einen Post"""
        self._scheduled_posts.append(post)
        self._scheduled_posts.sort(key=lambda p: p.scheduled_time or datetime.max)

    def process_scheduled(self) -> int:
        """Verarbeitet geplante Posts"""
        now = datetime.now()
        posted = 0

        for post in self._scheduled_posts[:]:
            if post.scheduled_time and post.scheduled_time <= now:
                if self.post(post.platform, post.content):
                    self._scheduled_posts.remove(post)
                    posted += 1

        return posted

    def get_post_history(self, platform: Platform = None, limit: int = 100) -> List[SocialPost]:
        """Gibt Post-Historie zurueck"""
        history = self._post_history

        if platform:
            history = [p for p in history if p.platform == platform]

        return history[-limit:]

    def notify_all(self, title: str, message: str) -> Dict[str, bool]:
        """Sendet Benachrichtigung auf alle Kanaele"""
        results = {}

        if self.discord._enabled:
            results["discord"] = self.discord.send_embed(title, message)

        if self.telegram._enabled:
            results["telegram"] = self.telegram.send_message(f"<b>{title}</b>\n\n{message}")

        return results


# Singleton
_manager: Optional[SocialMediaManager] = None


def get_social_media_manager() -> SocialMediaManager:
    """Gibt Social Media Manager zurueck"""
    global _manager
    if _manager is None:
        _manager = SocialMediaManager()
    return _manager
