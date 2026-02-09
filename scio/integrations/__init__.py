#!/usr/bin/env python3
"""
SCIO Integrations Module

Enterprise und Social Media Integrationen.
"""

from .social_media import (
    SocialMediaManager,
    TwitterClient,
    RedditClient,
    DiscordWebhook,
    TelegramBot,
    LinkedInClient,
    get_social_media_manager,
    # Data Classes
    SocialPost,
    SocialProfile,
    SocialMetrics,
    # Enums
    Platform,
    PostType,
)

from .enterprise import (
    EnterpriseManager,
    GoogleWorkspaceClient,
    SlackClient,
    GitHubClient,
    NotionClient,
    get_enterprise_manager,
    # Data Classes
    CalendarEvent,
    Document,
    SlackMessage,
    GitHubIssue,
    # Enums
    IntegrationType,
)

__all__ = [
    # Social Media
    "SocialMediaManager",
    "TwitterClient",
    "RedditClient",
    "DiscordWebhook",
    "TelegramBot",
    "LinkedInClient",
    "get_social_media_manager",
    "SocialPost",
    "SocialProfile",
    "SocialMetrics",
    "Platform",
    "PostType",
    # Enterprise
    "EnterpriseManager",
    "GoogleWorkspaceClient",
    "SlackClient",
    "GitHubClient",
    "NotionClient",
    "get_enterprise_manager",
    "CalendarEvent",
    "Document",
    "SlackMessage",
    "GitHubIssue",
    "IntegrationType",
]
