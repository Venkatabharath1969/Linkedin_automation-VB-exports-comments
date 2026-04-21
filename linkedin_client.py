"""
linkedin_client.py
──────────────────
Thin LinkedIn REST API wrapper for the engagement bot.
Handles: get posts, get comments, post like, post nested comment reply.

All patterns match the working linkedin_publisher.py from the carousel project.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

log = logging.getLogger(__name__)

BASE             = "https://api.linkedin.com"
LINKEDIN_VERSION = "202503"
HEADERS_BASE     = {
    "LinkedIn-Version": LINKEDIN_VERSION,
    "X-Restli-Protocol-Version": "2.0.0",
}
_VERIFY_SSL  = False
_MAX_RETRIES = 3
_INIT_DELAY  = 2


# ── Retry wrapper ─────────────────────────────────────────────────────────────

def _retry(fn, *args, **kwargs):
    delay = _INIT_DELAY
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = fn(*args, **kwargs)
            if 400 <= resp.status_code < 500 and resp.status_code != 429:
                log.error("LinkedIn %d — not retrying: %s", resp.status_code, resp.text[:300])
                resp.raise_for_status()
            resp.raise_for_status()
            return resp
        except requests.HTTPError as e:
            if attempt == _MAX_RETRIES:
                raise
            log.warning("Attempt %d/%d failed (%s) — retrying in %ds", attempt, _MAX_RETRIES, e, delay)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("_retry exhausted")


def _auth_headers(access_token: str) -> dict:
    return {**HEADERS_BASE, "Authorization": f"Bearer {access_token}"}


# ── Posts ─────────────────────────────────────────────────────────────────────

def get_recent_posts(access_token: str, author_urn: str, count: int = 10) -> list[dict]:
    """
    Returns up to `count` recent posts by the author.
    Each element has at least: id (ugcPost URN), created.time
    """
    import urllib.parse
    params = {"author": author_urn, "count": count, "q": "author"}
    try:
        resp = _retry(
            requests.get,
            f"{BASE}/rest/posts",
            params=params,
            headers=_auth_headers(access_token),
            timeout=15,
            verify=_VERIFY_SSL,
        )
        return resp.json().get("elements", [])
    except Exception as e:
        log.warning("get_recent_posts failed: %s", e)
        return []


# ── Comments ──────────────────────────────────────────────────────────────────

def get_comments(access_token: str, post_urn: str) -> list[dict]:
    """
    Returns all top-level comments on a post.
    Each element has: id, actor, message.text, created.time, commentUrn
    """
    import urllib.parse
    encoded = urllib.parse.quote(post_urn, safe="")
    try:
        resp = _retry(
            requests.get,
            f"{BASE}/rest/socialActions/{encoded}/comments",
            headers=_auth_headers(access_token),
            timeout=15,
            verify=_VERIFY_SSL,
        )
        return resp.json().get("elements", [])
    except Exception as e:
        log.warning("get_comments(%s) failed: %s", post_urn, e)
        return []


# ── Like a comment ────────────────────────────────────────────────────────────

def like_comment(access_token: str, comment_urn: str, actor_urn: str) -> bool:
    """
    Likes a comment. comment_urn is the full composite commentUrn.
    Returns True on success.
    """
    import urllib.parse
    encoded = urllib.parse.quote(comment_urn, safe="")
    try:
        _retry(
            requests.post,
            f"{BASE}/rest/socialActions/{encoded}/likes",
            headers={**_auth_headers(access_token), "Content-Type": "application/json"},
            json={"actor": actor_urn},
            timeout=15,
            verify=_VERIFY_SSL,
        )
        log.info("  Liked comment: %s", comment_urn[-30:])
        return True
    except Exception as e:
        log.warning("  like_comment failed: %s", e)
        return False


# ── Reply to a comment ────────────────────────────────────────────────────────

def reply_to_comment(
    access_token: str,
    post_urn: str,
    comment_urn: str,
    actor_urn: str,
    reply_text: str,
) -> bool:
    """
    Posts a nested reply to a comment.
    - post_urn:    the ugcPost URN (for URL path)
    - comment_urn: composite commentUrn (for parentComment field)
    - actor_urn:   person or org URN posting the reply
    Returns True on success.
    """
    import urllib.parse
    encoded_post = urllib.parse.quote(post_urn, safe="")
    try:
        resp = _retry(
            requests.post,
            f"{BASE}/rest/socialActions/{encoded_post}/comments",
            headers={**_auth_headers(access_token), "Content-Type": "application/json"},
            json={
                "actor":         actor_urn,
                "object":        post_urn,          # raw URN in body
                "message":       {"text": reply_text},
                "parentComment": comment_urn,       # makes it a nested reply
            },
            timeout=20,
            verify=_VERIFY_SSL,
        )
        log.info("  Reply posted (HTTP %d)", resp.status_code)
        return True
    except Exception as e:
        log.warning("  reply_to_comment failed: %s", e)
        return False
