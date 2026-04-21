"""
engagement_bot.py
─────────────────
Generic LinkedIn engagement bot.
Reads comments on recent posts, likes each one, and replies with AI.

Driven entirely by JSON profile configs — no code changes needed for new niches.
Add a new JSON file in config/profiles/ and it works automatically.

Usage:
  python engagement_bot.py                      # all active profiles
  python engagement_bot.py --profile vb_exports_personal
  python engagement_bot.py --dry-run            # simulate only, no API writes
  python engagement_bot.py --profile vb_exports_company --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys
import time
from datetime import datetime, timezone, timedelta

from linkedin_client import get_recent_posts, get_comments, like_comment, reply_to_comment
from reply_gen import generate_reply

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("engagement_bot")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = pathlib.Path(__file__).parent
PROFILES_DIR  = ROOT / "config" / "profiles"
STATE_FILE    = ROOT / "state" / "processed_comments.json"

# ── Cooldown between API calls (ms) ──────────────────────────────────────────
CALL_DELAY_S  = 1.5   # seconds between each API write call (rate-limit safety)


# ══════════════════════════════════════════════════════════════════════════════
# State management
# ══════════════════════════════════════════════════════════════════════════════

def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def _is_processed(state: dict, profile_id: str, comment_id: str) -> bool:
    return comment_id in state.get(profile_id, {})


def _mark_processed(state: dict, profile_id: str, comment_id: str) -> None:
    if profile_id not in state:
        state[profile_id] = {}
    state[profile_id][comment_id] = datetime.now(timezone.utc).isoformat()


# ══════════════════════════════════════════════════════════════════════════════
# Profile loading
# ══════════════════════════════════════════════════════════════════════════════

def _load_profiles(profile_filter: str | None) -> list[dict]:
    """Load all non-template profile JSON files, optionally filtered by profile_id."""
    profiles = []
    for path in sorted(PROFILES_DIR.glob("*.json")):
        if path.name.startswith("_"):
            continue  # skip _template.json
        try:
            p = json.loads(path.read_text(encoding="utf-8"))
            if profile_filter and p.get("profile_id") != profile_filter:
                continue
            profiles.append(p)
        except Exception as e:
            log.warning("Skipping %s: %s", path.name, e)
    return profiles


# ══════════════════════════════════════════════════════════════════════════════
# Resolve author URN from env (env var overrides hardcoded value)
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_urn(profile: dict) -> str:
    env_var = profile.get("author_urn_env", "")
    from_env = os.environ.get(env_var, "").strip() if env_var else ""
    return from_env or profile.get("author_urn", "")


# ══════════════════════════════════════════════════════════════════════════════
# Core engagement loop for one profile
# ══════════════════════════════════════════════════════════════════════════════

def run_profile(
    profile: dict,
    access_token: str,
    state: dict,
    dry_run: bool,
) -> dict:
    """
    Runs the engagement loop for a single profile.
    Returns updated state dict.
    """
    profile_id   = profile["profile_id"]
    label        = profile.get("label", profile_id)
    author_urn   = _resolve_urn(profile)
    lookback     = profile.get("post_lookback_days", 7)
    max_posts    = profile.get("max_posts_to_check", 10)
    like_all     = profile.get("like_all_comments", True)
    reply_on     = profile.get("reply_enabled", True)
    persona      = profile.get("reply_persona", "")
    max_chars    = profile.get("reply_max_chars", 250)
    signature    = profile.get("signature", "")
    niche        = profile.get("niche", "general")

    log.info("=" * 60)
    log.info("Profile: %s | Author: %s | Niche: %s", label, author_urn, niche)
    log.info("Dry run: %s", dry_run)

    if not author_urn:
        log.error("No author URN for profile %s — skipping", profile_id)
        return state

    # Cutoff timestamp for lookback window
    cutoff_ms = int((datetime.now(timezone.utc) - timedelta(days=lookback)).timestamp() * 1000)

    # Step 1: Get recent posts
    posts = get_recent_posts(access_token, author_urn, count=max_posts)
    log.info("Found %d recent posts", len(posts))

    total_liked  = 0
    total_replied = 0
    total_skipped = 0

    for post in posts:
        post_urn = post.get("id", "")
        created_time = post.get("created", {}).get("time", 0)

        # Skip posts older than lookback window
        if created_time and created_time < cutoff_ms:
            continue

        log.info("  Post: %s", post_urn)

        # Step 2: Get comments on this post
        comments = get_comments(access_token, post_urn)
        log.info("    %d comments found", len(comments))

        # Extract post context from post data for AI reply
        post_context = niche.replace("_", " ").title() + " post"

        for comment in comments:
            comment_id  = comment.get("id", "")
            comment_urn = comment.get("commentUrn", "")
            comment_text = comment.get("message", {}).get("text", "").strip()
            commenter_urn = comment.get("actor", "")

            if not comment_id or not comment_urn:
                continue

            # Skip own comments
            if commenter_urn == author_urn:
                continue

            # Skip already processed
            if _is_processed(state, profile_id, comment_id):
                total_skipped += 1
                continue

            log.info("    New comment [%s]: %s", comment_id, comment_text[:80])

            # Step 3: Like the comment
            if like_all:
                if dry_run:
                    log.info("      [DRY RUN] Would like comment %s", comment_id)
                else:
                    like_comment(access_token, comment_urn, author_urn)
                    time.sleep(CALL_DELAY_S)
                total_liked += 1

            # Step 4: Generate and post reply
            if reply_on and comment_text:
                reply_text = generate_reply(
                    comment_text=comment_text,
                    post_context=post_context,
                    persona=persona,
                    max_chars=max_chars,
                    signature=signature,
                )
                if reply_text:
                    if dry_run:
                        log.info("      [DRY RUN] Would reply: %s", reply_text[:100])
                    else:
                        reply_to_comment(
                            access_token=access_token,
                            post_urn=post_urn,
                            comment_urn=comment_urn,
                            actor_urn=author_urn,
                            reply_text=reply_text,
                        )
                        time.sleep(CALL_DELAY_S)
                    total_replied += 1

            # Mark as processed (even in dry-run so re-runs don't re-process)
            _mark_processed(state, profile_id, comment_id)

    log.info("Profile %s done — liked: %d | replied: %d | skipped (already done): %d",
             label, total_liked, total_replied, total_skipped)
    return state


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="LinkedIn Engagement Bot")
    parser.add_argument("--profile", default=None,
                        help="Run only this profile_id (default: all profiles)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate only — no API write calls")
    args = parser.parse_args()

    access_token = os.environ.get("LINKEDIN_ACCESS_TOKEN", "").strip()
    if not access_token:
        log.error("LINKEDIN_ACCESS_TOKEN not set")
        sys.exit(1)

    profiles = _load_profiles(args.profile)
    if not profiles:
        log.warning("No profiles found (filter=%s)", args.profile)
        sys.exit(0)

    log.info("Loaded %d profile(s)", len(profiles))
    state = _load_state()

    for profile in profiles:
        try:
            state = run_profile(profile, access_token, state, dry_run=args.dry_run)
        except Exception as e:
            log.error("Profile %s failed: %s", profile.get("profile_id"), e)

    _save_state(state)
    log.info("State saved to %s", STATE_FILE)
    log.info("Engagement bot run complete.")


if __name__ == "__main__":
    main()
