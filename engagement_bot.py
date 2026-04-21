"""
engagement_bot.py
-----------------
Generic LinkedIn engagement bot - production-grade, profile-driven.

What it does per run:
  - Fetches recent posts (within lookback window) for each configured profile
  - For every NEW top-level comment by someone else:
      1. Likes the comment (human-like random jitter delay)
      2. Generates a contextual AI reply via Gemini (skips emoji-only/short comments)
      3. Posts the reply as a nested comment
  - Tracks processed comment IDs in state/processed_comments.json
  - Prunes state entries older than 30 days (FIX 5 - keeps file lean forever)
  - Hard caps: 30 likes + 20 replies per run (FIX 4 - LinkedIn rate limit safety)

State saved locally every run, git-committed only once/day via midnight cron.
This eliminates 11 of 12 daily git commits (Phase 1 plan).

Usage:
  python engagement_bot.py                        # all profiles
  python engagement_bot.py --profile vb_exports_personal
  python engagement_bot.py --dry-run              # no API writes
  python engagement_bot.py --commit-state         # git-commit state (midnight only)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import random
import subprocess
import sys
import time
import unicodedata
from datetime import datetime, timezone, timedelta

from linkedin_client import get_recent_posts, get_comments, like_comment, reply_to_comment
from reply_gen import generate_reply

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("engagement_bot")

# --------------------------------------------------------------------------- #
# Paths & constants
# --------------------------------------------------------------------------- #
ROOT         = pathlib.Path(__file__).parent
PROFILES_DIR = ROOT / "config" / "profiles"
STATE_FILE        = ROOT / "state" / "processed_comments.json"
KNOWN_URNS_FILE   = ROOT / "state" / "known_post_urns.json"

MAX_LIKES_PER_RUN   = 30   # FIX 4: daily safe cap
MAX_REPLIES_PER_RUN = 20   # FIX 4: daily safe cap
STATE_PRUNE_DAYS    = 60   # prune window matches 60-day lookback


# --------------------------------------------------------------------------- #
# FIX 3 - Human-like jitter delay between API write calls
# --------------------------------------------------------------------------- #
def _human_delay() -> None:
    """Random 1-4 second pause - avoids bot detection patterns."""
    time.sleep(random.uniform(1.0, 4.0))


# --------------------------------------------------------------------------- #
# FIX 2 - Detect meaningful comment text (not just emojis / whitespace)
# --------------------------------------------------------------------------- #
def _is_meaningful(text: str, min_chars: int = 5) -> bool:
    """
    Returns True only if the comment has >= min_chars of real non-emoji content.
    A comment of only emojis strips to 0 chars and returns False.
    These still get a LIKE but no AI reply (looks natural, not robotic).
    """
    cleaned = "".join(
        ch for ch in text
        if unicodedata.category(ch) not in ("So", "Cs")  # So=Symbol, Cs=surrogate
    ).strip()
    return len(cleaned) >= min_chars


def _load_known_urns() -> dict:
    """Load state/known_post_urns.json — populated by the carousel bot after each post."""
    if KNOWN_URNS_FILE.exists():
        try:
            return json.loads(KNOWN_URNS_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("known_post_urns.json unreadable (%s)", e)
    return {}


# --------------------------------------------------------------------------- #
# State management
# --------------------------------------------------------------------------- #
def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("State file corrupt (%s) - starting fresh", e)
    return {}


def _prune_state(state: dict) -> tuple[dict, int]:
    """
    FIX 5: Remove entries older than STATE_PRUNE_DAYS.
    Runs on every startup - keeps processed_comments.json lean forever
    regardless of how many comments accumulate over time.
    """
    cutoff  = (datetime.now(timezone.utc) - timedelta(days=STATE_PRUNE_DAYS)).isoformat()
    removed = 0
    for pid in list(state.keys()):
        stale = [cid for cid, ts in state[pid].items() if ts < cutoff]
        for cid in stale:
            del state[pid][cid]
            removed += 1
        if not state[pid]:      # remove empty profile buckets too
            del state[pid]
    return state, removed


def _save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def _is_processed(state: dict, pid: str, cid: str) -> bool:
    return cid in state.get(pid, {})


def _mark_processed(state: dict, pid: str, cid: str) -> None:
    state.setdefault(pid, {})[cid] = datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# Profile loading
# --------------------------------------------------------------------------- #
def _load_profiles(profile_filter: str | None) -> list[dict]:
    """Load all non-template profile JSON files, optionally filtered."""
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


def _resolve_urn(profile: dict) -> str:
    """Env var overrides hardcoded URN - safe for CI where URN is a GitHub Secret."""
    env_var  = profile.get("author_urn_env", "")
    from_env = os.environ.get(env_var, "").strip() if env_var else ""
    return from_env or profile.get("author_urn", "")


# --------------------------------------------------------------------------- #
# Core engagement loop - one profile
# --------------------------------------------------------------------------- #
def run_profile(
    profile: dict,
    access_token: str,
    state: dict,
    dry_run: bool,
    likes_left: list[int],
    replies_left: list[int],
    known_urns: list[str] | None = None,
) -> dict:
    pid        = profile["profile_id"]
    label      = profile.get("label", pid)
    author_urn = _resolve_urn(profile)
    lookback   = profile.get("post_lookback_days", 7)
    max_posts  = profile.get("max_posts_to_check", 10)
    like_all   = profile.get("like_all_comments", True)
    reply_on   = profile.get("reply_enabled", True)
    persona    = profile.get("reply_persona", "")
    max_chars  = profile.get("reply_max_chars", 250)
    signature  = profile.get("signature", "")
    niche      = profile.get("niche", "general")

    log.info("=" * 60)
    log.info("Profile: %s | URN: %s | Niche: %s", label, author_urn, niche)

    if not author_urn:
        log.error("No author URN for profile %s - skipping", pid)
        return state

    cutoff_ms    = int((datetime.now(timezone.utc) - timedelta(days=lookback)).timestamp() * 1000)
    post_context = niche.replace("_", " ").title() + " post"

    posts = get_recent_posts(access_token, author_urn, count=max_posts, known_urns=known_urns)
    log.info("Found %d posts to scan", len(posts))

    total_liked = total_replied = total_skipped = 0

    for post in posts:
        # FIX 4: stop early when both caps exhausted - no point scanning further
        if likes_left[0] <= 0 and replies_left[0] <= 0:
            log.warning("Run caps reached (likes=%d, replies=%d) - stopping early",
                        MAX_LIKES_PER_RUN, MAX_REPLIES_PER_RUN)
            break

        post_urn  = post.get("id", "")
        created_t = post.get("created", {}).get("time", 0)
        if created_t and created_t < cutoff_ms:
            continue  # outside lookback window

        log.info("  Scanning post: %s", post_urn)
        comments = get_comments(access_token, post_urn)
        log.info("    %d comments found", len(comments))

        for c in comments:
            cid   = c.get("id", "")
            curn  = c.get("commentUrn", "")
            ctext = c.get("message", {}).get("text", "").strip()
            cact  = c.get("actor", "")

            if not cid or not curn:
                continue

            # Guard 1: never engage with own comments
            if cact == author_urn:
                continue

            # Guard 2 - FIX 1: skip nested replies entirely
            # parentComment field present = this is a reply-to-a-reply
            # Only engage with top-level comments to avoid reply chain noise
            if c.get("parentComment"):
                continue

            # Guard 3: already handled in a previous run
            if _is_processed(state, pid, cid):
                total_skipped += 1
                continue

            log.info("    New comment [%s]: %.80s", cid, ctext or "(no text)")

            # Action 1: Like the comment
            if like_all and likes_left[0] > 0:
                if dry_run:
                    log.info("      [DRY RUN] Would LIKE comment %s", cid)
                else:
                    if like_comment(access_token, curn, author_urn):
                        likes_left[0] -= 1
                    _human_delay()  # FIX 3: random jitter
                total_liked += 1

            # Action 2: Reply - but ONLY for meaningful text comments
            # FIX 2: emoji-only comments get a like but NO reply
            do_reply = (
                reply_on
                and replies_left[0] > 0
                and _is_meaningful(ctext, min_chars=5)
            )
            if do_reply:
                reply_text = generate_reply(ctext, post_context, persona, max_chars, signature)
                if reply_text:
                    if dry_run:
                        log.info("      [DRY RUN] Would REPLY: %.100s", reply_text)
                    else:
                        if reply_to_comment(access_token, post_urn, curn, author_urn, reply_text):
                            replies_left[0] -= 1
                        _human_delay()  # FIX 3: random jitter
                    total_replied += 1
            elif reply_on and not _is_meaningful(ctext, min_chars=5):
                log.info("      Liked only - short/emoji comment, no reply generated")

            # Mark processed regardless of dry-run
            _mark_processed(state, pid, cid)

    log.info(
        "Profile %s done - liked:%d replied:%d skipped:%d | cap left: likes=%d replies=%d",
        label, total_liked, total_replied, total_skipped, likes_left[0], replies_left[0],
    )
    return state


# --------------------------------------------------------------------------- #
# Daily git commit (Phase 1 plan - midnight cron only, not every 2h run)
# --------------------------------------------------------------------------- #
def _git_commit_state() -> None:
    """
    Commits state/processed_comments.json to git.
    Called ONLY from the midnight cron via --commit-state flag.
    2-hourly runs save state to disk only - no git involved.
    This reduces git noise from 12 commits/day down to 1 commit/day.
    """
    try:
        diff = subprocess.run(
            ["git", "diff", "--quiet", str(STATE_FILE)],
            cwd=ROOT, capture_output=True,
        )
        if diff.returncode == 0:
            log.info("State unchanged - no git commit needed")
            return
        subprocess.run(["git", "config", "user.name",  "engagement-bot[bot]"],
                       cwd=ROOT, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "engagement-bot@noreply.github.com"],
                       cwd=ROOT, check=True, capture_output=True)
        subprocess.run(["git", "add", str(STATE_FILE)],
                       cwd=ROOT, check=True, capture_output=True)
        tag = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        subprocess.run(
            ["git", "commit", "-m", f"chore: daily state snapshot {tag} [skip ci]"],
            cwd=ROOT, check=True, capture_output=True,
        )
        subprocess.run(["git", "push"], cwd=ROOT, check=True, capture_output=True)
        log.info("State committed (daily snapshot %s)", tag)
    except Exception as e:
        log.warning("Git commit failed (non-fatal): %s", e)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="LinkedIn Engagement Bot")
    parser.add_argument("--profile",      default=None, help="Run only this profile_id")
    parser.add_argument("--dry-run",      action="store_true", help="Simulate - no API writes")
    parser.add_argument("--commit-state", action="store_true",
                        help="Git-commit state after run (midnight cron only)")
    args = parser.parse_args()

    token = os.environ.get("LINKEDIN_ACCESS_TOKEN", "").strip()
    if not token:
        log.error("LINKEDIN_ACCESS_TOKEN not set")
        sys.exit(1)

    profiles = _load_profiles(args.profile)
    if not profiles:
        log.warning("No profiles found (filter=%s)", args.profile)
        sys.exit(0)

    log.info("Loaded %d profile(s) | dry_run=%s", len(profiles), args.dry_run)

    known_urns = _load_known_urns()
    log.info("Known post URNs loaded: %d profile(s) in sync file", len(known_urns))

    # FIX 5: prune stale entries every run before processing
    state, pruned = _prune_state(_load_state())
    if pruned:
        log.info("Pruned %d stale state entries (>%d days old)", pruned, STATE_PRUNE_DAYS)

    # FIX 4: mutable shared caps - all profiles in one run share the same budget
    likes_left   = [MAX_LIKES_PER_RUN]
    replies_left = [MAX_REPLIES_PER_RUN]

    for profile in profiles:
        try:
            state = run_profile(
                profile, token, state, args.dry_run,
                likes_left, replies_left,
                known_urns.get(profile.get("profile_id", ""), []),
            )
        except Exception as e:
            log.error("Profile %s failed: %s", profile.get("profile_id"), e)

    # Always save state to disk (cheap local write - no git)
    _save_state(state)
    log.info("State saved to %s", STATE_FILE)

    # Git commit only when explicitly triggered (midnight cron)
    if args.commit_state:
        _git_commit_state()

    log.info("Run complete - likes cap remaining: %d | replies cap remaining: %d",
             likes_left[0], replies_left[0])


if __name__ == "__main__":
    main()
