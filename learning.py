"""Server learning for llmcord.

Lets the bot be shaped by the server it lives in:
- remembers what people say and builds notes about the server (slang, running jokes)
- measures how people actually type (lowercase, message length, emoji, common phrases)
- keeps an evolving personality that drifts toward the crew, with fixed guardrails
- learns which of its own messages land (reactions, replies) and which flop
- keeps notes on each person, plus facts people tell it with /remember
- remembers gifs, images and videos people post so it can send them back

Everything is stored locally in data/memory.db (plus data/media/ for saved
files). Nothing leaves the computer the bot runs on.
"""

from array import array
import asyncio
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
import logging
import math
import os
import random
import re
import sqlite3
import time
from typing import Any, Awaitable, Callable, Optional
from urllib.parse import urlparse

import discord
import httpx

try:
    import numpy
except ImportError:  # optional, only makes memory recall faster
    numpy = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "memory.db")
MEDIA_DIR = os.path.join(DATA_DIR, "media")

URL_RE = re.compile(r"https?://[^\s<>]+")
MEDIA_TAG_RE = re.compile(r"\[\s*media\s*[:#]?\s*(\d+)\s*\]", re.IGNORECASE)
REACT_TAG_RE = re.compile(r"\[\s*react\s*[:#]?\s*(\d+)\s*\]", re.IGNORECASE)
CUSTOM_EMOJI_RE = re.compile(r"<a?:(\w+):(\d+)>")
WORD_RE = re.compile(r"[a-z0-9']+")
TOKEN_RE = re.compile(r"[a-z][a-z0-9']*")
EMOJI_RE = re.compile(r"<a?:\w+:\d+>|[\U0001F300-\U0001FAFF☀-➿]")
MENTION_ID_RE = re.compile(r"<@!?(\d+)>")

MAX_STORED_TEXT = 1000
PROMPT_CHAR_BUDGET = 8000  # keeps background prompts inside a small local model's context
RETRY_SECONDS = 600
HABITS_CACHE_SECONDS = 600
MAX_MEDIA_PER_MESSAGE = 5
FEEDBACK_DAYS = 30
HIT_DAYS = 7
OVERUSE_WINDOW = 40  # how many of the bot's recent messages are checked for repeats

Complete = Callable[[list[dict[str, str]]], Awaitable[str]]
Embed = Callable[[list[str], str], Awaitable[list[list[float]]]]  # (texts, "document" | "query") -> vectors

DEFAULT_REACTIONS = ["💀", "😭", "😂", "🫡", "👀", "🔥"]
MOMENT_WINDOW = 8  # max messages per remembered moment
MOMENT_GAP_SECONDS = 900  # a pause this long starts a new moment
MOMENT_SETTLE_SECONDS = 600  # wait this long so a conversation is finished before saving it
MOMENTS_EVERY = 20  # save new moments after this many unsaved messages

DISCORD_CDN_HOSTS = {"cdn.discordapp.com", "media.discordapp.net"}

EXTENSION_KINDS = {
    ".gif": "gif",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".webp": "image",
    ".mp4": "video",
    ".mov": "video",
    ".webm": "video",
}

GENERIC_NAME_WORDS = {
    "image", "images", "img", "unknown", "attachment", "video", "file", "photo", "screenshot", "gif", "clip",
    "tenor", "giphy", "download", "untitled", "mp4", "png", "jpg", "jpeg", "webp", "mov", "webm", "media",
}

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be", "been", "to", "of", "in", "on", "at", "for",
    "it", "its", "it's", "i", "im", "i'm", "you", "your", "you're", "u", "me", "my", "we", "he", "she", "they", "them",
    "this", "that", "with", "so", "just", "like", "do", "does", "did", "dont", "don't", "not", "no", "yes", "what",
    "when", "who", "how", "why", "can", "can't", "have", "has", "had", "if", "then", "there", "here", "about", "up",
    "out", "get", "got", "all", "some", "one", "said", "after", "captioned", "someone", "gif", "image", "video", "will",
    "would", "should", "could", "from", "into", "too", "very", "really", "also", "his", "her", "him", "our", "us", "am",
    "go", "going", "know", "think", "now", "then", "than", "that's", "there's", "what's", "a", "oh", "ok", "okay",
    "posted", "said", "want", "need", "make", "see", "yeah", "lol",
}

# Short words that are slang, not noise, so they count as phrases worth copying
SLANG_KEEP = {"lol", "lmao", "lmfao", "fr", "ngl", "tbh", "idk", "bro", "bruh", "ong", "istg", "wtf", "nah", "yeah", "yea", "ight", "aight", "deadass", "lowkey", "highkey"}

# Two-word phrases starting with these are just "the reactor" style noise
PHRASE_BAD_STARTS = {"the", "a", "an", "this", "that", "my", "your", "our", "their", "his", "her", "its", "of", "to", "in", "on", "at", "for"}

LAUGH_EMOJI = {"💀", "😭", "😂", "🤣", "☠", "😹", "💯", "🔥"}
NEGATIVE_EMOJI = {"👎", "🙄", "😐", "😑", "🤢", "🚮"}
LAUGH_RE = re.compile(r"\b(lmao+|lmfao|lol+|haha+|dead|dying|crying|hilarious|i'?m weak)\b|💀|😭|😂|🤣", re.IGNORECASE)
NEGATIVE_RE = re.compile(r"\b(shut up|stfu|cringe|unfunny|not funny|nobody asked|be quiet|annoying)\b|👎|🙄", re.IGNORECASE)


@dataclass
class LearnResult:
    stored_text: bool = False
    media_saved: int = 0
    lore_due: bool = False
    persona_due: bool = False
    person_due: bool = False
    moments_due: bool = False


@dataclass
class LearnedContext:
    text: str = ""
    media: dict[int, Any] = field(default_factory=dict)
    reactions: list[str] = field(default_factory=list)
    overused: list[str] = field(default_factory=list)


_media_bytes: Optional[int] = None
_running: set[tuple] = set()
_last_attempt: dict[tuple, float] = {}
_optouts: set[tuple[int, int]] = set()
_habits_cache: dict[int, tuple[float, str]] = {}
_moment_cache: dict[int, list[tuple]] = {}
_embed_failed_logged = False
_llm_lock = asyncio.Lock()


# ---------------------------------------------------------------- database


def _connect() -> sqlite3.Connection:
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def _db(sql: str, params: tuple = (), fetch: Optional[str] = None) -> Any:
    conn = _connect()
    try:
        cursor = conn.execute(sql, params)
        if fetch == "all":
            result = cursor.fetchall()
        elif fetch == "one":
            result = cursor.fetchone()
        else:
            result = cursor.rowcount
        conn.commit()
        return result
    finally:
        conn.close()


async def _adb(sql: str, params: tuple = (), fetch: Optional[str] = None) -> Any:
    return await asyncio.to_thread(_db, sql, params, fetch)


def init_db() -> None:
    conn = _connect()
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY,
                guild_id INTEGER,
                channel_id INTEGER,
                author_id INTEGER,
                author_name TEXT,
                content TEXT,
                created_at REAL,
                to_bot INTEGER DEFAULT 0,
                chunked INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_messages_guild ON messages (guild_id, id);
            CREATE INDEX IF NOT EXISTS idx_messages_channel ON messages (channel_id, id);
            CREATE INDEX IF NOT EXISTS idx_messages_author ON messages (guild_id, author_id, id);

            CREATE TABLE IF NOT EXISTS media (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                guild_id INTEGER,
                channel_id INTEGER,
                message_id INTEGER,
                author_id INTEGER,
                kind TEXT,
                url TEXT,
                file_path TEXT,
                description TEXT DEFAULT '',
                context TEXT DEFAULT '',
                created_at REAL,
                last_used REAL DEFAULT 0,
                UNIQUE (message_id, url)
            );
            CREATE INDEX IF NOT EXISTS idx_media_guild ON media (guild_id, id);

            CREATE TABLE IF NOT EXISTS lore (guild_id INTEGER PRIMARY KEY, notes TEXT, updated_at REAL);

            CREATE TABLE IF NOT EXISTS persona (guild_id INTEGER PRIMARY KEY, traits TEXT, updated_at REAL);

            CREATE TABLE IF NOT EXISTS bot_messages (
                id INTEGER PRIMARY KEY,
                guild_id INTEGER,
                channel_id INTEGER,
                content TEXT,
                created_at REAL
            );
            CREATE INDEX IF NOT EXISTS idx_bot_messages_guild ON bot_messages (guild_id, created_at);

            CREATE TABLE IF NOT EXISTS feedback (
                message_id INTEGER,
                user_id INTEGER,
                kind TEXT,
                key TEXT,
                weight REAL,
                created_at REAL,
                PRIMARY KEY (message_id, user_id, kind, key)
            );

            CREATE TABLE IF NOT EXISTS people (
                guild_id INTEGER,
                user_id INTEGER,
                name TEXT,
                notes TEXT,
                updated_at REAL,
                PRIMARY KEY (guild_id, user_id)
            );

            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                guild_id INTEGER,
                user_id INTEGER,
                fact TEXT,
                added_by INTEGER,
                created_at REAL
            );

            CREATE TABLE IF NOT EXISTS optouts (guild_id INTEGER, user_id INTEGER, PRIMARY KEY (guild_id, user_id));

            CREATE TABLE IF NOT EXISTS emoji_usage (guild_id INTEGER, emoji TEXT, count INTEGER DEFAULT 0, PRIMARY KEY (guild_id, emoji));

            CREATE TABLE IF NOT EXISTS moments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                guild_id INTEGER,
                channel_id INTEGER,
                start_id INTEGER,
                end_id INTEGER,
                author_ids TEXT,
                text TEXT,
                embedding BLOB,
                created_at REAL
            );
            CREATE INDEX IF NOT EXISTS idx_moments_guild ON moments (guild_id, created_at);
            """
        )

        # Databases made by older versions of this file are missing these columns
        columns = {row[1] for row in conn.execute("PRAGMA table_info(messages)")}
        if "to_bot" not in columns:
            conn.execute("ALTER TABLE messages ADD COLUMN to_bot INTEGER DEFAULT 0")
        if "chunked" not in columns:
            conn.execute("ALTER TABLE messages ADD COLUMN chunked INTEGER DEFAULT 0")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_unchunked ON messages (guild_id, chunked, created_at)")

        conn.commit()
        _optouts.clear()
        _optouts.update((row[0], row[1]) for row in conn.execute("SELECT guild_id, user_id FROM optouts"))
    finally:
        conn.close()


# ---------------------------------------------------------------- small helpers


def words(text: str) -> set[str]:
    return {w for w in WORD_RE.findall((text or "").lower()) if w not in STOPWORDS and len(w) > 2}


def kind_from_extension(path: str) -> Optional[str]:
    return EXTENSION_KINDS.get(os.path.splitext(path.lower())[1])


def kind_from_content_type(content_type: Optional[str]) -> Optional[str]:
    content_type = (content_type or "").lower()
    if content_type == "image/gif":
        return "gif"
    if content_type.startswith("image/"):
        return "image"
    if content_type.startswith("video/"):
        return "video"
    return None


def describe_filename(path: str) -> str:
    name = os.path.splitext(os.path.basename(path))[0]
    parts = [p for p in re.split(r"[-_.\s]+", name.lower()) if p and not p.isdigit() and p not in GENERIC_NAME_WORDS]
    return " ".join(parts)


def _slug_words(slug: str, drop_last: bool = False) -> str:
    parts = [p for p in slug.strip("/").split("-") if p]
    if drop_last:
        parts = parts[:-1]
    return " ".join(p for p in parts if not p.isdigit() and p.lower() not in GENERIC_NAME_WORDS)


def is_discord_hosted(url: str) -> bool:
    return (urlparse(url).hostname or "").lower() in DISCORD_CDN_HOSTS


def strip_query(url: str) -> str:
    return url.split("?", 1)[0]


def classify_link(url: str) -> Optional[tuple[str, str]]:
    """Return (kind, description) for links worth remembering as reaction media."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().removeprefix("www.")
    path = parsed.path

    if host in DISCORD_CDN_HOSTS:
        kind = kind_from_extension(path)
        return (kind, describe_filename(path)) if kind else None

    if host == "tenor.com" and path.startswith("/view/"):
        return "gif", _slug_words(path.split("/view/", 1)[1])
    if host == "media.tenor.com":
        return "gif", ""
    if host == "giphy.com" and path.startswith("/gifs/"):
        return "gif", _slug_words(path.split("/gifs/", 1)[1], drop_last=True)
    if host.endswith("giphy.com"):
        return "gif", ""
    if host == "i.imgur.com" and kind_from_extension(path):
        return kind_from_extension(path), ""
    if host == "imgur.com" and len(path) > 1:
        return "image", ""

    if (host in ("youtube.com", "m.youtube.com") and (path.startswith("/watch") or path.startswith("/shorts/"))) or host == "youtu.be":
        return "video", ""
    if host in ("tiktok.com", "vm.tiktok.com", "vt.tiktok.com") and len(path) > 1:
        return "video", ""
    if host in ("x.com", "twitter.com", "fxtwitter.com", "vxtwitter.com", "fixupx.com") and "/status/" in path:
        return "video", ""
    if host == "streamable.com" and len(path) > 1:
        return "video", ""
    if host == "v.redd.it":
        return "video", ""
    if host == "instagram.com" and (path.startswith("/reel/") or path.startswith("/p/")):
        return "video", ""

    return None


def clean_message_text(msg: Any) -> str:
    text = URL_RE.sub("", getattr(msg, "clean_content", None) or msg.content or "")
    return re.sub(r"\s+", " ", text).strip()[:MAX_STORED_TEXT]


def _normalize_emoji(emoji: str) -> str:
    return emoji.replace("️", "")


def reaction_weight(emoji: str, name: str = "") -> int:
    emoji = _normalize_emoji(emoji)
    if emoji in LAUGH_EMOJI or re.search(r"lol|lmao|laugh|kek|dead|skull|cry", name or "", re.IGNORECASE):
        return 2
    if emoji in NEGATIVE_EMOJI:
        return -2
    return 1


def reply_weight(text: str) -> int:
    if NEGATIVE_RE.search(text or ""):
        return -2
    if LAUGH_RE.search(text or ""):
        return 2
    return 0  # a reply isn't approval: people reply to mock things too


def _budgeted_lines(lines: list[str], budget: int) -> list[str]:
    """Keep lines (newest first in input) until the character budget runs out."""
    kept, used = [], 0
    for line in lines:
        if used + len(line) > budget:
            break
        kept.append(line)
        used += len(line)
    return kept


def can_learn_from(msg: Any, config: dict[str, Any]) -> bool:
    if not msg.guild or msg.author.bot or (msg.guild.id, msg.author.id) in _optouts:
        return False

    learn_cfg = config.get("learning") or {}
    permissions = config.get("permissions") or {}
    users = permissions.get("users") or {}
    roles = permissions.get("roles") or {}
    channels = permissions.get("channels") or {}

    if msg.author.id in (learn_cfg.get("ignored_user_ids") or []) or msg.author.id in (users.get("blocked_ids") or []):
        return False

    role_ids = {role.id for role in getattr(msg.author, "roles", ())}
    if role_ids & set(roles.get("blocked_ids") or []):
        return False

    channel_ids = set(
        filter(None, (msg.channel.id, getattr(msg.channel, "parent_id", None), getattr(msg.channel, "category_id", None)))
    )
    if channel_ids & set(channels.get("blocked_ids") or []):
        return False
    if (allowed := channels.get("allowed_ids")) and not channel_ids & set(allowed):
        return False
    if (learn_channels := learn_cfg.get("channel_ids")) and not channel_ids & set(learn_channels):
        return False

    return True


def _folder_size(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def _write_file(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as file:
        file.write(data)


def _remove_file(relative_path: Optional[str]) -> None:
    if relative_path:
        try:
            os.remove(os.path.join(DATA_DIR, relative_path))
        except OSError:
            pass


async def _download(http_client: httpx.AsyncClient, url: str, dest: str, max_bytes: int, max_total_bytes: int) -> Optional[str]:
    """Download url to dest (relative to DATA_DIR). Returns the relative path or None."""
    global _media_bytes

    if _media_bytes is None:
        _media_bytes = await asyncio.to_thread(_folder_size, MEDIA_DIR)
    if _media_bytes >= max_total_bytes:
        return None

    chunks, size = [], 0
    try:
        async with http_client.stream("GET", url, timeout=30) as resp:
            if resp.status_code != 200 or int(resp.headers.get("content-length") or 0) > max_bytes:
                return None
            async for chunk in resp.aiter_bytes():
                size += len(chunk)
                if size > max_bytes:
                    return None
                chunks.append(chunk)
    except Exception:
        logging.exception(f"Couldn't download media: {strip_query(url)}")
        return None

    await asyncio.to_thread(_write_file, os.path.join(DATA_DIR, dest), b"".join(chunks))
    _media_bytes += size
    return dest


# ---------------------------------------------------------------- background jobs


def _job_blocked(key: tuple) -> bool:
    return key in _running or time.time() - _last_attempt.get(key, 0) < RETRY_SECONDS


async def _run_job(key: tuple, force: bool, job: Callable[[], Awaitable[bool]]) -> bool:
    """Run one background model job at a time, with a retry cooldown so failures don't loop."""
    if key in _running or (not force and _job_blocked(key)):
        return False

    _running.add(key)
    _last_attempt[key] = time.time()
    try:
        async with _llm_lock:
            return await job()
    except Exception:
        logging.exception(f"Background learning job failed: {key[0]}")
        return False
    finally:
        _running.discard(key)


async def _count_since(guild_id: int, since: float, author_id: Optional[int] = None) -> int:
    if author_id is None:
        row = await _adb("SELECT COUNT(*) AS n FROM messages WHERE guild_id = ? AND created_at > ?", (guild_id, since), "one")
    else:
        row = await _adb("SELECT COUNT(*) AS n FROM messages WHERE guild_id = ? AND author_id = ? AND created_at > ?", (guild_id, author_id, since), "one")
    return row["n"]


# ---------------------------------------------------------------- learning from messages


async def learn_from_message(msg: Any, config: dict[str, Any], http_client: httpx.AsyncClient, bot_user_id: Optional[int] = None) -> LearnResult:
    """Store a server message, any media in it, and feedback on the bot's messages. Safe on every message."""
    result = LearnResult()
    learn_cfg = config.get("learning") or {}
    media_cfg = config.get("media") or {}

    if not (learn_cfg.get("enabled") or media_cfg.get("enabled")) or not can_learn_from(msg, config):
        return result

    text = clean_message_text(msg)
    guild_id = msg.guild.id

    # Is this aimed at the bot? (mentions it, or replies to one of its messages)
    replied_to_bot = None
    if (reference := getattr(msg, "reference", None)) and (ref_id := getattr(reference, "message_id", None)):
        replied_to_bot = await _adb("SELECT id FROM bot_messages WHERE id = ?", (ref_id,), "one")
    to_bot = bool(replied_to_bot) or any(getattr(user, "id", None) == bot_user_id for user in getattr(msg, "mentions", []))

    if replied_to_bot:
        await _adb(
            "INSERT OR REPLACE INTO feedback (message_id, user_id, kind, key, weight, created_at) VALUES (?, ?, 'reply', ?, ?, ?)",
            (replied_to_bot["id"], msg.author.id, str(msg.id), reply_weight(text), time.time()),
        )

    if learn_cfg.get("enabled") and text:
        await _adb(
            "INSERT OR REPLACE INTO messages (id, guild_id, channel_id, author_id, author_name, content, created_at, to_bot) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (msg.id, guild_id, msg.channel.id, msg.author.id, msg.author.display_name, text, msg.created_at.timestamp(), int(to_bot)),
        )
        result.stored_text = True

    if media_cfg.get("enabled"):
        result.media_saved = await _save_media(msg, text, media_cfg, http_client)

    # Track which emoji people use in their messages
    for emoji in EMOJI_RE.findall(msg.content or ""):
        await record_emoji_use(guild_id, emoji)

    if learn_cfg.get("enabled") and result.stored_text:
        result.lore_due = await _lore_due(guild_id, learn_cfg)
        result.persona_due = await _persona_due(guild_id, config)
        result.person_due = await _person_due(guild_id, msg.author.id, config)
        result.moments_due = await _moments_due(guild_id, config)

    return result


async def _save_media(msg: Any, text: str, media_cfg: dict[str, Any], http_client: httpx.AsyncClient) -> int:
    items = []

    for attachment in msg.attachments:
        if kind := kind_from_content_type(attachment.content_type) or kind_from_extension(attachment.filename):
            items.append(dict(kind=kind, url=attachment.url, description=describe_filename(attachment.filename), size=attachment.size))

    for url in URL_RE.findall(msg.content or ""):
        url = url.rstrip(").,>")
        if classified := classify_link(url):
            items.append(dict(kind=classified[0], url=url, description=classified[1], size=0))

    if not items:
        return 0

    context_parts = [f'captioned "{text[:120]}"'] if text else []
    previous = await _adb(
        "SELECT author_name, content FROM messages WHERE channel_id = ? AND id < ? ORDER BY id DESC LIMIT 1",
        (msg.channel.id, msg.id),
        "one",
    )
    if previous:
        context_parts.append(f'posted after {previous["author_name"]} said "{previous["content"][:120]}"')
    context = "; ".join(context_parts)

    save_files = media_cfg.get("save_files", True)
    max_bytes = int(media_cfg.get("max_file_mb", 8) * 1024 * 1024)
    max_total_bytes = int(media_cfg.get("max_total_mb", 1000) * 1024 * 1024)
    saved = 0

    for index, item in enumerate(items[:MAX_MEDIA_PER_MESSAGE]):
        file_path = None

        # Discord's own links expire after about a day, so keep a copy of those files
        if is_discord_hosted(item["url"]):
            if not save_files or item["size"] > max_bytes:
                continue
            ext = os.path.splitext(urlparse(item["url"]).path.lower())[1]
            ext = ext if ext in EXTENSION_KINDS else {"gif": ".gif", "video": ".mp4"}.get(item["kind"], ".png")
            dest = os.path.join("media", str(msg.guild.id), f"{msg.id}_{index}{ext}")
            if not (file_path := await _download(http_client, item["url"], dest, max_bytes, max_total_bytes)):
                continue

        stored_url = strip_query(item["url"]) if is_discord_hosted(item["url"]) else item["url"]
        changed = await _adb(
            "INSERT OR IGNORE INTO media (guild_id, channel_id, message_id, author_id, kind, url, file_path, description, context, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (msg.guild.id, msg.channel.id, msg.id, msg.author.id, item["kind"], stored_url, file_path, item["description"], context, msg.created_at.timestamp()),
        )
        saved += changed

    if saved:
        await update_media_from_embeds(msg)

    return saved


async def update_media_from_embeds(msg: Any) -> None:
    """Fill in titles (YouTube video names etc.) once Discord has generated link previews."""
    titled = [embed for embed in msg.embeds if embed.title]
    if not titled:
        return

    missing = await _adb("SELECT id, url FROM media WHERE message_id = ? AND (description IS NULL OR description = '')", (msg.id,), "all")
    if not missing:
        return

    if len(missing) == 1 and len(titled) == 1:
        await _adb("UPDATE media SET description = ? WHERE id = ?", (titled[0].title[:150], missing[0]["id"]))
        return

    for embed in titled:
        for row in missing:
            if embed.url and strip_query(embed.url) == strip_query(row["url"]):
                await _adb("UPDATE media SET description = ? WHERE id = ?", (embed.title[:150], row["id"]))


async def forget_messages(message_ids: list[int]) -> None:
    """Remove deleted messages and their saved media, so deleted stuff never gets reposted or recalled."""
    for message_id in message_ids:
        if stored := await _adb("SELECT guild_id, channel_id FROM messages WHERE id = ?", (message_id,), "one"):
            await _adb(
                "DELETE FROM moments WHERE channel_id = ? AND start_id <= ? AND end_id >= ?",
                (stored["channel_id"], message_id, message_id),
            )
            _moment_cache.pop(stored["guild_id"], None)

        rows = await _adb("SELECT file_path FROM media WHERE message_id = ?", (message_id,), "all") or []
        for row in rows:
            _remove_file(row["file_path"])
        await _adb("DELETE FROM media WHERE message_id = ?", (message_id,))
        await _adb("DELETE FROM messages WHERE id = ?", (message_id,))
        await _adb("DELETE FROM bot_messages WHERE id = ?", (message_id,))
        await _adb("DELETE FROM feedback WHERE message_id = ? OR (kind = 'reply' AND key = ?)", (message_id, str(message_id)))


# ---------------------------------------------------------------- feedback on the bot's own messages


async def record_bot_messages(guild_id: int, channel_id: int, message_ids: list[int], text: str) -> None:
    for message_id in message_ids:
        await _adb(
            "INSERT OR REPLACE INTO bot_messages (id, guild_id, channel_id, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (message_id, guild_id, channel_id, text[:1000], time.time()),
        )


async def record_reaction(message_id: int, user_id: int, emoji: str, emoji_name: str, added: bool) -> None:
    if not await _adb("SELECT id FROM bot_messages WHERE id = ?", (message_id,), "one"):
        return

    key = _normalize_emoji(emoji)
    if added:
        await _adb(
            "INSERT OR REPLACE INTO feedback (message_id, user_id, kind, key, weight, created_at) VALUES (?, ?, 'reaction', ?, ?, ?)",
            (message_id, user_id, key, reaction_weight(emoji, emoji_name), time.time()),
        )
    else:
        await _adb("DELETE FROM feedback WHERE message_id = ? AND user_id = ? AND kind = 'reaction' AND key = ?", (message_id, user_id, key))


async def get_hits(guild_id: int, limit: int) -> list[str]:
    """The bot's best-received lines lately."""
    rows = await _adb(
        "SELECT b.content, SUM(f.weight) AS score, COUNT(DISTINCT CASE WHEN f.weight > 0 THEN f.user_id END) AS fans "
        "FROM bot_messages b JOIN feedback f ON f.message_id = b.id "
        "WHERE b.guild_id = ? AND b.created_at > ? GROUP BY b.id HAVING score >= 3 AND fans >= 2 ORDER BY score DESC LIMIT ?",
        (guild_id, time.time() - HIT_DAYS * 86400, limit * 3),
        "all",
    )
    return list(dict.fromkeys(row["content"] for row in rows if row["content"]))[:limit]


async def get_flops(guild_id: int, limit: int) -> list[str]:
    """Lines that got negative reactions, or got ignored while people were actively chatting."""
    rows = await _adb(
        "SELECT b.id, b.channel_id, b.content, b.created_at, COALESCE(SUM(f.weight), 0) AS score "
        "FROM bot_messages b LEFT JOIN feedback f ON f.message_id = b.id "
        "WHERE b.guild_id = ? AND b.created_at > ? AND b.created_at < ? GROUP BY b.id ORDER BY b.created_at DESC LIMIT 100",
        (guild_id, time.time() - FEEDBACK_DAYS * 86400, time.time() - 600),
        "all",
    )

    flops = []
    for row in rows:
        if row["score"] < 0:
            flops.append(row["content"])
        elif row["score"] == 0:
            chatter = await _adb(
                "SELECT COUNT(*) AS n FROM messages WHERE channel_id = ? AND created_at > ? AND created_at < ?",
                (row["channel_id"], row["created_at"], row["created_at"] + 600),
                "one",
            )
            if chatter["n"] >= 5:
                flops.append(row["content"])

    return list(dict.fromkeys(content for content in flops if content))[:limit]


async def _recent_bot_texts(guild_id: int, limit: int = OVERUSE_WINDOW) -> list[str]:
    rows = await _adb(
        "SELECT content FROM bot_messages WHERE guild_id = ? ORDER BY created_at DESC LIMIT ?", (guild_id, limit * 3), "all"
    )
    texts = [row["content"] for row in rows if row["content"] and not row["content"].startswith("(sent a")]
    return list(dict.fromkeys(texts))[:limit]  # long replies are stored once per Discord message


def find_overused_phrases(texts: list[str], min_messages: int = 2, limit: int = 8) -> list[str]:
    """Phrases (3-6 words) the bot has used in several different messages."""
    counts: Counter = Counter()
    for text in texts:
        tokens = TOKEN_RE.findall(text.lower())
        grams = set()
        for size in range(3, 7):
            for start in range(len(tokens) - size + 1):
                gram = tokens[start : start + size]
                if sum(1 for t in gram if t not in STOPWORDS) >= 2:
                    grams.add(" ".join(gram))
        counts.update(grams)

    def trigrams(phrase: str) -> set[str]:
        """Word pairs in the phrase (skipping pairs of filler words), used to spot overlapping phrases."""
        tokens = phrase.split()
        return {f"{a} {b}" for a, b in zip(tokens, tokens[1:]) if not (a in STOPWORDS and b in STOPWORDS)}

    # Most-repeated first, longest first among ties; skip anything overlapping a phrase already picked,
    # so one repeated sentence doesn't flood the list with overlapping fragments
    repeated = sorted(((g, n) for g, n in counts.items() if n >= min_messages), key=lambda item: (item[1], len(item[0])), reverse=True)
    kept: list[str] = []
    taken: set[str] = set()
    for gram, _ in repeated:
        if trigrams(gram) & taken:
            continue
        kept.append(gram)
        taken |= trigrams(gram)
        if len(kept) >= limit:
            break
    return kept


async def get_overused_phrases(guild_id: int) -> list[str]:
    return find_overused_phrases(await _recent_bot_texts(guild_id))


def contains_overused(text: str, phrases: list[str]) -> list[str]:
    lowered = " ".join(TOKEN_RE.findall((text or "").lower()))
    return [phrase for phrase in phrases if phrase in lowered]


# ---------------------------------------------------------------- speech habits


async def get_speech_habits(guild_id: int) -> str:
    """Measure how people here actually type, as concrete rules. Cached for a few minutes."""
    if (cached := _habits_cache.get(guild_id)) and time.time() - cached[0] < HABITS_CACHE_SECONDS:
        return cached[1]

    rows = await _adb("SELECT author_id, content FROM messages WHERE guild_id = ? ORDER BY id DESC LIMIT 600", (guild_id,), "all")
    bot_text = " ".join(" ".join(TOKEN_RE.findall(t.lower())) for t in await _recent_bot_texts(guild_id, 100))
    text = compute_speech_habits([(row["author_id"], row["content"]) for row in rows], exclude_text=bot_text)
    _habits_cache[guild_id] = (time.time(), text)
    return text


def compute_speech_habits(rows: list[tuple[int, str]], exclude_text: str = "") -> str:
    texts = [(author, content) for author, content in rows if content and content.strip()]
    if len(texts) < 30:
        return ""

    lines = []
    lettered = [content for _, content in texts if re.search(r"[A-Za-z]", content)]
    if lettered:
        lowercase = sum(1 for content in lettered if content == content.lower()) / len(lettered)
        if lowercase >= 0.7:
            lines.append(f"Almost everyone types in all lowercase ({lowercase:.0%} of messages). You do too.")
        elif lowercase <= 0.3:
            lines.append("People here mostly capitalize normally.")

    periods = sum(1 for _, content in texts if content.rstrip().endswith(".") and not content.rstrip().endswith("..")) / len(texts)
    if periods <= 0.15:
        lines.append("Hardly anyone ends a message with a period. Don't.")
    elif periods >= 0.5:
        lines.append("People here use proper punctuation.")

    lengths = sorted(len(content.split()) for _, content in texts)
    median = lengths[len(lengths) // 2]
    lines.append(f"A typical message here is about {median} words. Match that; only go longer when actually explaining something.")

    emoji_counts = Counter(_normalize_emoji(e) for _, content in texts for e in EMOJI_RE.findall(content))
    emoji_rate = sum(1 for _, content in texts if EMOJI_RE.search(content)) / len(texts)
    if emoji_rate >= 0.2 and emoji_counts:
        favorites = " ".join(e for e, _ in emoji_counts.most_common(5))
        lines.append(f"Emoji are common here. Favorites: {favorites}")
    elif emoji_rate <= 0.05:
        lines.append("Emoji are rare here. Mostly skip them.")

    # Words and two-word phrases used by several different people
    users_by_phrase: dict[str, set[int]] = {}
    phrase_counts: Counter = Counter()
    for author, content in texts:
        tokens = TOKEN_RE.findall(content.lower())
        grams = [t for t in tokens if t in SLANG_KEEP or (t not in STOPWORDS and len(t) > 2)]
        grams += [
            f"{a} {b}"
            for a, b in zip(tokens, tokens[1:])
            if a not in PHRASE_BAD_STARTS and not (a in STOPWORDS and b in STOPWORDS) and len(a + b) > 4
        ]
        for gram in set(grams):
            phrase_counts[gram] += 1
            users_by_phrase.setdefault(gram, set()).add(author)

    common = [p for p, n in phrase_counts.most_common(200) if n >= 4 and len(users_by_phrase[p]) >= 2 and not (len(p) > 3 and p in exclude_text)]
    if common:
        lines.append("Words and phrases people here use a lot: " + ", ".join(common[:15]))

    return "\n".join(f"- {line}" for line in lines)


# ---------------------------------------------------------------- server notes (lore)


async def get_lore(guild_id: int) -> str:
    row = await _adb("SELECT notes FROM lore WHERE guild_id = ?", (guild_id,), "one")
    return row["notes"] if row else ""


async def _lore_due(guild_id: int, learn_cfg: dict[str, Any]) -> bool:
    if _job_blocked(("lore", guild_id)):
        return False
    row = await _adb("SELECT updated_at FROM lore WHERE guild_id = ?", (guild_id,), "one")
    return await _count_since(guild_id, row["updated_at"] if row else 0) >= learn_cfg.get("lore_every_messages", 200)


async def rebuild_lore(guild_id: int, config: dict[str, Any], complete: Complete, force: bool = False, fresh: bool = False) -> bool:
    """Ask the model to update its notes about the server from recent messages."""

    async def job() -> bool:
        learn_cfg = config.get("learning") or {}
        rows = await _adb(
            "SELECT author_name, content FROM messages WHERE guild_id = ? ORDER BY id DESC LIMIT ?",
            (guild_id, learn_cfg.get("lore_source_messages", 120)),
            "all",
        )
        if len(rows) < 15:
            return False

        lines = _budgeted_lines([f"{row['author_name']}: {row['content'][:200]}" for row in rows], PROMPT_CHAR_BUDGET)[::-1]
        old_notes = "(none yet)" if fresh else (await get_lore(guild_id) or "(none yet)")
        prompt = [
            dict(role="system", content="You keep short, accurate notes about a Discord community for a chatbot that hangs out there."),
            dict(
                role="user",
                content=(
                    "Recent messages from the server:\n"
                    + "\n".join(lines)
                    + f"\n\nYour current notes:\n{old_notes}\n\n"
                    "Rewrite the notes. Keep what is still true and add what is new. Cover: slang and phrases people use, "
                    "running jokes and memes, and what people talk about a lot. Only include things actually seen in the messages. "
                    "Something only counts as a running joke if several people bring it up at different times; don't turn one-off "
                    "comments into running jokes, and ignore people quoting or mocking the bot. "
                    "Skip anything private or sensitive (health, family problems, relationships, where people live). "
                    "Plain bullet points, under 200 words, no intro."
                ),
            ),
        ]

        notes = (await complete(prompt) or "").strip()[:2500]
        if not notes:
            return False

        await _adb("INSERT OR REPLACE INTO lore (guild_id, notes, updated_at) VALUES (?, ?, ?)", (guild_id, notes, time.time()))
        logging.info(f"Updated server notes for guild {guild_id}")
        return True

    return await _run_job(("lore", guild_id), force, job)


# ---------------------------------------------------------------- evolving personality


async def get_persona(guild_id: int, config: dict[str, Any]) -> str:
    row = await _adb("SELECT traits FROM persona WHERE guild_id = ?", (guild_id,), "one")
    return row["traits"] if row else ((config.get("persona") or {}).get("seed") or "").strip()


async def reset_persona(guild_id: int) -> None:
    await _adb("DELETE FROM persona WHERE guild_id = ?", (guild_id,))


async def _persona_due(guild_id: int, config: dict[str, Any]) -> bool:
    persona_cfg = config.get("persona") or {}
    if not persona_cfg.get("enabled") or _job_blocked(("persona", guild_id)):
        return False
    row = await _adb("SELECT updated_at FROM persona WHERE guild_id = ?", (guild_id,), "one")
    return await _count_since(guild_id, row["updated_at"] if row else 0) >= persona_cfg.get("evolve_every_messages", 300)


async def evolve_persona(guild_id: int, config: dict[str, Any], complete: Complete, bot_name: str = "the bot", force: bool = False) -> bool:
    """Let the bot's personality drift toward how this server talks and treats it."""

    async def job() -> bool:
        current = await get_persona(guild_id, config) or "(no personality written yet)"

        chat = await _adb("SELECT author_name, content FROM messages WHERE guild_id = ? ORDER BY id DESC LIMIT 150", (guild_id,), "all")
        if len(chat) < 30:
            return False
        chat_lines = _budgeted_lines([f"{row['author_name']}: {row['content'][:160]}" for row in chat], 4500)[::-1]

        to_bot = await _adb("SELECT author_name, content FROM messages WHERE guild_id = ? AND to_bot = 1 ORDER BY id DESC LIMIT 30", (guild_id,), "all")
        to_bot_lines = _budgeted_lines([f"{row['author_name']}: {row['content'][:160]}" for row in to_bot], 1500)[::-1]

        hits = await get_hits(guild_id, 5)
        flops = await get_flops(guild_id, 5)
        habits = await get_speech_habits(guild_id)

        sections = [
            f"Current personality:\n{current}",
            "How people here type:\n" + (habits or "(not enough data)"),
            "Recent chat in the server:\n" + "\n".join(chat_lines),
            "How people have been talking to you:\n" + ("\n".join(to_bot_lines) or "(nothing yet)"),
            "Your lines that got big laughs or reactions:\n" + ("\n".join(f"- {h}" for h in hits) or "(none yet)"),
            "Your lines that flopped or got ignored:\n" + ("\n".join(f"- {f}" for f in flops) or "(none yet)"),
        ]
        prompt = [
            dict(role="system", content=f"You write the personality profile for {bot_name}, a chatbot who is a regular member of a Discord friend group."),
            dict(
                role="user",
                content=(
                    "\n\n".join(sections)
                    + f"\n\nRewrite {bot_name}'s personality so it fits in with this crew. Change it gradually: keep most of the "
                    "current personality and adjust maybe 10-20% based on how people talk, what makes them laugh, and how they treat "
                    f"{bot_name}. Lean into what landed, drop what flopped. Keep the same name. Keep it subtle and human: a real person whose traits "
                    "show in how they talk, not in announcing them, catchphrases, or making every message about their background. "
                    "Never give the personality catchphrases, signature lines, favorite foods, or recurring objects or topics. "
                    "Write it in second person "
                    "(\"You are...\"), concrete and specific (attitude, humor, how you talk, what you care about), under 150 words, "
                    "no intro, no rules about safety."
                ),
            ),
        ]

        traits = (await complete(prompt) or "").strip()[:1500]
        if not traits:
            return False

        await _adb("INSERT OR REPLACE INTO persona (guild_id, traits, updated_at) VALUES (?, ?, ?)", (guild_id, traits, time.time()))
        logging.info(f"Personality evolved for guild {guild_id}")
        return True

    return await _run_job(("persona", guild_id), force, job)


# ---------------------------------------------------------------- people


async def _latest_name(guild_id: int, user_id: int) -> str:
    row = await _adb("SELECT author_name FROM messages WHERE guild_id = ? AND author_id = ? ORDER BY id DESC LIMIT 1", (guild_id, user_id), "one")
    if row:
        return row["author_name"]
    person = await _adb("SELECT name FROM people WHERE guild_id = ? AND user_id = ?", (guild_id, user_id), "one")
    return person["name"] if person else str(user_id)


async def _person_due(guild_id: int, user_id: int, config: dict[str, Any]) -> bool:
    people_cfg = config.get("people") or {}
    if not people_cfg.get("enabled") or _job_blocked(("person", guild_id, user_id)):
        return False
    row = await _adb("SELECT updated_at FROM people WHERE guild_id = ? AND user_id = ?", (guild_id, user_id), "one")
    return await _count_since(guild_id, row["updated_at"] if row else 0, user_id) >= people_cfg.get("notes_every_messages", 50)


async def get_facts(guild_id: int, user_id: int, limit: int = 8) -> list[Any]:
    return await _adb("SELECT * FROM facts WHERE guild_id = ? AND user_id = ? ORDER BY id DESC LIMIT ?", (guild_id, user_id, limit), "all")


async def add_fact(guild_id: int, user_id: int, fact: str, added_by: int) -> None:
    await _adb(
        "INSERT INTO facts (guild_id, user_id, fact, added_by, created_at) VALUES (?, ?, ?, ?, ?)",
        (guild_id, user_id, fact.strip()[:200], added_by, time.time()),
    )


async def get_person(guild_id: int, user_id: int) -> tuple[str, list[Any]]:
    row = await _adb("SELECT notes FROM people WHERE guild_id = ? AND user_id = ?", (guild_id, user_id), "one")
    return (row["notes"] if row else ""), await get_facts(guild_id, user_id)


async def rebuild_person(guild_id: int, user_id: int, config: dict[str, Any], complete: Complete, bot_name: str = "the bot", force: bool = False) -> bool:
    """Update the bot's notes on one person: what they're into, how they talk, how the bot feels about them."""

    async def job() -> bool:
        if (guild_id, user_id) in _optouts:
            return False

        name = await _latest_name(guild_id, user_id)
        rows = await _adb("SELECT content FROM messages WHERE guild_id = ? AND author_id = ? ORDER BY id DESC LIMIT 100", (guild_id, user_id), "all")
        facts = await get_facts(guild_id, user_id)
        if len(rows) < 10 and not facts:
            return False

        their_lines = _budgeted_lines([row["content"][:200] for row in rows], 3500)[::-1]
        to_bot = await _adb(
            "SELECT content FROM messages WHERE guild_id = ? AND author_id = ? AND to_bot = 1 ORDER BY id DESC LIMIT 15", (guild_id, user_id), "all"
        )
        reactions = await _adb(
            "SELECT SUM(CASE WHEN f.weight >= 2 THEN 1 ELSE 0 END) AS laughs, SUM(CASE WHEN f.weight < 0 THEN 1 ELSE 0 END) AS negative "
            "FROM feedback f JOIN bot_messages b ON b.id = f.message_id WHERE b.guild_id = ? AND f.user_id = ?",
            (guild_id, user_id),
            "one",
        )
        current, _ = await get_person(guild_id, user_id)

        sections = [
            f"Person: {name}",
            "Facts people told you about them:\n" + ("\n".join(f"- {f['fact']}" for f in facts) or "(none)"),
            "Their recent messages:\n" + "\n".join(their_lines),
            "Things they said to you:\n" + ("\n".join(f"- {row['content'][:200]}" for row in to_bot) or "(nothing yet)"),
            f"How they react to you: laughed at your messages {reactions['laughs'] or 0} times, reacted negatively {reactions['negative'] or 0} times.",
            f"Your current notes on them:\n{current or '(none yet)'}",
        ]
        prompt = [
            dict(role="system", content=f"You are {bot_name}, a regular in a Discord friend group. You keep private notes on each person."),
            dict(
                role="user",
                content=(
                    "\n\n".join(sections)
                    + "\n\nRewrite your notes on this person: what they're into, running bits about them, and how they talk. "
                    "End with one line starting \"Your take:\" on how you feel about them based on how they treat you "
                    "(shift it gradually, don't flip it overnight). Only include things supported by the messages or facts, "
                    "and don't turn a single comment into a running bit. "
                    "Skip anything private or sensitive (health, family problems, relationships, where they live). "
                    "Under 100 words, no intro."
                ),
            ),
        ]

        notes = (await complete(prompt) or "").strip()[:1200]
        if not notes:
            return False

        await _adb(
            "INSERT OR REPLACE INTO people (guild_id, user_id, name, notes, updated_at) VALUES (?, ?, ?, ?, ?)",
            (guild_id, user_id, name, notes, time.time()),
        )
        logging.info(f"Updated notes on user {user_id} in guild {guild_id}")
        return True

    return await _run_job(("person", guild_id, user_id), force, job)


async def most_active_users(guild_id: int, limit: int, min_messages: int = 15) -> list[int]:
    rows = await _adb(
        "SELECT author_id, COUNT(*) AS n FROM messages WHERE guild_id = ? GROUP BY author_id HAVING n >= ? ORDER BY n DESC LIMIT ?",
        (guild_id, min_messages, limit),
        "all",
    )
    return [row["author_id"] for row in rows]


async def forget_person(guild_id: int, user_id: int) -> None:
    """Wipe everything stored about someone and stop learning from them."""
    for row in await _adb("SELECT file_path FROM media WHERE guild_id = ? AND author_id = ?", (guild_id, user_id), "all"):
        _remove_file(row["file_path"])

    await _adb("DELETE FROM media WHERE guild_id = ? AND author_id = ?", (guild_id, user_id))
    await _adb("DELETE FROM messages WHERE guild_id = ? AND author_id = ?", (guild_id, user_id))
    await _adb("DELETE FROM moments WHERE guild_id = ? AND author_ids LIKE ?", (guild_id, f"%,{user_id},%"))
    _moment_cache.pop(guild_id, None)
    await _adb("DELETE FROM people WHERE guild_id = ? AND user_id = ?", (guild_id, user_id))
    await _adb("DELETE FROM facts WHERE guild_id = ? AND user_id = ?", (guild_id, user_id))
    await _adb(
        "DELETE FROM feedback WHERE user_id = ? AND message_id IN (SELECT id FROM bot_messages WHERE guild_id = ?)", (user_id, guild_id)
    )
    await _adb("INSERT OR IGNORE INTO optouts (guild_id, user_id) VALUES (?, ?)", (guild_id, user_id))
    _optouts.add((guild_id, user_id))
    _habits_cache.pop(guild_id, None)


async def reset_learning(guild_id: int) -> None:
    """Throw away the personality, notes and feedback (keeps the chat log, media, memories and /remember facts)."""
    await _adb("DELETE FROM persona WHERE guild_id = ?", (guild_id,))
    await _adb("DELETE FROM lore WHERE guild_id = ?", (guild_id,))
    await _adb("DELETE FROM people WHERE guild_id = ?", (guild_id,))
    await _adb("DELETE FROM feedback WHERE message_id IN (SELECT id FROM bot_messages WHERE guild_id = ?)", (guild_id,))
    await _adb("DELETE FROM bot_messages WHERE guild_id = ?", (guild_id,))
    _habits_cache.pop(guild_id, None)
    for key in [k for k in _last_attempt if len(k) > 1 and k[1] == guild_id]:
        _last_attempt.pop(key, None)


async def opt_back_in(guild_id: int, user_id: int) -> None:
    await _adb("DELETE FROM optouts WHERE guild_id = ? AND user_id = ?", (guild_id, user_id))
    _optouts.discard((guild_id, user_id))


def is_opted_out(guild_id: int, user_id: int) -> bool:
    return (guild_id, user_id) in _optouts


# ---------------------------------------------------------------- emoji


async def record_emoji_use(guild_id: int, emoji: str) -> None:
    await _adb(
        "INSERT INTO emoji_usage (guild_id, emoji, count) VALUES (?, ?, 1) ON CONFLICT (guild_id, emoji) DO UPDATE SET count = count + 1",
        (guild_id, _normalize_emoji(emoji)),
    )


async def get_server_emoji(guild_id: int, usable_custom_ids: Optional[set[int]] = None, limit: int = 8) -> list[str]:
    """The server's favorite emoji (from messages and reactions), limited to ones the bot can actually use."""
    usable_custom_ids = usable_custom_ids or set()
    rows = await _adb("SELECT emoji, count FROM emoji_usage WHERE guild_id = ? ORDER BY count DESC LIMIT 100", (guild_id,), "all")

    favorites = []
    for row in rows:
        if custom := CUSTOM_EMOJI_RE.fullmatch(row["emoji"]):
            if int(custom.group(2)) not in usable_custom_ids:
                continue
        favorites.append(row["emoji"])
        if len(favorites) >= limit:
            break

    for emoji in DEFAULT_REACTIONS:
        if len(favorites) >= max(4, min(limit, 6)):
            break
        if emoji not in favorites:
            favorites.append(emoji)

    return favorites


def emoji_label(emoji: str) -> str:
    """How an emoji is shown to the model: custom ones by name."""
    return f":{custom.group(1)}:" if (custom := CUSTOM_EMOJI_RE.fullmatch(emoji)) else emoji


def chosen_reaction(text: str, options: list[str]) -> Optional[str]:
    for match in REACT_TAG_RE.finditer(text):
        if 1 <= (number := int(match.group(1))) <= len(options):
            return options[number - 1]
    return None


def reaction_prompt(bot_name: str, author_name: str, text: str, options: list[str]) -> list[dict[str, str]]:
    """Tiny prompt for reaction-only chime-ins."""
    numbered = "\n".join(f"{i}. {emoji_label(e)}" for i, e in enumerate(options, 1))
    return [
        dict(role="system", content=f"You are {bot_name}, a regular in a Discord friend group. You react to messages with emoji like a friend would."),
        dict(role="user", content=f'{author_name} just said: "{text[:400]}"\n\nWhich emoji reaction fits best?\n{numbered}\n\nAnswer with only the number, or 0 if none fit.'),
    ]


def parse_reaction_choice(output: str, options: list[str]) -> Optional[str]:
    if match := re.search(r"\d+", output or ""):
        if 1 <= (number := int(match.group())) <= len(options):
            return options[number - 1]
    return None


# ---------------------------------------------------------------- mood and time


def time_of_day(now: datetime) -> str:
    hour = now.hour
    if hour < 5:
        return "late night"
    if hour < 9:
        return "early morning"
    if hour < 12:
        return "morning"
    if hour < 17:
        return "afternoon"
    if hour < 21:
        return "evening"
    return "night"


async def mood_section(guild_id: int, channel_id: int, now: datetime, current_message_id: Optional[int] = None) -> str:
    """Time of day, how busy chat is, and a mood based on how people have been treating the bot."""
    t = time.time()
    clock = f"{now.hour % 12 or 12}:{now.minute:02d} {'AM' if now.hour < 12 else 'PM'}"
    label = time_of_day(now)
    lines = [f"Right now it's {now.strftime('%A')} {label} ({clock})."]
    if label == "late night":
        lines.append("It's really late.")

    busy = await _adb("SELECT COUNT(*) AS n FROM messages WHERE channel_id = ? AND created_at > ?", (channel_id, t - 600), "one")
    previous = await _adb(
        "SELECT created_at FROM messages WHERE channel_id = ? AND id < ? ORDER BY id DESC LIMIT 1",
        (channel_id, current_message_id or (1 << 62)),
        "one",
    )
    if busy["n"] >= 15:
        lines.append(f"Chat is busy right now ({busy['n']} messages in the last 10 minutes).")
    elif previous and (gap_hours := (t - previous["created_at"]) / 3600) >= 3:
        lines.append(f"Chat was dead for about {round(gap_hours)} hours before this.")

    feedback = await _adb(
        "SELECT COALESCE(SUM(f.weight), 0) AS score, SUM(CASE WHEN f.weight < 0 THEN 1 ELSE 0 END) AS negative "
        "FROM feedback f JOIN bot_messages b ON b.id = f.message_id WHERE b.guild_id = ? AND f.created_at > ?",
        (guild_id, t - 3 * 3600),
        "one",
    )
    to_bot = await _adb("SELECT content, created_at FROM messages WHERE guild_id = ? AND to_bot = 1 AND created_at > ?", (guild_id, t - 12 * 3600), "all")
    roasts = sum(1 for row in to_bot if row["created_at"] > t - 3 * 3600 and NEGATIVE_RE.search(row["content"]))
    score = (feedback["score"] or 0) - 2 * roasts

    if roasts + (feedback["negative"] or 0) >= 3 and score < 0:
        mood = "grumpy: people have been roasting you or telling you to shut up. Let it color your tone a little (salty, defensive), don't announce it"
    elif score >= 6:
        mood = "in a great mood: your jokes have been landing. A bit more playful than usual"
    elif score >= 2:
        mood = "in a good mood"
    elif not to_bot:
        mood = "a little bored: nobody's talked to you in a while. You're glad someone's talking to you, even if you won't admit it"
    else:
        mood = ""

    if label == "late night":
        mood = f"{mood}, and tired" if mood else "tired"
    if mood:
        lines.append(f"Your mood: {mood}. Let it show subtly in your tone; don't narrate it.")

    return "\n".join(lines)


# ---------------------------------------------------------------- remembered moments ("remember when...")


async def _moments_due(guild_id: int, config: dict[str, Any]) -> bool:
    if not (config.get("recall") or {}).get("enabled") or _job_blocked(("moments", guild_id)):
        return False
    row = await _adb(
        "SELECT COUNT(*) AS n FROM messages WHERE guild_id = ? AND chunked = 0 AND created_at < ?",
        (guild_id, time.time() - MOMENT_SETTLE_SECONDS),
        "one",
    )
    return row["n"] >= MOMENTS_EVERY


async def _embed_safely(embed: Optional[Embed], texts: list[str], kind: str) -> Optional[list[list[float]]]:
    global _embed_failed_logged
    if not embed or not texts:
        return None
    try:
        vectors = []
        for start in range(0, len(texts), 32):
            vectors += await embed(texts[start : start + 32], kind)
        return vectors
    except Exception:
        if not _embed_failed_logged:
            logging.exception("Embedding model unavailable, falling back to keyword memory search")
            _embed_failed_logged = True
        return None


async def save_moments(guild_id: int, embed: Optional[Embed], force: bool = False) -> bool:
    """Group finished conversations into moments the bot can recall later."""

    async def job() -> bool:
        rows = await _adb(
            "SELECT id, channel_id, author_id, author_name, content, created_at FROM messages "
            "WHERE guild_id = ? AND chunked = 0 AND created_at < ? ORDER BY channel_id, id LIMIT 5000",
            (guild_id, time.time() - MOMENT_SETTLE_SECONDS),
            "all",
        )
        if not rows:
            return False

        groups, current = [], []
        for row in rows:
            if current and (
                row["channel_id"] != current[-1]["channel_id"]
                or row["created_at"] - current[-1]["created_at"] > MOMENT_GAP_SECONDS
                or len(current) >= MOMENT_WINDOW
            ):
                groups.append(current)
                current = []
            current.append(row)
        groups.append(current)

        moments = [g for g in groups if len(g) >= 3 or sum(len(r["content"]) for r in g) >= 120]
        texts = ["\n".join(f"{r['author_name']}: {r['content'][:300]}" for r in g) for g in moments]
        vectors = await _embed_safely(embed, texts, "document")

        for index, (group, text) in enumerate(zip(moments, texts)):
            blob = array("f", vectors[index]).tobytes() if vectors else None
            author_ids = "," + ",".join(str(a) for a in dict.fromkeys(r["author_id"] for r in group)) + ","
            await _adb(
                "INSERT INTO moments (guild_id, channel_id, start_id, end_id, author_ids, text, embedding, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (guild_id, group[0]["channel_id"], group[0]["id"], group[-1]["id"], author_ids, text, blob, group[-1]["created_at"]),
            )

        ids = [row["id"] for row in rows]
        for start in range(0, len(ids), 500):
            batch = ids[start : start + 500]
            await _adb(f"UPDATE messages SET chunked = 1 WHERE id IN ({','.join('?' * len(batch))})", tuple(batch))

        _moment_cache.pop(guild_id, None)
        logging.info(f"Saved {len(moments)} moments for guild {guild_id}")
        return True

    return await _run_job(("moments", guild_id), force, job)


async def _load_moments(guild_id: int) -> list[tuple]:
    if guild_id not in _moment_cache:
        rows = await _adb("SELECT id, created_at, text, embedding FROM moments WHERE guild_id = ?", (guild_id,), "all")
        loaded = []
        for row in rows:
            vector = norm = None
            if row["embedding"]:
                vector = array("f")
                vector.frombytes(row["embedding"])
                if numpy is not None:
                    vector = numpy.frombuffer(row["embedding"], dtype=numpy.float32)
                    norm = float(numpy.linalg.norm(vector)) or 1.0
                else:
                    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
            loaded.append((row["id"], row["created_at"], row["text"], vector, norm))
        _moment_cache[guild_id] = loaded
    return _moment_cache[guild_id]


def _cosine(query: Any, query_norm: float, vector: Any, norm: float) -> float:
    if numpy is not None and isinstance(vector, numpy.ndarray):
        return float(numpy.dot(query, vector)) / (query_norm * norm)
    return sum(a * b for a, b in zip(query, vector)) / (query_norm * norm)


def _ago(created_at: float) -> str:
    days = (time.time() - created_at) / 86400
    if days < 1:
        return "earlier today"
    if days < 2:
        return "yesterday"
    if days < 14:
        return f"{round(days)} days ago"
    if days < 60:
        return f"about {round(days / 7)} weeks ago"
    return f"about {round(days / 30)} months ago"


async def recall_moments(guild_id: int, query: str, recall_cfg: dict[str, Any], embed: Optional[Embed]) -> list[tuple[float, str]]:
    """Find old moments related to what's being said now. Uses embeddings, or keywords as a fallback."""
    if not query.strip():
        return []

    cutoff = time.time() - recall_cfg.get("min_age_hours", 6) * 3600
    candidates = [m for m in await _load_moments(guild_id) if m[1] < cutoff]
    if not candidates:
        return []

    scored = []
    embedded = [m for m in candidates if m[3] is not None]
    query_vectors = await _embed_safely(embed, [query], "query") if embedded else None

    if query_vectors:
        query_vector = numpy.asarray(query_vectors[0], dtype=numpy.float32) if numpy is not None else query_vectors[0]
        query_norm = math.sqrt(sum(v * v for v in query_vectors[0])) or 1.0
        threshold = recall_cfg.get("min_similarity", 0.6)
        for moment in embedded:
            if len(moment[3]) == len(query_vectors[0]) and (score := _cosine(query_vector, query_norm, moment[3], moment[4])) >= threshold:
                scored.append((score, moment))
    else:
        query_words = words(query)
        for moment in candidates:
            if (overlap := len(query_words & words(moment[2]))) >= recall_cfg.get("min_keyword_overlap", 3):
                scored.append((overlap, moment))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [(moment[1], moment[2]) for _, moment in scored[: recall_cfg.get("max_moments", 2)]]


# ---------------------------------------------------------------- using what was learned


def _media_label(row: Any) -> str:
    label = row["description"] or "no title"
    return f"{label} ({row['context']})" if row["context"] else label


async def _pick_media(guild_id: int, query: str, count: int, cooldown_hours: float) -> list[Any]:
    rows = await _adb(
        "SELECT * FROM media WHERE guild_id = ? AND last_used < ? AND (description != '' OR context != '') ORDER BY id DESC LIMIT 500",
        (guild_id, time.time() - cooldown_hours * 3600),
        "all",
    )
    rows = [row for row in rows if not row["file_path"] or os.path.isfile(os.path.join(DATA_DIR, row["file_path"]))]
    if not rows:
        return []

    query_words = words(query)
    scored = [(len(query_words & words(f"{row['description']} {row['context']}")), random.random(), row) for row in rows]
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)

    relevant = [row for score, _, row in scored if score > 0][: max(count - 2, 1)]
    relevant_ids = {row["id"] for row in relevant}
    others = [row for _, _, row in scored if row["id"] not in relevant_ids]
    random.shuffle(others)

    return relevant + others[: count - len(relevant)]


async def _people_section(guild_id: int, participant_ids: list[int], people_cfg: dict[str, Any]) -> str:
    entries = []
    for user_id in list(dict.fromkeys(participant_ids))[: people_cfg.get("max_in_context", 4)]:
        if (guild_id, user_id) in _optouts:
            continue
        notes, facts = await get_person(guild_id, user_id)
        if not notes and not facts:
            continue
        entry = f"<@{user_id}> ({await _latest_name(guild_id, user_id)}):"
        if notes:
            entry += f"\n{notes}"
        if facts:
            entry += "\nThings people told you about them: " + "; ".join(f["fact"] for f in facts[:5])
        entries.append(entry)

    return ("People in this conversation (your notes on them):\n" + "\n\n".join(entries)) if entries else ""


async def _roster_section(guild_id: int, size: int) -> str:
    rows = await _adb("SELECT author_id, author_name FROM messages WHERE guild_id = ? ORDER BY id DESC LIMIT 1000", (guild_id,), "all")
    names: dict[int, str] = {}
    counts: Counter = Counter()
    for row in rows:
        names.setdefault(row["author_id"], row["author_name"])
        counts[row["author_id"]] += 1

    regulars = [user_id for user_id, _ in counts.most_common(size) if (guild_id, user_id) not in _optouts]
    if not regulars:
        return ""
    return "Regulars here (use <@ID> to mention them): " + ", ".join(f"{names[user_id]} = <@{user_id}>" for user_id in regulars)


async def build_context(
    guild_id: int,
    query: str,
    config: dict[str, Any],
    participant_ids: Optional[list[int]] = None,
    *,
    channel_id: Optional[int] = None,
    message_id: Optional[int] = None,
    now: Optional[datetime] = None,
    usable_emoji_ids: Optional[set[int]] = None,
    embed: Optional[Embed] = None,
    extra_sections: Optional[list[tuple[int, str]]] = None,
) -> LearnedContext:
    """Build the extra system-prompt text, plus the media and reactions the bot may use this time."""
    learn_cfg = config.get("learning") or {}
    persona_cfg = config.get("persona") or {}
    people_cfg = config.get("people") or {}
    feedback_cfg = config.get("feedback") or {}
    media_cfg = config.get("media") or {}
    reactions_cfg = config.get("reactions") or {}
    mood_cfg = config.get("mood") or {}
    recall_cfg = config.get("recall") or {}
    anti_repeat_cfg = config.get("anti_repeat") or {}

    # (priority, text): when over budget, the lowest priority sections are dropped first
    sections: list[tuple[int, str]] = list(extra_sections or [])
    result = LearnedContext()

    # Phrases the bot keeps reusing get banned, so it can't loop on its own lines
    if anti_repeat_cfg.get("enabled", True):
        result.overused = await get_overused_phrases(guild_id)
        if result.overused:
            sections.append(
                (1000, "You've been repeating yourself. Do not use these phrases or jokes again, and don't bring up their topics "
                 "unless someone else does: " + "; ".join(f'"{phrase}"' for phrase in result.overused))
            )

    if persona_cfg.get("enabled"):
        if traits := await get_persona(guild_id, config):
            sections.append((100, f"Your personality (it keeps growing from hanging out with this crew):\n{traits}"))
        if guardrails := persona_cfg.get("guardrails"):
            sections.append((1000, "Always, no matter how your personality changes:\n" + "\n".join(f"- {rule}" for rule in guardrails)))

    if mood_cfg.get("enabled") and channel_id is not None:
        if mood := await mood_section(guild_id, channel_id, now or datetime.now().astimezone(), message_id):
            sections.append((88, mood))

    if learn_cfg.get("enabled"):
        if learn_cfg.get("speech_habits", True) and (habits := await get_speech_habits(guild_id)):
            sections.append((90, f"How people here actually type (measured from real messages):\n{habits}"))

        if notes := await get_lore(guild_id):
            sections.append((70, f"What you've picked up about this server (use it naturally, don't recite it):\n{notes}"))

        if (sample_size := learn_cfg.get("style_samples", 8)) > 0:
            recent = await _adb("SELECT author_name, content FROM messages WHERE guild_id = ? ORDER BY id DESC LIMIT 400", (guild_id,), "all")
            pool = [row for row in recent if 3 <= len(row["content"]) <= 200 and not contains_overused(row["content"], result.overused)]
            if sample := random.sample(pool, min(sample_size, len(pool))):
                sections.append(
                    (30, "Some real messages from people here, so you talk like you belong (don't quote them back):\n"
                     + "\n".join(f"{row['author_name']}: {row['content']}" for row in sample))
                )

    if recall_cfg.get("enabled") and (moments := await recall_moments(guild_id, query, recall_cfg, embed)):
        sections.append(
            (65, "Old moments from this server that relate to what's being said. You can call back to one like you remember it "
             "(\"wait didn't this already happen\"), but only if it really fits:\n"
             + "\n\n".join(f"[{_ago(created_at)}]\n{text}" for created_at, text in moments))
        )

    if people_cfg.get("enabled"):
        if people := await _people_section(guild_id, participant_ids or [], people_cfg):
            sections.append((80, people))
        if roster := await _roster_section(guild_id, people_cfg.get("roster_size", 15)):
            sections.append((40, roster))

    if feedback_cfg.get("enabled") and feedback_cfg.get("show_hits", 0) > 0 and (hits := await get_hits(guild_id, feedback_cfg["show_hits"])):
        sections.append((60, "Your lines that got big laughs here. That's the kind of humor that lands here; don't reuse the same bits or references:\n" + "\n".join(f"- {h}" for h in hits)))

    media_section = reaction_section = None
    if media_cfg.get("enabled") and random.random() < media_cfg.get("offer_chance", 0.4):
        if rows := await _pick_media(guild_id, query, media_cfg.get("candidates", 8), media_cfg.get("reuse_cooldown_hours", 12)):
            result.media = {row["id"]: row for row in rows}
            media_section = (
                "You can send ONE reaction gif, image or video that people here posted before, but only if it genuinely fits. "
                "Most replies shouldn't have one. To send it, end your reply with its tag, like [media:12]. Options:\n"
                + "\n".join(f"[media:{row['id']}] {row['kind']}: {_media_label(row)}" for row in rows)
            )
            sections.append((50, media_section))

    if reactions_cfg.get("enabled") and random.random() < reactions_cfg.get("offer_chance", 0.35):
        result.reactions = await get_server_emoji(guild_id, usable_emoji_ids, reactions_cfg.get("options", 8))
        reaction_section = (
            "You can also react to their message with an emoji, like people here do. To react, put [react:N] at the end of your reply, "
            "using a number from this list: " + ", ".join(f"{i} {emoji_label(e)}" for i, e in enumerate(result.reactions, 1))
        )
        sections.append((45, reaction_section))

    # Fit inside the budget, dropping low-priority sections first
    budget = learn_cfg.get("max_context_chars", 7000)
    kept, used = [], 0
    for priority, text in sorted(sections, key=lambda item: item[0], reverse=True):
        if priority < 1000 and used + len(text) > budget:
            if text is media_section:
                result.media = {}
            if text is reaction_section:
                result.reactions = []
            continue
        kept.append(text)
        used += len(text)

    result.text = "\n\n".join(kept)
    return result


def strip_media_tags(text: str) -> str:
    """Remove [media:N] and [react:N] tags before a reply is shown."""
    text = REACT_TAG_RE.sub("", MEDIA_TAG_RE.sub("", text))
    return re.sub(r"[ \t]+\n", "\n", text).strip()


def chosen_media(text: str, offered: dict[int, Any]) -> Optional[Any]:
    for match in MEDIA_TAG_RE.finditer(text):
        if (media_id := int(match.group(1))) in offered:
            return offered[media_id]
    return None


async def send_media(reply_target: Any, row: Any) -> Optional[Any]:
    """Post a remembered media item as a reply. Returns the sent message."""
    path = os.path.join(DATA_DIR, row["file_path"]) if row["file_path"] else None

    if path and os.path.isfile(path):
        ext = os.path.splitext(path)[1]
        sent = await reply_target.reply(file=discord.File(path, filename=f"{row['kind']}{ext}"), silent=True, mention_author=False)
    elif row["url"] and not is_discord_hosted(row["url"]):
        sent = await reply_target.reply(content=row["url"], silent=True, mention_author=False)
    else:
        return None

    await _adb("UPDATE media SET last_used = ? WHERE id = ?", (time.time(), row["id"]))
    return sent


async def backfill_channel(channel: Any, limit: int, config: dict[str, Any], http_client: httpx.AsyncClient, bot_user_id: Optional[int] = None) -> tuple[int, int]:
    """Learn from a channel's past messages, oldest first. Returns (messages, media)."""
    history = [msg async for msg in channel.history(limit=limit)]
    history.reverse()

    texts = media = 0
    for msg in history:
        if msg.author.bot:
            continue
        result = await learn_from_message(msg, config, http_client, bot_user_id)
        texts += result.stored_text
        media += result.media_saved

    _habits_cache.pop(getattr(channel.guild, "id", None), None)
    return texts, media
