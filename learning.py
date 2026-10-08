"""Server learning for llmcord.

Remembers what people in a server say and the media they share, so the bot can
pick up the server's slang, running jokes and vibe, and send reaction gifs,
images and videos that were posted before.

Everything is stored locally in data/memory.db (plus data/media/ for saved
files). Nothing leaves the computer the bot runs on.
"""

import asyncio
import logging
import os
import random
import re
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional
from urllib.parse import urlparse

import discord
import httpx

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "memory.db")
MEDIA_DIR = os.path.join(DATA_DIR, "media")

URL_RE = re.compile(r"https?://[^\s<>]+")
MEDIA_TAG_RE = re.compile(r"\[\s*media\s*[:#]?\s*(\d+)\s*\]", re.IGNORECASE)
WORD_RE = re.compile(r"[a-z0-9']+")

MAX_STORED_TEXT = 1000
LORE_CHAR_BUDGET = 8000  # keeps the notes prompt inside a small local model's context
LORE_RETRY_SECONDS = 600
MAX_MEDIA_PER_MESSAGE = 5

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
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be", "to", "of", "in", "on", "at", "for", "it",
    "its", "it's", "i", "im", "i'm", "you", "your", "u", "me", "my", "we", "he", "she", "they", "them", "this", "that",
    "with", "so", "just", "like", "do", "dont", "don't", "not", "no", "yes", "yeah", "lol", "lmao", "what", "when",
    "who", "how", "why", "can", "have", "has", "had", "if", "then", "there", "here", "about", "up", "out", "get",
    "got", "all", "some", "one", "said", "after", "captioned", "someone", "gif", "image", "video",
}


@dataclass
class LearnResult:
    stored_text: bool = False
    media_saved: int = 0
    lore_due: bool = False


_media_bytes: Optional[int] = None
_lore_running: set[int] = set()
_lore_last_attempt: dict[int, float] = {}


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
                created_at REAL
            );
            CREATE INDEX IF NOT EXISTS idx_messages_guild ON messages (guild_id, id);
            CREATE INDEX IF NOT EXISTS idx_messages_channel ON messages (channel_id, id);

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

            CREATE TABLE IF NOT EXISTS lore (
                guild_id INTEGER PRIMARY KEY,
                notes TEXT,
                updated_at REAL
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------- helpers


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
    if drop_last and len(parts) > 1:
        parts = parts[:-1]
    elif drop_last:
        parts = []
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


def can_learn_from(msg: Any, config: dict[str, Any]) -> bool:
    if not msg.guild or msg.author.bot:
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


# ---------------------------------------------------------------- learning


async def learn_from_message(msg: Any, config: dict[str, Any], http_client: httpx.AsyncClient) -> LearnResult:
    """Store a server message and any media in it. Safe to call on every message."""
    result = LearnResult()
    learn_cfg = config.get("learning") or {}
    media_cfg = config.get("media") or {}

    if not (learn_cfg.get("enabled") or media_cfg.get("enabled")) or not can_learn_from(msg, config):
        return result

    text = clean_message_text(msg)
    guild_id, channel_id = msg.guild.id, msg.channel.id

    if learn_cfg.get("enabled") and text:
        await _adb(
            "INSERT OR REPLACE INTO messages (id, guild_id, channel_id, author_id, author_name, content, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (msg.id, guild_id, channel_id, msg.author.id, msg.author.display_name, text, msg.created_at.timestamp()),
        )
        result.stored_text = True

    if media_cfg.get("enabled"):
        result.media_saved = await _save_media(msg, text, media_cfg, http_client)

    if learn_cfg.get("enabled") and result.stored_text:
        result.lore_due = await _lore_due(guild_id, learn_cfg)

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
    """Remove deleted messages and their saved media, so deleted stuff never gets reposted."""
    for message_id in message_ids:
        rows = await _adb("SELECT file_path FROM media WHERE message_id = ?", (message_id,), "all") or []
        for row in rows:
            if row["file_path"]:
                try:
                    os.remove(os.path.join(DATA_DIR, row["file_path"]))
                except OSError:
                    pass
        await _adb("DELETE FROM media WHERE message_id = ?", (message_id,))
        await _adb("DELETE FROM messages WHERE id = ?", (message_id,))


# ---------------------------------------------------------------- lore notes


async def get_lore(guild_id: int) -> str:
    row = await _adb("SELECT notes FROM lore WHERE guild_id = ?", (guild_id,), "one")
    return row["notes"] if row else ""


async def _lore_due(guild_id: int, learn_cfg: dict[str, Any]) -> bool:
    if guild_id in _lore_running or time.time() - _lore_last_attempt.get(guild_id, 0) < LORE_RETRY_SECONDS:
        return False

    row = await _adb("SELECT updated_at FROM lore WHERE guild_id = ?", (guild_id,), "one")
    since = row["updated_at"] if row else 0
    count = await _adb("SELECT COUNT(*) AS n FROM messages WHERE guild_id = ? AND created_at > ?", (guild_id, since), "one")
    return count["n"] >= learn_cfg.get("lore_every_messages", 200)


async def rebuild_lore(guild_id: int, config: dict[str, Any], complete: Callable[[list[dict[str, str]]], Awaitable[str]]) -> bool:
    """Ask the model to update its notes about the server from recent messages."""
    if guild_id in _lore_running:
        return False

    _lore_running.add(guild_id)
    _lore_last_attempt[guild_id] = time.time()

    try:
        learn_cfg = config.get("learning") or {}
        rows = await _adb(
            "SELECT author_name, content FROM messages WHERE guild_id = ? ORDER BY id DESC LIMIT ?",
            (guild_id, learn_cfg.get("lore_source_messages", 120)),
            "all",
        )
        if len(rows) < 15:
            return False

        lines, used = [], 0
        for row in rows:
            line = f"{row['author_name']}: {row['content'][:200]}"
            if used + len(line) > LORE_CHAR_BUDGET:
                break
            lines.append(line)
            used += len(line)
        lines.reverse()

        old_notes = await get_lore(guild_id) or "(none yet)"
        prompt = [
            dict(role="system", content="You keep short, accurate notes about a Discord community for a chatbot that hangs out there."),
            dict(
                role="user",
                content=(
                    "Recent messages from the server:\n"
                    + "\n".join(lines)
                    + f"\n\nYour current notes:\n{old_notes}\n\n"
                    "Rewrite the notes. Keep what is still true and add what is new. Cover: slang and phrases people use, "
                    "running jokes and memes, what people talk about a lot, and a few words on how each regular talks. "
                    "Only include things actually seen in the messages. Plain bullet points, under 200 words, no intro."
                ),
            ),
        ]

        notes = (await complete(prompt) or "").strip()[:2500]
        if not notes:
            return False

        await _adb("INSERT OR REPLACE INTO lore (guild_id, notes, updated_at) VALUES (?, ?, ?)", (guild_id, notes, time.time()))
        logging.info(f"Updated server notes for guild {guild_id}")
        return True

    except Exception:
        logging.exception("Couldn't rebuild server notes")
        return False

    finally:
        _lore_running.discard(guild_id)


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


async def build_context(guild_id: int, query: str, config: dict[str, Any]) -> tuple[str, dict[int, Any]]:
    """Return extra system-prompt text, plus the media the bot is allowed to send this time."""
    parts = []
    offered = {}
    learn_cfg = config.get("learning") or {}
    media_cfg = config.get("media") or {}

    if learn_cfg.get("enabled"):
        if notes := await get_lore(guild_id):
            parts.append(f"What you've picked up from hanging around this server (use it naturally, don't recite it):\n{notes}")

        if (sample_size := learn_cfg.get("style_samples", 12)) > 0:
            recent = await _adb("SELECT author_name, content FROM messages WHERE guild_id = ? ORDER BY id DESC LIMIT 400", (guild_id,), "all")
            pool = [row for row in recent if 3 <= len(row["content"]) <= 200]
            if sample := random.sample(pool, min(sample_size, len(pool))):
                parts.append(
                    "Some real messages from people here, so you talk like you belong (don't quote them back):\n"
                    + "\n".join(f"{row['author_name']}: {row['content']}" for row in sample)
                )

    if media_cfg.get("enabled") and random.random() < media_cfg.get("offer_chance", 0.4):
        if rows := await _pick_media(guild_id, query, media_cfg.get("candidates", 8), media_cfg.get("reuse_cooldown_hours", 12)):
            offered = {row["id"]: row for row in rows}
            parts.append(
                "You can send ONE reaction gif, image or video that people here posted before, but only if it genuinely fits. "
                "Most replies shouldn't have one. To send it, end your reply with its tag, like [media:12]. Options:\n"
                + "\n".join(f"[media:{row['id']}] {row['kind']}: {_media_label(row)}" for row in rows)
            )

    return "\n\n".join(parts), offered


def strip_media_tags(text: str) -> str:
    text = MEDIA_TAG_RE.sub("", text)
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


async def backfill_channel(channel: Any, limit: int, config: dict[str, Any], http_client: httpx.AsyncClient) -> tuple[int, int]:
    """Learn from a channel's past messages, oldest first. Returns (messages, media)."""
    history = [msg async for msg in channel.history(limit=limit)]
    history.reverse()

    texts = media = 0
    for msg in history:
        result = await learn_from_message(msg, config, http_client)
        texts += result.stored_text
        media += result.media_saved

    return texts, media
