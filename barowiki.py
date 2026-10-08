"""Barotrauma wiki lookups for llmcord.

When someone asks the bot a real game question ("how much power does a
coilgun need?"), this finds the matching page on the official Barotrauma wiki
and hands its text to the model, so the answer comes from the wiki instead of
being made up.
"""

import asyncio
from html.parser import HTMLParser
import json
import logging
import os
import re
import time
from typing import Optional

import httpx

import learning

API_URL = "https://barotraumagame.com/baro-wiki/api.php"
HEADERS = {"User-Agent": "llmcord-discord-bot (Barotrauma wiki lookups for a private friend server)"}
TITLES_PATH = os.path.join(learning.DATA_DIR, "wiki_titles.json")
TITLES_MAX_AGE = 7 * 86400
PAGE_CACHE_SECONDS = 86400
MAX_PAGE_CHARS = 2500

QUESTION_RE = re.compile(r"\?|^\s*(how|what|where|which|why|when|who|whats|does|do|can|is|are|should|will|would)\b", re.IGNORECASE)
GAME_HINT_RE = re.compile(
    r"\b(barotrauma|baro|submarine|europa|reactor|husks?|crawlers?|moloch|mudraptors?|coilgun|railgun|ballast|fabricator|"
    r"deconstructor|talents?|outposts?|wrecks?|beacon|thalamus|endworm|charybdis|watcher|hammerhead|fuel rods?|diving suit|"
    r"welding tool|plasma cutter|nav terminal|sonar|junction box|supercapacitor|oxygen generator|electrical|medical)\b",
    re.IGNORECASE,
)

# Page titles that match normal chat too easily
SKIP_TITLES = {
    "main page", "help", "guide", "guides", "items", "item", "characters", "character", "the", "game", "games", "news",
    "update", "updates", "changelog", "changelogs", "mods", "mod", "wiki", "home", "start", "test", "it", "is", "do", "go",
}
STOPWORDS = learning.STOPWORDS | {"much", "many", "need", "needs", "use", "used", "using", "does", "best", "good", "work", "works"}

_titles: dict[str, str] = {}
_titles_loading: Optional[asyncio.Task] = None
_page_cache: dict[str, tuple[float, str, str]] = {}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower())).strip()


async def _api(http_client: httpx.AsyncClient, **params) -> dict:
    resp = await http_client.get(API_URL, params=dict(format="json", **params), headers=HEADERS, timeout=10)
    resp.raise_for_status()
    return resp.json()


async def load_titles(http_client: httpx.AsyncClient, force: bool = False) -> int:
    """Load every wiki page title (cached on disk for a week)."""
    global _titles

    if not force and os.path.isfile(TITLES_PATH) and time.time() - os.path.getmtime(TITLES_PATH) < TITLES_MAX_AGE:
        with open(TITLES_PATH, encoding="utf-8") as file:
            titles = json.load(file)
    else:
        titles, params = [], dict(action="query", list="allpages", apnamespace=0, aplimit=500)
        for _ in range(30):
            data = await _api(http_client, **params)
            titles += [page["title"] for page in data.get("query", {}).get("allpages", [])]
            if not (cont := data.get("continue")):
                break
            params.update(cont)

        os.makedirs(os.path.dirname(TITLES_PATH), exist_ok=True)
        with open(TITLES_PATH, "w", encoding="utf-8") as file:
            json.dump(titles, file)

    _titles = {key: title for title in titles if len(key := normalize(title)) >= 3 and key not in SKIP_TITLES}
    logging.info(f"Loaded {len(_titles)} Barotrauma wiki titles")
    return len(_titles)


async def ensure_titles(http_client: httpx.AsyncClient) -> None:
    global _titles_loading
    if _titles:
        return
    if _titles_loading is None or _titles_loading.done():
        _titles_loading = asyncio.create_task(load_titles(http_client))
    try:
        await asyncio.wait_for(asyncio.shield(_titles_loading), timeout=15)
    except Exception:
        logging.exception("Couldn't load Barotrauma wiki titles")


def find_titles(text: str) -> list[str]:
    """Wiki pages named in the text, longest matches first (also tries singular forms)."""
    tokens = normalize(text).split()
    found = []
    for size in (4, 3, 2, 1):
        for start in range(len(tokens) - size + 1):
            gram = " ".join(tokens[start : start + size])
            for candidate in (gram, gram[:-1] if gram.endswith("s") else None, gram[:-2] if gram.endswith("es") else None):
                if candidate and candidate in _titles and _titles[candidate] not in found:
                    found.append(_titles[candidate])
    return found


def looks_like_game_question(text: str) -> bool:
    return bool(QUESTION_RE.search(text or "")) and bool(find_titles(text) or GAME_HINT_RE.search(text or ""))


class _TextExtractor(HTMLParser):
    """Pulls readable text out of a rendered wiki page, skipping navboxes, edit links and references."""

    SKIP_CLASSES = ("navbox", "mw-editsection", "reference", "toc", "noprint", "mw-references-wrap")
    VOID = {"br", "img", "hr", "input", "meta", "link", "source", "wbr", "area", "col"}
    BLOCKS = {"p", "li", "tr", "h1", "h2", "h3", "h4", "div", "table", "ul", "ol", "dd", "dt"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip_stack: list[bool] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in self.VOID:
            if tag == "br" and not any(self.skip_stack):
                self.parts.append("\n")
            return
        classes = dict(attrs).get("class") or ""
        skip = tag in ("style", "script", "sup") or any(c in classes for c in self.SKIP_CLASSES)
        self.skip_stack.append(skip)
        if not any(self.skip_stack):
            if tag in self.BLOCKS:
                self.parts.append("\n")
            elif tag in ("td", "th"):
                self.parts.append(" | ")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.VOID:
            return
        if self.skip_stack:
            self.skip_stack.pop()
        if tag in self.BLOCKS and not any(self.skip_stack):
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not any(self.skip_stack):
            self.parts.append(data)

    def text(self) -> str:
        text = "".join(self.parts)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" ?\| ?(\| ?)+", " | ", text)
        lines = [line.strip(" |") for line in text.splitlines()]
        return "\n".join(line for line in lines if line)


def html_to_text(html: str) -> str:
    extractor = _TextExtractor()
    extractor.feed(html)
    return extractor.text()


async def search(http_client: httpx.AsyncClient, text: str) -> Optional[str]:
    keywords = [w for w in normalize(text).split() if w not in STOPWORDS and len(w) > 2]
    if not keywords:
        return None
    data = await _api(http_client, action="query", list="search", srsearch=" ".join(keywords[:6]), srlimit=1)
    results = data.get("query", {}).get("search", [])
    return results[0]["title"] if results else None


async def page_text(http_client: httpx.AsyncClient, title: str) -> Optional[tuple[str, str]]:
    """(resolved title, readable text) for a wiki page, cached for a day."""
    if (cached := _page_cache.get(title)) and time.time() - cached[0] < PAGE_CACHE_SECONDS:
        return cached[1], cached[2]

    data = await _api(http_client, action="parse", page=title, prop="text", formatversion=2, redirects=1)
    if "parse" not in data:
        return None

    resolved = data["parse"].get("title", title)
    text = html_to_text(data["parse"].get("text", ""))[:MAX_PAGE_CHARS]
    _page_cache[title] = (time.time(), resolved, text)
    return resolved, text


async def lookup(http_client: httpx.AsyncClient, question: str) -> Optional[tuple[str, str]]:
    """If the text is a Barotrauma question, return (page title, page text) from the wiki."""
    if not QUESTION_RE.search(question or ""):
        return None

    await ensure_titles(http_client)
    titles = find_titles(question)
    if not titles and GAME_HINT_RE.search(question):
        if found := await search(http_client, question):
            titles = [found]
    if not titles:
        return None

    return await page_text(http_client, titles[0])


def prompt_section(title: str, text: str) -> str:
    return (
        f'Barotrauma wiki info on "{title}" (from the official wiki). Use it to answer the game question accurately, '
        "briefly and in your own voice. If it doesn't cover what they asked, say you're not sure instead of making things up:\n"
        + text
    )
