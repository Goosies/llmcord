import asyncio
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime
import io
import logging
import os
import random
import re
import time
from typing import Any, Literal, Optional
from urllib.parse import urlparse

import discord
from discord import app_commands
from discord.app_commands import Choice
from discord.ext import commands
from discord.ui import LayoutView, TextDisplay
from dotenv import load_dotenv
import httpx
from openai import AsyncOpenAI
from PIL import Image
import yaml

import learning

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

VISION_MODEL_TAGS = ("chat-latest", "claude", "gemini", "gemma", "gpt-4", "gpt-5", "gpt-latest", "grok-4", "llama", "vision", "vl")

EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()

STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1

MAX_MESSAGE_NODES = 500

GIF_MAX_FRAMES = 4  # how many frames to pull from an animated GIF
GIF_MAX_SIZE = 768  # longest side in pixels for each extracted frame

# Image links pasted in chat (not uploaded) are only fetched from these hosts
IMAGE_LINK_HOSTS = {"cdn.discordapp.com", "media.discordapp.net", "media.tenor.com", "i.imgur.com"}
IMAGE_LINK_HOST_SUFFIXES = (".discordapp.net", ".giphy.com")
IMAGE_LINK_EXTENSIONS = (".gif", ".png", ".jpg", ".jpeg", ".webp")
MAX_LINK_IMAGES = 3
MAX_LINK_IMAGE_BYTES = 10 * 1024 * 1024
URL_PATTERN = re.compile(r"https?://[^\s<>]+")


def resolve_env(node: Any) -> Any:
    if isinstance(node, dict):
        return {key.removesuffix("_env"): os.environ.get(value) if key.endswith("_env") else resolve_env(value) for key, value in node.items()}
    return node


def get_config(filename: str = "config.yaml") -> dict[str, Any]:
    with open(filename, encoding="utf-8") as file:
        return resolve_env(yaml.safe_load(file))


def gif_to_png_frames(data: bytes, max_frames: int = GIF_MAX_FRAMES) -> list[bytes]:
    """Return up to max_frames PNG frames spread evenly across a GIF."""
    frames = []

    with Image.open(io.BytesIO(data)) as gif:
        total = getattr(gif, "n_frames", 1)
        count = min(max_frames, total)
        indexes = sorted({round(i * (total - 1) / max(count - 1, 1)) for i in range(count)})

        for index in indexes:
            gif.seek(index)
            frame = gif.convert("RGB")
            frame.thumbnail((GIF_MAX_SIZE, GIF_MAX_SIZE))

            buffer = io.BytesIO()
            frame.save(buffer, format="PNG")
            frames.append(buffer.getvalue())

    return frames


def is_allowed_image_url(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    return parsed.scheme in ("http", "https") and (host in IMAGE_LINK_HOSTS or host.endswith(IMAGE_LINK_HOST_SUFFIXES))


def extract_image_urls(msg: Any) -> list[str]:
    """Find image/GIF links in a message's text, falling back to its embeds."""
    urls = []

    for url in URL_PATTERN.findall(msg.content or ""):
        url = url.rstrip(").,>")
        if is_allowed_image_url(url) and urlparse(url).path.lower().endswith(IMAGE_LINK_EXTENSIONS):
            urls.append(url)

    if not urls:
        for embed in msg.embeds:
            for media in (embed.image, embed.thumbnail):
                media_url = getattr(media, "url", None)
                if media_url and is_allowed_image_url(media_url):
                    urls.append(media_url)

    return list(dict.fromkeys(urls))[:MAX_LINK_IMAGES]


async def make_image_parts(content_type: str, data: bytes) -> list[dict[str, Any]]:
    """Turn image bytes into model-ready parts. GIFs become a few PNG frames."""
    if content_type == "image/gif":
        try:
            gif_frames = await asyncio.to_thread(gif_to_png_frames, data)
            return [dict(type="image_url", image_url=dict(url=f"data:image/png;base64,{b64encode(frame).decode('utf-8')}")) for frame in gif_frames]
        except Exception:
            logging.exception("Couldn't convert GIF to frames, sending it as-is")

    return [dict(type="image_url", image_url=dict(url=f"data:{content_type};base64,{b64encode(data).decode('utf-8')}"))]


config = get_config()
curr_model = next(iter(config["models"]))

msg_nodes = {}
last_task_time = 0
last_random_reply = 0.0

intents = discord.Intents.default()
intents.message_content = True
activity = discord.CustomActivity(name=(config.get("status_message") or "github.com/jakobdylanc/llmcord")[:128])
discord_bot = commands.Bot(intents=intents, activity=activity, command_prefix=None)

httpx_client = httpx.AsyncClient()

learning.init_db()
background_tasks = set()


def run_in_background(coro) -> None:
    task = asyncio.create_task(coro)
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)


async def simple_completion(prompt_messages: list[dict[str, str]]) -> str:
    """One-off, non-streamed request to the current model (used for building server notes)."""
    cfg = await asyncio.to_thread(get_config)
    provider, model = curr_model.removesuffix(":vision").split("/", 1)
    provider_config = cfg["providers"][provider]

    client = AsyncOpenAI(base_url=provider_config["base_url"], api_key=provider_config.get("api_key", "sk-no-key-required"))
    extra_body = (provider_config.get("extra_body") or {}) | (cfg["models"].get(curr_model) or {}) | {"temperature": 0.4}

    response = await client.chat.completions.create(
        model=model,
        messages=prompt_messages,
        stream=False,
        extra_headers=provider_config.get("extra_headers"),
        extra_query=provider_config.get("extra_query"),
        extra_body=extra_body,
    )
    return response.choices[0].message.content or ""


async def learn_in_background(msg: discord.Message, cfg: dict[str, Any]) -> None:
    try:
        result = await learning.learn_from_message(msg, cfg, httpx_client)
        if result.lore_due:
            await learning.rebuild_lore(msg.guild.id, cfg, simple_completion)
    except Exception:
        logging.exception("Error while learning from message")


@dataclass
class MsgNode:
    role: Literal["user", "assistant"] = "assistant"

    text: Optional[str] = None
    images: list[dict[str, Any]] = field(default_factory=list)

    has_bad_attachments: bool = False
    fetch_parent_failed: bool = False

    parent_msg: Optional[discord.Message] = None

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@discord_bot.tree.command(name="model", description="View or switch the current model")
async def model_command(interaction: discord.Interaction, model: str) -> None:
    global curr_model

    if model == curr_model:
        output = f"Current model: `{curr_model}`"
    else:
        if user_is_admin := interaction.user.id in config["permissions"]["users"]["admin_ids"]:
            curr_model = model
            output = f"Model switched to: `{model}`"
            logging.info(output)
        else:
            output = "You don't have permission to change the model."

    await interaction.response.send_message(output, ephemeral=(interaction.channel.type == discord.ChannelType.private))


@model_command.autocomplete("model")
async def model_autocomplete(interaction: discord.Interaction, curr_str: str) -> list[Choice[str]]:
    global config

    if curr_str == "":
        config = await asyncio.to_thread(get_config)

    choices = [Choice(name=f"◉ {curr_model} (current)", value=curr_model)] if curr_str.lower() in curr_model.lower() else []
    choices += [Choice(name=f"○ {model}", value=model) for model in config["models"] if model != curr_model and curr_str.lower() in model.lower()]

    return choices[:25]


@discord_bot.tree.command(name="lore", description="See what the bot has picked up about this server")
async def lore_command(interaction: discord.Interaction) -> None:
    if not interaction.guild:
        await interaction.response.send_message("This only works in a server.", ephemeral=True)
        return

    notes = await learning.get_lore(interaction.guild.id)
    await interaction.response.send_message((notes or "Nothing yet. I need more chat to learn from.")[:1900], ephemeral=True)


@discord_bot.tree.command(name="backfill", description="Admin: learn from this channel's past messages")
@app_commands.describe(limit="How many past messages to read")
async def backfill_command(interaction: discord.Interaction, limit: app_commands.Range[int, 1, 5000] = 500) -> None:
    cfg = await asyncio.to_thread(get_config)

    if interaction.user.id not in cfg["permissions"]["users"]["admin_ids"]:
        await interaction.response.send_message("You don't have permission to do that.", ephemeral=True)
        return
    if not interaction.guild:
        await interaction.response.send_message("This only works in a server.", ephemeral=True)
        return

    await interaction.response.defer(ephemeral=True, thinking=True)

    try:
        texts, media = await learning.backfill_channel(interaction.channel, limit, cfg, httpx_client)
    except discord.Forbidden:
        await interaction.followup.send("I can't read this channel's history.", ephemeral=True)
        return

    if (cfg.get("learning") or {}).get("enabled"):
        run_in_background(learning.rebuild_lore(interaction.guild.id, cfg, simple_completion))

    await interaction.followup.send(f"Learned from {texts} messages and saved {media} gifs/images/videos. Updating my notes in the background.", ephemeral=True)


@discord_bot.event
async def on_ready() -> None:
    if client_id := config.get("client_id"):
        logging.info(f"\n\nBOT INVITE URL:\nhttps://discord.com/oauth2/authorize?client_id={client_id}&permissions=412317240320&scope=bot\n")

    await discord_bot.tree.sync()


@discord_bot.event
async def on_message_edit(before: discord.Message, after: discord.Message) -> None:
    # Discord adds link previews (like YouTube titles) a moment after a message is sent
    if after.guild and not after.author.bot and len(after.embeds) > len(before.embeds):
        run_in_background(learning.update_media_from_embeds(after))


@discord_bot.event
async def on_raw_message_delete(payload: discord.RawMessageDeleteEvent) -> None:
    run_in_background(learning.forget_messages([payload.message_id]))


@discord_bot.event
async def on_raw_bulk_message_delete(payload: discord.RawBulkMessageDeleteEvent) -> None:
    run_in_background(learning.forget_messages(list(payload.message_ids)))


@discord_bot.event
async def on_message(new_msg: discord.Message) -> None:
    global last_task_time, last_random_reply

    is_dm = new_msg.channel.type == discord.ChannelType.private

    if new_msg.author.bot:
        return

    config = await asyncio.to_thread(get_config)

    # Learn from every server message, not just ones aimed at the bot
    if not is_dm:
        run_in_background(learn_in_background(new_msg, config))

    # Random chime-ins: occasionally reply to messages that don't @ the bot
    random_cfg = config.get("random_replies") or {}
    random_trigger = False

    if not is_dm and discord_bot.user not in new_msg.mentions and random_cfg.get("enabled"):
        random_channel_ids = random_cfg.get("channel_ids") or []
        in_random_channel = (
            not random_channel_ids
            or new_msg.channel.id in random_channel_ids
            or getattr(new_msg.channel, "parent_id", None) in random_channel_ids
        )
        now_ts = time.time()

        if (
            in_random_channel
            and new_msg.content
            and now_ts - last_random_reply >= random_cfg.get("cooldown_seconds", 600)
            and random.random() < random_cfg.get("chance", 0.03)
        ):
            random_trigger = True
            last_random_reply = now_ts

    if not is_dm and discord_bot.user not in new_msg.mentions and not random_trigger:
        return

    role_ids = set(role.id for role in getattr(new_msg.author, "roles", ()))
    channel_ids = set(filter(None, (new_msg.channel.id, getattr(new_msg.channel, "parent_id", None), getattr(new_msg.channel, "category_id", None))))

    allow_dms = config.get("allow_dms", True)

    permissions = config["permissions"]

    user_is_admin = new_msg.author.id in permissions["users"]["admin_ids"]

    (allowed_user_ids, blocked_user_ids), (allowed_role_ids, blocked_role_ids), (allowed_channel_ids, blocked_channel_ids) = (
        (perm["allowed_ids"], perm["blocked_ids"]) for perm in (permissions["users"], permissions["roles"], permissions["channels"])
    )

    allow_all_users = not allowed_user_ids if is_dm else not allowed_user_ids and not allowed_role_ids
    is_good_user = user_is_admin or allow_all_users or new_msg.author.id in allowed_user_ids or any(id in allowed_role_ids for id in role_ids)
    is_bad_user = not is_good_user or new_msg.author.id in blocked_user_ids or any(id in blocked_role_ids for id in role_ids)

    allow_all_channels = not allowed_channel_ids
    is_good_channel = user_is_admin or allow_dms if is_dm else allow_all_channels or any(id in allowed_channel_ids for id in channel_ids)
    is_bad_channel = not is_good_channel or any(id in blocked_channel_ids for id in channel_ids)

    if is_bad_user or is_bad_channel:
        return

    provider_slash_model = curr_model
    provider, model = provider_slash_model.removesuffix(":vision").split("/", 1)

    provider_config = config["providers"][provider]

    base_url = provider_config["base_url"]
    api_key = provider_config.get("api_key", "sk-no-key-required")
    openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    model_parameters = config["models"].get(provider_slash_model, None)

    extra_headers = provider_config.get("extra_headers")
    extra_query = provider_config.get("extra_query")
    extra_body = (provider_config.get("extra_body") or {}) | (model_parameters or {}) or None

    accept_images = any(x in provider_slash_model.lower() for x in VISION_MODEL_TAGS)

    max_text = config.get("max_text", 100000)
    max_images = config.get("max_images", 5) if accept_images else 0
    max_messages = config.get("max_messages", 25)

    # Build message chain and set user warnings
    messages = []
    user_warnings = set()
    curr_msg = new_msg

    while curr_msg != None and len(messages) < max_messages:
        curr_node = msg_nodes.setdefault(curr_msg.id, MsgNode())

        async with curr_node.lock:
            if curr_node.text == None:
                cleaned_content = curr_msg.content.removeprefix(discord_bot.user.mention).lstrip()

                good_attachments = [att for att in curr_msg.attachments if att.content_type and any(att.content_type.startswith(x) for x in ("text", "image"))]

                attachment_responses = await asyncio.gather(*[httpx_client.get(att.url) for att in good_attachments])

                curr_node.role = "assistant" if curr_msg.author == discord_bot.user else "user"

                curr_node.text = "\n".join(
                    ([cleaned_content] if cleaned_content else [])
                    + ["\n".join(filter(None, (embed.title, embed.description, embed.footer.text))) for embed in curr_msg.embeds]
                    + [component.content for component in curr_msg.components if component.type == discord.ComponentType.text_display]
                    + [resp.text for att, resp in zip(good_attachments, attachment_responses) if att.content_type.startswith("text")]
                )

                curr_node.images = []

                # Uploaded images (GIFs are turned into a few still frames since Ollama can't read them)
                for att, resp in zip(good_attachments, attachment_responses):
                    if att.content_type.startswith("image"):
                        curr_node.images += await make_image_parts(att.content_type, resp.content)

                # Image/GIF links pasted in the message
                for link_url in extract_image_urls(curr_msg):
                    try:
                        link_resp = await httpx_client.get(link_url, timeout=15)
                    except Exception:
                        logging.exception(f"Couldn't fetch image link: {link_url}")
                        continue

                    link_type = link_resp.headers.get("content-type", "").split(";")[0].strip().lower()
                    if link_resp.status_code == 200 and link_type.startswith("image/") and len(link_resp.content) <= MAX_LINK_IMAGE_BYTES:
                        curr_node.images += await make_image_parts(link_type, link_resp.content)

                if curr_node.role == "user" and (curr_node.text or curr_node.images):
                    curr_node.text = f"<@{curr_msg.author.id}>: {curr_node.text}"

                curr_node.has_bad_attachments = len(curr_msg.attachments) > len(good_attachments)

                try:
                    if (
                        curr_msg.reference == None
                        and discord_bot.user.mention not in curr_msg.content
                        and (prev_msg_in_channel := ([m async for m in curr_msg.channel.history(before=curr_msg, limit=1)] or [None])[0])
                        and prev_msg_in_channel.type in (discord.MessageType.default, discord.MessageType.reply)
                        and prev_msg_in_channel.author == (discord_bot.user if curr_msg.channel.type == discord.ChannelType.private else curr_msg.author)
                    ):
                        curr_node.parent_msg = prev_msg_in_channel
                    else:
                        is_public_thread = curr_msg.channel.type == discord.ChannelType.public_thread
                        parent_is_thread_start = is_public_thread and curr_msg.reference == None and curr_msg.channel.parent.type == discord.ChannelType.text

                        if parent_msg_id := curr_msg.channel.id if parent_is_thread_start else getattr(curr_msg.reference, "message_id", None):
                            if parent_is_thread_start:
                                curr_node.parent_msg = curr_msg.channel.starter_message or await curr_msg.channel.parent.fetch_message(parent_msg_id)
                            else:
                                curr_node.parent_msg = curr_msg.reference.cached_message or await curr_msg.channel.fetch_message(parent_msg_id)

                except (discord.NotFound, discord.HTTPException):
                    logging.exception("Error fetching next message in the chain")
                    curr_node.fetch_parent_failed = True

            if curr_node.images[:max_images]:
                content = [dict(type="text", text=curr_node.text[:max_text])] + curr_node.images[:max_images]
            else:
                content = curr_node.text[:max_text]

            if content != "":
                messages.append(dict(content=content, role=curr_node.role))

            if len(curr_node.text) > max_text:
                user_warnings.add(f"⚠️ Max {max_text:,} characters per message")
            if len(curr_node.images) > max_images:
                user_warnings.add(f"⚠️ Max {max_images} image{'' if max_images == 1 else 's'} per message" if max_images > 0 else "⚠️ Can't see images")
            if curr_node.has_bad_attachments:
                user_warnings.add("⚠️ Unsupported attachments")
            if curr_node.fetch_parent_failed or (curr_node.parent_msg != None and len(messages) == max_messages):
                user_warnings.add(f"⚠️ Only using last {len(messages)} message{'' if len(messages) == 1 else 's'}")

            curr_msg = curr_node.parent_msg

    if random_trigger:
        # Use the recent channel chat as context instead of a reply chain
        messages = [dict(role="user", content=f"<@{new_msg.author.id}>: {new_msg.content}"[:max_text])]
        user_warnings.clear()

        async for past_msg in new_msg.channel.history(limit=random_cfg.get("context_messages", 8), before=new_msg):
            past_text = "\n".join(
                filter(
                    None,
                    [past_msg.content]
                    + [embed.description for embed in past_msg.embeds]
                    + [component.content for component in past_msg.components if component.type == discord.ComponentType.text_display],
                )
            )
            if not past_text:
                continue

            if past_msg.author == discord_bot.user:
                messages.append(dict(role="assistant", content=past_text[:max_text]))
            else:
                messages.append(dict(role="user", content=f"<@{past_msg.author.id}>: {past_text}"[:max_text]))

        messages = messages[:max_messages]

    logging.info(f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}, conversation length: {len(messages)}, random chime-in: {random_trigger}):\n{new_msg.content}")

    # What the bot has learned from this server, plus media it may send back
    learned_text, offered_media = "", {}
    if not is_dm and new_msg.guild:
        query = " ".join(m["content"] if isinstance(m["content"], str) else m["content"][0].get("text", "") for m in messages[:4])
        try:
            learned_text, offered_media = await learning.build_context(new_msg.guild.id, query, config)
        except Exception:
            logging.exception("Error while building learned context")

    if system_prompt := config.get("system_prompt"):
        now = datetime.now().astimezone()

        system_prompt = system_prompt.replace("{date}", now.strftime("%B %d %Y")).replace("{time}", now.strftime("%H:%M:%S %Z%z")).strip()

        if random_trigger:
            system_prompt += "\n\nNobody @'d you. You are chiming into this chat on your own. React naturally to the latest message, stay in character, and keep it to one or two short sentences."

    if system_prompt := "\n\n".join(filter(None, (system_prompt, learned_text))):
        messages.append(dict(role="system", content=system_prompt))

    # Generate and send response message(s) (can be multiple if response is long)
    curr_content = finish_reason = None
    response_msgs = []
    response_contents = []

    openai_kwargs = dict(model=model, messages=messages[::-1], stream=True, extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body)

    if use_plain_responses := config.get("use_plain_responses", False):
        max_message_length = 4000
    else:
        max_message_length = 4096 - len(STREAMING_INDICATOR)
        embed = discord.Embed.from_dict(dict(fields=[dict(name=warning, value="", inline=False) for warning in sorted(user_warnings)]))

    async def reply_helper(**reply_kwargs) -> None:
        reply_target = new_msg if not response_msgs else response_msgs[-1]
        response_msg = await reply_target.reply(**reply_kwargs)
        response_msgs.append(response_msg)

        msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
        await msg_nodes[response_msg.id].lock.acquire()

    try:
        async with new_msg.channel.typing():
            async for chunk in await openai_client.chat.completions.create(**openai_kwargs):
                if finish_reason != None:
                    break

                if not (choice := chunk.choices[0] if chunk.choices else None):
                    continue

                finish_reason = choice.finish_reason

                prev_content = curr_content or ""
                curr_content = choice.delta.content or ""

                new_content = prev_content if finish_reason == None else (prev_content + curr_content)

                if response_contents == [] and new_content == "":
                    continue

                if start_next_msg := response_contents == [] or len(response_contents[-1] + new_content) > max_message_length:
                    response_contents.append("")

                response_contents[-1] += new_content

                if not use_plain_responses:
                    time_delta = datetime.now().timestamp() - last_task_time

                    ready_to_edit = time_delta >= EDIT_DELAY_SECONDS
                    msg_split_incoming = finish_reason == None and len(response_contents[-1] + curr_content) > max_message_length
                    is_final_edit = finish_reason != None or msg_split_incoming
                    is_good_finish = finish_reason != None and finish_reason.lower() in ("stop", "end_turn")

                    if start_next_msg or ready_to_edit or is_final_edit:
                        shown_text = learning.strip_media_tags(response_contents[-1]) or "​"
                        embed.description = shown_text if is_final_edit else (shown_text + STREAMING_INDICATOR)
                        embed.color = EMBED_COLOR_COMPLETE if msg_split_incoming or is_good_finish else EMBED_COLOR_INCOMPLETE

                        if start_next_msg:
                            await reply_helper(embed=embed, silent=True)
                        else:
                            await asyncio.sleep(EDIT_DELAY_SECONDS - time_delta)
                            await response_msgs[-1].edit(embed=embed)

                        last_task_time = datetime.now().timestamp()

            if use_plain_responses:
                for content in response_contents:
                    if content := learning.strip_media_tags(content):
                        await reply_helper(view=LayoutView().add_item(TextDisplay(content=content)))

    except Exception:
        logging.exception("Error while generating response")

    full_response = "".join(response_contents)
    final_text = learning.strip_media_tags(full_response)

    # Send a remembered gif/image/video if the model picked one it was offered
    if chosen := learning.chosen_media(full_response, offered_media):
        reply_target = response_msgs[-1] if response_msgs else new_msg
        try:
            if media_msg := await learning.send_media(reply_target, chosen):
                msg_nodes[media_msg.id] = MsgNode(text=f"(sent a {chosen['kind']}: {chosen['description'] or 'reaction'})", parent_msg=reply_target)
                logging.info(f"Sent remembered {chosen['kind']} (media ID: {chosen['id']})")
        except Exception:
            logging.exception("Error while sending remembered media")

    for response_msg in response_msgs:
        msg_nodes[response_msg.id].text = final_text
        msg_nodes[response_msg.id].lock.release()

    # Delete oldest MsgNodes (lowest message IDs) from the cache
    if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
        for msg_id in sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]:
            async with msg_nodes.setdefault(msg_id, MsgNode()).lock:
                msg_nodes.pop(msg_id, None)


async def main() -> None:
    await discord_bot.start(config["bot_token"])


try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass
