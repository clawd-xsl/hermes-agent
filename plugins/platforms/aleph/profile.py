"""The agent's profile on Aleph: its name and avatar (protocol "Agent profile").

Only the agent changes them, with the ``aleph`` tools (``tools.py``); the server keeps them and is
the one truth for every device. This module holds what does not need the running adapter: name
validation, reading an avatar source (an absolute local path or an ``https`` URL) and turning it
into the stored form, a square JPEG of at most 512 pixels a side.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import os
import unicodedata
from dataclasses import dataclass
from pathlib import Path

DEFAULT_NAME = "Aleph"
MAX_NAME_CHARS = 32
AVATAR_SIDE = 512
MAX_AVATAR_SOURCE_BYTES = 20 * 1024 * 1024
FETCH_TIMEOUT_SECONDS = 20.0
MAX_REDIRECTS = 5
# Decompression bombs: a source this many pixels or more is refused before it is decoded.
MAX_SOURCE_PIXELS = 64_000_000
SUPPORTED_FORMATS = {"PNG", "JPEG", "MPO", "HEIF", "WEBP", "GIF"}
JPEG_QUALITY = 90


class ProfileError(ValueError):
    """What the agent did wrong, said so it can fix it (the tool's ``isError`` text)."""


@dataclass(frozen=True)
class Avatar:
    jpeg: bytes
    width: int
    height: int

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.jpeg).hexdigest()


def clean_name(raw: object) -> str:
    """A valid display name, or ``ProfileError``: 1 to 32 characters after trimming, no line breaks
    or other control characters. Characters are Unicode code points."""
    if not isinstance(raw, str):
        raise ProfileError("name must be a string.")
    name = raw.strip()
    if not name:
        raise ProfileError("name is empty. Give 1 to 32 characters.")
    if any(unicodedata.category(ch) in ("Cc", "Zl", "Zp") for ch in name):
        raise ProfileError("name must be one line: no line breaks or control characters.")
    if len(name) > MAX_NAME_CHARS:
        raise ProfileError(f"name is {len(name)} characters; the limit is {MAX_NAME_CHARS}. Choose a shorter one.")
    return name


# ── reading a source ───────────────────────────────────────────────────────────────────────────


def read_local(path_text: str) -> bytes:
    path = Path(os.path.expanduser(path_text))
    if not path.is_absolute():
        raise ProfileError(f"avatar path {path_text!r} is not absolute. Pass an absolute path, such as the one an "
                           "image tool returned, or an https URL.")
    if not path.exists():
        raise ProfileError(f"avatar file {path} does not exist.")
    if not path.is_file():
        raise ProfileError(f"avatar path {path} is not a file.")
    size = path.stat().st_size
    if size > MAX_AVATAR_SOURCE_BYTES:
        raise ProfileError(f"avatar file is {size // (1024 * 1024)} MB; the limit is 20 MB.")
    return path.read_bytes()


def _check_https(url: str) -> None:
    from urllib.parse import urlsplit
    if urlsplit(url).scheme.lower() != "https":
        raise ProfileError(f"avatar URL must use https: {url}")
    from tools.url_safety import is_safe_url
    if not is_safe_url(url):
        raise ProfileError(f"avatar URL points at a private or blocked address: {url}")


async def fetch_https(url: str, *, client_factory=None) -> bytes:
    """GET an https URL (every redirect https too, and not to a private address), at most 20 MB."""
    import httpx
    await asyncio.to_thread(_check_https, url)
    factory = client_factory or (lambda: httpx.AsyncClient(timeout=FETCH_TIMEOUT_SECONDS, follow_redirects=False))
    async with factory() as client:
        for _ in range(MAX_REDIRECTS + 1):
            try:
                async with client.stream("GET", url) as response:
                    if response.status_code in (301, 302, 303, 307, 308):
                        location = response.headers.get("location")
                        if not location:
                            raise ProfileError(f"avatar URL redirected without a location: {url}")
                        url = str(response.url.join(location))
                        await asyncio.to_thread(_check_https, url)
                        continue
                    if response.status_code != 200:
                        raise ProfileError(f"fetching the avatar failed: HTTP {response.status_code} from {url}")
                    declared = response.headers.get("content-length")
                    if declared and declared.isdigit() and int(declared) > MAX_AVATAR_SOURCE_BYTES:
                        raise ProfileError("the avatar at that URL is over the 20 MB limit.")
                    data = bytearray()
                    async for chunk in response.aiter_bytes():
                        data += chunk
                        if len(data) > MAX_AVATAR_SOURCE_BYTES:
                            raise ProfileError("the avatar at that URL is over the 20 MB limit.")
                    return bytes(data)
            except httpx.HTTPError as exc:
                raise ProfileError(f"fetching the avatar failed: {exc.__class__.__name__}: {exc}") from exc
    raise ProfileError(f"avatar URL redirected more than {MAX_REDIRECTS} times: {url}")


async def read_source(source: object, **fetch_kwargs) -> bytes:
    if not isinstance(source, str) or not source.strip():
        raise ProfileError("avatar must be an absolute file path or an https URL.")
    source = source.strip()
    if "://" in source:
        return await fetch_https(source, **fetch_kwargs)
    return await asyncio.to_thread(read_local, source)


# ── normalizing ────────────────────────────────────────────────────────────────────────────────


def normalize(data: bytes) -> Avatar:
    """Upright, center-cropped square, at most 512 pixels, flattened onto white, as a JPEG without
    metadata. ``ProfileError`` if it is not a PNG, JPEG, HEIC, WebP or GIF image."""
    from PIL import Image, ImageOps, UnidentifiedImageError
    try:
        import pillow_heif  # HEIC (iPhone photos); a core dependency, but stay usable without it
        pillow_heif.register_heif_opener()
    except Exception:
        pass
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in SUPPORTED_FORMATS:
                raise ProfileError(f"the avatar is a {image.format} file; use a PNG, JPEG, HEIC, WebP or GIF image.")
            if image.width * image.height >= MAX_SOURCE_PIXELS:
                raise ProfileError(f"the avatar is {image.width}x{image.height} pixels; use a smaller image.")
            image.seek(0)  # a GIF's (or WebP's) first frame
            image.load()
            upright = ImageOps.exif_transpose(image)
    except ProfileError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        raise ProfileError(f"the avatar is not a readable image ({exc.__class__.__name__}). Use a PNG, JPEG, HEIC, "
                           "WebP or GIF file.") from exc
    rgba = upright.convert("RGBA")
    flat = Image.new("RGB", rgba.size, (255, 255, 255))
    flat.paste(rgba, mask=rgba.getchannel("A"))
    side = min(flat.size)
    left, top = (flat.width - side) // 2, (flat.height - side) // 2
    square = flat.crop((left, top, left + side, top + side))
    if side > AVATAR_SIDE:
        square = square.resize((AVATAR_SIDE, AVATAR_SIDE), Image.Resampling.LANCZOS)
    out = io.BytesIO()
    square.save(out, format="JPEG", quality=JPEG_QUALITY, optimize=True, progressive=True)
    return Avatar(out.getvalue(), square.width, square.height)


async def load_avatar(source: object, **fetch_kwargs) -> Avatar:
    data = await read_source(source, **fetch_kwargs)
    return await asyncio.to_thread(normalize, data)
