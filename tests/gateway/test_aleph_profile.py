"""The agent's profile on Aleph (plugins/platforms/aleph/profile.py, tools.py): name validation, avatar
sources and normalization, the aleph tools as MCP tools, hello and the agent.profile event."""

from __future__ import annotations

import asyncio
import io
import json

import httpx
import jsonschema
import pytest
import pytest_asyncio

from gateway.config import PlatformConfig
from plugins.platforms.aleph import tools as aleph_tools
from plugins.platforms.aleph.profile import ProfileError, clean_name, fetch_https, normalize, read_local
from tests.gateway.aleph_phone import connect, pair_device


@pytest_asyncio.fixture
async def adapter(tmp_path, monkeypatch):
    from plugins.platforms.aleph.adapter import AlephAdapter
    config = PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "port": 0,
                                                 "state_dir": str(tmp_path / "aleph")})
    instance = AlephAdapter(config)

    async def handler(event):
        return None
    instance.set_message_handler(handler)
    assert await instance.connect()
    instance.device, instance.keys = pair_device(instance)
    # The tools find the adapter the way they do in the gateway; here it is the test's.
    monkeypatch.setattr(aleph_tools, "_live", lambda: (None, instance))
    yield instance
    await instance.disconnect()


def _image(fmt="PNG", size=(800, 400), color=(255, 0, 0), mode="RGB", **save):
    from PIL import Image
    buffer = io.BytesIO()
    Image.new(mode, size, color).save(buffer, format=fmt, **save)
    return buffer.getvalue()


def _open(data):
    from PIL import Image
    image = Image.open(io.BytesIO(data))
    image.load()
    return image


def _result(text, tool):
    """Parse a CallToolResult and check it against the MCP shape and the tool's outputSchema."""
    result = json.loads(text)
    assert [block["type"] for block in result["content"]] == ["text"]
    if result.get("isError"):
        assert "structuredContent" not in result
    else:
        jsonschema.validate(result["structuredContent"], tool["outputSchema"])
    return result


async def _set(**args):
    return _result(await aleph_tools.set_profile(args), aleph_tools.SET_PROFILE)


async def _get():
    return _result(await aleph_tools.get_profile({}), aleph_tools.GET_PROFILE)


# ── the tool definitions ─────────────────────────────────────────────────────────────────────


def test_tools_are_complete_mcp_definitions():
    for tool in aleph_tools.TOOLS:
        assert set(tool) == {"name", "title", "description", "inputSchema", "outputSchema", "annotations"}
        jsonschema.Draft202012Validator.check_schema(tool["inputSchema"])
        jsonschema.Draft202012Validator.check_schema(tool["outputSchema"])
        assert tool["inputSchema"]["type"] == "object" and tool["outputSchema"]["type"] == "object"
        schema = aleph_tools.hermes_schema(tool)
        assert schema == {"name": tool["name"], "description": tool["description"], "parameters": tool["inputSchema"]}
    assert aleph_tools.GET_PROFILE["annotations"] == {"readOnlyHint": True, "openWorldHint": False}
    assert aleph_tools.SET_PROFILE["annotations"] == {"readOnlyHint": False, "destructiveHint": False,
                                                       "idempotentHint": True, "openWorldHint": True}
    description = aleph_tools.SET_PROFILE["description"]
    for hint in ("absolute file path", "https URL", "reset_avatar", "at once", "isError"):
        assert hint in description


def test_the_tools_are_offered_only_where_aleph_runs(monkeypatch):
    monkeypatch.setattr(aleph_tools, "_live", lambda: (None, None))
    assert aleph_tools.aleph_connected() is False
    monkeypatch.setattr(aleph_tools, "_live", lambda: (None, object()))
    assert aleph_tools.aleph_connected() is True


@pytest.mark.asyncio
async def test_outside_the_gateway_the_tools_say_so(monkeypatch):
    monkeypatch.setattr(aleph_tools, "_live", lambda: (None, None))
    for result in (await aleph_tools.get_profile({}), await aleph_tools.set_profile({"name": "Iris"})):
        result = json.loads(result)
        assert result["isError"] is True and "not running" in result["content"][0]["text"]


# ── names ────────────────────────────────────────────────────────────────────────────────────


def test_names():
    assert clean_name("  Iris ") == "Iris"
    assert clean_name("x" * 32) == "x" * 32
    assert clean_name("小C ✨") == "小C ✨"
    for bad, hint in (("", "empty"), ("   ", "empty"), ("x" * 33, "limit is 32"), ("two\nlines", "one line"),
                      ("tab\there", "one line"), ("a b", "one line"), (5, "string")):
        with pytest.raises(ProfileError, match=hint):
            clean_name(bad)


# ── avatar sources ───────────────────────────────────────────────────────────────────────────


def test_local_sources(tmp_path, monkeypatch):
    from plugins.platforms.aleph import profile
    path = tmp_path / "a.png"
    path.write_bytes(b"data")
    assert read_local(str(path)) == b"data"
    for bad, hint in (("a.png", "not absolute"), (str(tmp_path / "missing.png"), "does not exist"),
                      (str(tmp_path), "not a file")):
        with pytest.raises(ProfileError, match=hint):
            read_local(bad)
    monkeypatch.setattr(profile, "MAX_AVATAR_SOURCE_BYTES", 3)
    with pytest.raises(ProfileError, match="20 MB"):
        read_local(str(path))


def _client(handler):
    return lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False)


@pytest.mark.asyncio
async def test_url_sources_are_https_only_and_capped(monkeypatch):
    from plugins.platforms.aleph import profile
    import tools.url_safety
    monkeypatch.setattr(tools.url_safety, "is_safe_url", lambda url: "internal" not in url)
    png = _image()

    def handler(request):
        path = request.url.path
        if path == "/ok.png":
            return httpx.Response(200, content=png)
        if path == "/hop":
            return httpx.Response(302, headers={"location": "/ok.png"})
        if path == "/downgrade":
            return httpx.Response(302, headers={"location": "http://cdn.example.com/ok.png"})
        if path == "/inside":
            return httpx.Response(302, headers={"location": "https://internal.example.com/x"})
        if path == "/loop":
            return httpx.Response(302, headers={"location": "/loop"})
        if path == "/big":
            return httpx.Response(200, content=b"x" * 64)
        return httpx.Response(404)
    assert await fetch_https("https://cdn.example.com/ok.png", client_factory=_client(handler)) == png
    assert await fetch_https("https://cdn.example.com/hop", client_factory=_client(handler)) == png
    cases = (("http://cdn.example.com/ok.png", "https"), ("https://internal.example.com/x", "private"),
             ("https://cdn.example.com/downgrade", "https"), ("https://cdn.example.com/inside", "private"),
             ("https://cdn.example.com/loop", "redirected more than"), ("https://cdn.example.com/nope", "HTTP 404"))
    for url, hint in cases:
        with pytest.raises(ProfileError, match=hint):
            await fetch_https(url, client_factory=_client(handler))
    monkeypatch.setattr(profile, "MAX_AVATAR_SOURCE_BYTES", 10)
    with pytest.raises(ProfileError, match="20 MB"):
        await fetch_https("https://cdn.example.com/big", client_factory=_client(handler))

    def broken(request):
        raise httpx.ConnectTimeout("too slow")
    with pytest.raises(ProfileError, match="ConnectTimeout"):
        await fetch_https("https://cdn.example.com/ok.png", client_factory=_client(broken))


# ── normalizing ──────────────────────────────────────────────────────────────────────────────


def test_large_images_are_cropped_square_and_scaled_to_512():
    out = normalize(_image(size=(2000, 1000)))
    assert (out.width, out.height) == (512, 512)
    image = _open(out.jpeg)
    assert image.format == "JPEG" and image.size == (512, 512)


def test_small_images_are_not_scaled_up():
    out = normalize(_image(size=(100, 80)))
    assert (out.width, out.height) == (80, 80)


def test_exif_orientation_is_applied_and_metadata_dropped():
    from PIL import Image
    # Stored landscape: left half red, right half blue; EXIF says rotate 90° clockwise to show it.
    image = Image.new("RGB", (800, 400), (0, 0, 255))
    image.paste((255, 0, 0), (0, 0, 400, 400))
    exif = Image.Exif()
    exif[0x0112] = 6  # orientation
    exif[0x010F] = "Camera Maker"
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif.tobytes(), quality=95)
    out = _open(normalize(buffer.getvalue()).jpeg)
    assert out.size == (400, 400)
    red, _green, blue = out.getpixel((350, 50))
    assert red > 200 and blue < 60  # upright: the red half is on top, where the unrotated crop is blue
    assert not out.getexif()


def test_transparency_is_flattened_onto_white():
    out = _open(normalize(_image(size=(64, 64), color=(0, 0, 0, 0), mode="RGBA")).jpeg)
    assert all(channel > 245 for channel in out.getpixel((10, 10)))


def test_formats():
    from PIL import Image
    gif = io.BytesIO()
    frames = [Image.new("P", (40, 40), 1), Image.new("P", (40, 40), 2)]
    frames[0].save(gif, format="GIF", save_all=True, append_images=frames[1:])
    for data in (_image("JPEG"), _image("WEBP"), gif.getvalue()):
        assert normalize(data).width > 0
    import pillow_heif
    pillow_heif.register_heif_opener()
    assert normalize(_image("HEIF", size=(64, 48))).width == 48
    with pytest.raises(ProfileError, match="BMP"):
        normalize(_image("BMP"))
    with pytest.raises(ProfileError, match="not a readable image"):
        normalize(b"definitely not an image")


def test_huge_images_are_refused_before_decoding(monkeypatch):
    from plugins.platforms.aleph import profile
    monkeypatch.setattr(profile, "MAX_SOURCE_PIXELS", 1000)
    with pytest.raises(ProfileError, match="smaller image"):
        normalize(_image(size=(100, 100)))


# ── the tools against the channel ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_default_profile_and_hello(adapter):
    result = await _get()
    assert result["structuredContent"] == {"name": "Aleph", "avatar": None}
    assert "Aleph" in result["content"][0]["text"]
    phone = await connect(adapter, adapter.keys)
    try:
        answer, _ = await phone.hello()
        agent = answer["result"]["agent"]
        assert agent["name"] == "Aleph" and agent["avatar"] is None
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_renaming_logs_one_event_and_reaches_hello_and_pairing(adapter):
    phone = await connect(adapter, adapter.keys)
    try:
        await phone.hello()
        result = await _set(name="  Iris ")
        assert result["structuredContent"] == {"name": "Iris", "avatar": None, "changed": ["name"]}
        event = (await phone.logged())["params"]
        assert event["type"] == "agent.profile"
        assert event["payload"]["name"] == "Iris" and event["payload"]["avatar"] is None
        assert event["payload"]["changed"] == ["name"]
        from datetime import datetime
        assert datetime.fromisoformat(event["payload"]["at"]).tzinfo is not None
        latest = adapter.store.latest()
        again = await _set(name="Iris")
        assert again["structuredContent"]["changed"] == [] and "Nothing changed" in again["content"][0]["text"]
        assert adapter.store.latest() == latest  # nothing logged
    finally:
        await phone.close()
    phone = await connect(adapter, adapter.keys)
    try:
        answer, _ = await phone.hello()
        assert answer["result"]["agent"]["name"] == "Iris"
    finally:
        await phone.close()
    secret, _ = adapter.store.new_pairing()
    from plugins.platforms.aleph import noise
    paired = await adapter.complete_pairing(secret, noise.KeyPair.generate().public, name="iPad", model="iPad")
    assert paired["agent"] == {"name": "Iris"}


@pytest.mark.asyncio
async def test_an_avatar_is_stored_encrypted_downloadable_and_survives_a_restart(adapter, tmp_path):
    source = tmp_path / "me.png"
    source.write_bytes(_image(size=(1200, 900)))
    phone = await connect(adapter, adapter.keys)
    try:
        await phone.hello()
        result = await _set(avatar=str(source))
        assert result["structuredContent"]["changed"] == ["avatar"]
        avatar_info = result["structuredContent"]["avatar"]
        assert (avatar_info["width"], avatar_info["height"]) == (512, 512) and avatar_info["updated_at"]
        payload = (await phone.logged())["params"]["payload"]
        avatar = payload["avatar"]
        assert payload["changed"] == ["avatar"]
        assert {k: avatar[k] for k in ("name", "mime", "kind", "width", "height")} == {
            "name": "avatar.jpg", "mime": "image/jpeg", "kind": "image", "width": 512, "height": 512}
        jpeg = await phone.download(adapter, avatar)
        assert _open(jpeg).size == (512, 512) and len(jpeg) == avatar["size"]
        stored = adapter.store.file(avatar["id"])
        from pathlib import Path
        assert b"JFIF" not in Path(stored.path).read_bytes()  # ciphertext at rest
        # The same picture again changes nothing and keeps the id.
        same = await _set(avatar=str(source))
        assert same["structuredContent"]["changed"] == []
    finally:
        await phone.close()
    await adapter.disconnect()
    assert await adapter.connect()
    phone = await connect(adapter, adapter.keys)
    try:
        answer, _ = await phone.hello()
        agent = answer["result"]["agent"]
        assert agent["avatar"]["id"] == avatar["id"] and agent["avatar"]["url"].startswith("/aleph/v1/files/t_")
        assert await phone.download(adapter, agent["avatar"]) == jpeg
        fresh = (await phone.call("file.ticket", {"id": avatar["id"]}))["result"]
        assert await phone.download(adapter, {**avatar, "url": fresh["url"]}) == jpeg
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_a_new_avatar_gets_a_new_id_and_reset_goes_back_to_the_default(adapter, tmp_path):
    first, second = tmp_path / "1.png", tmp_path / "2.png"
    first.write_bytes(_image(color=(255, 0, 0)))
    second.write_bytes(_image(color=(0, 255, 0)))
    await _set(avatar=str(first))
    old_id = adapter._avatar["id"]
    await _set(avatar=str(second), name="Iris")
    new_id = adapter._avatar["id"]
    assert new_id != old_id and adapter.store.file(old_id) is None  # the old one is gone
    phone = await connect(adapter, adapter.keys)
    try:
        await phone.hello()
        assert (await phone.call("file.ticket", {"id": old_id}))["error"]["code"] == -32602
        result = await _set(reset_avatar=True)
        assert result["structuredContent"] == {"name": "Iris", "avatar": None, "changed": ["avatar"]}
        payload = (await phone.logged())["params"]["payload"]
        assert payload["avatar"] is None and payload["changed"] == ["avatar"]
        assert adapter.store.file(new_id) is None
        assert (await _set(reset_avatar=True))["structuredContent"]["changed"] == []
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_errors_change_nothing(adapter, tmp_path):
    good = tmp_path / "ok.png"
    good.write_bytes(_image())
    junk = tmp_path / "junk.png"
    junk.write_bytes(b"not an image")
    latest = adapter.store.latest()
    cases = (({}, "Nothing to change"), ({"reset_avatar": False}, "Nothing to change"),
             ({"avatar": str(good), "reset_avatar": True}, "not both"), ({"name": ""}, "empty"),
             ({"name": "x" * 40}, "limit is 32"), ({"name": "Iris", "avatar": str(junk)}, "not a readable image"),
             ({"avatar": "relative.png"}, "not absolute"), ({"avatar": "http://example.com/a.png"}, "https"),
             ({"nickname": "x"}, "Unknown argument"), ({"reset_avatar": "yes"}, "true or false"))
    for args, hint in cases:
        result = await _set(**args)
        assert result["isError"] is True, args
        assert hint in result["content"][0]["text"], (args, result)
    assert adapter.store.latest() == latest
    assert (await _get())["structuredContent"] == {"name": "Aleph", "avatar": None}


@pytest.mark.asyncio
async def test_profile_changes_are_not_pushed(adapter, tmp_path):
    from tests.gateway.test_aleph_push import FakeSender
    adapter._push = FakeSender()
    adapter.store.set_push_token(adapter.device.id, "ab" * 32, "production")
    avatar = tmp_path / "a.png"
    avatar.write_bytes(_image())
    await _set(name="Iris", avatar=str(avatar))
    for _ in range(20):
        await asyncio.sleep(0.01)
    assert adapter._push.sent == []
