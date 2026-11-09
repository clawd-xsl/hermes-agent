"""Aleph attachments both ways (file tickets, per-file encryption, the media cache, what the phone
says about a file) and shared locations, driven through a real listener."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import pytest_asyncio

from gateway.config import PlatformConfig
from tests.gateway.aleph_phone import connect, pair_device, url as _url


@pytest_asyncio.fixture
async def adapter(tmp_path):
    from plugins.platforms.aleph.adapter import AlephAdapter
    config = PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "port": 0,
                                                 "state_dir": str(tmp_path / "aleph")})
    instance = AlephAdapter(config)
    received = []

    async def handler(event):
        received.append(event)
    instance.set_message_handler(handler)
    instance.received = received
    assert await instance.connect()
    instance.device, instance.keys = pair_device(instance)
    yield instance
    await instance.disconnect()


async def _phone(adapter, keys=None):
    return await connect(adapter, keys or adapter.keys)


# ── files ─────────────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_upload_attach_and_download(adapter):
    png = _png_bytes()
    phone = await _phone(adapter)
    try:
        await phone.hello()
        attachment = await phone.upload(adapter, png, name="shot one.png", mime="image/png")
        bad = await phone.call("message.send", {"chat": "main", "id": "m-bad", "text": "", "attachments": [
            {"ticket": "t_nope", "key": attachment["key"], "name": "x", "mime": "image/png"}]})
        assert bad["error"]["code"] == -32602
        await phone.call("message.send", {"chat": "main", "id": "m-img", "text": "", "attachments": [attachment]})
        echo = (await phone.logged())["params"]["payload"]
        (description,) = echo["attachments"]
        assert description["name"] == "shot one.png" and description["kind"] == "image"
        assert description["mime"] == "image/png" and description["size"] == len(png)
        assert (description["width"], description["height"]) == (3, 2)
        assert description["key"] == attachment["key"] and description["url"].startswith("/aleph/v1/files/t_")
        assert await phone.download(adapter, description) == png
        async with phone.session.get(_url(adapter, description["url"][len("/aleph/v1"):])) as resp:
            assert resp.status == 404  # single use
        fresh = (await phone.call("file.ticket", {"id": description["id"]}))["result"]
        assert await phone.download(adapter, {**description, "url": fresh["url"]}) == png
        await asyncio.sleep(0.6)
        (event,) = [e for e in adapter.received if e.message_id == "m-img"]
        assert event.media_types == ["image/png"] and event.message_type.value == "photo"
        assert event.media_urls and Path(event.media_urls[0]).read_bytes() == png  # the agent gets plaintext
        # The upload ticket was spent.
        again = await phone.call("message.send", {"chat": "main", "id": "m-again", "text": "", "attachments": [attachment]})
        assert again["error"]["code"] == -32602
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_upload_tickets_check_size_key_and_use(adapter):
    from plugins.platforms.aleph.files import encrypt_file
    from plugins.platforms.aleph.noise import b64url
    phone = await _phone(adapter)
    try:
        await phone.hello()
        for size in (0, 15, -1, "12", 300 * 1024 * 1024):
            assert (await phone.call("file.ticket", {"size": size}))["error"]["code"] == -32602
        assert (await phone.call("file.ticket", {"id": "f_missing"}))["error"]["code"] == -32602
        key, ciphertext = encrypt_file(b"hello")
        ticket = (await phone.call("file.ticket", {"size": len(ciphertext)}))["result"]
        base = _url(adapter, "")[:-len("/aleph/v1")]
        async with phone.session.put(base + ticket["url"], data=ciphertext + b"x") as resp:
            assert resp.status == 400  # not the announced size
        async with phone.session.put(base + ticket["url"], data=ciphertext) as resp:
            assert resp.status == 204
        async with phone.session.put(base + ticket["url"], data=ciphertext) as resp:
            assert resp.status == 404  # one PUT per ticket
        async with phone.session.put(base + "/aleph/v1/files/t_made_up", data=ciphertext) as resp:
            assert resp.status == 404
        wrong = await phone.call("message.send", {"chat": "main", "id": "m-wrong", "text": "", "attachments": [
            {"ticket": ticket["ticket"], "key": b64url(b"\x01" * 32), "name": "a.txt", "mime": "text/plain"}]})
        assert wrong["error"]["code"] == -32602
        right = await phone.call("message.send", {"chat": "main", "id": "m-right", "text": "", "attachments": [
            {"ticket": ticket["ticket"], "key": b64url(key), "name": "a.txt", "mime": "text/plain"}]})
        assert "result" in right  # a wrong key did not spend the ticket
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_agent_files_become_downloadable_attachments(adapter, tmp_path):
    report = tmp_path / "report.pdf"
    report.write_bytes(b"%PDF-1.4 test")
    phone = await _phone(adapter)
    try:
        await phone.hello()
        sent = await adapter.send_document("main", str(report), caption="Here it is")
        assert sent.success
        payload = (await phone.event())["params"]["payload"]
        assert payload["text"] == "Here it is" and payload["sender"] == "agent"
        (attachment,) = payload["attachments"]
        assert attachment["name"] == "report.pdf" and attachment["kind"] == "file"
        assert attachment["mime"] == "application/pdf" and attachment["size"] == len(b"%PDF-1.4 test")
        report.unlink()  # Aleph keeps its own (encrypted) copy
        stored = adapter.store.file(attachment["id"])
        assert b"PDF" not in Path(stored.path).read_bytes()
        assert await phone.download(adapter, attachment) == b"%PDF-1.4 test"
        missing = await adapter.send_image_file("main", str(tmp_path / "nope.png"))
        assert not missing.success
    finally:
        await phone.close()


def _png_bytes():
    import io
    from PIL import Image
    buffer = io.BytesIO()
    Image.new("RGB", (3, 2), (255, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


async def _send(phone, **params):
    import uuid
    return await phone.call("message.send", {"chat": "main", "id": str(uuid.uuid4()), "text": "", **params})


async def _received(adapter, message_id=None):
    for _ in range(50):
        if adapter.received and (message_id is None or adapter.received[-1].message_id == message_id):
            return adapter.received[-1]
        await asyncio.sleep(0.02)
    raise AssertionError("nothing reached the gateway")


# ── what the phone says about a file, and the kind ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_kind_comes_from_the_mime_and_phone_hints_pass_through(adapter):
    phone = await _phone(adapter)
    try:
        await phone.hello()
        video = await phone.upload(adapter, b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64, name="clip.mov",
                                   mime="video/quicktime")
        audio = await phone.upload(adapter, b"ID3" + b"\x00" * 64, name="memo.m4a", mime="audio/mp4")
        voice = await phone.upload(adapter, b"OggS" + b"\x00" * 64, name="note.ogg", mime="audio/ogg")
        doc = await phone.upload(adapter, b"%PDF-1.4 x", name="paper.pdf", mime="application/pdf")
        image = await phone.upload(adapter, _png_bytes(), name="shot.png", mime="image/png")
        answer = await _send(phone, attachments=[
            {**video, "width": 1920, "height": 1080, "duration": 12.5},
            {**audio, "duration": 3, "width": 10, "height": 10},   # an audio's size is meaningless: dropped
            {**voice, "duration": "long"},                           # junk: ignored
            {**doc, "width": -1, "duration": True},                  # junk: ignored
            {**image, "width": 999, "height": 999},                  # an image is measured, not trusted
        ])
        assert "result" in answer, answer
        echo = (await phone.logged())["params"]["payload"]
        by_name = {a["name"]: a for a in echo["attachments"]}
        assert {n: a["kind"] for n, a in by_name.items()} == {
            "clip.mov": "video", "memo.m4a": "audio", "note.ogg": "voice", "paper.pdf": "file", "shot.png": "image"}
        assert (by_name["clip.mov"]["width"], by_name["clip.mov"]["height"], by_name["clip.mov"]["duration"]) == \
            (1920, 1080, 12.5)
        assert by_name["memo.m4a"]["duration"] == 3.0 and "width" not in by_name["memo.m4a"]
        assert "duration" not in by_name["note.ogg"]
        assert not {"width", "height", "duration"} & set(by_name["paper.pdf"])
        assert (by_name["shot.png"]["width"], by_name["shot.png"]["height"]) == (3, 2)
        assert all(a["key"] and a["url"].startswith("/aleph/v1/files/t_") for a in echo["attachments"])
        event = await _received(adapter)
        assert event.message_type.value == "video" and len(event.media_urls) == 5
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_the_upload_cap_follows_the_gateway_media_cap(adapter, monkeypatch):
    from gateway.platforms import base
    from plugins.platforms.aleph import adapter as mod
    phone = await _phone(adapter)
    try:
        await phone.hello()
        assert mod.upload_limit() == min(200 * 1024 * 1024, base.get_inbound_media_max_bytes())
        over = await phone.call("file.ticket", {"size": 200 * 1024 * 1024 + 17})
        assert over["error"]["code"] == -32602 and "too large" in over["error"]["message"]
        monkeypatch.setattr(base, "get_inbound_media_max_bytes", lambda: 1024 * 1024)
        assert mod.upload_limit() == 1024 * 1024
        refused = await phone.call("file.ticket", {"size": 1024 * 1024 + 17})
        assert refused["error"]["data"] == {"max_bytes": 1024 * 1024}
        assert "result" in await phone.call("file.ticket", {"size": 1024 * 1024 + 16})
        monkeypatch.setattr(base, "get_inbound_media_max_bytes", lambda: 0)  # 0 disables the gateway cap
        assert mod.upload_limit() == 200 * 1024 * 1024
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_a_file_the_gateway_refuses_fails_the_send_and_spends_no_ticket(adapter, monkeypatch):
    from plugins.platforms.aleph import adapter as mod
    phone = await _phone(adapter)
    try:
        await phone.hello()
        first = await phone.upload(adapter, b"%PDF-1.4 a", name="a.pdf", mime="application/pdf")
        second = await phone.upload(adapter, b"\x00" * 32, name="big.mp4", mime="video/mp4")
        real = mod.cache_media_bytes_async

        async def picky(data, *, filename="", mime_type="", **kw):
            if filename == "big.mp4":
                raise ValueError("Inbound video payload is too large")
            return await real(data, filename=filename, mime_type=mime_type, **kw)
        monkeypatch.setattr(mod, "cache_media_bytes_async", picky)
        answer = await _send(phone, attachments=[first, second])
        assert answer["error"]["code"] == -32602 and "big.mp4 was refused" in answer["error"]["message"]
        monkeypatch.setattr(mod, "cache_media_bytes_async", real)
        assert "result" in await _send(phone, attachments=[first])  # the first ticket was not spent
    finally:
        await phone.close()


# ── locations ─────────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_location_alone_is_logged_and_reaches_the_agent_as_a_pin(adapter):
    phone = await _phone(adapter)
    try:
        await phone.hello()
        point = {"latitude": 31.2304, "longitude": 121.4737, "name": "The Bund", "address": "Zhongshan East Rd,\n Shanghai"}
        answer = await _send(phone, location=point)
        assert "result" in answer, answer
        echo = (await phone.logged())["params"]["payload"]
        assert echo["text"] == "" and echo["attachments"] == []
        assert echo["location"] == {"latitude": 31.2304, "longitude": 121.4737, "name": "The Bund",
                                    "address": "Zhongshan East Rd, Shanghai"}
        event = await _received(adapter)
        assert event.message_type.value == "location"
        assert event.text == ("[The user shared a location pin.]\nPlace: The Bund\nAddress: Zhongshan East Rd, Shanghai\n"
                              "latitude: 31.2304\nlongitude: 121.4737\n"
                              "Map: https://maps.apple.com/?ll=31.2304,121.4737&q=The%20Bund")
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_a_location_with_text_and_without_a_name(adapter):
    phone = await _phone(adapter)
    try:
        await phone.hello()
        assert "result" in await _send(phone, text="Meet here?", location={"latitude": -33.8568, "longitude": 151})
        echo = (await phone.logged())["params"]["payload"]
        assert echo["location"] == {"latitude": -33.8568, "longitude": 151.0} and echo["text"] == "Meet here?"
        event = await _received(adapter)
        assert event.text == ("[The user shared a location pin.]\nlatitude: -33.8568\nlongitude: 151.0\n"
                              "Map: https://maps.apple.com/?ll=-33.8568,151.0\n\nMeet here?")
        # Plain text stays plain; no location key on the echo.
        assert "result" in await _send(phone, text="just text")
        assert "location" not in (await phone.logged())["params"]["payload"]
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_bad_locations_are_refused(adapter):
    phone = await _phone(adapter)
    try:
        await phone.hello()
        for location in ("here", {"latitude": 91, "longitude": 0}, {"latitude": 0, "longitude": -180.5},
                         {"latitude": True, "longitude": 0}, {"latitude": "1", "longitude": 0}, {"longitude": 0},
                         {"latitude": 0, "longitude": 0, "name": 5}, {"latitude": 0, "longitude": 0, "address": "x" * 201}):
            answer = await _send(phone, location=location)
            assert answer["error"]["code"] == -32602, location
        assert (await _send(phone))["error"]["message"] == "a message needs text, attachments or a location"
    finally:
        await phone.close()
    from plugins.platforms.aleph.adapter import clean_location
    from plugins.platforms.aleph.server import RpcError
    for value in (float("nan"), float("inf")):
        with pytest.raises(RpcError):
            clean_location({"latitude": value, "longitude": 0})


@pytest.mark.asyncio
async def test_a_resend_that_never_reached_the_gateway_keeps_its_location(adapter, monkeypatch):
    phone = await _phone(adapter)
    try:
        await phone.hello()
        point = {"latitude": 1.5, "longitude": 2.5, "name": "Spot"}

        async def gateway_gone(*args, **kwargs):
            raise RuntimeError("gateway gone")
        with monkeypatch.context() as patch:
            patch.setattr(adapter, "_dispatch_user_message", gateway_gone)
            answer = await phone.call("message.send", {"chat": "main", "id": "m-loc", "text": "", "location": point})
        assert answer["error"]["code"] == -32603  # logged, never handed over
        assert "result" in await phone.call("message.send", {"chat": "main", "id": "m-loc", "text": "", "location": point})
        event = await _received(adapter, "m-loc")
        assert event.message_type.value == "location" and "Place: Spot" in event.text
    finally:
        await phone.close()


# ── the agent's files ────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_every_agent_file_send_carries_key_url_and_kind(adapter, tmp_path):
    files = {"pic.png": _png_bytes(), "clip.mp4": b"\x00\x00\x00\x18ftypmp42", "note.ogg": b"OggS\x00",
             "doc.txt": b"hello"}
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)
    phone = await _phone(adapter)
    try:
        await phone.hello()
        assert (await adapter.send_image_file("main", str(tmp_path / "pic.png"))).success
        assert (await adapter.send_video("main", str(tmp_path / "clip.mp4"))).success
        assert (await adapter.send_voice("main", str(tmp_path / "note.ogg"))).success
        assert (await adapter.send_document("main", str(tmp_path / "doc.txt"))).success
        kinds = []
        for _ in range(4):
            (attachment,) = (await phone.logged())["params"]["payload"]["attachments"]
            assert attachment["key"] and attachment["url"].startswith("/aleph/v1/files/t_")
            assert await phone.download(adapter, attachment) == files[attachment["name"]]
            kinds.append(attachment["kind"])
        assert kinds == ["image", "video", "voice", "file"]
    finally:
        await phone.close()



# ── text files the agent reads ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_small_text_files_are_inlined_and_the_gateway_says_so_truthfully(adapter):
    from gateway.run_inbound import GatewayInboundMixin
    phone = await _phone(adapter)
    try:
        await phone.hello()
        notes = await phone.upload(adapter, b"# Plan\n\n- fly Tuesday\n", name="plan.md", mime="text/markdown")
        big = await phone.upload(adapter, b"x" * (100 * 1024 + 1), name="huge.log", mime="text/plain")
        pdf = await phone.upload(adapter, b"%PDF-1.4 binary", name="paper.pdf", mime="application/pdf")
        assert "result" in await _send(phone, text="Have a look", attachments=[notes, big, pdf])
        event = await _received(adapter)
        assert event.media_text_inlined == [True, False, False]
        assert event.media_filenames == ["plan.md", "huge.log", "paper.pdf"]
        assert event.text == "[Content of plan.md]:\n# Plan\n\n- fly Tuesday\n\n\nHave a look"
        # What the gateway then puts in front, per file.
        prompt = GatewayInboundMixin._prepend_inbound_document_notes(event, event.text)
        assert "text document: 'plan.md'. Its content has been included below" in prompt
        assert "text document: 'huge.log'" in prompt and "Its content is not inlined here" in prompt
        assert "document: 'paper.pdf'" in prompt and "extract the document's text yourself" in prompt
    finally:
        await phone.close()
