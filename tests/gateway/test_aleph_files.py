"""Aleph attachments both ways: file tickets, per-file encryption, the media cache
(plugins/platforms/aleph/files.py), driven through a real listener."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio

from gateway.config import PlatformConfig
from tests.gateway.aleph_phone import connect, pair_device, url as _url

def _register():
    from plugins.platforms.aleph.register import register
    captured = {}
    register(SimpleNamespace(register_platform=lambda **kw: captured.update(kw),
                             register_cli_command=lambda **kw: captured.setdefault("cli", kw)))
    return captured


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
        bad = await phone.call("message.send", {"id": "m-bad", "text": "", "attachments": [
            {"ticket": "t_nope", "key": attachment["key"], "name": "x", "mime": "image/png"}]})
        assert bad["error"]["code"] == -32602
        await phone.call("message.send", {"id": "m-img", "text": "", "attachments": [attachment]})
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
        again = await phone.call("message.send", {"id": "m-again", "text": "", "attachments": [attachment]})
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
        wrong = await phone.call("message.send", {"id": "m-wrong", "text": "", "attachments": [
            {"ticket": ticket["ticket"], "key": b64url(b"\x01" * 32), "name": "a.txt", "mime": "text/plain"}]})
        assert wrong["error"]["code"] == -32602
        right = await phone.call("message.send", {"id": "m-right", "text": "", "attachments": [
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
