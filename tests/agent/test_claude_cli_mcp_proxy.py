from __future__ import annotations

import base64

from agent.claude_cli_mcp_proxy import (
    _mcp_execution_id,
    _normalized_mcp_content,
)


def test_mcp_execution_id_namespaces_json_rpc_ids_by_proxy_instance():
    assert _mcp_execution_id("proxy-a", 7) == "proxy-a:7"
    assert _mcp_execution_id("proxy-a", "7") == 'proxy-a:"7"'
    assert _mcp_execution_id("proxy-b", 7) != _mcp_execution_id("proxy-a", 7)


def test_loopback_client_uses_native_turn_timeout(monkeypatch):
    from agent.claude_cli_mcp_proxy import LoopbackClient

    monkeypatch.setenv("HERMES_CLAUDE_LOOPBACK_PORT", "1234")
    monkeypatch.setenv("HERMES_CLAUDE_LOOPBACK_TOKEN", "secret")
    monkeypatch.setenv("HERMES_CLAUDE_LOOPBACK_TIMEOUT_SECONDS", "1800")

    assert LoopbackClient()._request_timeout == 1_800.0


def test_normalized_mcp_content_preserves_multimodal_tool_images():
    encoded = base64.b64encode(b"image-bytes").decode("ascii")

    assert _normalized_mcp_content(
        [
            {"type": "text", "text": "Screenshot captured"},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{encoded}",
                },
            },
        ]
    ) == [
        {"type": "text", "text": "Screenshot captured"},
        {"type": "image", "data": encoded, "mimeType": "image/png"},
    ]


def test_normalized_mcp_content_keeps_remote_image_as_explicit_reference():
    assert _normalized_mcp_content(
        [{"type": "image_url", "image_url": "https://example.test/x.png"}]
    ) == [
        {
            "type": "text",
            "text": "[Image available at URL: https://example.test/x.png]",
        }
    ]
