"""Per-route webhook ``model``/``provider`` overrides ride a route-scoped parent channel."""
import json

from gateway.config import ChannelOverride, PlatformConfig
from gateway.platforms.webhook import WebhookAdapter, _INSECURE_NO_AUTH


def _adapter(routes, **extra) -> WebhookAdapter:
    return WebhookAdapter(PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "port": 0, "routes": routes,
                                                              **extra}))


def _route(**kw):
    return {"secret": _INSECURE_NO_AUTH, "prompt": "Mail from {sender}: {subject}", **kw}


class TestModelOverrides:
    def test_route_model_beats_the_platform_default_and_rides_a_parent_channel(self):
        adapter = _adapter({"a": _route(model="anthropic/claude-sonnet-4.6", provider="openrouter"), "b": _route()},
                           model="sonnet", provider="anthropic")
        source = adapter._admission_source("a", None, "d")
        override = adapter.config.channel_overrides[source.parent_chat_id]
        assert (override.model, override.provider) == ("anthropic/claude-sonnet-4.6", "openrouter")
        inherited = adapter.config.channel_overrides[adapter._admission_source("b", None, "d").parent_chat_id]
        assert (inherited.model, inherited.provider) == ("sonnet", "anthropic")

    def test_the_gateway_channel_resolver_reads_the_override_for_a_delivery(self):
        from gateway.config import GatewayConfig, Platform
        from gateway.run import _get_channel_override
        adapter = _adapter({"a": _route(model="m1", provider="p1")})
        source = adapter._admission_source("a", None, "delivery-1")
        config = GatewayConfig(platforms={Platform.WEBHOOK: adapter.config})
        found = _get_channel_override(config, Platform.WEBHOOK, source.chat_id, parent_id=source.parent_chat_id)
        assert (found.model, found.provider) == ("m1", "p1")

    def test_no_model_config_keeps_normal_resolution(self):
        adapter = _adapter({"a": _route()})
        assert adapter._admission_source("a", None, "d").parent_chat_id is None
        assert adapter.config.channel_overrides == {}

    def test_dynamic_reload_registers_and_drops_overrides_but_keeps_explicit_ones(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        subs = tmp_path / "webhook_subscriptions.json"
        subs.write_text(json.dumps({"gmail": {"secret": "s3cret-value", "prompt": "x", "model": "haiku"}}))
        config = PlatformConfig(enabled=True, extra={"routes": {}},
                                channel_overrides={"__webhook_model__:pinned": ChannelOverride(model="explicit")})
        adapter = WebhookAdapter(config)
        adapter._reload_dynamic_routes()
        assert adapter.config.channel_overrides["__webhook_model__:gmail"].model == "haiku"
        subs.unlink()
        adapter._reload_dynamic_routes()
        assert "__webhook_model__:gmail" not in adapter.config.channel_overrides
        assert adapter.config.channel_overrides["__webhook_model__:pinned"].model == "explicit"
