# (C) Datadog, Inc. 2026-present
# All rights reserved
# Licensed under a 3-clause BSD style license (see LICENSE)

import json
from functools import partial
from unittest.mock import AsyncMock

import anthropic
import httpx2

from ddev.ai.agent.anthropic_client import check_gateway_messages, direct_messages_client, gateway_messages_client
from ddev.ai.model_catalog import ModelResolver, Route
from ddev.ai.runtime.gateway_transport import GatewayTransport


def _message(model: str) -> httpx2.Response:
    return httpx2.Response(
        200,
        json={
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    )


async def test_gateway_messages_sends_only_gateway_credential_to_public_endpoint(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "direct-secret")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://different.example")
    monkeypatch.setenv(
        "ANTHROPIC_CUSTOM_HEADERS",
        "X-API-Key: direct-secret\nML-App-ID: different-app\nX-AI-Gateway-Provider: anthropic",
    )
    model = ModelResolver().resolve("haiku", route=Route.AI_GATEWAY)

    def handle(request: httpx2.Request) -> httpx2.Response:
        assert request.url == "https://ai-gateway.datadoghq.com/ai/v1/messages"
        assert request.headers.get_list("authorization") == ["Bearer ddpat_gateway_secret"]
        assert "x-api-key" not in request.headers
        assert "x-ai-gateway-provider" not in request.headers
        assert request.headers["ml-app-id"] == "togo"
        assert json.loads(request.content)["model"] == model.binding.model
        assert b"ddpat_gateway_secret" not in request.content
        return _message("claude-haiku-4-5-20251001")

    transport = GatewayTransport(AsyncMock(return_value="ddpat_gateway_secret"), httpx2.MockTransport(handle))
    async with gateway_messages_client(transport) as client:
        result = await check_gateway_messages(client, model)

    assert result.status == 200
    assert result.model_confirmed


async def test_direct_messages_sends_only_provider_credential(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "Authorization: Bearer ddpat_unrelated\nML-App-ID: gateway-app")

    def handle(request: httpx2.Request) -> httpx2.Response:
        assert request.headers.get_list("authorization") == ["Bearer direct-secret"]
        assert "ml-app-id" not in request.headers
        assert "x-ai-gateway-provider" not in request.headers
        return _message("claude-haiku-4-5")

    monkeypatch.setattr(
        anthropic,
        "DefaultAsyncHttpxClient",
        partial(anthropic.DefaultAsyncHttpxClient, transport=httpx2.MockTransport(handle)),
    )
    async with direct_messages_client(AsyncMock(return_value="direct-secret")) as client:
        await client.messages.create(
            model="claude-haiku-4-5",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=1,
            extra_headers={"x-ai-gateway-provider": "anthropic"},
        )
