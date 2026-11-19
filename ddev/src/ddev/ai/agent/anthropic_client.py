# (C) Datadog, Inc. 2026-present
# All rights reserved
# Licensed under a 3-clause BSD style license (see LICENSE)

import asyncio
from collections.abc import Awaitable, Callable

import anthropic
import httpx2

from ddev.ai.agent.exceptions import AgentConnectionError, AgentError
from ddev.ai.agent.provider import ModelAccessCheck
from ddev.ai.model_catalog import ResolvedModel
from ddev.ai.runtime.gateway_auth import LOGIN_HELP, GatewayAuthError
from ddev.ai.runtime.gateway_transport import (
    ATTRIBUTION_HEADER,
    CREDENTIAL_HEADERS,
    GATEWAY_BASE_URL,
    PROVIDER_HEADER,
    GatewayTransport,
)

CHECK_TIMEOUT = 15


def gateway_messages_client(transport: GatewayTransport) -> anthropic.AsyncAnthropic:
    """Build a Messages client that leaves Gateway authentication to the transport."""
    # The SDK appends /v1/messages itself. Give it only the prefix before /v1.
    return anthropic.AsyncAnthropic(
        api_key="gateway-transport",
        base_url=GATEWAY_BASE_URL.removesuffix("/v1"),
        http_client=anthropic.DefaultAsyncHttpxClient(
            transport=transport,
            follow_redirects=False,
            trust_env=False,
        ),
        max_retries=0,
    )


def direct_messages_client(token: Callable[[], Awaitable[str]]) -> anthropic.AsyncAnthropic:
    """Build a Messages client that sends the provider key, not Gateway headers."""

    async def attach_provider_key(request: httpx2.Request) -> None:
        for header in (*CREDENTIAL_HEADERS, PROVIDER_HEADER, ATTRIBUTION_HEADER):
            request.headers.pop(header, None)
        request.headers["Authorization"] = f"Bearer {await token()}"

    return anthropic.AsyncAnthropic(
        api_key="provider-transport",
        http_client=anthropic.DefaultAsyncHttpxClient(
            event_hooks={"request": [attach_provider_key]}, follow_redirects=False
        ),
    )


async def check_gateway_messages(client: anthropic.AsyncAnthropic, model: ResolvedModel) -> ModelAccessCheck:
    """Verify Messages access with a one-token request and report failures without response bodies."""
    requested_model = model.binding.model
    try:
        async with asyncio.timeout(CHECK_TIMEOUT):
            response = await client.messages.with_raw_response.create(
                model=requested_model,
                max_tokens=1,
                messages=[{"role": "user", "content": "Hi"}],
                extra_headers=dict(model.binding.headers),
            )
            message = await response.parse()
    except (TimeoutError, anthropic.APITimeoutError):
        raise GatewayAuthError(f"AI Gateway startup check timed out. Retry the run. {LOGIN_HELP}") from None
    except anthropic.APIConnectionError as error:
        if isinstance(error.__cause__, AgentError):
            raise error.__cause__ from None
        raise AgentConnectionError(f"AI Gateway startup connection failed. Retry the run. {LOGIN_HELP}") from None
    except anthropic.APIError:
        raise GatewayAuthError(f"AI Gateway startup check failed. Retry the run. {LOGIN_HELP}") from None
    return ModelAccessCheck(
        response.status_code,
        requested_model,
        message.model in (requested_model, requested_model.partition("/")[2]),
    )
