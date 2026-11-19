# (C) Datadog, Inc. 2026-present
# All rights reserved
# Licensed under a 3-clause BSD style license (see LICENSE)

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Final

import anthropic

from ddev.ai.agent.anthropic_agent import DEFAULT_MAX_TOKENS, AnthropicAgent
from ddev.ai.agent.anthropic_client import check_gateway_messages, direct_messages_client, gateway_messages_client
from ddev.ai.agent.base import BaseAgent
from ddev.ai.config.models import AgentConfig
from ddev.ai.model_catalog import ANTHROPIC, ModelResolver, ResolvedModel, Route
from ddev.ai.runtime.gateway_auth import GatewayAuthError
from ddev.ai.runtime.gateway_transport import GatewayTransport
from ddev.ai.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from ddev.ai.agent.provider import ModelAccessCheck
    from ddev.ai.runtime.tokens import TokenProducer

DEFAULT_MODEL: Final[str] = "sonnet"
GATEWAY_RUNTIME_REQUIRED: Final[str] = "AI Gateway execution requires prepared run resources"


class AnthropicProvider:
    """Resolves model bindings and builds agents with an Anthropic Messages client."""

    def __init__(
        self,
        api_key: str | None,
        *,
        route: Route = Route.DIRECT,
        resolver: ModelResolver | None = None,
        tokens: TokenProducer | None = None,
    ) -> None:
        self._api_key = api_key
        self._route = route
        self._resolver = ModelResolver() if resolver is None else resolver
        self._tokens = tokens
        self._client: anthropic.AsyncAnthropic | None = None

    def for_run(self, route: Route, tokens: TokenProducer) -> AnthropicProvider:
        return AnthropicProvider(None, route=route, resolver=self._resolver, tokens=tokens)

    @property
    def client(self) -> anthropic.AsyncAnthropic:
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _create_client(self) -> anthropic.AsyncAnthropic:
        if self._tokens is not None:
            token = partial(self._tokens.token_for, ANTHROPIC)
            if self._route is Route.AI_GATEWAY:
                return gateway_messages_client(GatewayTransport(token))
            return direct_messages_client(token)
        if self._route is Route.AI_GATEWAY:
            raise ValueError(GATEWAY_RUNTIME_REQUIRED)
        if self._api_key is None:
            raise ValueError("Direct Anthropic execution requires an API key")
        return anthropic.AsyncAnthropic(api_key=self._api_key)

    async def check_access(self, agent_config: AgentConfig) -> ModelAccessCheck:
        model = self.resolve_model(agent_config.model)
        return await check_gateway_messages(self.client, model)

    async def close(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            await client.close()

    def default_model(self) -> str:
        return DEFAULT_MODEL

    def supported_models(self) -> frozenset[str]:
        """The names this provider handles: aliases and canonical models."""
        return self._resolver.names_for(ANTHROPIC)

    def resolve_model(self, model: str) -> ResolvedModel:
        """Resolve a model name to its binding on the captured route."""
        return self._resolver.resolve(model, route=self._route, provider=ANTHROPIC)

    def validate_config(self, agent_config: AgentConfig):
        """Validate Anthropic-specific agent configuration."""
        resolved = None
        if self._route is Route.AI_GATEWAY and self._tokens is None:
            # Browsing cannot know whether run startup will need the direct binding.
            canonical, spec = self._resolver.model_spec(agent_config.model, provider=ANTHROPIC)
        else:
            resolved = self.resolve_model(agent_config.model)
            canonical, spec = resolved.canonical, resolved.model
        requested = DEFAULT_MAX_TOKENS if agent_config.max_tokens is None else agent_config.max_tokens
        if spec.max_output_tokens is not None and requested > spec.max_output_tokens:
            raise ValueError(
                f"max_tokens {requested} exceeds the {spec.max_output_tokens}-token output limit of model {canonical!r}"
            )
        if self._route is Route.AI_GATEWAY and resolved is not None:
            self._validate_gateway_binding(resolved)

    @staticmethod
    def _validate_gateway_binding(model: ResolvedModel) -> None:
        slug, _, model_id = model.binding.model.partition("/")
        if not slug or not model_id:
            raise GatewayAuthError(f"AI Gateway binding for {model.canonical!r} must use <provider>/<model>.")

    def build_agent(
        self,
        agent_config: AgentConfig,
        *,
        tools: ToolRegistry,
        system_prompt: str,
        owner_id: str,
    ) -> BaseAgent[Any]:
        if self._route is Route.AI_GATEWAY and self._tokens is None:
            raise ValueError(GATEWAY_RUNTIME_REQUIRED)
        resolved = self.resolve_model(agent_config.model)
        kwargs: dict[str, Any] = {}
        if agent_config.max_tokens is not None:
            kwargs["max_tokens"] = agent_config.max_tokens
        return AnthropicAgent(
            client=self.client,
            tools=tools,
            system_prompt=system_prompt,
            name=owner_id,
            model=resolved,
            **kwargs,
        )
