# Togo: provider boundary

These rules supplement the [repository guidelines](../../../../AGENTS.md).

Togo is model independent. Keep these layers separate:

- **Orchestration**: everything under `ddev/src/ddev/ai/` outside the adapters, plus the CLI and TUI in
  `ddev/src/ddev/cli/meta/ai/`, uses provider-neutral contracts.
- **Adapters**: provider and protocol implementations in `agent/` own SDK clients, model resolution, access probes,
  request and response translation, and history formats behind `AgentProvider` and `BaseAgent`. Keep history opaque
  above the boundary. Inspect or change it through neutral operations or within the adapter.

The Gateway is a token producer, not a model provider. The run selects a token producer by route and asks it for a
provider's credential. The Gateway producer supplies the same `dd-auth` token for all providers; the direct producer
loads the configured key for the requested provider. Concrete wiring belongs at the composition root. Config and
the model catalog may map models to providers and protocols.

Adding a model API must not require provider branches in orchestration. If you find or need provider-specific logic
above the boundary, raise it to the human. Get explicit confirmation that the exception is acceptable or that a
plan with a tracked issue will remove it.
