---
name: geti-ui-dev
description: Develop and validate changes in `application/ui/` for the React and TypeScript frontend. Use when touching `application/ui/src/**`, frontend tests, RSBuild or Rstest config, Playwright setup, package scripts, or generated API typings under `src/api`. Helps with Node and npm requirements, install and build commands, lint, typecheck, test workflows, and coordination with backend OpenAPI changes.
---

# Geti UI Development

> For the full architecture reference (feature-folder layout, data fetching,
> and generated API types) read the `application/ui/AGENTS.md` file.

## Quick Start

- Work from `application/ui/`.
- Use Node `>=24.2.0` and npm `>=11.14.0`.
- Install or refresh dependencies with `npm ci`.
- Start with `npm run format:check`, `npm run lint`, `npm run cyclic-deps-check`, and `npm run type-check`.

## Workflow

1. Keep the change inside the existing UI structure under `src/` unless the task explicitly calls for build or tooling updates.
2. Use existing component, routing, testing, and styling patterns instead of introducing a new structure.
3. Regenerate API types instead of hand-editing them when the backend contract changes.
4. Escalate to component (Playwright) or e2e tests only when the change affects rendered browser behavior, i.e. prefer unit over Playwright over e2e tests.

## Verification

- Use `npm run format:check` for Prettier verification.
- Use `npm run lint`, `npm run cyclic-deps-check`, and `npm run type-check` for normal code changes.
- Use `npm run test:unit` or `npm run test:unit:coverage` for logic and component behavior covered by Rstest.
- Use `npm run test:component` or `npm run test:e2e` only when the task reaches Playwright coverage.
- Use `npm run build` before finishing broader UI changes.

## API Type Notes

- `npm run build:api` reads `src/api/openapi-spec.json` and regenerates `src/api/openapi-spec.d.ts`.
- `npm run update-spec` downloads the spec from `https://localhost:7860` (self-signed TLS, fetched with `--insecure`) and then rebuilds the TypeScript types.
- Use `$geti-openapi-sync` when backend API changes are part of the task.

## Coordination Notes

- `@geti-ui/ui` and `@geti-ui/smart-tools` are consumed as regular published npm
  dependencies (`npm install`/`npm ci`) — no clone/preinstall step, do not vendor them locally.
- Keep generated API artifacts and the consuming UI changes in the same change set when the contract changes.
