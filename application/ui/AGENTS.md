# Geti UI Agent Guide

Component guide for `application/ui/` — the React + TypeScript frontend. Read this
together with the repo-wide `../../AGENTS.md` and the matching skill
`.agents/skills/geti-ui-dev/`.

## What This Component Is

- React 19 + TypeScript single-page app, bundled with RSBuild.
- Also packaged as a desktop app via Tauri (`src-tauri/`).
- Node `>=24.2.0`, npm `>=11.14.0`.
- Talks to the `geti` backend through a generated, typed API client.

## Source Layout (`src/`)

| Path                     | Responsibility                                                                                                                       |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------ |
| `index.tsx`              | App bootstrap.                                                                                                                       |
| `providers.tsx`          | Global providers (React Query, theming, routing context).                                                                            |
| `router.tsx` / `routes/` | Routing configuration and route components.                                                                                          |
| `layout.tsx`             | Top-level layout shell.                                                                                                              |
| `features/`              | Domain features: `dataset/`, `inference/`, `license/`, `models/`, `project/`. Each groups components, hooks, and tests for a domain. |
| `modules/`               | Large reusable capabilities consumed by several features (currently `annotator/`).                                                   |
| `components/`            | Shared, reusable UI components.                                                                                                      |
| `hooks/`                 | Shared React hooks.                                                                                                                  |
| `api/`                   | **Generated** OpenAPI client (`openapi-spec.json`, `openapi-spec.d.ts`) and API access helpers.                                      |
| `query-client/`          | React Query client setup.                                                                                                            |
| `shared/`                | Cross-feature, domain-agnostic utilities and types.                                                                                  |
| `constants/`             | App-wide constants.                                                                                                                  |
| `platform/`              | Platform/environment abstractions (web vs. desktop).                                                                                 |
| `test-utils/`            | Testing helpers.                                                                                                                     |
| `assets/`                | Static assets.                                                                                                                       |

## Data Fetching & API Types

- Server state goes through **React Query** (`useQuery` / `useMutation`). Never call
  `fetch` / `axios` directly from components, and do not add another data-fetching
  library.
- **API types are generated** from the backend OpenAPI spec:
    - `src/api/openapi-spec.json` — the spec.
    - `src/api/openapi-spec.d.ts` — generated TypeScript types (**never hand-edit**).
- Only `src/constants/shared-types.ts` and the client-building layer (`src/api/**`, `src/query-client/**`) may import types from `src/api/openapi-spec.d.ts` directly; everywhere else, import domain types from `src/constants/shared-types.ts`.
- Regenerate types with:
    - `npm run update-spec` — pull the spec from a running backend on `:7860`, then rebuild types.
    - `npm run build:api` — rebuild types from an existing local `openapi-spec.json`.
- When the backend contract changes, use the `geti-openapi-sync` skill and keep the
  generated spec, generated `.d.ts`, and consuming UI changes in one change set.

## Conventions

- TypeScript: no `any` — use `unknown` and narrow.
- Prefer `interface` for object shapes that do composition with `extends` instead of
  `type` intersections (`A & B`): interfaces flatten to a single object type, catch
  property conflicts, display better in errors, and let the compiler cache type
  relationships.
- Use `type` for what an interface can't express: object shapes, unions, tuples, function
  signatures, and mapped/conditional or other computed types. Give a complex or
  reused computed type a named `type` alias so the compiler can cache the result.
- Function components + hooks only. Co-locate styles as CSS Modules
  (`*.module.scss`); do not add CSS-in-JS beyond what `@geti-ui/ui` already uses.
- Layers, lowest first: foundation (`api`, `components`, `constants`, `hooks`, `i18n`, `platform`,
  `query-client`, `shared`, `test-utils`) → `modules/` → `features/` → `routes/`. Import only from
  lower layers; a module or feature never imports a sibling. Move code needed by several features down
  a layer, or compose in a route. Enforced by `import/no-restricted-paths` in `eslint.config.js`.
- New backend endpoints need a corresponding mock handler under `mocks/` — reuse
  existing handlers.

## Testing

- Unit tests sit next to source as `*.test.ts(x)` and run under Rstest.
- Component and E2E tests live under `tests/` and run under Playwright.
- Only reach for component/E2E tests when the change affects rendered browser
  behavior.

## Commands

Work from `application/ui/`. See `.github/instructions/ui.instructions.md` for the
full table.

| Task                | Command                     |
| ------------------- | --------------------------- |
| Install deps        | `npm ci`                    |
| Dev server          | `npm run start`             |
| Desktop dev (Tauri) | `npm run start:desktop`     |
| Production build    | `npm run build`             |
| Format check        | `npm run format:check`      |
| Lint                | `npm run lint`              |
| Lint + fix          | `npm run lint:fix`          |
| Cyclic-deps check   | `npm run cyclic-deps-check` |
| Type-check          | `npm run type-check`        |
| Unit tests          | `npm run test:unit`         |
| Component tests     | `npm run test:component`    |
| E2E tests           | `npm run test:e2e`          |
| Update API types    | `npm run update-spec`       |
| Rebuild API types   | `npm run build:api`         |

## Guardrails

- Do not hand-edit generated API types.
- Do not introduce another data-fetching or CSS-in-JS library.
- See `application/ui/README.md` for detailed architecture, API integration
  examples, and contributing guidelines.
