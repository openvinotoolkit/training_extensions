# Geti UI

Modern React application for AI model training and inference, built with Rsbuild and TypeScript.

The Geti™ applications aim to provide a user experience and design language consistent with the main [Geti application](https://github.com/open-edge-platform/geti). To achieve this, we reuse many architectural decisions from Geti™, including the shared `@geti-ui/ui` and `@geti-ui/smart-tools` packages.

## Goals

- **Developer Experience**: Setting up a developer environment for both the UI and server should take only seconds.
- **API Adaptability**: Adapting the UI to REST API changes should require minimal effort.
- **Consistency**: Maintain a unified look, feel, and user experience across the whole Geti ecosystem through shared design language, architectural patterns, and reusable components.

## 🚀 Quick Start

### Prerequisites

- Node.js >= v24.2.0 (see `.nvmrc` — `nvm use`)
- npm >= v11.14.0

### Installation

```bash
npm install
```

The UI library and smart tools are consumed as published [`@geti-ui/ui`](https://github.com/MarkRedeman/geti-ui) and `@geti-ui/smart-tools` npm packages (regular dependencies), not clones.

#### OpenCV WASM binary

`@geti-ui/smart-tools` uses a custom-compiled `opencv.js` build for GrabCut, Intelligent Scissors, Watershed, SSIM, RITM, and Segment Anything's mask-to-polygon postprocessing. The package doesn't bundle or serve that binary itself — it must be compiled separately (Docker + Emscripten, see [its README](https://github.com/MarkRedeman/geti-ui/blob/main/packages/smart-tools/README.md)) and vendored here at `vendor/opencv/4.9.0/opencv.js`; `rsbuild.config.ts`'s `output.copy` copies it to `dist/opencv/`.

`src/features/annotator/webworkers/opencv-source.ts` calls `setOpenCVSourceUrl(...)` before any OpenCV-backed tool runs (imported for its side effect by every worker that needs it). Note: `setOpenCVSourceUrl` resolves string paths against the running app's origin (`location.origin`), not the configured asset prefix, so `ASSET_PREFIX` (e.g. `/html` in the Docker/`install.sh` production build) must be prepended manually — same handling as the ORT wasm path in `segment-anything.worker.ts`.

If `@geti-ui/smart-tools` bumps its OpenCV version, update both the vendored binary and the version segment of the `output.copy` path in `rsbuild.config.ts`.

### Development

```bash
npm start            # Start dev server at http://localhost:3000
npm run server       # Start backend (Hypercorn, HTTP/2 + TLS) at https://localhost:7860 (separate terminal, requires `uv` and `just`)
```

The backend serves HTTP/2 over TLS using a self-signed certificate, auto-generated on first run into `application/backend/data/certs/` (see `just gen-certs`). Your browser/OS may need to trust this cert, or you can accept the self-signed warning when visiting `https://localhost:7860` directly.

The dev server proxies API requests to `PUBLIC_API_BASE_URL` (set in `.env.development`, defaults to `https://localhost:7860`). The `PUBLIC_` prefix is required for Rsbuild client exposure.
While the backend is running on `localhost:7860`, use `npm run update-spec` to download a fresh OpenAPI spec and regenerate types.

```bash
npm run update-spec      # Download openapi spec and generate TypeScript types from src/api/openapi-spec.json
```

### Build

```bash
npm run build        # Production build to dist/
npm run preview      # Preview production build
```

## 📁 Project Structure

```
.
├── src/
│   ├── api/                 # OpenAPI client (openapi-fetch + openapi-react-query)
│   ├── assets/              # Images, illustrations, icons
│   ├── components/          # Reusable UI components
│   ├── constants/           # Route paths (static-path), shared domain types
│   ├── features/            # Domain modules (see below)
│   │   ├── annotator/       # Annotation tools, canvas, providers
│   │   ├── dataset/         # Data collection, gallery, media preview
│   │   ├── inference/       # Live inference, WebRTC streaming
│   │   └── project/         # Project creation, configuration
│   ├── hooks/               # Cross-feature hooks, including hooks/api/
│   ├── platform/            # Cross-platform abstractions (web vs Tauri via *.tauri.ts)
│   ├── query-client/        # QueryClient config and typed query-key system
│   ├── routes/              # Route-level components and loaders
│   ├── shared/              # Cross-feature types, providers, annotation utilities
│   ├── test-utils/          # Custom render/renderHook wrapped with providers
│   ├── index.tsx            # Application entrypoint
│   ├── layout.tsx           # Top-level layout
│   ├── providers.tsx        # Global providers (QueryClient, Theme, Router, etc.)
│   └── router.tsx           # Routing setup
├── src-tauri/               # Tauri desktop wrapper (Rust)
├── mocks/                   # MSW handlers and mock entity factories
└── tests/                   # Playwright component and E2E tests
```

### Key Directories

- **`api/`** — Type-safe API client. `$api` provides `useQuery`, `useSuspenseQuery`, `useMutation`, etc., all typed from `src/api/openapi-spec.d.ts` (auto-generated, gitignored — never edit).
- **`components/`** — Locally reusable UI primitives (use `*.component.tsx` suffix); promote mature components upstream to `@geti-ui/ui`.
- **`features/`** — Domain-driven modules. Each owns its own components, hooks, providers, and API hooks.
- **`hooks/api/`** — Custom reusable API hooks composed on top of `$api`.
- **`platform/`** — Web/Tauri-specific implementations. Files ending in `.tauri.ts` are swapped in for the desktop build.
- **`query-client/`** — TanStack Query configuration. Query keys are typed tuples `[method, path, params?]`. Global mutation handlers auto-show error toasts and auto-invalidate via `meta.invalidateQueries`.
- **`routes/`** / **`router.tsx`** — Routing setup; keep route files minimal — extract complex routes into feature modules.
- **`packages/`** - Shared libraries from Geti ecosystem

## 🔧 Common Tasks

### Update API Types

```bash
npm run build:api          # Regenerate types from existing spec
npm run update-spec        # Download fresh spec from backend, then regenerate
```

### Testing

```bash
npm run test:unit          # Unit tests (Vitest + jsdom)
npm run test:unit:watch    # Watch mode
npm run test:unit:coverage # Coverage report (v8)
npm run test:component     # Component tests (Playwright + MSW)
npm run test:e2e           # E2E tests (Playwright against real backend)
```

Run a single unit test: `npm run test:unit -- src/path/to/file.test.tsx`
Run a single Playwright spec: `npm run test:component -- tests/path/to/file.spec.ts`

### Internationalization

The UI uses i18next and react-i18next for internationalization. English is the
default, fallback, and currently the only supported language.

### Code Quality

```bash
npm run format             # Prettier (120 chars, 4-space indent, single quotes)
npm run lint               # ESLint with zero-warnings policy
npm run lint:fix           # Auto-fix
npm run cyclic-deps-check  # Detect circular dependencies (madge)
npm run type-check         # tsc --noEmit
```

## 🏗️ Architecture

### Core Pillars

The application is built on four main pillars:

1. **Build System**: Modern web tooling for React-based applications
2. **Application Architecture**: Type-safe API integration with minimal effort
3. **Testing & CI/CD**: Robust testing setup that works locally and in CI
4. **AI Algorithms**: Interactive AI with low-latency algorithms

#### Build System

- **React** - Component-based UI architecture
- **TypeScript** - Static typing for reliability and maintainability
- **Rsbuild** - Fast and robust build toolchain for bundling, optimization, and environment targeting
- **Tauri** - Cross-platform desktop app packaging
- **ESLint & Prettier** - Enforced for code consistency and best practices

#### Application Architecture

- **React Router** - Single-page application navigation with dynamic routing
- **Tanstack Query** & [`openapi-react-query`](https://openapi-ts.dev/openapi-react-query/) - Server state management and type-safe API consumption
- **@geti-ui/ui** - Shared visual components for consistent UX
- **React Context & Local State** - Local state via `useState`, shared state via `createContext`

#### Testing & CI/CD

- **Vitest** - Fast unit and integration testing
- **Playwright** - Component and end-to-end testing with shared MSW configuration for auto-mocking REST endpoints
- **Testing Library** - User-centric React component testing; follow [guiding principles](https://testing-library.com/docs/guiding-principles) for accessibility
- **MSW + OpenAPI** - Mock Service Worker with OpenAPI specs to simulate API responses in tests
- **GitHub Actions** - Automated CI/CD pipelines for building, testing, and deploying

#### Algorithms & AI

- **@geti-ui/smart-tools** - Suite of intelligent tools for advanced functionality and optimization
- **WebRTC API** - Live video feeds with prediction overlays
- **WebAssembly** - High-performance, browser-executed code for compute-intensive tasks
- **OpenCV** - Image processing and computer vision
- **ONNXRuntime** - In-browser machine learning model execution for predictive analytics

### Feature Structure

Recommended structure for new features:

```
feature-name/
├── feature-name.component.tsx
├── feature-name.module.scss
├── feature-name.test.tsx
├── feature-name-provider.component.tsx
├── api/                      # Feature-specific API hooks
└── hooks/                    # Feature-specific hooks composing API + state
```

### State Management

- **Context Providers** for domain state (annotations, zoom, visibility)
- **React Query** for server state (via `$api`)
- Local state with `useState` for UI-only concerns

### Code Conventions

- **File naming**: `*.component.tsx`, `*.hook.ts(x)`, `*-provider.component.tsx`, `*.module.scss`. Kebab-case throughout.
- **Copyright header required** on every source file (enforced by ESLint):
    ```
    // Copyright (C) 2025-2026 Intel Corporation
    // SPDX-License-Identifier: Apache-2.0
    ```
- Prefer **`type`** over `interface` (unless declaration merging is needed).
- **No direct** `@adobe/react-spectrum`, `@react-spectrum/*`, `@react-types/*`, `@spectrum-icons` imports — go through `@geti-ui/ui`.
- Import sorting handled automatically by `@ianvs/prettier-plugin-sort-imports` — do not reorder manually.
- Path aliases: `test-utils/*`, `hooks/*`, `mocks/*` (see `tsconfig.json`).

## 🤝 Contributing

Before opening a PR, run in this order (matches CI):

```bash
npm run format
npm run lint
npm run cyclic-deps-check
npm run type-check
npm run test:unit
npm run test:component
npm run build
```

Keep changes focused — one fix per PR.

### Adding a New Feature

1. Create a folder in `src/features/`
2. Add components, hooks, providers, and any feature-specific API hooks
3. Register routes in `src/router.tsx`
4. Add unit tests (`*.test.tsx`) and/or component tests (`tests/`)

## 🔗 API Integration

The app uses `openapi-fetch` with auto-generated types and custom hooks for commonly used endpoints.

### Direct API Usage (one-off calls)

```tsx
import { $api } from '../api/client';

// Query
const { data } = $api.useQuery('get', '/api/sources');

// Mutation
const mutation = $api.useMutation('post', '/api/sources');
mutation.mutate({
    body: {/* ... */},
});
```

### Custom Hooks (preferred for reusable endpoints)

Custom hooks in `src/hooks/api/` provide better ergonomics, type completion, and automatic cache invalidation:

```tsx
import { useCreateProject, useProjects } from '../hooks/api/project.hook';

const { data: projects } = useProjects();

const createProject = useCreateProject();
createProject.mutate({
    body: {
        name: 'My Project',
        task: {/* ... */},
    },
});
```

## 🖥️ Desktop App

```bash
npm run start:desktop      # tauri dev
npm run build:tauri        # Tauri-targeted production build
```

Built with [Tauri](https://tauri.app). Web vs desktop code is split via the `src/platform/` abstraction — files suffixed `.tauri.ts` are substituted in the desktop bundle (e.g. `download-file.tauri.ts`, `storage-cleanup.tauri.ts`).
More details in this [document](./src-tauri/README.md).

---

For backend setup, see `../backend/readme.md`.
