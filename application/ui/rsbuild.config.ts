// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { defineConfig, loadEnv } from '@rsbuild/core';
import { pluginBabel } from '@rsbuild/plugin-babel';
import { pluginReact } from '@rsbuild/plugin-react';
import { pluginSass } from '@rsbuild/plugin-sass';
import { pluginSvgr } from '@rsbuild/plugin-svgr';

const { publicVars } = loadEnv({ prefixes: ['PUBLIC_'] });

// Platform target selection. When building for the Tauri desktop shell we
// prepend `.tauri.*` extensions so the bundler resolves platform-specific
// overrides (e.g. `foo.tauri.ts` wins over `foo.ts`). Files not shadowed by
// a `.tauri.*` twin resolve as usual. This keeps Tauri-specific code out of
// the web graph entirely, and removes the need for runtime `isTauri` checks.
const isTauriBuild = process.env.BUILD_TARGET === 'tauri';

// `TAURI_ENV_DEBUG` is set by the Tauri CLI: `tauri dev` / `start:desktop`
// propagate it as `true`, and `tauri build` sets it to `false`. We disable
// minification and emit inline JS source maps for debug desktop builds so
// stack traces are readable inside the embedded WebView.
const isTauriDebugBuild = isTauriBuild && process.env.TAURI_ENV_DEBUG === 'true';

const platformExtensions = isTauriBuild ? ['.tauri.tsx', '.tauri.ts', '.tauri.jsx', '.tauri.js', '.tauri.scss'] : [];
// `.scss` is appended unconditionally so extensionless SCSS imports (used
// to opt in to the platform-override mechanism, e.g. `import './foo'`)
// still resolve to `foo.scss` on the web build.
const styleExtensions = ['.scss'];

// API base URL injected into the bundle. Web builds resolve to '' (relative
// paths, served same-origin behind a reverse proxy). Tauri builds load the
// frontend from a custom protocol (`tauri://localhost`) that has no API, so
// fetches must point at the absolute backend URL — the sidecar always binds
// to localhost:7860. The web dev server reads the same value from .env.development.
const getPublicApiUrl = () => {
    if (publicVars['import.meta.env.PUBLIC_API_BASE_URL'] !== undefined) {
        return JSON.parse(publicVars['import.meta.env.PUBLIC_API_BASE_URL']);
    }

    if (isTauriBuild) {
        return 'https://localhost:7860';
    }

    return '';
};
const publicApiUrl = getPublicApiUrl();
const publicApiUrlJson = JSON.stringify(publicApiUrl);

// Static assets (including the `onnxruntime-web` wasm/mjs artifacts copied
// below) are served from wherever the backend mounts the built UI — the
// origin root in dev, but under `ASSET_PREFIX` (e.g. `/html`) in production
// (see application/docker/Dockerfile, install.sh, and
// application/backend/app/main.py's `static_dir` mount). Consumers that build
// absolute asset URLs at runtime (e.g. the SAM worker's `setOrtWasmPaths`)
// must prefix them with this value, or they 404 behind the `/html` mount.
//
// rsbuild's OWN `output.assetPrefix` must never be an empty string: rsbuild
// treats `''` as "emit relative asset paths" in index.html (e.g.
// `static/js/index.js` instead of `/static/js/index.js`). Relative paths
// resolve fine at `/`, but break on any deep client-side route (e.g.
// `/projects/:id/dataset/:itemId`) because the browser resolves them against
// the current URL path, 404s, and gets the SPA's index.html fallback back
// instead of the script — `SyntaxError: Unexpected token '<'` — which blanks
// the whole app. Fall back to the root-absolute `/` when no explicit prefix
// is configured.
const assetPrefix = process.env.ASSET_PREFIX ?? '/';

// Runtime consumers (opencv-source.ts, segment-anything.worker.ts) prepend
// this value to an already-leading-slash path (`` + `/opencv/opencv.js``), so
// the trailing slash must be stripped: `'/' + '/opencv/opencv.js'` yields the
// protocol-relative URL `//opencv/opencv.js`, which the browser resolves to
// the host `opencv` and the CSP `connect-src 'self'` then blocks.
const runtimeAssetPrefixJson = JSON.stringify(assetPrefix.replace(/\/+$/, ''));

export default defineConfig({
    plugins: [
        pluginReact(),

        // React Compiler
        pluginBabel({
            include: /\.(?:tsx)$/,
            babelLoaderOptions(opts) {
                opts.plugins?.unshift('babel-plugin-react-compiler');
            },
        }),

        pluginSass(),

        pluginSvgr({
            parallel: true,
            svgrOptions: {
                exportType: 'named',
            },
        }),
    ],
    output: {
        assetPrefix,
        distPath: { root: isTauriBuild ? 'dist-tauri' : 'dist' },
        minify: isTauriDebugBuild ? false : undefined,
        sourceMap: isTauriDebugBuild
            ? {
                  js: 'inline-source-map',
                  css: false,
              }
            : undefined,
        copy: [
            // Only the `jsep` build is ever fetched at runtime: it is the sole variant
            // referenced by onnxruntime-web's browser entrypoints, and it serves both
            // execution providers smart-tools can select (`webgpu` and `cpu`). Copying the
            // whole `dist/*.{wasm,mjs}` glob shipped ~59 MB of never-requested binaries
            // (`.jspi`, `.asyncify`, the plain CPU build, and every `ort.*.mjs` entrypoint).
            //
            // These binaries must come from the same `onnxruntime-web` copy the bundler
            // links against, or the JS glue and the wasm disagree at runtime. Keep our pin
            // equal to `@geti-ui/smart-tools`' own pin so npm hoists a single install.
            {
                from: 'node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.{wasm,mjs}',
                to: 'ort/[name][ext]',
            },
            { from: 'vendor/opencv/4.9.0/opencv.js', to: 'opencv/[name][ext]' },
        ],
    },
    source: {
        define: {
            ...publicVars,
            'import.meta.env.PUBLIC_API_BASE_URL': publicApiUrlJson,
            'process.env.PUBLIC_API_BASE_URL': publicApiUrlJson,
            'process.env.ASSET_PREFIX': runtimeAssetPrefixJson,
            // Needed to prevent an issue with spectrum's picker
            // eslint-disable-next-line max-len
            // https://github.com/adobe/react-spectrum/blob/6173beb4dad153aef74fc81575fd97f8afcf6cb3/packages/%40react-spectrum/overlays/src/OpenTransition.tsx#L40
            'process.env': {},
        },
    },
    html: {
        template: './public/index.html',
        title: 'Geti™',
        favicon: './src/assets/icons/favicon.ico',
        meta: {
            description:
                'Geti™ provides a "recipe" for every supported task type, which consolidates ' +
                'necessary information to build a model. Model templates are validated on ' +
                'various datasets and serve as a one-stop shop for obtaining the best models in general.',
        },
    },
    performance: {
        preload: {
            type: 'initial',
            include: [
                /inter-v20-latin-variable.*\.woff2$/,
                // The branded loading spinner is the LCP element on the initial
                // route (it's rendered by the root <Suspense> fallback while the
                // route chunk loads). Without a preload, the browser can't
                // discover its URL until ~2 MB of JS parses and React mounts,
                // pushing LCP to ~4 s. Preloading shrinks resourceLoadDelay
                // dramatically and lets the spinner paint near FCP.
                /intel-loading\..*\.webp$/,
            ],
        },
    },
    tools: {
        rspack: (config) => {
            // `resolve.extensions` is order-sensitive: the first match wins.
            // Rsbuild's defaults put `.ts` near the front, so a plain object
            // merge would let it shadow our `.tauri.ts` overrides. Prepend
            // explicitly and dedupe to keep the platform suffixes first.
            const existing = config.resolve?.extensions ?? [];
            const extensions = Array.from(new Set([...platformExtensions, ...existing, ...styleExtensions]));

            // onnxruntime-web's default browser export inlines its ~26 MB wasm binary as a
            // bundler asset. We already serve that binary from `ort/` (see `output.copy`) and
            // point ORT at it via `setOrtWasmPaths`, so the inlined copy is emitted but never
            // fetched. This export condition selects ORT's external-wasm entry instead.
            // `'...'` must be kept: it expands to rspack's own defaults (`webpack`,
            // `production`, `browser`, …), and dropping them breaks every package whose
            // `exports` map is keyed on `browser`.
            const conditionNames = ['onnxruntime-web-use-extern-wasm', '...'];

            // `@geti-ui/smart-tools` still builds both model URLs eagerly at module scope
            // (`new URL('./mobile_sam.encoder.onnx', import.meta.url)`), which makes rspack
            // emit the 27 MB encoder regardless. `emit: false` keeps the reference
            // resolvable while dropping the binary from the output.
            const rules = [
                ...(config.module?.rules ?? []),
                {
                    test: /[\\/]@geti-ui[\\/]smart-tools[\\/].*mobile_sam\.encoder\.onnx$/,
                    type: 'asset/resource' as const,
                    generator: { emit: false },
                },
            ];

            return {
                ...config,
                module: { ...config.module, rules },
                resolve: { ...config.resolve, extensions, conditionNames },
                watchOptions: { ...config.watchOptions, ignored: ['**/src-tauri/**'] },
            };
        },
    },
    server: {
        headers: {
            'Cross-Origin-Embedder-Policy': 'credentialless',
            'Cross-Origin-Opener-Policy': 'same-origin',
            // Must stay `no-cache` (revalidate via ETag). Dev assets — including the
            // HMR client and async chunks — are not content-hashed, so an `immutable`
            // policy makes the browser serve a stale HMR client that can no longer
            // apply updates. It then falls back to a full reload, which loads the same
            // stale client again: an infinite reload loop. Production assets are hashed
            // and served with their own cache headers by the backend/Tauri, so this
            // only affects the local rsbuild dev/preview server.
            'Cache-Control': 'no-cache',
            'Content-Security-Policy':
                "default-src 'self'; " +
                "script-src 'self' 'unsafe-eval' blob:; " +
                "worker-src 'self' blob:; " +
                `connect-src 'self' ${publicApiUrl} data:; ` +
                `img-src 'self' ${publicApiUrl} data: blob:; ` +
                `media-src 'self' ${publicApiUrl} blob: data:; ` +
                "style-src 'self' 'unsafe-inline';",
        },
    },
});
