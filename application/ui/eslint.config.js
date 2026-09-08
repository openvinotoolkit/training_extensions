// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { FlatCompat } from '@eslint/eslintrc';
import js from '@eslint/js';

import sharedEslintConfig from './eslint.shared.config.js';

const filename = fileURLToPath(import.meta.url);
const dirname = path.dirname(filename);
const currentYear = new Date().getFullYear();
const allowedYears = Array.from({ length: Math.max(currentYear - 2025 + 1, 1) }, (_, index) => 2025 + index);
const compat = new FlatCompat({
    baseDirectory: dirname,
    recommendedConfig: js.configs.recommended,
    allConfig: js.configs.all,
});

const restrictedImportPaths = [
    {
        name: '@adobe/react-spectrum',
        message: 'Use component from the @geti-ui/ui package instead.',
    },
];

const restrictedImportPatterns = [
    {
        group: ['@react-spectrum'],
        message: 'Use component from the @geti-ui/ui package instead.',
    },
    {
        group: ['@react-types/*'],
        message: 'Use type from the @geti-ui/ui package instead.',
    },
    {
        group: ['@spectrum-icons'],
        message: 'Use icons from the @geti-ui/ui/icons package instead.',
    },
    {
        group: ['src/*'],
        message: 'Use relative imports instead of absolute "src/" imports.',
    },
];

// Containment rule for Tauri APIs. The bundler picks `*.tauri.{ts,tsx}` files
// for the Tauri build via `resolve.extensions`, so only those files should
// import `@tauri-apps/*`. Applied to every source file *except* tauri twins.
const tauriRestrictedImportPattern = {
    group: ['@tauri-apps/*'],
    message:
        'Import Tauri plugins only from `*.tauri.{ts,tsx}` files. Consumers should ' +
        'import the capability module (e.g. ./download-file) so the bundler can ' +
        'swap implementations per build target.',
};

// Containment rule for generated OpenAPI types. Only the client-building layer
// (src/api/**, src/query-client/**) and the shared type re-export barrel
// (src/api/shared-types.ts) may import the generated spec directly.
// Everywhere else must import domain types from `@/api/types`.
const openapiSpecRestrictedImportPattern = {
    group: ['**/openapi-spec'],
    message: 'Do not import generated OpenAPI types directly. Import them from `@/api/types` instead.',
};

// Containment rule for the shared-types re-export barrel. Only src/api/** and
// src/query-client/** may reach it with a relative import (they sit next to
// it or build the client). Everywhere else must go through the `@/api/types`
// alias so the concrete file location stays an implementation detail.
const sharedTypesRestrictedImportPattern = {
    group: ['**/shared-types'],
    message: 'Do not import `shared-types` with a relative path. Import domain types from `@/api/types` instead.',
};

const apiBarrelRestrictedImportPattern = {
    group: ['**/api/client', '**/api/fetch-sse'],
    message: 'Do not import the `@/api` barrel from within `src/api/`. Use a direct relative import instead.',
};

// Containment rule for the `@/api` barrel. Files inside src/api/ are the ones
// building that barrel, so importing it back via the alias creates a
// self-import / circular-reference risk. Use direct relative imports
// (e.g. './client', './fetch-sse') from within src/api/ instead.
const apiBarrelRestrictedImportPath = {
    name: '@/api',
    message: 'Do not import the `@/api` barrel from within `src/api/`. Use a direct relative import instead.',
};

// Forbid `isTauri()` runtime branching. Per-platform behaviour must be selected
// at build time by the bundler via `*.tauri.{ts,tsx}` file overrides — see
// src-tauri/README.md.
const isTauriRestrictedSyntax = {
    selector: "CallExpression[callee.name='isTauri']",
    message:
        'Do not branch on `isTauri()` at runtime. Add or split a capability module via a `.tauri.{ts,tsx}` twin instead.',
};

// Containment rule for the shared `src/components/` folder: it must be reached
// through the `@/components/*` alias. A relative `../components/` specifier is
// ambiguous on its own, so the trees that own a local `components/` folder are
// exempted rather than resolved.
const sharedComponentsAliasConfig = {
    files: ['src/**/*.{ts,tsx}'],
    ignores: ['src/components/**', 'src/platform/**', 'src/features/models/**'],
    rules: {
        'no-restricted-syntax': [
            'error',
            isTauriRestrictedSyntax,
            {
                selector:
                    ':matches(ImportDeclaration, ImportExpression, ExportNamedDeclaration, ExportAllDeclaration)' +
                    '[source.value=/^(\\.\\.\\/)+components\\//]',
                message:
                    'Do not import `src/components/` with a relative path. Use the `@/components/*` alias instead.',
            },
        ],
    },
};

export default [
    {
        ignores: [...sharedEslintConfig[0].ignores, 'src/api/openapi-spec.d.ts'],
    },
    ...sharedEslintConfig,
    {
        rules: {
            'no-restricted-imports': [
                'error',
                {
                    paths: restrictedImportPaths,
                    patterns: restrictedImportPatterns,
                },
            ],
            'header/header': [
                'warn',
                'line',
                [
                    {
                        pattern: ` Copyright \\(C\\) ((?:${allowedYears.join('|')})|2025-(?:${allowedYears.join('|')})) Intel Corporation`,
                        template: ` Copyright (C) 2025-${currentYear} Intel Corporation`,
                    },
                    ' SPDX-License-Identifier: Apache-2.0',
                ],
            ],
            'no-restricted-syntax': ['error', isTauriRestrictedSyntax],
        },
    },
    sharedComponentsAliasConfig,
    {
        files: ['**/*.test.ts', '**/*.test.tsx', '**/*mock*.ts', '**/*.spec.ts'],
        rules: {
            'max-len': ['off'],
        },
    },
    {
        // Every source file *except* `.tauri.{ts,tsx}` twins must not import
        // `@tauri-apps/*` directly.
        files: ['src/**/*.{ts,tsx}'],
        ignores: ['src/**/*.tauri.{ts,tsx}'],
        rules: {
            'no-restricted-imports': [
                'error',
                {
                    paths: restrictedImportPaths,
                    patterns: [...restrictedImportPatterns, tauriRestrictedImportPattern],
                },
            ],
        },
    },
    {
        files: ['src/**/*.{ts,tsx}'],
        ignores: ['src/**/*.tauri.{ts,tsx}', 'src/api/**', 'src/query-client/**'],
        rules: {
            'no-restricted-imports': [
                'error',
                {
                    paths: restrictedImportPaths,
                    patterns: [
                        ...restrictedImportPatterns,
                        tauriRestrictedImportPattern,
                        openapiSpecRestrictedImportPattern,
                        apiBarrelRestrictedImportPattern,
                        sharedTypesRestrictedImportPattern,
                    ],
                },
            ],
        },
    },
    {
        files: ['src/**/*.tauri.{ts,tsx}'],
        ignores: ['src/api/**', 'src/query-client/**'],
        rules: {
            'no-restricted-imports': [
                'error',
                {
                    paths: restrictedImportPaths,
                    patterns: [
                        ...restrictedImportPatterns,
                        openapiSpecRestrictedImportPattern,
                        apiBarrelRestrictedImportPattern,
                        sharedTypesRestrictedImportPattern,
                    ],
                },
            ],
        },
    },
    {
        // Files inside src/api/ build the `@/api` barrel; importing it back
        // from here would be a self-import through the alias. Use direct
        // relative imports to sibling modules instead (e.g. './client').
        files: ['src/api/**/*.{ts,tsx}'],
        rules: {
            'no-restricted-imports': [
                'error',
                {
                    paths: [...restrictedImportPaths, apiBarrelRestrictedImportPath],
                    patterns: [...restrictedImportPatterns, tauriRestrictedImportPattern],
                },
            ],
        },
    },
    ...compat.extends('plugin:playwright/playwright-test').map((config) => ({
        ...config,
        files: ['tests/**/*.ts'],
    })),
    {
        files: ['tests/**/*.ts'],

        rules: {
            'playwright/no-wait-for-selector': ['off'],
            'playwright/no-conditional-expect': ['off'],
            'playwright/no-standalone-expect': ['off'],
            'playwright/missing-playwright-await': ['warn'],
            'playwright/valid-expect': ['warn'],
            'playwright/no-useless-not': ['warn'],
            'playwright/no-page-pause': ['warn'],
            'playwright/prefer-to-have-length': ['warn'],
            'playwright/no-conditional-in-test': ['off'],
            'playwright/expect-expect': ['off'],
            'playwright/no-skipped-test': ['off'],
            'playwright/no-wait-for-timeout': ['off'],
            'playwright/no-nested-step': ['off'],
        },
    },
];
