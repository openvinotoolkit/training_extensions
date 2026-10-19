// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { withRsbuildConfig } from '@rstest/adapter-rsbuild';
import { defineConfig } from '@rstest/core';

const CI = !!process.env.CI;

export default defineConfig({
    extends: withRsbuildConfig(),
    source: {
        // rsbuild.config.ts defines `process.env` as `{}`, which hides NODE_ENV=test from react-aria/spectrum
        define: { 'process.env': 'process.env' },
    },
    testEnvironment: 'jsdom',

    // This is needed to use globals like describe, expect or rs
    globals: true,
    include: ['./src/**/*.test.{ts,tsx}'],
    setupFiles: ['./src/setup-tests.ts'],

    coverage: {
        provider: 'v8',
        reporters: [CI ? 'json-summary' : 'text'],
        reportOnFailure: true,
        include: ['src/**/*.{ts,tsx}'],
    },
});
