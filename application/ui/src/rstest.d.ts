// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// `@testing-library/jest-dom` matchers are registered at runtime via `expect.extend` (see setup-tests.ts).
// Type them by augmenting Rstest's `Assertion`, as documented in
// https://rstest.rs/api/runtime-api/test-api/expect.md (custom matchers typing).

import type { TestingLibraryMatchers } from '@testing-library/jest-dom/matchers';

/* eslint-disable @typescript-eslint/no-unused-vars, @typescript-eslint/no-empty-object-type */
declare module '@rstest/core' {
    interface Assertion<T> extends TestingLibraryMatchers<unknown, void> {}
}
/* eslint-enable @typescript-eslint/no-unused-vars, @typescript-eslint/no-empty-object-type */
