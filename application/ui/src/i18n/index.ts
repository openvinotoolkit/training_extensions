// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { TFunction } from 'i18next';

import { createI18nInstance } from './config';

/** Shared instance used by the app and by non-React code that needs to translate at call time. */
export const i18n = createI18nInstance();

/**
 * Single entry point for the translation engine. Everything outside `src/i18n/` imports from here
 * (enforced by `no-restricted-imports`) so swapping the engine stays contained to this folder.
 */
export { Trans, useTranslation } from 'react-i18next';

/** Engine-agnostic alias for the translate function, for modules that take `t` as an argument. */
export type TranslateFn = TFunction;

export { createI18nInstance, LANGUAGE_STORAGE_KEY } from './config';
export { DEFAULT_LANGUAGE, SUPPORTED_LANGUAGES } from './locales';
