// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createI18nInstance } from './config';

/** Shared instance used by the app and by non-React code that needs to translate at call time. */
export const i18n = createI18nInstance();

export { createI18nInstance, LANGUAGE_STORAGE_KEY } from './config';
export { DEFAULT_LANGUAGE, SUPPORTED_LANGUAGES } from './locales';
