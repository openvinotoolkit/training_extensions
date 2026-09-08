// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createI18nInstance } from '../i18n/config';

/**
 * Deterministic English instance for unit tests: pinning `lng` skips browser detection, so tests
 * never read or write the language storage key. Imported by setup-tests so it becomes the default
 * instance for every `useTranslation` call.
 */
export const testI18n = createI18nInstance({ lng: 'en' });
