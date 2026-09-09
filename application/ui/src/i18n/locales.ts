// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Resource } from 'i18next';

import en from './locales/en.json';

export const DEFAULT_LANGUAGE = 'en';

export const resources: Resource = {
    [DEFAULT_LANGUAGE]: { translation: en },
};

export const SUPPORTED_LANGUAGES = Object.keys(resources);
