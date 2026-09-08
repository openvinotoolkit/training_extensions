// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import 'i18next';

import type en from './locales/en.json';

// Makes `t()` reject keys that are missing from the English catalog.
declare module 'i18next' {
    interface CustomTypeOptions {
        defaultNS: 'translation';
        resources: {
            translation: typeof en;
        };
    }
}
