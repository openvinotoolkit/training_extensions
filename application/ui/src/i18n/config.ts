// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createInstance, type i18n as I18n, type InitOptions } from 'i18next';
import LanguageDetector from 'i18next-browser-languagedetector';
import { initReactI18next } from 'react-i18next';

import { DEFAULT_LANGUAGE, resources, SUPPORTED_LANGUAGES } from './locales';

export const LANGUAGE_STORAGE_KEY = 'geti-language';

/**
 * Creates a ready-to-use i18next instance. Catalogs are bundled and the detector is synchronous, so
 * initialization should complete quickly after this returns; callers should initialize i18n before
 * rendering components that call `useTranslation`.
 *
 * Passing `lng` pins the language and skips browser/storage detection, which is what tests want.
 */
export const createI18nInstance = (overrides: InitOptions = {}): I18n => {
    const instance = createInstance();

    if (overrides.lng === undefined) {
        instance.use(LanguageDetector);
    }

    void instance.use(initReactI18next).init({
        resources,
        fallbackLng: DEFAULT_LANGUAGE,
        supportedLngs: SUPPORTED_LANGUAGES,
        detection: {
            order: ['localStorage', 'navigator'],
            lookupLocalStorage: LANGUAGE_STORAGE_KEY,
            caches: ['localStorage'],
        },
        interpolation: { escapeValue: false },
        react: { useSuspense: false },
        ...overrides,
    });

    return instance;
};
