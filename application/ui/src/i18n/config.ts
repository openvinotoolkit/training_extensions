// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createInstance, type i18n as I18n } from 'i18next';
import LanguageDetector from 'i18next-browser-languagedetector';
import { initReactI18next } from 'react-i18next';

import en from './locales/en.json';
import zhCN from './locales/zh-CN.json';
import zhHK from './locales/zh-HK.json';
import zhMO from './locales/zh-MO.json';
import zhTW from './locales/zh-TW.json';

const SUPPORTED_LANGUAGES = ['en', 'zh-CN', 'zh-TW', 'zh-HK', 'zh-MO'];

const resources = {
    en: { translation: en },
    'zh-CN': { translation: zhCN },
    'zh-HK': { translation: zhHK },
    'zh-MO': { translation: zhMO },
    'zh-TW': { translation: zhTW },
};

export const LANGUAGE_STORAGE_KEY = 'geti-language';

export const resolveLanguage = (language: string): string => {
    if (!language || typeof language !== 'string') return 'en';

    const tag = language.trim();
    if (tag === '') return 'en';

    try {
        const locale = new Intl.Locale(tag);

        if (locale.language === 'en') return 'en';
        if (locale.language !== 'zh') return 'en';

        const maximized = locale.maximize();

        if (maximized.script === 'Hant') {
            if (maximized.region === 'HK') return 'zh-HK';
            if (maximized.region === 'MO') return 'zh-MO';
            return 'zh-TW';
        }

        return 'zh-CN';
    } catch {
        return 'en';
    }
};

const i18next = createInstance();

if (!i18next.isInitialized) {
    void i18next
        .use(LanguageDetector)
        .use(initReactI18next)
        .init({
            resources,
            fallbackLng: {
                'zh-MO': ['zh-HK', 'zh-TW', 'en'],
                'zh-HK': ['zh-TW', 'en'],
                default: ['en'],
            },
            supportedLngs: [...SUPPORTED_LANGUAGES],
            detection: {
                order: ['localStorage', 'navigator'],
                lookupLocalStorage: LANGUAGE_STORAGE_KEY,
                caches: ['localStorage'],
                convertDetectedLanguage: (language) => resolveLanguage(language),
            },
            interpolation: { escapeValue: false },
            react: { useSuspense: false },
        });
}

export const i18n: I18n = i18next;
