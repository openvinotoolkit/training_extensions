// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { afterEach, describe, expect, it } from 'vitest';

import { createI18nInstance, LANGUAGE_STORAGE_KEY } from './config';
import { resources } from './locales';

const setNavigatorLanguage = (language: string) => {
    Object.defineProperty(window.navigator, 'language', { value: language, configurable: true });
    Object.defineProperty(window.navigator, 'languages', { value: [language], configurable: true });
};

describe('createI18nInstance', () => {
    afterEach(() => {
        localStorage.clear();
        setNavigatorLanguage('en-US');
    });

    beforeEach(() => {
        createI18nInstance({
            lng: 'en',
            resources: structuredClone(resources),
        });
    });

    it('is initialized and can translate synchronously', () => {
        const instance = createI18nInstance({ lng: 'en' });

        expect(instance.isInitialized).toBe(true);
        expect(instance.t('navigation.dataset')).toBe('Dataset');
    });

    it('does not touch language storage when a language is pinned', () => {
        createI18nInstance({ lng: 'en' });

        expect(localStorage.getItem(LANGUAGE_STORAGE_KEY)).toBeNull();
    });

    it('prefers the stored language over the navigator language', () => {
        localStorage.setItem(LANGUAGE_STORAGE_KEY, 'zh-TW');
        setNavigatorLanguage('zh-CN');

        expect(
            createI18nInstance({
                supportedLngs: ['en', 'zh-TW', 'zh-CN'],
            }).language
        ).toBe('zh-TW');
    });

    it.each(['en', 'en-US', 'en-GB'])('renders English for browser language %s', (language) => {
        setNavigatorLanguage(language);
        const instance = createI18nInstance();

        expect(instance.language).toBe('en');
        expect(instance.resolvedLanguage).toBe('en');
        expect(instance.t('navigation.dataset')).toBe('Dataset');
        expect(localStorage.getItem(LANGUAGE_STORAGE_KEY)).toBe('en');
    });

    it.each(['zh-CN', 'zh-TW', 'zh-Hant-HK', 'zh-MO', 'fr-FR', 'not a language', '---', ''])(
        'renders English for unsupported browser language %s',
        (language) => {
            setNavigatorLanguage(language);
            const instance = createI18nInstance();

            expect(instance.resolvedLanguage).toBe('en');
            expect(instance.t('navigation.dataset')).toBe('Dataset');
        }
    );

    it.each(['zh-CN', 'zh-TW', 'zh-HK', 'zh-MO', 'klingon', 'not a language'])(
        'renders English for unsupported stored language %s',
        (language) => {
            localStorage.setItem(LANGUAGE_STORAGE_KEY, language);
            const instance = createI18nInstance();

            expect(instance.resolvedLanguage).toBe('en');
            expect(instance.t('navigation.dataset')).toBe('Dataset');
        }
    );

    it('renders English for a pinned unsupported language', () => {
        const instance = createI18nInstance({ lng: 'zh-CN' });

        expect(instance.resolvedLanguage).toBe('en');
        expect(instance.t('navigation.dataset')).toBe('Dataset');
    });
});

describe('English fallback', () => {
    it('uses English for missing translations in a newly registered language', () => {
        const instance = createI18nInstance({
            lng: 'zh-TW',
            supportedLngs: ['en', 'zh-TW'],
            resources: {
                en: resources.en,
                'zh-TW': { translation: { navigation: { dataset: 'TW dataset' } } },
            },
        });

        expect(instance.t('navigation.dataset')).toBe('TW dataset');
        expect(instance.t('navigation.models')).toBe('Models');
    });
});
