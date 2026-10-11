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

    it('is initialized and can translate synchronously', () => {
        const instance = createI18nInstance({ lng: 'en' });

        expect(instance.isInitialized).toBe(true);
        expect(instance.t('common.labels.dataset')).toBe('Dataset');
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
        expect(instance.t('common.labels.dataset')).toBe('Dataset');
        expect(localStorage.getItem(LANGUAGE_STORAGE_KEY)).toBe('en');
    });

    it.each(['fr-FR', 'not a language', '---', ''])(
        'renders English for unsupported browser language %s',
        (language) => {
            setNavigatorLanguage(language);
            const instance = createI18nInstance();

            expect(instance.resolvedLanguage).toBe('en');
            expect(instance.t('common.labels.dataset')).toBe('Dataset');
        }
    );

    it.each(['zh', 'zh-CN', 'zh-Hant-HK'])('renders Simplified Chinese for browser language %s', (language) => {
        setNavigatorLanguage(language);
        const instance = createI18nInstance();

        expect(instance.resolvedLanguage).toBe('zh-CN');
        expect(instance.t('common.labels.dataset')).toBe('数据集');
    });

    it.each(['zh-TW'])('renders Traditional Chinese (Taiwan) for browser language %s', (language) => {
        setNavigatorLanguage(language);
        const instance = createI18nInstance();

        expect(instance.resolvedLanguage).toBe('zh-TW');
        expect(instance.t('common.labels.dataset')).toBe('資料集');
    });

    it.each(['zh-HK'])('renders Traditional Chinese (Hong Kong) for browser language %s', (language) => {
        setNavigatorLanguage(language);
        const instance = createI18nInstance();

        expect(instance.resolvedLanguage).toBe('zh-HK');
        expect(instance.t('common.labels.dataset')).toBe('數據集');
    });

    it.each(['zh-MO'])('renders Traditional Chinese (Macau) for browser language %s', (language) => {
        setNavigatorLanguage(language);
        const instance = createI18nInstance();

        expect(instance.resolvedLanguage).toBe('zh-MO');
        expect(instance.t('common.labels.dataset')).toBe('數據集');
    });

    it.each(['klingon', 'not a language'])('renders English for unsupported stored language %s', (language) => {
        localStorage.setItem(LANGUAGE_STORAGE_KEY, language);
        const instance = createI18nInstance();

        expect(instance.resolvedLanguage).toBe('en');
        expect(instance.t('common.labels.dataset')).toBe('Dataset');
    });

    it('renders the stored Simplified Chinese language', () => {
        localStorage.setItem(LANGUAGE_STORAGE_KEY, 'zh-CN');
        const instance = createI18nInstance();

        expect(instance.resolvedLanguage).toBe('zh-CN');
        expect(instance.t('common.labels.dataset')).toBe('数据集');
    });

    it('renders English for a pinned unsupported language', () => {
        const instance = createI18nInstance({ lng: 'fr-FR' });

        expect(instance.resolvedLanguage).toBe('en');
        expect(instance.t('common.labels.dataset')).toBe('Dataset');
    });
});

describe('English fallback', () => {
    it('uses English for missing translations in a newly registered language', () => {
        const instance = createI18nInstance({
            lng: 'zh-TW',
            supportedLngs: ['en', 'zh-TW'],
            resources: {
                en: resources.en,
                'zh-TW': { translation: { common: { labels: { dataset: 'TW dataset' } } } },
            },
        });

        expect(instance.t('common.labels.dataset')).toBe('TW dataset');
        expect(instance.t('navigation.models')).toBe('Models');
    });
});
