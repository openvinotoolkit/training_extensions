// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { act, render, renderHook, screen } from '@testing-library/react';
import type { i18n as I18n } from 'i18next';
import { I18nextProvider, useTranslation } from 'react-i18next';

import { createI18nInstance } from './config';
import { resources } from './locales';
import { useDocumentLanguage } from './use-document-language.hook';

// A test-only second locale; the production Chinese catalogs are intentionally still empty.
const createIsolatedInstance = (language: string): I18n =>
    createI18nInstance({
        lng: language,
        supportedLngs: ['en', 'zh-TW'],
        resources: {
            en: structuredClone(resources.en),
            'zh-TW': { translation: { navigation: { dataset: '資料集' } } },
        },
    });

const wrapper = (instance: I18n) =>
    function Wrapper({ children }: { children: React.ReactNode }) {
        return <I18nextProvider i18n={instance}>{children}</I18nextProvider>;
    };

const setNavigatorLanguage = (language: string) => {
    Object.defineProperty(window.navigator, 'language', { value: language, configurable: true });
};

describe('useDocumentLanguage', () => {
    beforeEach(() => {
        setNavigatorLanguage('en-US');
    });

    afterEach(() => {
        setNavigatorLanguage('en-US');
    });

    it('sets the document language and direction from the rendered language', () => {
        const instance = createIsolatedInstance('en');

        renderHook(() => useDocumentLanguage(), { wrapper: wrapper(instance) });

        expect(document.documentElement.lang).toBe('en');
        expect(document.documentElement.dir).toBe('ltr');
    });

    it('keeps the regional formatting locale when it matches the rendered language', () => {
        setNavigatorLanguage('en-GB');
        const instance = createIsolatedInstance('en');

        const { result } = renderHook(() => useDocumentLanguage(), { wrapper: wrapper(instance) });

        expect(result.current).toBe('en-GB');
    });

    it('ignores the regional formatting locale of another language', () => {
        setNavigatorLanguage('en-GB');
        const instance = createIsolatedInstance('zh-TW');

        const { result } = renderHook(() => useDocumentLanguage(), { wrapper: wrapper(instance) });

        expect(result.current).toBe('zh-TW');
    });

    it('follows a language change without remounting the tree', async () => {
        const instance = createIsolatedInstance('en');

        const Representative = () => {
            const { t } = useTranslation();
            const locale = useDocumentLanguage();

            return (
                <>
                    <h1>{t('navigation.dataset')}</h1>
                    <span data-testid='locale'>{locale}</span>
                </>
            );
        };

        render(<Representative />, { wrapper: wrapper(instance) });

        const heading = screen.getByRole('heading');
        expect(heading).toHaveTextContent('Dataset');
        expect(document.documentElement.lang).toBe('en');
        expect(screen.getByTestId('locale')).toHaveTextContent(/^en-US$/);

        await act(async () => {
            await instance.changeLanguage('zh-TW');
        });

        // Same DOM node: the subtree re-rendered instead of remounting.
        expect(screen.getByRole('heading')).toBe(heading);
        expect(heading).toHaveTextContent('資料集');
        expect(document.documentElement.lang).toBe('zh-TW');
        expect(screen.getByTestId('locale')).toHaveTextContent(/^zh-TW$/);
    });
});
