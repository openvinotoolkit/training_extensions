// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect } from 'react';

import { useTranslation } from 'react-i18next';

// Keep the user's regional formatting (e.g. en-GB dates) when it belongs to the language we render.
const getFormattingLocale = (language: string): string => {
    const systemLocale = typeof navigator === 'undefined' ? '' : navigator.language;

    return systemLocale.toLowerCase().startsWith(language.toLowerCase()) ? systemLocale : language;
};

/**
 * Keeps `<html lang>` and `<html dir>` in sync with the language i18next actually renders, and
 * returns the locale to use for date and number formatting.
 */
export const useDocumentLanguage = (): string => {
    const { i18n } = useTranslation();
    const language = i18n.resolvedLanguage ?? i18n.language;

    useEffect(() => {
        document.documentElement.lang = language;
        document.documentElement.dir = i18n.dir(language);
    }, [i18n, language]);

    return getFormattingLocale(language);
};
