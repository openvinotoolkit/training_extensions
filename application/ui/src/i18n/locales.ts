// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Resource } from 'i18next';

import en from './locales/en.json';
import zhCN from './locales/zh-CN.json';
import zhHK from './locales/zh-HK.json';
import zhMO from './locales/zh-MO.json';
import zhTW from './locales/zh-TW.json';

export type TranslationCatalog = typeof en;

type PartialCatalog = {
    [Key in keyof TranslationCatalog]?: Partial<TranslationCatalog[Key]>;
};

export type LocaleDefinition = {
    /** BCP 47 tag used as the i18next language and as the catalog file name. */
    tag: string;
    /** Endonym shown in language pickers; never translated. */
    displayName: string;
    /** Catalogs other than the default language stay partial until they are reviewed. */
    catalog: PartialCatalog;
    /** Extra languages tried before the default language, most specific first. */
    fallbacks?: string[];
};

export const DEFAULT_LANGUAGE = 'en';

/**
 * Single source of truth for the languages the app ships. Adding a language means adding a catalog
 * file and one entry here; no resolver, fallback or registration code needs to change.
 */
export const LOCALES: readonly LocaleDefinition[] = [
    { tag: DEFAULT_LANGUAGE, displayName: 'English', catalog: en },
    { tag: 'zh-CN', displayName: '简体中文', catalog: zhCN },
    { tag: 'zh-TW', displayName: '繁體中文', catalog: zhTW },
    { tag: 'zh-HK', displayName: '繁體中文（香港）', catalog: zhHK, fallbacks: ['zh-TW'] },
    { tag: 'zh-MO', displayName: '繁體中文（澳門）', catalog: zhMO, fallbacks: ['zh-HK', 'zh-TW'] },
];

export const SUPPORTED_LANGUAGES: string[] = LOCALES.map(({ tag }) => tag);

export const resources: Resource = Object.fromEntries(
    LOCALES.map(({ tag, catalog }) => [tag, { translation: catalog }])
);

export const fallbackLng: Record<string, string[]> = {
    default: [DEFAULT_LANGUAGE],
    ...Object.fromEntries(
        LOCALES.filter(({ fallbacks }) => fallbacks !== undefined).map(({ tag, fallbacks }) => [
            tag,
            [...(fallbacks ?? []), DEFAULT_LANGUAGE],
        ])
    ),
};

type MaximizedLocale = {
    tag: string;
    language: string;
    script?: string;
    region?: string;
};

const maximize = (tag: unknown): MaximizedLocale | null => {
    try {
        const requested = String(tag).trim();
        const maximized = new Intl.Locale(requested).maximize();

        return {
            tag: requested,
            language: maximized.language,
            script: maximized.script,
            region: maximized.region,
        };
    } catch {
        return null;
    }
};

const MAXIMIZED_LOCALES = LOCALES.map(({ tag }) => maximize(tag)).filter((locale) => locale !== null);

/**
 * Maps any BCP 47 tag onto a registered language, preferring the most specific match:
 * exact tag, then language + script + region, then language + script, then language.
 * Unsupported or malformed tags resolve to the default language.
 */
export const resolveLanguage = (language: string): string => {
    const requested = maximize(language);

    if (requested === null) {
        return DEFAULT_LANGUAGE;
    }

    const exact = SUPPORTED_LANGUAGES.find((tag) => tag.toLowerCase() === requested.tag.toLowerCase());

    if (exact !== undefined) {
        return exact;
    }

    const matchers: ((candidate: MaximizedLocale) => boolean)[] = [
        ({ language: lang, script, region }) =>
            lang === requested.language && script === requested.script && region === requested.region,
        ({ language: lang, script }) => lang === requested.language && script === requested.script,
        ({ language: lang }) => lang === requested.language,
    ];

    for (const matches of matchers) {
        const candidate = MAXIMIZED_LOCALES.find(matches);

        if (candidate !== undefined) {
            return candidate.tag;
        }
    }

    return DEFAULT_LANGUAGE;
};
