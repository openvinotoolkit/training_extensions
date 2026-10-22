// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Resource } from 'i18next';

import en from './locales/en.json';
import es from './locales/es.json';
import fr from './locales/fr.json';
import it from './locales/it.json';
import nl from './locales/nl.json';
import pl from './locales/pl.json';
import pt from './locales/pt.json';
import ru from './locales/ru.json';
import zhCN from './locales/zh-CN.json';
import zhHK from './locales/zh-HK.json';
import zhMO from './locales/zh-MO.json';
import zhTW from './locales/zh-TW.json';

export const DEFAULT_LANGUAGE = 'en';
type Catalog = typeof en;

export const resources: Resource = {
    [DEFAULT_LANGUAGE]: { translation: en },
    es: { translation: es satisfies Catalog },
    fr: { translation: fr satisfies Catalog },
    it: { translation: it satisfies Catalog },
    nl: { translation: nl satisfies Catalog },
    pl: { translation: pl satisfies Catalog },
    pt: { translation: pt satisfies Catalog },
    ru: { translation: ru satisfies Catalog },
    'zh-CN': { translation: zhCN satisfies Catalog },
    'zh-HK': { translation: zhHK satisfies Catalog },
    'zh-MO': { translation: zhMO satisfies Catalog },
    'zh-TW': { translation: zhTW satisfies Catalog },
};

export const SUPPORTED_LANGUAGES = Object.keys(resources);
