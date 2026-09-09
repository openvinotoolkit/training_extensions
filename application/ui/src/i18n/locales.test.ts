// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { DEFAULT_LANGUAGE, resources, SUPPORTED_LANGUAGES } from './locales';

describe('locale registry', () => {
    it('registers a translation resource for every locale', () => {
        expect(Object.keys(resources)).toEqual(SUPPORTED_LANGUAGES);
        SUPPORTED_LANGUAGES.forEach((tag) => {
            expect(resources[tag]).toHaveProperty('translation');
        });
    });

    it('ships English as the default and only supported language', () => {
        expect(DEFAULT_LANGUAGE).toBe('en');
        expect(SUPPORTED_LANGUAGES).toEqual(['en']);
        expect(resources.en.translation).toHaveProperty('navigation.dataset', 'Dataset');
    });
});
