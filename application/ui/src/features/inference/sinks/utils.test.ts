// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createI18nInstance } from '@/i18n';

import { formatRateLimit, getObjectFromFormData, rateLimitFromFormData } from './utils';

describe('getObjectFromFormData', () => {
    it('return an object mapping keys to values', () => {
        const keys = ['a', 'b', 'c'];
        const values = ['1', '2', '3'];
        expect(getObjectFromFormData(keys, values)).toEqual({ a: '1', b: '2', c: '3' });
    });

    it('skip entries with empty keys', () => {
        const keys = ['', 'b', ''];
        const values = ['1', '2', '3'];
        expect(getObjectFromFormData(keys, values)).toEqual({ b: '2' });
    });

    it('skip entries with empty values', () => {
        const keys = ['a', 'b', 'c'];
        const values = ['', '2', ''];
        expect(getObjectFromFormData(keys, values)).toEqual({ b: '2' });
    });

    it('return an empty object if all keys are empty', () => {
        const keys = ['', '', ''];
        const values = ['1', '2', '3'];
        expect(getObjectFromFormData(keys, values)).toEqual({});
    });

    it('return an empty object if all values are empty', () => {
        const keys = ['a', 'b', 'c'];
        const values = ['', '', ''];
        expect(getObjectFromFormData(keys, values)).toEqual({});
    });

    it('handle different lengths of keys and values', () => {
        const keys = ['a', 'b'];
        const values = ['1', '2', '3'];
        expect(getObjectFromFormData(keys, values)).toEqual({ a: '1', b: '2' });
    });

    it('handle keys and values with whitespace', () => {
        const keys = [' ', 'b', 'c'];
        const values = ['1', ' ', '3'];
        expect(getObjectFromFormData(keys, values)).toEqual({ c: '3' });
    });
});

describe('rateLimitFromFormData', () => {
    it('returns null when fields are missing', () => {
        const formData = new FormData();

        expect(rateLimitFromFormData(formData)).toBeNull();
    });

    it('returns null for non-numeric values', () => {
        const formData = new FormData();
        formData.set('rate_limit_samples', 'abc');
        formData.set('rate_limit_seconds', '1');

        expect(rateLimitFromFormData(formData)).toBeNull();
    });

    it('returns null for NaN values', () => {
        const formData = new FormData();
        formData.set('rate_limit_samples', 'NaN');
        formData.set('rate_limit_seconds', '1');

        expect(rateLimitFromFormData(formData)).toBeNull();
    });

    it('returns null for empty string values', () => {
        const formData = new FormData();
        formData.set('rate_limit_samples', '');
        formData.set('rate_limit_seconds', '2');

        expect(rateLimitFromFormData(formData)).toBeNull();
    });

    it('returns null when values are not finite numbers', () => {
        const formData = new FormData();
        formData.set('rate_limit_samples', 'Infinity');
        formData.set('rate_limit_seconds', '2');

        expect(rateLimitFromFormData(formData)).toBeNull();
    });

    it('returns null for non-positive values', () => {
        const formData = new FormData();
        formData.set('rate_limit_samples', '0');
        formData.set('rate_limit_seconds', '2');

        expect(rateLimitFromFormData(formData)).toBeNull();
    });

    it('returns computed rate for valid values', () => {
        const formData = new FormData();
        formData.set('rate_limit_samples', '10');
        formData.set('rate_limit_seconds', '2');

        expect(rateLimitFromFormData(formData)).toBe(5);
    });

    it('returns null for decimal comma values', () => {
        const formData = new FormData();
        formData.set('rate_limit_samples', '0,1');
        formData.set('rate_limit_seconds', '1');

        expect(rateLimitFromFormData(formData)).toBeNull();
    });

    it('returns null for invalid mixed separators', () => {
        const formData = new FormData();
        formData.set('rate_limit_samples', '1,2.3');
        formData.set('rate_limit_seconds', '1');

        expect(rateLimitFromFormData(formData)).toBeNull();
    });
});

describe('formatRateLimit', () => {
    const { t } = createI18nInstance({ lng: 'en' });

    it('returns "Not set" for nullish or non-positive values', () => {
        expect(formatRateLimit(undefined, t)).toBe('Not set');
        expect(formatRateLimit(null, t)).toBe('Not set');
        expect(formatRateLimit(0, t)).toBe('Not set');
        expect(formatRateLimit(Number.NaN, t)).toBe('Not set');
    });

    it('formats singular sample and second labels', () => {
        expect(formatRateLimit(1, t)).toBe('1 sample every 1 second');
    });

    it('formats plural samples for rates above one', () => {
        expect(formatRateLimit(2, t)).toBe('2 samples every 1 second');
    });

    it('formats rates below one as canonical ratio', () => {
        expect(formatRateLimit(0.5, t)).toBe('1 sample every 2 seconds');
    });
});
