// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createI18nInstance } from '@/i18n';

import { generateUniqueProjectName } from './utils';

describe('generateUniqueProjectName', () => {
    const { t } = createI18nInstance({ lng: 'en' });

    it('returns "Project #1" when list is empty', () => {
        const result = generateUniqueProjectName([], t);
        expect(result).toBe('Project #1');
    });

    it('increments the highest existing project number', () => {
        const result = generateUniqueProjectName(['Project #1', 'Project #2'], t);
        expect(result).toBe('Project #3');
    });

    it('ignores non-matching names', () => {
        const result = generateUniqueProjectName(['Alpha', 'Beta'], t);
        expect(result).toBe('Project #1');
    });

    it('handles mixed names and picks max + 1', () => {
        const result = generateUniqueProjectName(['Project #1', 'Alpha', 'Project #3'], t);
        expect(result).toBe('Project #4');
    });

    it('handles duplicate numbers correctly', () => {
        const result = generateUniqueProjectName(['Project #2', 'Project #2'], t);
        expect(result).toBe('Project #3');
    });

    it('handles large numbers', () => {
        const result = generateUniqueProjectName(['Project #999'], t);
        expect(result).toBe('Project #1000');
    });
});
