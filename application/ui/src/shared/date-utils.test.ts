// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { formatDateRangeEnd, formatDateRangeStart, formatDateTime, formatTrainingDateTime } from './date-utils';

// Dates are formatted in the local timezone, so the inputs are built from local time components
const localISOString = (year: number, month: number, day: number, hours = 0, minutes = 0) =>
    new Date(year, month - 1, day, hours, minutes).toISOString();

describe('date-utils', () => {
    describe('formatDateRangeStart', () => {
        it('formats the start of a range', () => {
            expect(formatDateRangeStart(localISOString(2026, 1, 1))).toBe('From 01/01/2026 00:00');
        });

        it('formats the time in a 24 hour cycle', () => {
            expect(formatDateRangeStart(localISOString(2026, 12, 31, 23, 59))).toBe('From 31/12/2026 23:59');
        });
    });

    describe('formatDateRangeEnd', () => {
        it('formats the end of a range', () => {
            expect(formatDateRangeEnd(localISOString(2026, 1, 31))).toBe('To 31/01/2026 00:00');
        });

        it('formats the time in a 24 hour cycle', () => {
            expect(formatDateRangeEnd(localISOString(2026, 1, 31, 23, 59))).toBe('To 31/01/2026 23:59');
        });
    });

    describe('formatDateTime', () => {
        it('formats a date and time', () => {
            expect(formatDateTime(localISOString(2025, 10, 1, 11, 7))).toBe('01 Oct 2025, 11:07 AM');
        });

        it.each([null, undefined, ''])('returns the fallback for %p', (date) => {
            expect(formatDateTime(date)).toBe('-');
        });

        it('returns the fallback for an unparsable date', () => {
            expect(formatDateTime('not a date')).toBe('-');
        });

        it('returns a custom fallback', () => {
            expect(formatDateTime(null, 'N/A')).toBe('N/A');
        });
    });

    describe('formatTrainingDateTime', () => {
        it('puts the time on a second line', () => {
            expect(formatTrainingDateTime(localISOString(2025, 10, 1, 11, 7))).toBe('01 Oct 2025\n11:07 AM');
        });

        it.each([null, undefined, '', 'not a date'])('returns a placeholder for %p', (date) => {
            expect(formatTrainingDateTime(date)).toBe('-');
        });
    });
});
