// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState, type ReactNode } from 'react';

import { Button, ThemeProvider } from '@geti-ui/ui';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { localISOString } from 'test-utils/local-iso-string';
import { render, renderHook } from 'test-utils/render';

import { DATE_TIME_FORMAT, useFormatDate } from './use-format-date.hook';

const localISOStringNoOffset = (year: number, month: number, day: number, hours = 0, minutes = 0) => {
    const pad = (n: number) => String(n).padStart(2, '0');

    return `${year}-${pad(month)}-${pad(day)}T${pad(hours)}:${pad(minutes)}:00`;
};

const withLocale = (locale: string) => {
    const wrapper = ({ children }: { children: ReactNode }) => (
        <ThemeProvider locale={locale}>{children}</ThemeProvider>
    );

    return wrapper;
};

describe('useFormatDate', () => {
    it.each([null, undefined, '', 'not a date'])('returns null for %p', (value) => {
        const { result } = renderHook(() => useFormatDate(DATE_TIME_FORMAT));

        expect(result.current(value)).toBeNull();
    });

    it('formats an ISO string with a Z offset and one without an offset identically (local time)', () => {
        const { result } = renderHook(() => useFormatDate(DATE_TIME_FORMAT));

        const withZ = result.current(localISOString(2025, 10, 1, 11, 7));
        const withoutOffset = result.current(localISOStringNoOffset(2025, 10, 1, 11, 7));

        expect(withZ).toBe(withoutOffset);
    });

    it('formats en-US', () => {
        const { result } = renderHook(() => useFormatDate(DATE_TIME_FORMAT), { wrapper: withLocale('en-US') });

        expect(result.current(localISOString(2025, 10, 1, 11, 7))).toBe('Oct 01, 2025, 11:07 AM');
    });

    it('formats fr-FR', () => {
        const { result } = renderHook(() => useFormatDate(DATE_TIME_FORMAT), { wrapper: withLocale('fr-FR') });

        expect(result.current(localISOString(2025, 10, 1, 23, 7))).toBe('01 oct. 2025, 23:07');
    });

    it('re-renders with a different output when the locale changes', async () => {
        const date = localISOString(2025, 10, 1, 11, 7);

        const Probe = () => {
            const formatDate = useFormatDate(DATE_TIME_FORMAT);

            return <span>{formatDate(date)}</span>;
        };

        const Harness = () => {
            const [locale, setLocale] = useState('en-US');

            return (
                <ThemeProvider locale={locale}>
                    <Probe />
                    <Button onPress={() => setLocale('fr-FR')}>switch</Button>
                </ThemeProvider>
            );
        };

        render(<Harness />);

        expect(screen.getByText('Oct 01, 2025, 11:07 AM')).toBeInTheDocument();

        await userEvent.click(screen.getByRole('button', { name: 'switch' }));

        expect(screen.getByText('01 oct. 2025, 11:07')).toBeInTheDocument();
    });
});
