// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useCallback } from 'react';

import { useDateFormatter } from '@geti-ui/ui';
import type { DateFormatterOptions } from 'react-aria';

export const DATE_TIME_FORMAT: DateFormatterOptions = {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
};
export const DATE_FORMAT: DateFormatterOptions = { day: '2-digit', month: 'short', year: 'numeric' };
export const TIME_FORMAT: DateFormatterOptions = { hour: '2-digit', minute: '2-digit' };
export const LONG_DATE_TIME_FORMAT: DateFormatterOptions = {
    day: 'numeric',
    month: 'long',
    year: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
};

/**
 * Locale-aware date formatting bound to the ThemeProvider locale. Accepts raw API values and
 * returns `null` for missing or invalid input so the caller chooses the fallback.
 */
export const useFormatDate = (options: DateFormatterOptions) => {
    const formatter = useDateFormatter(options);

    return useCallback(
        (value: string | null | undefined): string | null => {
            if (!value) return null;

            const date = new Date(value);

            return Number.isNaN(date.getTime()) ? null : formatter.format(date);
        },
        [formatter]
    );
};
