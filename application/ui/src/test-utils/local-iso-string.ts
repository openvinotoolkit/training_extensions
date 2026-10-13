// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * Builds an ISO string from local time components (not UTC), so date fixtures render the same
 * wall-clock time regardless of which timezone the test runs in (local dev machine vs. CI, which
 * defaults to UTC).
 */
export const localISOString = (year: number, month: number, day: number, hours = 0, minutes = 0): string =>
    new Date(year, month - 1, day, hours, minutes).toISOString();
