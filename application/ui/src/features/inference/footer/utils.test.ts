// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { PipelineStatus } from '@/api/types';
import { createI18nInstance } from '@/i18n';

import { getComponentStatusMeta, getOverallStatusMeta, shouldShowPipelineHealthDetails } from './utils';

const { t } = createI18nInstance({ lng: 'en' });

const getStatus = (custom?: Partial<PipelineStatus>): PipelineStatus => ({
    status: 'ok',
    updated_at: '2026-01-01T00:00:00Z',
    message: null,
    ...custom,
});

describe('getOverallStatusMeta', () => {
    it.each([
        ['running', 'Running', 'positive'],
        ['idle', 'Idle', 'neutral'],
        ['error', 'Problems detected', 'negative'],
    ])('maps known status "%s" to label "%s" and variant "%s"', (status, label, variant) => {
        expect(getOverallStatusMeta(status, t)).toEqual({ label, variant });
    });

    it('falls back to a capitalized label and neutral variant for an unknown status', () => {
        expect(getOverallStatusMeta('mystery_state', t)).toEqual({ label: 'Mystery_state', variant: 'neutral' });
    });
});

describe('getComponentStatusMeta', () => {
    it.each([
        ['ok', 'Healthy', 'positive'],
        ['finished', 'Finished', 'info'],
        ['unavailable', 'Unavailable', 'neutral'],
        ['error', 'Error', 'negative'],
    ])('maps known status "%s" (no message) to label "%s" and variant "%s"', (status, label, variant) => {
        expect(getComponentStatusMeta(getStatus({ status, message: null }), t)).toEqual({
            label,
            variant,
            message: null,
        });
    });

    it('falls back to a capitalized label and neutral variant for an unknown status', () => {
        expect(getComponentStatusMeta(getStatus({ status: 'mystery_state', message: null }), t)).toEqual({
            label: 'Mystery_state',
            variant: 'neutral',
            message: null,
        });
    });

    it('prefers the raw message over the friendly status label when a message is present', () => {
        const component = getStatus({ status: 'error', message: 'Connection refused' });

        expect(getComponentStatusMeta(component, t)).toEqual({
            label: 'Error',
            variant: 'negative',
            message: 'Connection refused',
        });
    });

    it('falls back to the friendly "Error" label when status is error but message is null', () => {
        const component = getStatus({ status: 'error', message: null });

        expect(getComponentStatusMeta(component, t)).toEqual({ label: 'Error', variant: 'negative', message: null });
    });
});

describe('shouldShowPipelineHealthDetails', () => {
    it('returns false when components is null', () => {
        expect(shouldShowPipelineHealthDetails(null)).toBe(false);
    });

    it('returns false when components is undefined', () => {
        expect(shouldShowPipelineHealthDetails(undefined)).toBe(false);
    });

    it('returns false when no component has a message', () => {
        expect(
            shouldShowPipelineHealthDetails({
                source: getStatus({ message: null }),
                sink: getStatus({ message: null }),
                model: getStatus({ message: null }),
            })
        ).toBe(false);
    });

    it('returns false when status is error but the message is null (edge case)', () => {
        expect(
            shouldShowPipelineHealthDetails({
                source: getStatus({ status: 'error', message: null }),
                sink: getStatus({ message: null }),
                model: getStatus({ message: null }),
            })
        ).toBe(false);
    });

    it.each(['source', 'sink', 'model'] as const)('returns true when the %s component has a message', (key) => {
        expect(
            shouldShowPipelineHealthDetails({
                source: getStatus({ message: null }),
                sink: getStatus({ message: null }),
                model: getStatus({ message: null }),
                [key]: getStatus({ status: 'error', message: 'Something went wrong' }),
            })
        ).toBe(true);
    });

    it('returns true when multiple components have messages simultaneously', () => {
        expect(
            shouldShowPipelineHealthDetails({
                source: getStatus({ status: 'error', message: 'Camera disconnected' }),
                sink: getStatus({ message: null }),
                model: getStatus({ status: 'error', message: 'Model load failed' }),
            })
        ).toBe(true);
    });
});
