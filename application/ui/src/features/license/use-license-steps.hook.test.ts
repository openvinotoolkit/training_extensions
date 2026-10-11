// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { act, renderHook } from '@testing-library/react';

import type { LicenseNotice } from './license-notices';
import { useLicenseSteps } from './use-license-steps.hook';

const getLicense = (id: string, isRequired: boolean): LicenseNotice => ({
    id,
    name: id,
    notice: '',
    href: '',
    isRequired,
});

const licenses = [getLicense('a', true), getLicense('b', true), getLicense('c', false)];

describe('useLicenseSteps', () => {
    it('only lets the user reach the first pending required license', () => {
        const { result } = renderHook(() => useLicenseSteps(licenses));

        expect(result.current.lastReachableIndex).toBe(0);
        expect(result.current.canProceed).toBe(false);

        act(() => result.current.setCurrentAgreed(true));

        expect(result.current.canProceed).toBe(true);
        expect(result.current.lastReachableIndex).toBe(1);
    });

    it('does not require agreeing to optional licenses', () => {
        const { result } = renderHook(() => useLicenseSteps(licenses));

        act(() => result.current.setCurrentAgreed(true));
        act(() => result.current.goToNext());
        act(() => result.current.setCurrentAgreed(true));
        act(() => result.current.goToNext());

        expect(result.current.isLast).toBe(true);
        expect(result.current.canProceed).toBe(true);
        expect(result.current.areAllRequiredAgreed).toBe(true);
        expect(result.current.lastReachableIndex).toBe(2);
    });

    it('blocks acceptance again when a required license is unchecked', () => {
        const { result } = renderHook(() => useLicenseSteps(licenses));

        act(() => result.current.setCurrentAgreed(true));
        act(() => result.current.goToNext());
        act(() => result.current.setCurrentAgreed(true));
        act(() => result.current.goTo(0));
        act(() => result.current.setCurrentAgreed(false));

        expect(result.current.areAllRequiredAgreed).toBe(false);
        expect(result.current.lastReachableIndex).toBe(0);
    });
});
