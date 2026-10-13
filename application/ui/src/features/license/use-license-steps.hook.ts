// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import type { LicenseNotice } from './license-notices';

export const useLicenseSteps = (licenses: LicenseNotice[]) => {
    const [currentIndex, setCurrentIndex] = useState(0);
    const [agreedIds, setAgreedIds] = useState<ReadonlySet<string>>(new Set());

    const current = licenses[currentIndex];
    const isCurrentAgreed = agreedIds.has(current.id);
    const firstPendingIndex = licenses.findIndex(({ id, isRequired }) => isRequired && !agreedIds.has(id));
    const areAllRequiredAgreed = firstPendingIndex === -1;

    const goToPrevious = () => setCurrentIndex((index) => index - 1);
    const goToNext = () => setCurrentIndex((index) => index + 1);

    const setCurrentAgreed = (isAgreed: boolean) => {
        setAgreedIds((previous) => {
            const next = new Set(previous);

            if (isAgreed) {
                next.add(current.id);
            } else {
                next.delete(current.id);
            }

            return next;
        });
    };

    return {
        current,
        currentIndex,
        agreedIds,
        isFirst: currentIndex === 0,
        isLast: currentIndex === licenses.length - 1,
        isCurrentAgreed,
        canProceed: isCurrentAgreed || !current.isRequired,
        areAllRequiredAgreed,
        lastReachableIndex: areAllRequiredAgreed ? licenses.length - 1 : firstPendingIndex,
        goTo: setCurrentIndex,
        goToPrevious,
        goToNext,
        setCurrentAgreed,
    };
};
