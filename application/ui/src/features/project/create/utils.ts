// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { TranslateFn } from '@/i18n';

export const generateUniqueProjectName = (existingNames: string[], t: TranslateFn): string => {
    const usedNumbers: number[] = [];

    existingNames.forEach((name) => {
        const match = name.match(/^Project #(\d+)$/);
        if (match) {
            usedNumbers.push(Number(match[1]));
        }
    });

    const nextNumber = usedNumbers.length === 0 ? 1 : Math.max(...usedNumbers) + 1;

    return t('project.create.defaultName', { number: nextNumber });
};
