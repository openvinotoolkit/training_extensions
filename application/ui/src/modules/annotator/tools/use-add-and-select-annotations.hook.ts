// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useCallback } from 'react';

import type { Label } from '@/api/types';

import type { Shape } from '../../../shared/types';
import { useAnnotationCommands } from '../annotation-document-provider.component';
import { useSelectedAnnotations } from '../select-annotation-provider.component';

export const useAddAndSelectAnnotations = () => {
    const { addAnnotations } = useAnnotationCommands();
    const { setSelectedAnnotations } = useSelectedAnnotations();

    const addAndSelectAnnotations = useCallback(
        (shapes: Shape[], labels: Label[]): string[] => {
            const newIds = addAnnotations(
                shapes,
                labels.map(({ id }) => ({ id }))
            );
            setSelectedAnnotations(new Set(newIds));

            return newIds;
        },
        [addAnnotations, setSelectedAnnotations]
    );

    return { addAndSelectAnnotations };
};
