// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useCallback } from 'react';

import type { Label } from '@/api/types';
import { isEmpty } from 'lodash-es';

import type { Shape } from '../../../shared/types';
import { useAnnotationCommands, useAnnotations } from '../annotation-document-provider.component';
import { useSelectedAnnotations } from '../select-annotation-provider.component';

export const useAddAndSelectAnnotations = () => {
    const { addAnnotations, deleteAnnotations } = useAnnotationCommands();
    const { annotations } = useAnnotations();
    const { setSelectedAnnotations } = useSelectedAnnotations();

    const addAndSelectAnnotations = useCallback(
        (shapes: Shape[], labels: Label[]): string[] => {
            // If there is a global annotation, and we are trying to add new annotations,
            // delete that global annotation first
            const globalAnnotations = annotations.filter((annotation) => annotation.shape.type === 'full_image');

            if (!isEmpty(globalAnnotations)) {
                deleteAnnotations(globalAnnotations.map(({ id }) => id));
            }

            const newIds = addAnnotations(
                shapes,
                labels.map(({ id }) => ({ id }))
            );
            setSelectedAnnotations(new Set(newIds));

            return newIds;
        },
        [addAnnotations, annotations, deleteAnnotations, setSelectedAnnotations]
    );

    return { addAndSelectAnnotations };
};
