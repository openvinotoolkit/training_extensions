// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useHotkeys } from 'react-hotkeys-hook';

import { HOTKEYS } from '../../../shared/hotkeys-definition';
import { Annotation } from '../../../shared/types';

export type UseSelectNextAnnotationProps = {
    annotations: Annotation[];
    selectedAnnotationsIds: Set<string>;
    updateSelectedAnnotationsIds: (ids: Set<string>) => void;
};

export const useSelectNextAnnotation = ({
    annotations,
    selectedAnnotationsIds,
    updateSelectedAnnotationsIds,
}: UseSelectNextAnnotationProps) => {
    useHotkeys(
        HOTKEYS.selectNextAnnotation,
        (event) => {
            const selectedAnnotationIdx = annotations.findIndex((annotation) =>
                selectedAnnotationsIds.has(annotation.id)
            );

            if (selectedAnnotationIdx < 0) {
                return;
            }

            event.preventDefault();

            const nextAnnotationIdx = (selectedAnnotationIdx + 1) % annotations.length;
            updateSelectedAnnotationsIds(new Set([annotations[nextAnnotationIdx].id]));
        },
        [annotations, selectedAnnotationsIds, updateSelectedAnnotationsIds],
        {
            enabled: selectedAnnotationsIds.size === 1,
        }
    );
};
