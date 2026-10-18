// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Media } from '@/api/types';

import { useAnnotationCommands } from '../../../modules/annotator/annotation-document-provider.component';
import { useSelectedAnnotations } from '../../../modules/annotator/select-annotation-provider.component';
import { useSelectedMediaItem } from '../../../modules/annotator/selected-media-item-provider.component';

type UseAnnotatorMediaTransitionProps = {
    onSelectedMediaItem: (item: Media) => void;
};
export const useAnnotatorMediaTransition = ({ onSelectedMediaItem }: UseAnnotatorMediaTransitionProps) => {
    const { setMediaItem } = useSelectedMediaItem();
    const { setSelectedAnnotations } = useSelectedAnnotations();
    const { resetAnnotations } = useAnnotationCommands();

    return (item: Media) => {
        setSelectedAnnotations(new Set());
        resetAnnotations();
        setMediaItem(item);
        onSelectedMediaItem(item);
    };
};
