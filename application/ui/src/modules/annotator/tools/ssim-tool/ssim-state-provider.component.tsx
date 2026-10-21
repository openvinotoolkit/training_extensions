// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, PropsWithChildren, useCallback, useContext, useEffect, useMemo, useState } from 'react';

import { useProjectTask } from 'hooks/use-project-task.hook';

import { isDetectionTask } from '../../../../shared/task-type-guards';
import { useAnnotations } from '../../annotation-document-provider.component';
import { useAnnotatorLabels } from '../../annotator-labels-provider.component';
import { useSelectedMediaItem } from '../../selected-media-item-provider.component';
import { useTool } from '../../tool-provider.component';
import { useAddAndSelectAnnotations } from '../use-add-and-select-annotations.hook';
import { useSSIM } from './use-ssim.hook';

type SSIMState = ReturnType<typeof useSSIM> & {
    hasPreview: boolean;
    accept: () => void;
    reject: () => void;
    setNumberOfItems: (numberOfItems: number) => void;
    autoMergeDuplicates: boolean;
    setAutoMergeDuplicates: (autoMergeDuplicates: boolean) => void;
};

const SSIMStateContext = createContext<SSIMState | null>(null);

export const SSIMStateProvider = ({ children }: PropsWithChildren) => {
    const shapeType = isDetectionTask(useProjectTask()) ? 'rectangle' : 'polygon';
    const ssim = useSSIM(shapeType);
    const { activeTool } = useTool();
    const { mediaItem } = useSelectedMediaItem();
    const { mode } = useAnnotations();
    const { selectedLabel } = useAnnotatorLabels();
    const { addAndSelectAnnotations } = useAddAndSelectAnnotations();
    const { reset, toolState, updateToolState, rerun } = ssim;
    const [autoMergeDuplicates, setAutoMergeDuplicatesState] = useState(true);

    useEffect(() => {
        reset();
    }, [activeTool, mediaItem, mode, reset]);

    const accept = useCallback(() => {
        addAndSelectAnnotations(toolState.shapes, selectedLabel ? [selectedLabel] : []);
        reset();
    }, [addAndSelectAnnotations, toolState.shapes, selectedLabel, reset]);

    const setNumberOfItems = useCallback(
        (numberOfItems: number) => updateToolState({ threshold: numberOfItems }),
        [updateToolState]
    );

    const setAutoMergeDuplicates = useCallback(
        (value: boolean) => {
            setAutoMergeDuplicatesState(value);
            rerun({ autoMergeDuplicates: value });
        },
        [rerun]
    );

    const value = useMemo<SSIMState>(
        () => ({
            ...ssim,
            hasPreview: toolState.shapes.length > 0,
            accept,
            reject: reset,
            setNumberOfItems,
            autoMergeDuplicates,
            setAutoMergeDuplicates,
        }),
        [ssim, toolState.shapes, accept, reset, setNumberOfItems, autoMergeDuplicates, setAutoMergeDuplicates]
    );

    return <SSIMStateContext.Provider value={value}>{children}</SSIMStateContext.Provider>;
};

export const useSSIMState = (): SSIMState => {
    const context = useContext(SSIMStateContext);

    if (context === null) {
        throw new Error('useSSIMState must be used within a SSIMStateProvider');
    }

    return context;
};
