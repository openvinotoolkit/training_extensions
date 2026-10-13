// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, PropsWithChildren, useContext, useEffect } from 'react';

import { useProjectTask } from 'hooks/use-project-task.hook';

import { isDetectionTask } from '../../../../shared/task-type-guards';
import { useSelectedMediaItem } from '../../selected-media-item-provider.component';
import { useTool } from '../../tool-provider.component';
import { useSSIM } from './use-ssim.hook';

type SSIMState = ReturnType<typeof useSSIM>;

const SSIMStateContext = createContext<SSIMState | null>(null);

export const SSIMStateProvider = ({ children }: PropsWithChildren) => {
    const shapeType = isDetectionTask(useProjectTask()) ? 'rectangle' : 'polygon';
    const ssim = useSSIM(shapeType);
    const { activeTool } = useTool();
    const { mediaItem } = useSelectedMediaItem();
    const { reset } = ssim;

    useEffect(() => {
        reset();
    }, [activeTool, mediaItem, reset]);

    return <SSIMStateContext.Provider value={ssim}>{children}</SSIMStateContext.Provider>;
};

export const useSSIMState = (): SSIMState => {
    const context = useContext(SSIMStateContext);

    if (context === null) {
        throw new Error('useSSIMState must be used within a SSIMStateProvider');
    }

    return context;
};
