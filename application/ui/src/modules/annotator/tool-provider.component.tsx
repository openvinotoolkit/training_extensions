// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
    createContext,
    useContext,
    useEffect,
    useState,
    type Dispatch,
    type ReactNode,
    type SetStateAction,
} from 'react';

import type { TaskType } from '@/api/types';
import { useProject } from 'hooks/api/project.hook';

import { isClassificationTask, isSegmentationTask } from '../../shared/task-type-guards';
import type { ToolType } from './tool-type';

type ToolContextValue = {
    activeTool: ToolType | null;
    setActiveTool: Dispatch<SetStateAction<ToolType | null>>;
};

const ToolContext = createContext<ToolContextValue | null>(null);

export const useTool = (): ToolContextValue => {
    const context = useContext(ToolContext);

    if (context === null) {
        throw new Error('useTool must be used within a ToolProvider');
    }

    return context;
};

export const getDefaultTool = (taskType: TaskType | null): ToolType | null => {
    if (isClassificationTask(taskType)) {
        return null;
    }

    if (isSegmentationTask(taskType)) {
        return 'polygon';
    }

    return 'bounding-box';
};

type ToolProviderProps = {
    children: ReactNode;
};

export const ToolProvider = ({ children }: ToolProviderProps) => {
    const { data: selectedProject } = useProject();
    const [activeTool, setActiveTool] = useState<ToolType | null>(() => getDefaultTool(selectedProject.task.task_type));

    useEffect(() => {
        setActiveTool(getDefaultTool(selectedProject.task.task_type));
    }, [selectedProject.task.task_type]);

    return <ToolContext.Provider value={{ activeTool, setActiveTool }}>{children}</ToolContext.Provider>;
};
