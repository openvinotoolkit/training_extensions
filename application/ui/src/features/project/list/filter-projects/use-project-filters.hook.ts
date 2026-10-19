// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useDeferredValue, useState } from 'react';

import type { Project, TaskType } from '@/api/types';

import { filterProjects } from './utils';

interface UseProjectFiltersResult {
    searchName: string;
    setSearchName: (value: string) => void;
    selectedTaskTypes: TaskType[];
    setSelectedTaskTypes: (taskTypes: TaskType[]) => void;
    onlyWithTrainedModels: boolean;
    setOnlyWithTrainedModels: (value: boolean) => void;
    filteredProjects: Project[];
    isFiltering: boolean;
}

export const useProjectFilters = (projects: Project[]): UseProjectFiltersResult => {
    const [searchName, setSearchName] = useState<string>('');
    const [selectedTaskTypes, setSelectedTaskTypes] = useState<TaskType[]>([]);
    const [onlyWithTrainedModels, setOnlyWithTrainedModels] = useState<boolean>(false);
    const deferredSearchName = useDeferredValue(searchName);

    const filteredProjects = filterProjects(projects, deferredSearchName, selectedTaskTypes, onlyWithTrainedModels);

    const isFiltering = searchName.trim() !== '' || selectedTaskTypes.length > 0 || onlyWithTrainedModels;

    return {
        searchName,
        setSearchName,
        selectedTaskTypes,
        setSelectedTaskTypes,
        onlyWithTrainedModels,
        setOnlyWithTrainedModels,
        filteredProjects,
        isFiltering,
    };
};
