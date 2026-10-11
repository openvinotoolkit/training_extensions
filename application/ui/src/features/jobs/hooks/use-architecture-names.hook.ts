// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from 'react';

import { $api } from '@/api';
import type { Project } from '@/api/types';
import { useQueries } from '@tanstack/react-query';

export const useArchitectureNames = (projects: Project[]): ((architectureId: string) => string) => {
    const taskTypes = Array.from(new Set(projects.map(({ task }) => task.task_type)));

    const results = useQueries({
        queries: taskTypes.map((task) =>
            $api.queryOptions(
                'get',
                '/api/model_architectures',
                { params: { query: { task } } },
                { staleTime: Infinity }
            )
        ),
    });

    const names = useMemo(() => {
        const namesMap = new Map<string, string>();
        results.forEach(({ data }) => data?.model_architectures.forEach(({ id, name }) => namesMap.set(id, name)));

        return namesMap;
    }, [results]);

    return (architectureId: string) => names.get(architectureId) ?? architectureId;
};
