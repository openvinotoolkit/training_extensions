// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Project } from '@/api/types';
import { orderBy } from 'lodash-es';

export const SORT_BY_OPTIONS = [
    [
        { nameKey: 'project.list.sort.nameAscending', key: 'name-ascending' },
        { nameKey: 'project.list.sort.nameDescending', key: 'name-descending' },
    ],
    [
        { nameKey: 'project.list.sort.createdAtDescending', key: 'createdAt-descending' },
        { nameKey: 'project.list.sort.createdAtAscending', key: 'createdAt-ascending' },
    ],
] as const;

export type SortBy = (typeof SORT_BY_OPTIONS)[number][number]['key'];

export const SORT_BY_HANDLERS: Record<SortBy, (projects: Project[]) => Project[]> = {
    'name-ascending': (projects) => orderBy(projects, (project) => project.name.toLocaleLowerCase(), 'asc'),
    'name-descending': (projects) => orderBy(projects, (project) => project.name.toLocaleLowerCase(), 'desc'),
    'createdAt-ascending': (projects) => orderBy(projects, (project) => project.created_at, 'asc'),
    'createdAt-descending': (projects) => orderBy(projects, (project) => project.created_at, 'desc'),
};
