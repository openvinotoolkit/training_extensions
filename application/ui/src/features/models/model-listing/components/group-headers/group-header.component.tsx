// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { ModelArchitectureWithPerformanceCategory } from '@/api/types';

import { useModelListing } from '../../provider/model-listing-provider';
import type { ArchitectureGroup, DatasetGroup } from '../../types';
import { ArchitectureGroupHeader } from './architecture-group-header.component';
import { DatasetGroupHeader } from './dataset-group-header.component';

type GroupHeaderProps = {
    data: DatasetGroup | ArchitectureGroup;
    modelArchitectures: ModelArchitectureWithPerformanceCategory[];
};

export const GroupHeader = ({ data, modelArchitectures }: GroupHeaderProps) => {
    const { groupBy } = useModelListing();

    if (groupBy === 'dataset') {
        return <DatasetGroupHeader dataset={data as DatasetGroup} />;
    }

    const architecture = data as ArchitectureGroup;
    const modelArchitecture = modelArchitectures.find(({ id }) => id === architecture.id);

    return <ArchitectureGroupHeader architectureId={architecture.id} architecture={modelArchitecture} />;
};
