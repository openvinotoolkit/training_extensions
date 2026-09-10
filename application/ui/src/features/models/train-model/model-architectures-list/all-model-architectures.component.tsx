// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import type { ModelArchitecture as ModelArchitectureType } from '@/api/types';
import { Flex } from '@geti-ui/ui';
import { partition } from 'lodash-es';

import { SortModelArchitectures } from '../sort-model-architectures/sort-model-architectures.component';
import { SORT_OPTIONS, SORTING_HANDLERS, SortingOptions } from '../sort-model-architectures/utils';
import { TIMM_MODEL_ARCHITECTURE_ID } from '../timm-model-configuration/utils';
import { DetailedModelArchitecture } from './model-architecture.component';
import { ModelArchitecturesListLayout } from './model-architectures-list-layout/model-architectures-list-layout.component';

type AllModelArchitecturesProps = {
    modelArchitectures: ModelArchitectureType[];
    selectedModelArchitectureId: string | null;
    onSelectedModelArchitectureIdChange: (modelArchitectureId: string | null) => void;
};

export const AllModelArchitectures = ({
    modelArchitectures,
    onSelectedModelArchitectureIdChange,
    selectedModelArchitectureId,
}: AllModelArchitecturesProps) => {
    const [sortBy, setSortBy] = useState<SortingOptions>(SortingOptions.NAME_ASC);
    const [[timmCard], sortableModelArchitectures] = partition(
        modelArchitectures,
        (modelArchitecture) => modelArchitecture.id === TIMM_MODEL_ARCHITECTURE_ID
    );
    const sortedModelArchitectures = SORTING_HANDLERS[sortBy](sortableModelArchitectures);
    // The TIMM card always stays last, regardless of the active sort.
    const modelArchitecturesToRender =
        timmCard === undefined ? sortedModelArchitectures : [...sortedModelArchitectures, timmCard];

    return (
        <Flex direction={'column'} gap={'size-200'}>
            <SortModelArchitectures sortBy={sortBy} onSort={setSortBy} items={SORT_OPTIONS} />
            <ModelArchitecturesListLayout
                selectedModelArchitectureId={selectedModelArchitectureId}
                onSelectedModelArchitectureIdChange={onSelectedModelArchitectureIdChange}
                ariaLabel={'ALL model architectures'}
            >
                {modelArchitecturesToRender.map((modelArchitecture) => (
                    <DetailedModelArchitecture
                        key={modelArchitecture.id}
                        modelArchitecture={modelArchitecture}
                        selectedModelArchitectureId={selectedModelArchitectureId}
                        onSelectedModelArchitectureIdChange={onSelectedModelArchitectureIdChange}
                    />
                ))}
            </ModelArchitecturesListLayout>
        </Flex>
    );
};
