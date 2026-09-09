// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Model } from '@/api/types';
import { Disclosure, DisclosurePanel, DisclosureTitle, Flex } from '@geti-ui/ui';

import { useGetTaskModelArchitectures } from '../../../hooks/api/use-get-model-architectures.hook';
import { ModelDetailsTabs } from '../../model-details/model-details-tabs.component';
import { useModelListing } from '../../provider/model-listing-provider';
import { ArchitectureGroup, DatasetGroup } from '../../types';
import { isFailedModel } from '../../utils/utils';
import { GroupHeader } from '../group-headers/group-header.component';
import { ModelRowContainer } from '../model-row/model-row-container.component';
import { ModelsTableHeader } from '../models-table-header.component';

import classes from './group-models-container.module.scss';

interface GroupModelsContainerProps {
    group: DatasetGroup | ArchitectureGroup;
    models: Model[];
}

export const GroupModelsContainer = ({ group, models }: GroupModelsContainerProps) => {
    const { expandedModelIds, onExpandModel } = useModelListing();
    const { modelArchitectures } = useGetTaskModelArchitectures();

    return (
        <Flex
            direction={'column'}
            UNSAFE_className={classes.groupModelsContainer}
            data-testid={`model-group-${group.id}`}
        >
            <GroupHeader data={group} modelArchitectures={modelArchitectures} />
            <ModelsTableHeader groupId={group.id} />

            {models.map((model) => {
                const modelId = model.id;
                const modelArchitecture = modelArchitectures.find(({ id }) => id === model.architecture);

                return (
                    <Disclosure
                        key={modelId}
                        isQuiet
                        UNSAFE_className={classes.disclosure}
                        isExpanded={expandedModelIds.has(modelId)}
                        isDisabled={isFailedModel(model)}
                        onExpandedChange={() => onExpandModel(modelId)}
                        data-testid={`model-disclosure-${modelId}`}
                    >
                        <DisclosureTitle UNSAFE_className={classes.disclosureItem}>
                            <ModelRowContainer model={model} modelArchitecture={modelArchitecture} />
                        </DisclosureTitle>
                        <DisclosurePanel aria-label={`Model details for ${model.name}`}>
                            <ModelDetailsTabs modelId={modelId} />
                        </DisclosurePanel>
                    </Disclosure>
                );
            })}
        </Flex>
    );
};
