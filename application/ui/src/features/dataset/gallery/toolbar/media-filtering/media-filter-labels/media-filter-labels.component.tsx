// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { FilterPopoverButton } from '@/components/filter-popover-button/filter-popover-button.component';
import { MultiSelectList } from '@/components/multi-select-list/multi-select-list.component';
import { useDatasetFiltersSearchParams } from 'hooks/use-dataset-filters-search-params.hook';
import { useProjectLabels } from 'hooks/use-project-labels.hook';
import { isEmpty } from 'lodash-es';

import { pluralize } from '../../../../../../shared/util';

export const MediaFilterLabels = () => {
    const labels = useProjectLabels();
    const { selectedLabelIds, setSelectedLabelIds } = useDatasetFiltersSearchParams();

    const summary = isEmpty(selectedLabelIds)
        ? null
        : `${selectedLabelIds.length} ${pluralize(selectedLabelIds.length, 'label', 'labels')} selected`;

    return (
        <FilterPopoverButton
            ariaLabel='Filter by labels'
            placeholder='Search labels'
            summary={summary}
            minWidth='size-3000'
            dialogWidth='size-5000'
        >
            <MultiSelectList
                name='labels'
                items={labels}
                maxHeight='size-2000'
                selectAllLabel='Toggle all'
                onSelectionChange={setSelectedLabelIds}
                defaultSelectedKeys={new Set(selectedLabelIds)}
            />
        </FilterPopoverButton>
    );
};
