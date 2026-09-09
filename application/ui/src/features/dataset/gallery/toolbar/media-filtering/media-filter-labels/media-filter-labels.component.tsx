// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { FilterPopoverButton } from '@/components/filter-popover-button/filter-popover-button.component';
import { MultiSelectList } from '@/components/multi-select-list/multi-select-list.component';
import { useTranslation } from '@/i18n';
import { useDatasetFiltersSearchParams } from 'hooks/use-dataset-filters-search-params.hook';
import { useProjectLabels } from 'hooks/use-project-labels.hook';
import { isEmpty } from 'lodash-es';

export const MediaFilterLabels = () => {
    const { t } = useTranslation();
    const labels = useProjectLabels();
    const { selectedLabelIds, setSelectedLabelIds } = useDatasetFiltersSearchParams();

    const summary = isEmpty(selectedLabelIds)
        ? null
        : t('dataset.filters.labelsSelected', { count: selectedLabelIds.length });

    return (
        <FilterPopoverButton
            ariaLabel={t('dataset.filters.labels')}
            placeholder={t('dataset.filters.searchLabels')}
            summary={summary}
            minWidth='size-3000'
            dialogWidth='size-5000'
        >
            <MultiSelectList
                name='labels'
                items={labels}
                maxHeight='size-2000'
                selectAllLabel={t('dataset.filters.toggleAll')}
                onSelectionChange={setSelectedLabelIds}
                defaultSelectedKeys={new Set(selectedLabelIds)}
            />
        </FilterPopoverButton>
    );
};
