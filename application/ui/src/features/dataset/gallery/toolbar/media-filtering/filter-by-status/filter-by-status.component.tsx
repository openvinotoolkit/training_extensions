// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { FilterByStatusKey } from '@/api/types';
import { useTranslation } from '@/i18n';
import { DimensionValue, Item, Picker } from '@geti-ui/ui';
import { useDatasetFiltersSearchParams } from 'hooks/use-dataset-filters-search-params.hook';

type FilterByStatusProps = {
    width?: DimensionValue;
};

export const FilterByStatus = ({ width }: FilterByStatusProps) => {
    const { t } = useTranslation();
    const { annotationStatus, setAnnotationStatus } = useDatasetFiltersSearchParams();
    const filterByStatusOptions: { name: string; key: FilterByStatusKey }[] = [
        { name: t('dataset.filters.allMedia'), key: 'all' },
        { name: t('dataset.filters.withAnnotations'), key: 'with_annotations' },
        { name: t('dataset.filters.missingAnnotations'), key: 'missing_annotations' },
    ];

    return (
        <Picker
            width={width}
            aria-label={'media status'}
            items={filterByStatusOptions}
            selectedKey={annotationStatus ?? filterByStatusOptions[0].key}
            onSelectionChange={(status) => setAnnotationStatus(status as FilterByStatusKey)}
        >
            {(item) => <Item>{item.name}</Item>}
        </Picker>
    );
};
