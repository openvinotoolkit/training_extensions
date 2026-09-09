// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Label } from '@/api/types';
import { useTranslation } from '@/i18n';
import { ActionButton, Divider, Flex } from '@geti-ui/ui';
import { useDatasetFiltersSearchParams } from 'hooks/use-dataset-filters-search-params.hook';
import { useProjectLabels } from 'hooks/use-project-labels.hook';
import { capitalize, isEmpty } from 'lodash-es';

import { formatDateRangeEnd, formatDateRangeStart } from '../../../../shared/date-utils';
import { isNonEmptyArray } from '../../../../shared/util';
import { FilterChips } from '../toolbar/media-filtering/filter-chips/filter-chips.component';

export const ActiveFiltersList = () => {
    const { t } = useTranslation();
    const labels = useProjectLabels();
    const {
        selectedLabelIds,
        setSelectedLabelIds,
        annotationStatus,
        setAnnotationStatus,
        startDate,
        setStartDate,
        endDate,
        setEndDate,
        selectedSubsets,
        setSelectedSubsets,
    } = useDatasetFiltersSearchParams();

    const handleRemoveLabel = (id: string) => {
        setSelectedLabelIds(selectedLabelIds.filter((selectedId) => selectedId !== id));
    };

    const selectedLabels = selectedLabelIds
        .map((id) => labels.find((label) => label.id === id))
        .filter(Boolean) as Label[];

    return (
        <>
            {selectedLabels.map((label) => (
                <FilterChips key={label.id} name={label.name} onClose={() => handleRemoveLabel(label.id)} />
            ))}

            {annotationStatus !== null && (
                <FilterChips
                    name={
                        annotationStatus === 'with_annotations'
                            ? t('dataset.filters.withAnnotations')
                            : t('dataset.filters.missingAnnotations')
                    }
                    onClose={() => setAnnotationStatus(null)}
                />
            )}

            {startDate !== null && (
                <FilterChips name={formatDateRangeStart(startDate)} onClose={() => setStartDate(null)} />
            )}

            {endDate !== null && <FilterChips name={formatDateRangeEnd(endDate)} onClose={() => setEndDate(null)} />}

            {isNonEmptyArray(selectedSubsets) &&
                selectedSubsets.map((subset) => (
                    <FilterChips
                        key={subset}
                        name={capitalize(subset)}
                        onClose={() => setSelectedSubsets(selectedSubsets.filter((sub) => sub !== subset))}
                    />
                ))}
        </>
    );
};

export const useHasActiveFilters = () => {
    const { selectedLabelIds, annotationStatus, startDate, endDate, selectedSubsets } = useDatasetFiltersSearchParams();

    return (
        !isEmpty(selectedLabelIds) ||
        annotationStatus !== null ||
        startDate !== null ||
        endDate !== null ||
        !isEmpty(selectedSubsets)
    );
};

export const useClearAllFilters = () => {
    const { setSelectedLabelIds, setAnnotationStatus, setStartDate, setEndDate, setSelectedSubsets } =
        useDatasetFiltersSearchParams();

    const handleClearAll = () => {
        setSelectedLabelIds([]);
        setAnnotationStatus(null);
        setStartDate(null);
        setEndDate(null);
        setSelectedSubsets([]);
    };

    return handleClearAll;
};

export const ActiveFilters = () => {
    const { t } = useTranslation();
    const hasActiveFilters = useHasActiveFilters();
    const handleClearAll = useClearAllFilters();

    if (!hasActiveFilters) {
        return null;
    }

    return (
        <Flex gap={'size-150'} wrap={'wrap'} alignItems={'center'} aria-label={t('dataset.filtersActive.active')}>
            <ActionButton isQuiet onPress={handleClearAll}>
                {t('dataset.filtersActive.clearAll')}
            </ActionButton>

            <Divider orientation={'vertical'} size={'S'} />

            <ActiveFiltersList />
        </Flex>
    );
};
