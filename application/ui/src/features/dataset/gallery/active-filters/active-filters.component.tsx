// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Label } from '@/api/types';
import { useTranslation } from '@/i18n';
import { ActionButton, Divider, Flex } from '@geti-ui/ui';
import dayjs from 'dayjs';
import { useDatasetFiltersSearchParams } from 'hooks/use-dataset-filters-search-params.hook';
import { DATE_TIME_FORMAT, useFormatDate } from 'hooks/use-format-date.hook';
import { useProjectLabels } from 'hooks/use-project-labels.hook';
import { capitalize, isEmpty } from 'lodash-es';

import { SUBSET_LABEL_KEYS } from '../../../../shared/subsets';
import { isNonEmptyArray } from '../../../../shared/util';
import { FilterChips } from '../toolbar/media-filtering/filter-chips/filter-chips.component';

const formatFilterDate = (date: string): string => dayjs(date).format('DD/MM/YYYY HH:mm');

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
    const formatDate = useFormatDate(DATE_TIME_FORMAT);

    const handleRemoveLabel = (id: string) => {
        setSelectedLabelIds(selectedLabelIds.filter((selectedId) => selectedId !== id));
    };

    const selectedLabels = selectedLabelIds
        .map((id) => labels.find((label) => label.id === id))
        .filter(Boolean) as Label[];

    return (
        <>
            {selectedLabels.map((label) => (
                <FilterChips
                    key={label.id}
                    name={label.name}
                    ariaLabel={`Remove ${label.name} filter`}
                    onClose={() => handleRemoveLabel(label.id)}
                />
            ))}

            {annotationStatus !== null && (
                <FilterChips
                    name={
                        annotationStatus === 'with_annotations'
                            ? t('dataset.filters.withAnnotations')
                            : t('dataset.filters.missingAnnotations')
                    }
                    ariaLabel={`Remove ${
                        annotationStatus === 'with_annotations'
                            ? 'Media with annotations'
                            : 'Media with missing annotations'
                    } filter`}
                    onClose={() => setAnnotationStatus(null)}
                />
            )}

            {startDate !== null && (
                <FilterChips
                    name={t('dataset.filters.dateRange.from', { date: formatDate(startDate) ?? '-' })}
                    ariaLabel={`Remove From ${formatFilterDate(startDate)} filter`}
                    onClose={() => setStartDate(null)}
                />
            )}

            {endDate !== null && (
                <FilterChips
                    name={t('dataset.filters.dateRange.to', { date: formatDate(endDate) ?? '-' })}
                    ariaLabel={`Remove To ${formatFilterDate(endDate)} filter`}
                    onClose={() => setEndDate(null)}
                />
            )}

            {isNonEmptyArray(selectedSubsets) &&
                selectedSubsets.map((subset) => (
                    <FilterChips
                        key={subset}
                        name={t(SUBSET_LABEL_KEYS[subset])}
                        ariaLabel={`Remove ${capitalize(subset)} filter`}
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
    const { clearAllFilters } = useDatasetFiltersSearchParams();

    return clearAllFilters;
};

export const ActiveFilters = () => {
    const { t } = useTranslation();
    const hasActiveFilters = useHasActiveFilters();
    const handleClearAll = useClearAllFilters();

    if (!hasActiveFilters) {
        return null;
    }

    return (
        <Flex gap={'size-150'} wrap={'wrap'} alignItems={'center'} aria-label={'Active filters'}>
            <ActionButton isQuiet onPress={handleClearAll}>
                {t('common.actions.clearAll')}
            </ActionButton>

            <Divider orientation={'vertical'} size={'S'} />

            <ActiveFiltersList />
        </Flex>
    );
};
