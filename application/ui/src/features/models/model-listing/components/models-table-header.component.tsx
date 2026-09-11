// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useMemo } from 'react';

import { useTranslation } from '@/i18n';
import { dimensionValue, Grid } from '@geti-ui/ui';
import { useProjectTask } from 'hooks/use-project-task.hook';

import { GRID_COLUMNS } from '../constants';
import { useModelListing } from '../provider/model-listing-provider';
import type { SortBy } from '../types';
import { DEFAULT_SORT } from '../utils/sorting';
import { ColumnHeader } from './column-header.component';
import { getPerformanceColumnAriaLabel, getPerformanceColumnLabel } from './model-row/utils';

// NOTE: We cannot have DisclosureGroup inside TableView when using Spectrum, so this grid mimics a table.
export const ModelsTableHeader = ({ groupId }: { groupId: string }) => {
    const { t } = useTranslation();
    const { groupBy, sortBy, onSortChange, groupedModels } = useModelListing();
    const taskType = useProjectTask();

    const groupSortBy = sortBy[groupId] ?? DEFAULT_SORT;
    const handleSortChange = (key: SortBy) => onSortChange(groupId, key);

    const performanceColumnLabel = useMemo(() => {
        const models = groupedModels.flatMap((group) => group.models);

        return getPerformanceColumnLabel(models, taskType, t);
    }, [groupedModels, taskType, t]);

    const performanceColumnAriaLabel = useMemo(() => {
        const models = groupedModels.flatMap((group) => group.models);

        return getPerformanceColumnAriaLabel(models, taskType);
    }, [groupedModels, taskType]);

    return (
        <Grid
            columns={GRID_COLUMNS}
            alignItems={'center'}
            width={'100%'}
            columnGap={'size-200'}
            UNSAFE_style={{
                backgroundColor: 'var(--spectrum-global-color-gray-200)',
                padding: `${dimensionValue('size-150')} ${dimensionValue('size-600')}
                    ${dimensionValue('size-150')} ${dimensionValue('size-1000')}`,
            }}
        >
            <ColumnHeader
                label={t('models.list.columns.modelName')}
                ariaLabel={'Model Name'}
                sortKey={'name'}
                sortBy={groupSortBy}
                onSortChange={handleSortChange}
            />
            <ColumnHeader
                label={t('models.list.columns.trained')}
                ariaLabel={'Trained'}
                sortKey={'trained'}
                sortBy={groupSortBy}
                onSortChange={handleSortChange}
            />
            <ColumnHeader
                label={
                    groupBy === 'architecture'
                        ? t('models.list.columns.dataset')
                        : t('models.list.columns.architecture')
                }
                ariaLabel={groupBy === 'architecture' ? 'Dataset' : 'Architecture'}
                sortKey={groupBy === 'architecture' ? 'dataset' : 'architecture'}
                sortBy={groupSortBy}
                onSortChange={handleSortChange}
            />
            <ColumnHeader
                label={t('models.list.columns.device')}
                ariaLabel={'Device'}
                sortKey={'device'}
                sortBy={groupSortBy}
                onSortChange={handleSortChange}
            />
            <ColumnHeader
                label={t('models.list.columns.totalSize')}
                ariaLabel={'Total size'}
                sortKey={'size'}
                sortBy={groupSortBy}
                onSortChange={handleSortChange}
            />
            <ColumnHeader
                label={performanceColumnLabel}
                ariaLabel={performanceColumnAriaLabel}
                sortKey={'score'}
                sortBy={groupSortBy}
                onSortChange={handleSortChange}
            />
            <div />
        </Grid>
    );
};
