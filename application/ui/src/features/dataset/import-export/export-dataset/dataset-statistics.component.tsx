// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { DatasetStatistics } from '@/components/dataset-statistics/dataset-statistics.component';
import { useTranslation } from '@/i18n';

import { useGetDatasetItems } from '../../../../hooks/use-get-dataset-items.hook';

export const MainDatasetStatistics = () => {
    const { t } = useTranslation();
    const { totalCount: totalMediaItems } = useGetDatasetItems();
    const { totalCount: totalAnnotatedItems } = useGetDatasetItems({ annotationStatus: 'with_annotations' });

    return (
        <DatasetStatistics
            label={t('dataset.export.itemsLabel')}
            totalMediaItems={totalMediaItems}
            totalAnnotatedItems={totalAnnotatedItems}
        />
    );
};
