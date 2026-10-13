// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useTranslation } from '@/i18n';
import { dimensionValue, Grid, Text } from '@geti-ui/ui';

export const RUNNING_JOB_GRID_COLUMNS = [
    'minmax(0, 2fr) minmax(0, 2fr) minmax(0, 2fr) minmax(auto, var(--spectrum-global-dimension-size-1000))',
];

export const RunningJobTableHeader = () => {
    const { t } = useTranslation();

    return (
        <Grid
            columns={RUNNING_JOB_GRID_COLUMNS}
            alignItems={'center'}
            width={'100%'}
            columnGap={'size-200'}
            UNSAFE_style={{
                backgroundColor: 'var(--spectrum-global-color-gray-200)',
                padding: `${dimensionValue('size-150')} ${dimensionValue('size-600')}
                    ${dimensionValue('size-150')} ${dimensionValue('size-1000')}`,
            }}
        >
            <Text>{t('models.columns.modelName')}</Text>
            <Text>{t('common.labels.architecture')}</Text>
            <Text>{t('common.labels.dataset')}</Text>
            <div />
        </Grid>
    );
};
