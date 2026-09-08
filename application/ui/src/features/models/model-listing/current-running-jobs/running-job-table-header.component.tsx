// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { dimensionValue, Grid, Text } from '@geti-ui/ui';

import { GroupByMode } from '../types';

export const RUNNING_JOB_GRID_COLUMNS = ['2fr 2fr minmax(auto, var(--spectrum-global-dimension-size-1000))'];

type RunningJobTableHeaderProps = {
    groupBy: GroupByMode;
};

export const RunningJobTableHeader = ({ groupBy }: RunningJobTableHeaderProps) => {
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
            <Text>{'Model Name'}</Text>
            <Text>{groupBy === 'architecture' ? 'Dataset' : 'Architecture'}</Text>
            <div />
        </Grid>
    );
};
