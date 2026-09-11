// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { Evaluation } from '@/api/types';
import { useTranslation } from '@/i18n';
import { Grid, Text } from '@geti-ui/ui';

import { Box } from '../components/box/box.component';
import { getTestingMetrics } from '../components/model-row/utils';

const formatEvaluationValue = (value: number): string => {
    return `${(value * 100).toFixed(1)}%`;
};

type ModelEvaluationMetrics = {
    evaluations: Evaluation[];
};

export const ModelEvaluations = ({ evaluations }: ModelEvaluationMetrics) => {
    const { t } = useTranslation();
    const testingMetrics = getTestingMetrics(evaluations);

    if (testingMetrics.length === 0) {
        return (
            <Box
                title={t('models.metrics.evaluationsTitle')}
                content={
                    <Text UNSAFE_style={{ color: 'var(--spectrum-global-color-gray-900)' }}>
                        {t('models.metrics.noEvaluations')}
                    </Text>
                }
            />
        );
    }

    return (
        <Grid columns={['1fr', '1fr', '1fr']} gap={'size-200'}>
            {testingMetrics.map(({ name, value }) => (
                <Box
                    key={name}
                    title={name}
                    content={
                        <Text UNSAFE_style={{ color: 'var(--spectrum-global-color-gray-900)' }}>
                            {formatEvaluationValue(value)}
                        </Text>
                    }
                />
            ))}
        </Grid>
    );
};
